"""Подготовка событий исполнения из SQLite и проверенных тиковых ZIP Финама.

Запуск: python backtest/backtest_duration.py --symbols RTS MIX
Проверки: python -m unittest -v tests.test_duration_data
Исходные базы и архивы открываются только для чтения.
"""

from dataclasses import asdict, dataclass, replace
from contextlib import closing
from datetime import time
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import sqlite3
from zipfile import ZipFile
import numpy as np
import pandas as pd
from .duration_engine import Day, Event
from .chart_data import laguerre

CACHE_VERSION = 1


@dataclass(frozen=True)
class EntryFilter:
    """Выбирает дополнительный фильтр входа и параметр ALF просмотрщика."""
    mode: str = 'alf'
    alpha: float = 0.4

    def __post_init__(self):
        """Проверяет режим и допустимый диапазон параметра Лагерра."""
        if self.mode not in ('none', 'alf'):
            raise ValueError('Фильтр входа должен быть none или alf')
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError('Параметр ALF α должен быть больше 0 и не больше 1')

    def describe(self, reverse=False):
        """Описывает фактическую сторону сделки после обычного либо обратного входа."""
        rising, falling = ('Short', 'Long') if reverse else ('Long', 'Short')
        rule = (f'Длительность завершённого бара < порога входа; {rising} при close > open, '
                f'{falling} при close < open.')
        if self.mode == 'alf':
            rule += (f' Фильтр ALF (α={self.alpha:g}): {rising} только при close > ALF, '
                     f'{falling} только при close < ALF; при равенстве входа нет.')
        else:
            rule += ' Фильтр ALF отключён.'
        return rule + ' Исполнение на следующем тике. Выход по длительности не зависит от ALF.'


def apply_entry_filter(day, bars, settings):
    """Фильтрует входы по завершённым барам, сохраняя выходы и исходные события кэша."""
    if bars.bar_index.duplicated().any():
        raise ValueError(f'{day.date}: повторные номера сигнальных баров')
    lookup = bars.set_index('bar_index')
    if settings.mode == 'alf' and ('alf' not in lookup or not np.isfinite(lookup.alf).all()):
        raise ValueError(f'{day.date}: отсутствуют конечные значения ALF')
    events = []
    for event in day.events:
        if event.signal_bar < 0:
            events.append(replace(event))
            continue
        if event.signal_bar not in lookup.index:
            raise ValueError(f'{day.date}: не найден сигнальный бар {event.signal_bar}')
        bar = lookup.loc[event.signal_bar]
        close = float(bar.close)
        alf = float(bar.alf) if settings.mode == 'alf' else None
        if not np.isfinite(close):
            raise ValueError(f'{day.date}: некорректное закрытие сигнального бара')
        allowed = (alf is None or (event.direction > 0 and close > alf)
                   or (event.direction < 0 and close < alf))
        events.append(replace(event, enter=bool(event.enter and allowed),
                              signal_close=close, signal_alf=alf))
    return Day(day.date, events)


@dataclass(frozen=True)
class Session:
    """Задаёт московские часы сигнала, прекращения входов и дневного выхода."""
    start: str = '10:00:00'
    entry_end: str = '18:30:00'
    close: str = '18:40:00'

    def __post_init__(self):
        """Проверяет порядок часов и приводит их к полному формату."""
        for field in ('start','entry_end','close'):
            value=time.fromisoformat(getattr(self,field))
            if value.tzinfo or value.microsecond:
                raise ValueError('Используйте московские часы HH:MM:SS без долей секунды')
            object.__setattr__(self,field,value.isoformat())
        if not self.start < self.entry_end < self.close:
            raise ValueError('Нужно начало < прекращение входов < закрытие')


def read_tick_zip(path, expected_sha):
    """Проверяет SHA256 архива и читает его единственный CSV без изменения файлов."""
    payload=Path(path).read_bytes()
    if sha256(payload).hexdigest()!=expected_sha:
        raise ValueError(f'SHA256 не совпал с журналом базы: {path}. Сначала обновите бары.')
    with ZipFile(io.BytesIO(payload)) as archive:
        names=[x for x in archive.namelist() if x.lower().endswith('.csv')]
        if len(names)!=1:
            raise ValueError(f'Ожидался один CSV: {path}')
        with archive.open(names[0]) as stream:
            return pd.read_csv(stream,usecols=['datetime','last'],dtype={'datetime':str,'last':float})


def build_day(date, bars, ticks, session):
    """Связывает бары с исходными строками и находит исполнимое закрытие по времени."""
    audit=dict(date=date,status='included',reason='',bars=len(bars),ticks=len(ticks))
    if bars.empty or ticks.empty:
        return None,dict(audit,status='excluded',reason='Нет баров или тиков')
    bars=bars.sort_values('bar_index').reset_index(drop=True)
    starts=bars.start_row.to_numpy(dtype=int)-1
    ends=bars.end_row.to_numpy(dtype=int)-1
    if (starts<0).any() or (ends>=len(ticks)).any() or (ends<starts).any():
        raise ValueError(f'{date}: неверные номера строк баров')
    if len(starts)>1 and not np.array_equal(starts[1:],ends[:-1]+1):
        raise ValueError(f'{date}: нарушена последовательность строк баров')
    prices=ticks['last'].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices<=0).any():
        raise ValueError(f'{date}: некорректные цены тиков')
    if not np.array_equal(prices[starts],bars.open.to_numpy()) or not np.array_equal(prices[ends],bars.close.to_numpy()):
        raise ValueError(f'{date}: цены тиков не соответствуют базе')
    stamps=ticks.datetime.astype(str).to_numpy()
    seconds=np.asarray([x[:19] for x in stamps])
    # Проверка секунд не зависит от порядка искусственных дробных частей.
    if np.any(seconds[1:]<seconds[:-1]) or any(x[:10]!=date for x in seconds):
        raise ValueError(f'{date}: нарушены дата или порядок тиков')
    for source_rows,column in ((starts,'start_time'),(ends,'end_time')):
        actual=pd.to_datetime(stamps[source_rows],format='ISO8601').to_numpy()
        expected=pd.to_datetime(bars[column],format='ISO8601').to_numpy()
        if not np.array_equal(actual,expected):
            raise ValueError(f'{date}: время тиков не соответствует базе')
    target=f'{date} {session.close}'
    closing=int(np.searchsorted(seconds,target,side='left'))
    if closing==len(ticks):
        return None,dict(audit,status='excluded',reason=f'Нет тика в {session.close} или позже')
    delay=int((pd.Timestamp(str(seconds[closing]))-pd.Timestamp(target)).total_seconds())
    audit.update(close_time=stamps[closing],close_row=closing+1,close_delay_seconds=delay)
    duration=((pd.to_datetime(seconds[ends])-pd.to_datetime(seconds[starts])).total_seconds()).astype(int)
    direction=np.sign(bars.close.to_numpy()-bars.open.to_numpy()).astype(int)
    complete=bars.is_complete.to_numpy()==1
    events=[]
    start_limit=f'{date} {session.start}'
    entry_limit=f'{date} {session.entry_end}'
    for i,(a,b) in enumerate(zip(starts,ends)):
        if a>=closing:
            break
        if seconds[a]>=start_limit:
            valid=i>0 and complete[i-1] and seconds[starts[i-1]]>=start_limit
            # Нельзя входить/выходить по запоздавшему сигналу после дневного отсечения.
            valid=bool(valid and seconds[ends[i-1]]<target and seconds[a]<target)
            events.append(Event(stamps[a],int(a+1),float(prices[a]),
                int(duration[i-1]) if valid else -1,int(direction[i-1]) if valid else 0,
                bool(valid and seconds[a]<entry_limit),valid,False,
                int(bars.iloc[i-1].bar_index) if valid else -1))
        if b<closing and seconds[b]>=start_limit:
            events.append(Event(stamps[b],int(b+1),float(prices[b])))
    events.append(Event(stamps[closing],closing+1,float(prices[closing]),force=True))
    return Day(date,events),audit


def prepare_database(path, symbol, session, cache_dir, start='2022-01-01', end='9999-12-31', dataset_id=None,
                     progress=print, entry_filter=None):
    """Проверяет сырые события и применяет ALF с прогревом на истории до начала теста."""
    entry_filter = entry_filter or EntryFilter()
    path=Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Нет базы: {path}')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        sets=db.execute('SELECT DISTINCT dataset_id FROM bars WHERE symbol=?',(symbol,)).fetchall()
        if dataset_id is None:
            if len(sets)!=1:
                raise ValueError(f'{symbol}: выберите --dataset-id, найдено наборов {len(sets)}')
            dataset_id=sets[0][0]
        config=db.execute('SELECT config_json FROM datasets WHERE dataset_id=?',(dataset_id,)).fetchone()
        if config is None:
            raise ValueError('Набор параметров отсутствует')
        config=json.loads(config[0])
        bars=pd.read_sql_query('SELECT * FROM bars WHERE symbol=? AND dataset_id=? AND day>=? AND day<=? ORDER BY day,bar_index',
                              db,params=(symbol,dataset_id,start,end))
        sources=pd.read_sql_query("SELECT * FROM days WHERE symbol=? AND dataset_id=? AND day>=? AND day<=? AND status='ready' ORDER BY day",
                                 db,params=(symbol,dataset_id,start,end))
        history = None
        if entry_filter.mode == 'alf':
            history = pd.read_sql_query('SELECT day,bar_index,close FROM bars WHERE symbol=? AND dataset_id=? '
                'AND day<=? ORDER BY day,bar_index', db, params=(symbol, dataset_id, end))
    if bars.empty:
        raise ValueError(f'{symbol}: нет баров в заданном диапазоне')
    alf_by_day = {}
    if history is not None:
        if history.duplicated(['day', 'bar_index']).any() or not np.isfinite(history.close).all():
            raise ValueError(f'{symbol}: некорректная история для расчёта ALF')
        # Ровно та же рекурсия, порядок и дневные остатки, что в просмотрщике.
        history['alf'] = laguerre(history.close.to_numpy(), entry_filter.alpha)
        alf_by_day = {day: frame.set_index('bar_index').alf for day, frame in history.groupby('day', sort=False)}
    grouped={day:frame for day,frame in bars.groupby('day',sort=False)}
    prepared,audits=[],[]
    cache_dir=Path(cache_dir)/symbol
    cache_dir.mkdir(parents=True,exist_ok=True)
    for index,source in enumerate(sources.itertuples()):
        if source.day not in grouped:
            continue
        frame=grouped[source.day]
        # Проверяется содержимое ZIP даже при попадании в кэш: изменённые тики не скрываются.
        archive_path=Path(source.file_path)
        digest=sha256(archive_path.read_bytes()).hexdigest()
        if digest!=source.sha256:
            raise ValueError(f'SHA256 не совпал: {archive_path}. Пересоздайте затронутые бары.')
        encoded=frame.to_json(orient='split',double_precision=15)
        key=sha256(json.dumps([CACHE_VERSION,digest,encoded,asdict(session)],sort_keys=True).encode()).hexdigest()
        cache=cache_dir/(source.day+'_'+key+'.json.gz')
        if cache.is_file():
            with gzip.open(cache,'rt',encoding='utf-8') as f:
                saved=json.load(f)
            day=Day(source.day,[Event(**x) for x in saved['events']]) if saved['events'] is not None else None
            audit=saved['audit']
        else:
            ticks=read_tick_zip(archive_path,source.sha256)
            if len(ticks)!=source.tick_count:
                raise ValueError(f'{source.day}: число тиков не совпало с базой')
            day,audit=build_day(source.day,frame,ticks,session)
            temp=cache.with_suffix('.tmp')
            with gzip.open(temp,'wt',encoding='utf-8') as f:
                json.dump(dict(audit=audit,events=[asdict(x) for x in day.events] if day else None),f,ensure_ascii=False)
            temp.replace(cache)
        if day is not None:
            # На диске остаются исходные события без фильтра. Сторона ALF всегда
            # проверяется заново, в том числе после изменения α или истории прогрева.
            signal_bars = frame.assign(alf=frame.bar_index.map(alf_by_day[source.day])) if history is not None else frame
            prepared.append(apply_entry_filter(day, signal_bars, entry_filter))
        audits.append(audit)
        if progress and ((index+1)%100==0 or index+1==len(sources)):
            progress(f'{symbol}: проверены источники {index+1}/{len(sources)}',flush=True)
    if not prepared:
        raise ValueError('Нет дней с исполнимым закрытием по времени')
    metadata=dict(db=str(path),symbol=symbol,dataset_id=dataset_id,bar_config=config,
        entry_filter=asdict(entry_filter),entry_rule=entry_filter.describe(),
        session=asdict(session),source_first_day=str(bars.day.min()),source_last_day=str(bars.day.max()),
        source_bars=len(bars),included_days=len(prepared),excluded_days=len(audits)-len(prepared),
        source_manifest_sha256=sha256(sources[['day','sha256']].to_json(orient='records').encode()).hexdigest(),
        bars_sha256=sha256(bars.to_json(orient='split',double_precision=15).encode()).hexdigest())
    if history is not None:
        metadata.update(alf_history_start=str(history.day.iloc[0]),
            alf_warmup_bars=int((history.day < start).sum()),
            alf_history_sha256=sha256(history[['day','bar_index','close']].to_json(
                orient='records',double_precision=15).encode()).hexdigest())
    return prepared,audits,metadata
