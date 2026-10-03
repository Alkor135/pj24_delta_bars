"""Строит дельта-бары из сделок QUIK и сохраняет отдельный журнал восстановления.

Используется двумя графиками; самостоятельно запускать модуль не требуется.
Примеры из корня проекта:
    python chart_delta_bar_semafor_realtime_RTS.py --threshold 100
    python -m unittest -v tests.test_realtime_core
Tick rule, неделимая сделка и сброс дня повторяют исторический конвертер.
SQLite журнала не связана с исторической базой баров.
"""

from dataclasses import asdict, dataclass
from datetime import date
import math
from pathlib import Path
import sqlite3
from threading import RLock

from source.delta_core import NS_SECOND, format_time


def positive_integer(value, name, minimum=1):
    """Возвращает целое value >= minimum; name используется в сообщении ошибки."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: требуется целое число")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer() or abs(value) >= 2**53):
        raise ValueError(f"{name}: дробное или неточное целое число")
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: требуется целое число") from exc
    if isinstance(value, str) and value != str(result):
        raise ValueError(f"{name}: требуется десятичное целое число")
    if result < minimum:
        raise ValueError(f"{name}: требуется число не меньше {minimum}")
    return result


@dataclass(frozen=True)
class Trade:
    """Хранит дату МСК, наносекунды дня, точный номер, контракт, цену и qty."""

    day: str
    time_ns: int
    trade_num: str
    class_code: str
    sec_code: str
    price: float
    qty: int

    def __post_init__(self):
        """Проверяет поля новой сделки; некорректные числа/даты вызывают ValueError."""
        date.fromisoformat(self.day)
        time_ns = positive_integer(self.time_ns, "Время сделки", 0)
        if time_ns >= 86400 * NS_SECOND:
            raise ValueError("Время сделки выходит за границы дня")
        number = positive_integer(self.trade_num, "Номер сделки")
        qty = positive_integer(self.qty, "Количество контрактов")
        price = float(self.price)
        if not math.isfinite(price) or price <= 0 or not self.class_code or not self.sec_code:
            raise ValueError("Нужны положительная цена, класс и код контракта")
        object.__setattr__(self, "time_ns", time_ns)
        object.__setattr__(self, "trade_num", str(number))
        object.__setattr__(self, "qty", qty)
        object.__setattr__(self, "price", price)

    @property
    def key(self):
        """Возвращает идентификатор сделки с учётом даты, класса и контракта."""
        return self.day, self.class_code, self.sec_code, self.trade_num

    @property
    def order(self):
        """Возвращает ключ порядка: дата, биржевое время и числовой номер сделки."""
        return self.day, self.time_ns, int(self.trade_num)

    def to_dict(self):
        """Возвращает JSON-совместимые поля сделки без преобразования номера в float."""
        return asdict(self)


def parse_trade(data, price_scale=1):
    """Преобразует таблицу OnAllTrade data в Trade; price_scale задаёт масштаб цены.

    Используется qty (контракты), а не денежный value. datetime интерпретируется
    как время биржи по Москве. mcs — микросекундная часть секунды, если доступна;
    иначе используется ms. Некорректная таблица вызывает ValueError.
    """
    try:
        stamp = data["datetime"]
        day = date(int(stamp["year"]), int(stamp["month"]), int(stamp["day"])).isoformat()
        hour, minute, second = (positive_integer(stamp[k], "Время сделки", 0) for k in ("hour", "min", "sec"))
        if hour > 23 or minute > 59 or second > 59:
            raise ValueError("Неверное биржевое время")
        microseconds = (positive_integer(stamp["mcs"], "Микросекунды", 0) if stamp.get("mcs") is not None
                        else positive_integer(stamp.get("ms", 0), "Миллисекунды", 0) * 1000)
        if microseconds >= 1_000_000:
            raise ValueError("Неверная дробная часть времени")
        scale = float(price_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Масштаб цены должен быть положительным")
        return Trade(day, (hour * 3600 + minute * 60 + second) * NS_SECOND + microseconds * 1000,
                     data["trade_num"], data["class_code"], data["sec_code"], float(data["price"]) * scale, data["qty"])
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Неполная таблица обезличенной сделки QUIK") from exc


class DayBuilder:
    """Добавляет сделки в день, пересчитывая его только при появлении поздней сделки."""

    def __init__(self, day, threshold, symbol="RTS"):
        """Создаёт пустой день day с постоянным threshold и логическим symbol."""
        self.day = date.fromisoformat(str(day)).isoformat()
        self.threshold = positive_integer(threshold, "Порог дельты")
        self.symbol = symbol
        self.trades = {}
        self._ordered = []
        self._reset()

    def _reset(self):
        """Сбрасывает бары и tick rule для повторного расчёта текущего дня."""
        self._closed, self._current = [], None
        self._previous_price, self._direction, self._row = None, 0, 0

    def _consume(self, trade):
        """Учитывает одну уже упорядоченную trade; закрывает бар при достижении порога."""
        self._row += 1
        if self._previous_price is not None and trade.price != self._previous_price:
            self._direction = 1 if trade.price > self._previous_price else -1
        if self._current is None:
            self._current = dict(symbol=self.symbol, day=self.day, bar_index=len(self._closed),
                start_time=format_time(self.day, trade.time_ns), end_time=format_time(self.day, trade.time_ns),
                start_row=self._row, end_row=self._row, open=trade.price, high=trade.price, low=trade.price,
                close=trade.price, volume=0, delta=0, up_volume=0, down_volume=0, neutral_volume=0,
                tick_count=0, threshold=self.threshold, is_complete=0, close_reason="live")
        bar = self._current
        bar["high"], bar["low"], bar["close"] = max(bar["high"], trade.price), min(bar["low"], trade.price), trade.price
        bar["end_time"], bar["end_row"] = format_time(self.day, trade.time_ns), self._row
        bar["volume"] += trade.qty
        bar["delta"] += self._direction * trade.qty
        bar["tick_count"] += 1
        bar[{1: "up_volume", -1: "down_volume", 0: "neutral_volume"}[self._direction]] += trade.qty
        self._previous_price = trade.price
        if abs(bar["delta"]) >= self.threshold:
            bar.update(is_complete=1, close_reason="threshold")
            self._closed.append(bar)
            self._current = None

    def ingest(self, trades):
        """Принимает Trade из trades; возвращает True при новых данных, исключает повторы.

        Поздние сделки вызывают повторный расчёт всего дня. Другой день, другая
        серия или противоречивое содержимое уже известного номера запрещены.
        """
        new = {}
        for trade in trades:
            if trade.day != self.day:
                raise ValueError("Сделка относится к другому календарному дню")
            if self._ordered and (trade.class_code, trade.sec_code) != (self._ordered[0].class_code, self._ordered[0].sec_code):
                raise ValueError("Нельзя смешивать контракты внутри дня")
            previous = self.trades.get(trade.key, new.get(trade.key))
            if previous is not None and previous != trade:
                raise ValueError("Номер сделки уже существует с другими данными")
            if previous is None:
                new[trade.key] = trade
        if not new:
            return False
        incoming = sorted(new.values(), key=lambda item: item.order)
        if len({(t.class_code, t.sec_code) for t in incoming}) != 1:
            raise ValueError("Нельзя смешивать контракты внутри дня")
        late = bool(self._ordered and incoming[0].order < self._ordered[-1].order)
        self.trades.update(new)
        self._ordered.extend(incoming)
        if late:
            self._ordered.sort(key=lambda item: item.order)
            self._reset()
            incoming = self._ordered
        for trade in incoming:
            self._consume(trade)
        return True

    def snapshot(self, final=False):
        """Возвращает копии баров; final помечает остаток как day_end вместо live."""
        bars = [dict(bar) for bar in self._closed]
        if self._current is not None:
            bars.append(dict(self._current, close_reason="day_end" if final else "live"))
        return bars


class TradeJournal:
    """Хранит сделки отдельно от истории, выдавая новые записи по курсору вставки."""

    def __init__(self, path):
        """Открывает/создаёт журнал path; разрешает безопасное обращение из потоков."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, day TEXT NOT NULL,
            time_ns INTEGER NOT NULL, trade_num TEXT NOT NULL, class_code TEXT NOT NULL,
            sec_code TEXT NOT NULL, price REAL NOT NULL, qty INTEGER NOT NULL,
            UNIQUE(symbol,day,class_code,sec_code,trade_num))""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS trades_symbol_id ON trades(symbol,id)")
        self.connection.commit()

    def append(self, symbol, trades):
        """Записывает новые trades для symbol и возвращает число; конфликт номера откатывает пакет."""
        with self.lock, self.connection:
            before = self.connection.total_changes
            for trade in trades:
                cursor = self.connection.execute("""INSERT OR IGNORE INTO trades
                    (symbol,day,time_ns,trade_num,class_code,sec_code,price,qty) VALUES (?,?,?,?,?,?,?,?)""",
                    (symbol, trade.day, trade.time_ns, trade.trade_num, trade.class_code, trade.sec_code, trade.price, trade.qty))
                if cursor.rowcount == 0:
                    row = self.connection.execute("""SELECT * FROM trades WHERE
                        symbol=? AND day=? AND class_code=? AND sec_code=? AND trade_num=?""",
                        (symbol, trade.day, trade.class_code, trade.sec_code, trade.trade_num)).fetchone()
                    if {key: row[key] for key in Trade.__dataclass_fields__} != trade.to_dict():
                        raise ValueError(f"Сделка {trade.trade_num} повторилась с другими данными")
            return self.connection.total_changes - before

    def read(self, symbol, after=0, limit=20000, day=None):
        """Возвращает trades/cursor/more для symbol после after; day ограничивает дату."""
        after = positive_integer(after, "Курсор", 0)
        limit = positive_integer(limit, "Размер пакета")
        clause, params = (" AND day=?", [day]) if day else ("", [])
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM trades WHERE symbol=? AND id>?" + clause + " ORDER BY id LIMIT ?",
                [symbol, after, *params, limit + 1]).fetchall()
        selected = rows[:limit]
        return dict(cursor=selected[-1]["id"] if selected else after, more=len(rows) > limit,
                    trades=[{k: row[k] for k in Trade.__dataclass_fields__} for row in selected])

    def close(self):
        """Закрывает соединение журнала после завершения его потребителей."""
        with self.lock:
            self.connection.close()
