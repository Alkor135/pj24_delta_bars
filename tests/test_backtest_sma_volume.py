r"""Проверяет причинный фильтр объёма, SMA и тиковое исполнение бэктеста.

Запуск из корня:
    .\.venv\Scripts\python.exe -m unittest -v tests.test_backtest_sma_volume

Ручные примеры ловят использование будущего объёма, вход внутри сигнального
бара, повторный фильтр переворота, неверные затраты и потерю внутритиковой
просадки. Пустой следующий бар не допускает запоздавший вход.
"""

import importlib.util
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

if importlib.util.find_spec("source.sma_volume_engine"):
    from source.sma_volume_engine import accumulated_volumes, sma_frame, make_signals, run_day, paired_bootstrap

NS = 1_000_000_000
START = 36000 * NS
CLOSE = 67500 * NS
WIDTH = 300 * NS


def target(index, side, passed=True, signal_ns=START+WIDTH):
    """Возвращает сигнал с индексом исполнения index, стороной side и фильтром passed."""
    return dict(entry_index=index, side=side, volume_pass=passed, signal_ns=signal_ns,
                signal_bar=1, current_volume=100, median_volume=90, volume_ratio=100/90)


class SmaVolumeTests(unittest.TestCase):
    """Проверяет модель по независимым ручным ожиданиям."""

    def setUp(self):
        """Требует наличие движка до исполнения поведенческих проверок."""
        self.assertIsNotNone(importlib.util.find_spec("source.sma_volume_engine"))

    def test_volume_stops_before_decision_and_starts_at_ten(self):
        """Тик нового бара и будущие объёмы не попадают в решение закрытого бара."""
        times=np.array([START-1,START,START+WIDTH-1,START+WIDTH,START+WIDTH+1])
        result=accumulated_volumes(times,np.array([999,2,3,100,1000]))
        self.assertEqual(result[0],5)
        self.assertEqual(result[1],1105)

    def test_sma_formula_and_future_causality(self):
        """SMA 3/34 совпадают с ручными средними, будущая цена не меняет прошлые значения."""
        bars=pd.DataFrame(dict(day=["2026-01-05"]*102,close=np.arange(1,103),segment=0))
        old=sma_frame(bars)
        self.assertEqual(old.sma3.iloc[33],33)
        self.assertEqual(old.sma34.iloc[33],17.5)
        bars.loc[101,"close"]=10000
        new=sma_frame(bars)
        pd.testing.assert_frame_equal(old.iloc[:101],new.iloc[:101])
        self.assertFalse(old.warmed.iloc[99])
        self.assertTrue(old.warmed.iloc[100])

    def test_equalities_bridge_crossings_but_night_does_not(self):
        """Равенство SMA между знаками сохраняет пересечение, ночной переход его не создаёт."""
        bars=pd.DataFrame(dict(day=["2026-01-05"]*105,close=np.r_[np.full(100,100),101,99,98,100,103],segment=0))
        result=sma_frame(bars)
        self.assertEqual(result.cross.iloc[100:].tolist(),[0,0,-1,0,1])
        bars.loc[104,"day"]="2026-01-06"
        self.assertEqual(sma_frame(bars).cross.iloc[104],0)

    def test_loader_rejects_changed_ohlcv_cache(self):
        """Изменённый high свечи должен остановить запуск до моделирования сделок."""
        self.assertIsNotNone(importlib.util.find_spec("backtest.backtest_sma_volume"))
        from backtest.backtest_sma_volume import load_verified_day
        from research.volume_trend_data import time_bars
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1],prefix="sma_volume_test_") as directory:
            root=Path(directory)
            cache=root/"cache"/"RTS"
            cache.mkdir(parents=True)
            archive=root/"20260105.zip"
            with ZipFile(archive,"w") as output:
                output.writestr("ticks.csv","datetime,last,volume\n2026-01-05 10:00:00.000000000,100,1\n2026-01-05 10:00:00.000000001,110,2\n2026-01-05 18:45:00.000000000,100,1\n")
            digest=sha256(archive.read_bytes()).hexdigest()
            times=np.array([START,START+1,CLOSE])
            prices=np.array([100.,110.,100.])
            volumes=np.array([1,2,1])
            np.savez_compressed(cache/"20260105.npz",times=times,prices=prices,volumes=volumes,sha256=digest)
            bars=time_bars(times,prices,volumes,5)
            bars.loc[0,"high"]=120
            bars.to_pickle(cache/"20260105_5m.pkl")
            row=pd.Series(dict(day="2026-01-05",file=str(archive),sha256=digest,tick_count=3,volume=4,first_ns=START,last_ns=CLOSE))
            with self.assertRaisesRegex(ValueError,"свеч"):
                load_verified_day(root,"RTS",row)

    def test_signal_uses_same_time_median_and_next_bar_tick(self):
        """Вход следует после границы бара, медиана берётся именно из того же времени 20 дней."""
        bars=pd.DataFrame(dict(start_ns=[START],end_ns=[START+WIDTH-1],cross=[1],
                               sma3=[101.],sma34=[100.],bar_index=[1],warmed=[True]))
        times=np.array([START,START+WIDTH-1,START+WIDTH, CLOSE])
        curves=np.tile(np.arange(104)+5,(20,1))
        curves[10:,:]+=2
        signals=make_signals(bars,times,np.repeat(7,104),curves)
        self.assertEqual(len(signals),1)
        self.assertEqual(signals[0]["median_volume"],6)
        self.assertEqual(signals[0]["entry_index"],2)
        self.assertTrue(signals[0]["volume_pass"])
        signals=make_signals(bars,times,np.repeat(6,104),curves)
        self.assertFalse(signals[0]["volume_pass"])

    def test_next_empty_bar_cannot_execute_two_bars_later(self):
        """Без тика в следующей свече старый сигнал не исполняется в последующей."""
        bars=pd.DataFrame(dict(start_ns=[START],end_ns=[START+WIDTH-1],cross=[1],
                               sma3=[101.],sma34=[100.],bar_index=[1],warmed=[True]))
        times=np.array([START,START+2*WIDTH,CLOSE])
        self.assertEqual(make_signals(bars,times,np.repeat(10,104),np.ones((20,104))),[])

    def test_first_entry_filtered_then_reversals_ignore_volume(self):
        """Первый ложный фильтр запрещает вход; обратный сигнал после входа переворачивает."""
        times=np.array([START,START+WIDTH,START+2*WIDTH,START+3*WIDTH,CLOSE])
        result=run_day("2026-01-05",times,np.array([100,105,110,108,100]),
                       [target(1,-1,False),target(2,1,True),target(3,-1,False)],1,4,"volume")
        self.assertEqual(len(result["trades"]),2)
        a,b=result["trades"]
        self.assertEqual((a["entry_row"],a["entry_price"],a["exit_row"],a["gross_pnl"]),(3,110,4,-2))
        self.assertEqual((b["entry_row"],b["side"],b["gross_pnl"]),(4,"Short",8))
        self.assertEqual(result["daily"]["net_pnl"],-2)
        self.assertEqual(b["reason"],"Закрытие сессии")

    def test_closing_priority_and_tick_drawdown(self):
        """Сигнал в момент закрытия не открывает сделку; просадка включает движение между сигналами."""
        times=np.array([START,START+WIDTH,START+WIDTH+1,START+WIDTH+2,CLOSE,CLOSE+1])
        result=run_day("2026-01-05",times,np.array([100,100,120,90,110,500]),
                       [target(1,1),target(4,-1,signal_ns=CLOSE)],1,4,"baseline")
        self.assertEqual(len(result["trades"]),1)
        self.assertEqual(result["trades"][0]["exit_price"],110)
        self.assertEqual(result["daily"]["net_pnl"],6)
        self.assertEqual(result["daily"]["max_drawdown"],30)

    def test_zero_trades_remain_a_daily_observation(self):
        """День без разрешённых входов остаётся нулевым исходом для парного сравнения."""
        result=run_day("2026-01-05",np.array([START,CLOSE]),np.array([100,105]),[],1,4,"volume")
        self.assertEqual(result["daily"]["net_pnl"],0)
        self.assertEqual(result["daily"]["trades"],0)

    def test_result_tables_preserve_modes_and_zero_days(self):
        """Сводка читает колонку mode, сохраняет нулевой день и парную разность -6."""
        from backtest.backtest_sma_volume import result_tables
        daily=pd.DataFrame([dict(day="2026-01-05",symbol="RTS",mode="baseline",cost_ticks=4,segment=0,
                                trades=1,gross_pnl=10,net_pnl=6,equity_peak=10,equity_min=-2,max_drawdown=12),
                            dict(day="2026-01-05",symbol="RTS",mode="volume",cost_ticks=4,segment=0,
                                trades=0,gross_pnl=0,net_pnl=0,equity_peak=0,equity_min=0,max_drawdown=0)])
        trades=pd.DataFrame([dict(day="2026-01-05",symbol="RTS",mode="baseline",cost_ticks=4,side="Long",gross_pnl=10,net_pnl=6)])
        summary,comparisons=result_tables(trades,daily,[4],100)
        current=summary[summary.period.eq("2026")].set_index("mode")
        self.assertEqual(current.loc["baseline","net_pnl"],6)
        self.assertEqual(current.loc["volume","days"],1)
        self.assertEqual(comparisons[comparisons.period.eq("2026")].difference.tolist(),[-6,-6,-6])

    def test_paired_bootstrap_keeps_pairing_and_short_segments(self):
        """Парный эффект оценивается по дневным разностям; короткие сегменты не дают ложную точность."""
        result=paired_bootstrap(np.tile([1.,3.],100),np.zeros(200),repetitions=300)
        self.assertEqual(result["difference"],2)
        self.assertLess(result["p"],.05)
        self.assertTrue(np.isnan(paired_bootstrap(np.ones(20),np.zeros(20),repetitions=300)["p"]))


if __name__=="__main__":
    unittest.main()
