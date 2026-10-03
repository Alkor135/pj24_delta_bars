"""Проверки стохастика, причинности сигналов и исполнения на последующих тиках.

Запуск из корня проекта:
    python -m unittest -v tests.test_backtest_stochastic

Синтетические цены позволяют проверить расчёт индикатора вручную, порядок
срабатывания стопа и цели, комиссии и просадку без обращения к котировкам.
Исследовательский скрипт импортируется из пакета backtest.
"""

import importlib.util
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

if importlib.util.find_spec("backtest.backtest_stochastic"):
    from backtest import backtest_stochastic as stochastic_module
    from backtest.backtest_stochastic import entry_signals, simulate_day, stochastic_frame


def sample_bars(closes, day="2026-09-01"):
    """Возвращает полные минутные бары по списку закрытий для проверок индикатора."""
    prices = np.asarray(closes, dtype=float)
    times = pd.date_range(day + " 10:00:00", periods=len(prices), freq="min")
    return pd.DataFrame(dict(day=day, bar_index=np.arange(len(prices)),
        start_time=times.astype(str), end_time=(times + pd.Timedelta(seconds=30)).astype(str),
        open=prices, high=prices + 2, low=prices - 2, close=prices,
        is_complete=1, start_row=np.arange(len(prices)) * 2 + 1,
        end_row=np.arange(len(prices)) * 2 + 2))


def run_ticks(prices, signals, cost_ticks=0, close_row=None):
    """Моделирует один день на ценах; возвращает сделки и полную тиковую просадку."""
    stamps = [f"2026-09-01 10:00:{i:02d}" for i in range(len(prices))]
    return simulate_day("2026-09-01", signals, np.asarray(prices, dtype=float),
                        stamps, close_row or len(prices), 1, 2, cost_ticks)


class StochasticTests(unittest.TestCase):
    """Проверяет формулу, прогрев и независимость прошлых сигналов от будущего."""

    def setUp(self):
        """Требует наличие нового исследовательского скрипта перед каждой проверкой."""
        self.assertIsNotNone(importlib.util.find_spec("backtest.backtest_stochastic"),
                             "Нужен backtest/backtest_stochastic.py")

    def test_formula_and_warmup(self):
        """Закрытие 12 в диапазоне 8–14 даёт 66,67; прогрев остаётся неопределённым."""
        frame = stochastic_frame(sample_bars([10, 11, 12]), 3, 1, 1)
        self.assertTrue(frame.k.iloc[:2].isna().all())
        self.assertAlmostEqual(frame.k.iloc[2], 100 * 4 / 6)
        self.assertEqual(frame.k.iloc[2], frame.d.iloc[2])

    def test_zero_range_has_no_signal(self):
        """Нулевой ценовой диапазон не создаёт вымышленную перепроданность."""
        bars = sample_bars([100] * 20)
        bars[["high", "low"]] = 100
        frame = stochastic_frame(bars, 3, 1, 1)
        self.assertTrue(frame.k.isna().all())
        self.assertEqual(entry_signals(frame, tick_size=1), [])

    def test_partial_bar_not_in_stochastic(self):
        """Экстремумы неполного остатка не входят в окно стохастика."""
        bars = sample_bars([10, 11, 12, 13, 14, 15])
        bars.loc[2, ["is_complete", "high", "low"]] = [0, 1000, 1]
        frame = stochastic_frame(bars, 3, 1, 1)
        reference = stochastic_frame(bars[bars.is_complete == 1].copy(), 3, 1, 1)
        self.assertTrue(pd.isna(frame.k.iloc[2]))
        np.testing.assert_allclose(frame.loc[bars.is_complete == 1, "k"],
                                   reference.k, equal_nan=True)

    def test_future_does_not_change_indicators(self):
        """Добавление будущих экстремумов не меняет индикаторы уже закрытых баров."""
        original = sample_bars([100, 95, 90, 94, 99, 103, 105, 98])
        extended = sample_bars([100, 95, 90, 94, 99, 103, 105, 98, 1000, 1])
        a = stochastic_frame(original, 3, 2, 2)
        b = stochastic_frame(extended, 3, 2, 2).iloc[:len(a)]
        np.testing.assert_allclose(a[["k", "d", "alf"]], b[["k", "d", "alf"]],
                                   equal_nan=True)

    def test_upward_cross_and_alf(self):
        """Пересечение 20 вверх разрешает покупку только при растущем ALF."""
        frame = sample_bars([100, 102, 103, 104])
        frame["k"] = [10, 18, 24, 27]
        frame["d"] = [14, 17, 20, 25]
        frame["alf"] = [99, 100, 101, 102]
        signals = entry_signals(frame, tick_size=1)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["side"], "Long")
        self.assertEqual(signals[0]["entry_row"], 7)
        self.assertEqual(signals[0]["stop_price"], 97)
        frame["alf"] = [104, 103, 102, 101]
        self.assertEqual(entry_signals(frame, tick_size=1), [])

    def test_downward_cross_and_alf(self):
        """Пересечение 80 вниз разрешает продажу при падающем ALF."""
        frame = sample_bars([104, 102, 101, 100])
        frame["k"] = [90, 85, 76, 70]
        frame["d"] = [86, 84, 79, 75]
        frame["alf"] = [105, 104, 103, 102]
        signals = entry_signals(frame, tick_size=1)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["side"], "Short")
        self.assertEqual(signals[0]["stop_price"], 107)

    def test_baseline_removes_only_stochastic_condition(self):
        """База сравнения сохраняет ALF и стоп, снимая только условие стохастика."""
        frame = sample_bars([100, 102, 103, 104])
        frame["k"] = [10, 18, 24, 27]
        frame["d"] = [14, 17, 20, 25]
        frame["alf"] = [99, 100, 101, 102]
        a = entry_signals(frame, tick_size=1)
        b = entry_signals(frame, mode="alf", tick_size=1)
        self.assertEqual(len(b), 2)
        self.assertEqual(a[0], b[0])

    def test_stop_lookback_cannot_cross_day(self):
        """Вчерашние бары нельзя использовать как локальный экстремум отката."""
        frame = sample_bars([100, 102, 103])
        frame.loc[0, "day"] = "2026-08-31"
        frame["k"] = [10, 18, 24]
        frame["d"] = [14, 17, 20]
        frame["alf"] = [99, 100, 101]
        self.assertEqual(entry_signals(frame, tick_size=1), [])

    def test_intrabar_extremes_must_match_ticks(self):
        """Изменённый максимум базы выявляется по внутренней сделке исходного бара."""
        validate = getattr(stochastic_module, "validate_extremes", None)
        self.assertTrue(callable(validate), "Нужна сверка high/low с исходными тиками")
        bars = sample_bars([100])
        bars.loc[0, ["start_row", "end_row", "high", "low"]] = [1, 3, 110, 100]
        self.assertIsNone(validate(bars, np.array([100, 110, 100])))
        bars.loc[0, "high"] = 100
        with self.assertRaisesRegex(ValueError, "high/low"):
            validate(bars, np.array([100, 110, 100]))

    def test_missing_first_bar_breaks_full_coverage(self):
        """Удаление первого бара не позволяет скрыть исходные строки дня."""
        validate = getattr(stochastic_module, "validate_extremes", None)
        self.assertTrue(callable(validate), "Нужна проверка полного покрытия строк")
        bars = sample_bars([100, 101])
        bars.loc[:, ["high", "low"]] = [[100, 100], [101, 101]]
        prices = np.array([100, 100, 101, 101])
        self.assertIsNone(validate(bars, prices, expected_count=2))
        with self.assertRaisesRegex(ValueError, "покрытие"):
            validate(bars.iloc[1:], prices)
        with self.assertRaisesRegex(ValueError, "число баров"):
            validate(bars, prices, expected_count=3)

    def test_reader_keeps_warmup_in_coverage(self):
        """Журнал сохраняет день прогрева, даже когда для него нет баров."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quotes.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE datasets (dataset_id TEXT, config_json TEXT)")
                db.execute("INSERT INTO datasets VALUES ('test', '{}')")
                pd.DataFrame([dict(dataset_id="test", symbol="MIX", day="2022-01-03",
                                   status="warmup", sha256="warmup"),
                              dict(dataset_id="test", symbol="MIX", day="2022-01-31",
                                   status="ready", sha256="ready")]).to_sql("days", db, index=False)
                bars = sample_bars([100, 101, 102], "2022-01-31")
                bars.assign(dataset_id="test", symbol="MIX").to_sql("bars", db, index=False)
                db.commit()
            settings = SimpleNamespace(period=3, smooth_k=1, smooth_d=1, alf_alpha=0.4)
            _, sources, _ = stochastic_module._load_database(path, "MIX", "2022-01-01",
                                                              "2022-01-31", None, settings)
            self.assertEqual(sources.status.tolist(), ["warmup", "ready"])

    def test_future_year_is_a_separate_period(self):
        """Данные 2027 года получают свою строку, не попадая в сводку за 2026."""
        periods = getattr(stochastic_module, "report_periods", None)
        self.assertTrue(callable(periods), "Нужны точные календарные границы периодов")
        result = periods("2022-01-03", "2027-01-04")
        self.assertIn(("2026", "2026-01-01", "2027-01-01"), result)
        self.assertIn(("2027", "2027-01-01", "2028-01-01"), result)


class ExecutionTests(unittest.TestCase):
    """Проверяет тиковое исполнение, пропуски сигналов и учёт затрат."""

    def setUp(self):
        """Требует реализацию скрипта перед проверкой исполнения."""
        self.assertIsNotNone(importlib.util.find_spec("backtest.backtest_stochastic"),
                             "Нужен backtest/backtest_stochastic.py")

    def test_target_executes_on_next_tick(self):
        """Касание цели на третьем тике исполняется на четвёртом, по цене 109."""
        result = run_ticks([100, 104, 110, 109, 108],
                           [dict(entry_row=1, side="Long", stop_price=95)])
        trade = result["trades"][0]
        self.assertEqual(trade["reason"], "target")
        self.assertEqual(trade["trigger_row"], 3)
        self.assertEqual(trade["exit_row"], 4)
        self.assertEqual(trade["gross_pnl"], 9)

    def test_stop_does_not_fill_at_unavailable_price(self):
        """Пробой стопа 95 исполняется следующим тиком 92, учитывая разрыв цены."""
        result = run_ticks([100, 98, 94, 92, 96],
                           [dict(entry_row=1, side="Long", stop_price=95)])
        trade = result["trades"][0]
        self.assertEqual(trade["reason"], "stop")
        self.assertEqual(trade["exit_price"], 92)
        self.assertEqual(trade["gross_pnl"], -8)

    def test_short_target_and_round_trip_cost(self):
        """Короткая сделка вычитает затраты за круг один раз."""
        result = run_ticks([100, 97, 90, 91, 93],
                           [dict(entry_row=1, side="Short", stop_price=105)], 4)
        trade = result["trades"][0]
        self.assertEqual(trade["gross_pnl"], 9)
        self.assertEqual(trade["net_pnl"], 5)

    def test_full_tick_drawdown(self):
        """Просадка измеряет движение 105→96 внутри открытой позиции."""
        result = run_ticks([100, 105, 96, 110, 110],
                           [dict(entry_row=1, side="Long", stop_price=90)], 4)
        self.assertEqual(result["daily"]["max_drawdown"], 9)
        self.assertEqual(result["daily"]["equity_min"], -6)
        self.assertEqual(result["daily"]["net_pnl"], 6)

    def test_no_overlapping_or_same_tick_reentry(self):
        """Сигналы в позиции и на тике её закрытия пропускаются."""
        signals = [dict(entry_row=row, side="Long", stop_price=95 if row < 6 else 103)
                   for row in [1, 3, 5, 6]]
        result = run_ticks([100, 102, 103, 110, 109, 108, 100, 99, 98], signals)
        self.assertEqual([t["entry_row"] for t in result["trades"]], [1, 6])

    def test_wrong_side_stop_and_late_entry_are_skipped(self):
        """Непригодный стоп после разрыва цены и поздний вход не создают сделки."""
        signals = [dict(entry_row=1, side="Long", stop_price=101),
                   dict(entry_row=3, side="Long", stop_price=95)]
        result = run_ticks([100, 102, 103], signals)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["daily"]["invalid_stops"], 1)

    def test_actual_entry_time_must_be_before_cutoff(self):
        """Следующий тик после 18:30 не исполняет старый разрешённый сигнал."""
        result = simulate_day("2026-09-01", [dict(entry_row=1, side="Long", stop_price=95)],
            np.array([100, 101, 102]), ["2026-09-01 18:31:00", "2026-09-01 18:32:00",
                                      "2026-09-01 18:40:00"], 3, 1)
        self.assertEqual(result["trades"], [])


if __name__ == "__main__":
    unittest.main()
