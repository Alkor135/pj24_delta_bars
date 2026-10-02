"""Проверки Heikin Ashi без других индикаторов и исполнения по реальным тикам.

Формулы проверяются на ручных примерах; будущие и неполные бары не должны
изменять прошлые сигналы. Отдельно проверяются подтверждение цвета, перевороты,
запрет позднего входа, закрытие сессии и просадка внутри открытой позиции.
Временная база и ZIP проверяют сверку истории прогрева до начала периода:
повреждённый экстремум обязан остановить запуск до моделирования сделок.
Запуск из корня: python -m unittest -v tests.test_backtest_heikin_ashi
Все проверки: python -m unittest discover -s tests -v
"""

import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from hashlib import sha256
import zipfile

import numpy as np
import pandas as pd

if importlib.util.find_spec("backtest_heikin_ashi") is not None:
    from backtest_heikin_ashi import heikin_ashi_frame, color_targets, simulate_day, _load_database, main


def sample_bars():
    """Возвращает три полных бара с вручную вычисляемыми OHLC и границами строк."""
    return pd.DataFrame(dict(day=["2026-09-01"] * 3, bar_index=[1, 2, 3],
        open=[100, 102, 104], high=[104, 108, 106], low=[98, 100, 100],
        close=[102, 106, 102], is_complete=[1, 1, 1],
        start_time=[f"2026-09-01 10:0{i}:00" for i in range(3)],
        end_time=[f"2026-09-01 10:0{i}:59" for i in range(3)],
        start_row=[1, 3, 5], end_row=[2, 4, 6]))


def signal(row, side, bar=1):
    """Возвращает сигнал после строки row с направлением side и номером bar."""
    return dict(day="2026-09-01", signal_bar=bar, signal_time="2026-09-01 10:00:00",
                trigger_row=row, side=side, ha_open=100, ha_close=101)


def run_ticks(prices, targets, cost=0, stamps=None, close_row=None):
    """Возвращает моделирование дня по prices/targets с затратами cost за круг."""
    stamps = stamps or [f"2026-09-01 10:00:{i:02d}" for i in range(len(prices))]
    return simulate_day("2026-09-01", targets, np.asarray(prices, dtype=float),
                        stamps, close_row or len(prices), tick_size=1, cost_ticks=cost)


class HeikinAshiTests(unittest.TestCase):
    """Проверяет свечи, причинность сигналов и полную модель исполнения."""

    def setUp(self):
        """Требует исследовательский скрипт; его отсутствие даёт понятный провал."""
        self.assertIsNotNone(importlib.util.find_spec("backtest_heikin_ashi"),
                             "Нужен backtest_heikin_ashi.py")

    def test_recursive_formula(self):
        """Средние цены и рекурсивное открытие совпадают с ручным расчётом."""
        result = heikin_ashi_frame(sample_bars())
        np.testing.assert_allclose(result.ha_close, [101, 104, 103])
        np.testing.assert_allclose(result.ha_open, [101, 101, 102.5])
        np.testing.assert_allclose(result.ha_high, [104, 108, 106])
        np.testing.assert_allclose(result.ha_low, [98, 100, 100])
        self.assertEqual(result.ha_color.tolist(), [0, 1, 1])

    def test_partial_bar_does_not_enter_recursion(self):
        """Неполный дневной остаток не участвует ни в сигналах, ни в рекурсии."""
        bars = sample_bars()
        bars.loc[1, "is_complete"] = 0
        bars.loc[1, ["open", "high", "low", "close"]] = 10000
        result = heikin_ashi_frame(bars)
        self.assertTrue(pd.isna(result.loc[1, "ha_open"]))
        self.assertEqual(result.loc[2, "ha_open"], 101)
        self.assertEqual(len(color_targets(result)), 1)

    def test_future_does_not_change_prefix(self):
        """Изменение будущего бара не меняет прошлые свечи и сигналы."""
        bars = sample_bars()
        old = heikin_ashi_frame(bars)
        bars.loc[2, ["open", "high", "low", "close"]] = 9999
        new = heikin_ashi_frame(bars)
        pd.testing.assert_frame_equal(old.iloc[:2], new.iloc[:2])
        self.assertEqual(color_targets(old.iloc[:2]), color_targets(new.iloc[:2]))

    def test_recursion_crosses_day_but_confirmation_does_not(self):
        """Свечи используют прошлую историю, а серия подтверждения начинается заново."""
        bars = sample_bars()
        bars.loc[2, "day"] = "2026-09-02"
        bars.loc[2, ["start_time", "end_time"]] = ["2026-09-02 10:00:00", "2026-09-02 10:00:59"]
        result = heikin_ashi_frame(bars)
        self.assertEqual(result.loc[2, "ha_open"], 102.5)
        self.assertEqual(color_targets(result, confirmation=2), [])

    def test_confirmation_and_doji(self):
        """Две свечи подтверждают цвет; доджи сбрасывает серию, но не даёт выхода."""
        bars = pd.concat([sample_bars(), sample_bars()], ignore_index=True)
        bars.bar_index = np.arange(1, 7)
        bars.end_row = np.arange(1, 7) * 2
        bars["ha_open"] = 100
        bars["ha_close"] = [101, 101, 100, 99, 101, 101]
        bars["ha_color"] = [1, 1, 0, -1, 1, 1]
        targets = color_targets(bars, confirmation=2)
        self.assertEqual([x["signal_bar"] for x in targets], [2, 6])
        self.assertEqual([x["side"] for x in targets], [1, 1])

    def test_bar_crossing_session_start_cannot_signal(self):
        """Бар, начатый до 10:00, не используется для торгового подтверждения."""
        bars = heikin_ashi_frame(sample_bars())
        bars.loc[1, "start_time"] = "2026-09-01 09:59:59"
        self.assertEqual(color_targets(bars, confirmation=2), [])

    def test_next_raw_tick_and_reversal_costs(self):
        """Переворот исполняется следующим реальным тиком и оплачивает обе сделки."""
        result = run_ticks([100, 105, 110, 108, 102, 100],
                           [signal(1, 1), signal(3, -1, 2)], cost=4)
        a, b = result["trades"]
        self.assertEqual((a["entry_row"], a["entry_price"], a["exit_row"], a["exit_price"]),
                         (2, 105, 4, 108))
        self.assertEqual((b["entry_row"], b["entry_price"], b["exit_price"]), (4, 108, 100))
        self.assertEqual([a["net_pnl"], b["net_pnl"]], [-1, 4])
        self.assertEqual(result["daily"]["net_pnl"], 3)

    def test_same_color_does_not_add_positions(self):
        """Повторные свечи одного цвета оставляют одну позицию без новых затрат."""
        result = run_ticks([100, 100, 102, 104], [signal(1, 1), signal(2, 1)], cost=2)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["daily"]["net_pnl"], 2)

    def test_actual_late_entry_is_rejected(self):
        """Сигнал до ограничения не разрешает вход, если следующий тик уже в 18:30."""
        stamps = ["2026-09-01 18:29:59", "2026-09-01 18:30:00", "2026-09-01 18:40:00"]
        result = run_ticks([100, 110, 120], [signal(1, 1)], stamps=stamps)
        self.assertEqual(result["trades"], [])

    def test_late_opposite_color_closes_without_reversal(self):
        """После 18:30 противоположный цвет закрывает позицию без открытия новой."""
        stamps = ["2026-09-01 10:00:00", "2026-09-01 10:00:01",
                  "2026-09-01 18:31:00", "2026-09-01 18:31:01", "2026-09-01 18:40:00"]
        result = run_ticks([100, 100, 90, 95, 80], [signal(1, 1), signal(3, -1)], stamps=stamps)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["exit_price"], 95)

    def test_session_close_has_priority(self):
        """Сигнал, исполняемый в строке закрытия, не создаёт новую позицию."""
        result = run_ticks([100, 100, 105, 110, 80], [signal(1, 1), signal(3, -1)], close_row=4)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["reason"], "Закрытие сессии")
        self.assertEqual(result["trades"][0]["exit_price"], 110)

    def test_intratrade_drawdown_and_split_fees(self):
        """Просадка учитывает падение внутри прибыльной сделки и плату за вход/выход."""
        result = run_ticks([100, 100, 110, 80, 120], [signal(1, 1)], cost=4)
        self.assertEqual(result["daily"]["net_pnl"], 16)
        self.assertEqual(result["daily"]["equity_min"], -22)
        self.assertEqual(result["daily"]["equity_peak"], 18)
        self.assertEqual(result["daily"]["max_drawdown"], 30)

    def test_short_uses_real_price(self):
        """Продажа с искусственным HA-close 101 исполняется по реальной цене 110."""
        result = run_ticks([100, 110, 100, 90], [signal(1, -1)], cost=2)
        self.assertEqual(result["trades"][0]["entry_price"], 110)
        self.assertEqual(result["daily"]["net_pnl"], 18)

    def test_invalid_confirmation_rejected(self):
        """Нецелое либо нулевое число подтверждений запрещено."""
        for value in [0, 1.5]:
            with self.assertRaises(ValueError):
                color_targets(heikin_ashi_frame(sample_bars()), confirmation=value)

    def test_warmup_sources_are_loaded_for_validation(self):
        """Журнал включает историю до start для сверки, а границы отчёта остаются заданными."""
        before, after = sample_bars(), sample_bars()
        before["day"] = "2026-08-31"
        bars = pd.concat([before, after], ignore_index=True)
        bars["dataset_id"], bars["symbol"] = "test", "RTS"
        days = pd.DataFrame(dict(day=["2026-08-31", "2026-09-01"],
            dataset_id=["test"] * 2, symbol=["RTS"] * 2, status=["ready"] * 2,
            sha256=["before", "after"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                bars.to_sql("bars", db, index=False)
                days.to_sql("days", db, index=False)
                pd.DataFrame(dict(dataset_id=["test"], config_json=[json.dumps({})])).to_sql(
                    "datasets", db, index=False)
            frame, sources, metadata = _load_database(path, "RTS", "2026-09-01", "2026-09-30", None)
        self.assertEqual(sources.day.tolist(), ["2026-08-31", "2026-09-01"])
        self.assertEqual(len(frame), 6)
        self.assertEqual(metadata["first_day"], "2026-09-01")
        self.assertEqual(metadata["source_days"], 1)
        self.assertEqual(metadata["warmup_bars"], 3)

    def test_corrupt_warmup_fails_before_trading(self):
        """Запуск с поздним start обязан отвергнуть повреждённый high истории прогрева."""
        with tempfile.TemporaryDirectory() as directory:
            folder, bars, sources = Path(directory), [], []
            for day in ["2026-08-31", "2026-09-01"]:
                stamps = [day + " 10:00:00", day + " 10:00:01", day + " 18:40:00"]
                archive = folder / (day + ".zip")
                with zipfile.ZipFile(archive, "w") as target:
                    target.writestr("ticks.csv", pd.DataFrame(dict(datetime=stamps, last=[100, 110, 100])).to_csv(index=False))
                sources.append(dict(day=day, symbol="RTS", dataset_id="test", status="ready",
                    sha256=sha256(archive.read_bytes()).hexdigest(), file_path=str(archive), bar_count=2, tick_count=3))
                bars.extend([dict(day=day, symbol="RTS", dataset_id="test", bar_index=1,
                    open=100, close=110, high=111 if day < "2026-09-01" else 110, low=100,
                    start_time=stamps[0], end_time=stamps[1], start_row=1, end_row=2, is_complete=1),
                    dict(day=day, symbol="RTS", dataset_id="test", bar_index=2,
                    open=100, close=100, high=100, low=100, start_time=stamps[2], end_time=stamps[2],
                    start_row=3, end_row=3, is_complete=0)])
            database = folder / "bars.sqlite3"
            with closing(sqlite3.connect(database)) as db:
                pd.DataFrame(bars).to_sql("bars", db, index=False)
                pd.DataFrame(sources).to_sql("days", db, index=False)
                pd.DataFrame(dict(dataset_id=["test"], config_json=["{}"])).to_sql("datasets", db, index=False)
            with self.assertRaisesRegex(ValueError, "Экстремумы"):
                main(["--db", str(database), "--symbols", "RTS", "--start", "2026-09-01",
                      "--end", "2026-09-30", "--output", str(folder / "results")])


if __name__ == "__main__":
    unittest.main()
