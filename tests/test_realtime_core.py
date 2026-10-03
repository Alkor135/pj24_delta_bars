"""Проверяет реальные сделки, tick rule, поздние данные и отдельный журнал.

Запуск: .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_realtime_core
Сравнение с пакетным расчётом использует одинаковые тики и порог; QUIK не нужен.
"""

from array import array
from datetime import date
import importlib.util
from pathlib import Path
import tempfile
import unittest

from source.delta_core import DeltaPath, TickDay, build_bars


class RealtimeCoreTests(unittest.TestCase):
    """Проверяет поведение построителя, а не устройство его внутренних объектов."""

    def setUp(self):
        """Требует новый модуль и подготавливает тип сделки для каждого теста."""
        self.assertIsNotNone(importlib.util.find_spec("source.realtime_core"),
                             "Потоковый построитель ещё не реализован")
        from source.realtime_core import Trade
        self.Trade = Trade

    def trade(self, number, price, qty, day="2026-10-02"):
        """Возвращает сделку с заданными номером, ценой, объёмом и днём."""
        return self.Trade(day, number * 1_000_000_000, str(number), "SPBFUT", "RIZ6", price, qty)

    def test_tick_rule_overshoot_and_equal_price(self):
        """Равная цена переносит направление между барами, превышение не переносится."""
        from source.realtime_core import DayBuilder
        builder = DayBuilder("2026-10-02", 5)
        builder.ingest([self.trade(1, 100, 2), self.trade(2, 101, 3),
                        self.trade(3, 101, 4), self.trade(4, 101, 2),
                        self.trade(5, 99, 7), self.trade(6, 99, 1)])
        bars = builder.snapshot()
        self.assertEqual([b["delta"] for b in bars], [7, -5, -1])
        self.assertEqual([b["volume"] for b in bars], [9, 9, 1])
        self.assertEqual([b["tick_count"] for b in bars], [3, 2, 1])
        self.assertEqual([b["close_reason"] for b in bars], ["threshold", "threshold", "live"])
        self.assertEqual(bars[0]["neutral_volume"], 2)
        self.assertEqual(bars[1]["open"], 101)
        self.assertEqual(bars[1]["low"], 99)

    def test_matches_batch_and_rebuilds_late_trade(self):
        """Поздняя сделка пересчитывает день; повтор не изменяет объём и OHLC."""
        from source.realtime_core import DayBuilder
        trades = [self.trade(i + 1, p, q) for i, (p, q) in enumerate(
            [(100, 2), (101, 3), (101, 4), (101, 2), (99, 7), (99, 1)])]
        builder = DayBuilder("2026-10-02", 5)
        builder.ingest([trades[0], *trades[2:]])
        self.assertTrue(builder.ingest([trades[1]]))
        self.assertFalse(builder.ingest(trades))
        ticks = TickDay(date(2026, 10, 2), array("q", [t.time_ns for t in trades]),
                        array("d", [t.price for t in trades]), array("q", [t.qty for t in trades]),
                        DeltaPath(array("q", [0, 3, 7, 9, 2, 1])), 1, 19)
        expected = build_bars(ticks, 5)
        actual = builder.snapshot(final=True)
        for bar, reference in zip(actual, expected):
            for key in reference:
                self.assertEqual(bar[key], reference[key], key)
        self.assertEqual(len(actual), len(expected))

    def test_new_day_is_neutral_and_invalid_trade_is_rejected(self):
        """Новый день не наследует направление, неверные сделки отвергаются."""
        from source.realtime_core import DayBuilder
        builder = DayBuilder("2026-10-03", 5)
        builder.ingest([self.trade(1, 150, 100, day="2026-10-03")])
        self.assertEqual(builder.snapshot()[0]["delta"], 0)
        with self.assertRaises(ValueError):
            builder.ingest([self.trade(2, 151, 1)])
        with self.assertRaises(ValueError):
            DayBuilder("2026-10-03", 0)

    def test_quik_timestamp_quantity_and_large_identity(self):
        """Парсер сохраняет целый номер, время МСК и qty вместо денежного value."""
        from source.realtime_core import parse_trade
        data = dict(trade_num=2**53 + 9, class_code="SPBFUT", sec_code="RIZ6",
                    price=100, qty=3, value=300000,
                    datetime=dict(year=2026, month=10, day=2, hour=9, min=1, sec=2, ms=123))
        trade = parse_trade(data)
        self.assertEqual(trade.trade_num, str(2**53 + 9))
        self.assertEqual(trade.qty, 3)
        self.assertEqual(trade.time_ns, (9 * 3600 + 62) * 1_000_000_000 + 123_000_000)
        for field, value in (("qty", 1.5), ("price", float("nan")), ("trade_num", float(2**53 + 9))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_trade(dict(data, **{field: value}))

    def test_journal_recovers_and_cursor_keeps_late_trades(self):
        """Журнал после открытия сохраняет повтороустойчивость и поздние новые записи."""
        from source.realtime_core import TradeJournal
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trades.sqlite3"
            journal = TradeJournal(path)
            self.assertEqual(journal.append("RTS", [self.trade(2, 101, 3)]), 1)
            first = journal.read("RTS")
            journal.close()
            journal = TradeJournal(path)
            try:
                self.assertEqual(journal.append("RTS", [self.trade(2, 101, 3), self.trade(1, 100, 2)]), 1)
                late = journal.read("RTS", after=first["cursor"])
                self.assertEqual([t["trade_num"] for t in late["trades"]], ["1"])
                self.assertEqual(journal.read("MIX")["trades"], [])
            finally:
                journal.close()

    def test_conflicting_duplicate_does_not_silently_change_journal(self):
        """Повторный номер с другим объёмом отклоняется, сохраняя исходную сделку."""
        from source.realtime_core import TradeJournal
        with tempfile.TemporaryDirectory() as folder:
            journal = TradeJournal(Path(folder) / "trades.sqlite3")
            try:
                journal.append("RTS", [self.trade(1, 100, 2)])
                with self.assertRaises(ValueError):
                    journal.append("RTS", [self.trade(1, 100, 3)])
                self.assertEqual(journal.read("RTS")["trades"][0]["qty"], 2)
            finally:
                journal.close()


if __name__ == "__main__":
    unittest.main()
