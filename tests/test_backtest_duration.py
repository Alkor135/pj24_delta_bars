"""Проверки границ перебора и сохранения результатов backtest.backtest_duration.

Запуск: python -m unittest -v tests.test_backtest_duration
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from source.duration_engine import Day,Event,parameter_grid,simulate_grid

if importlib.util.find_spec('backtest.backtest_duration'):
    from backtest.backtest_duration import parse_grid,build_payload,save_tables


class RunnerTests(unittest.TestCase):
    """Проверяет сверку подробного журнала и результатов сетки."""

    def setUp(self):
        """Требует реализации точки входа до начала интеграционных проверок."""
        self.assertIsNotNone(importlib.util.find_spec('backtest.backtest_duration'),'Нужен backtest/backtest_duration.py')

    def test_parse_grid(self):
        """Сетка включает правую границу и не допускает нулевой шаг."""
        self.assertEqual(parse_grid('1:5:2'),[1,3,5])
        with self.assertRaises(ValueError): parse_grid('1:5:0')

    def test_payload_and_save(self):
        """Обрабатывает пустой контрольный участок и сохраняет журнал и метаданные."""
        day=Day('2022-01-03',[Event('2022-01-03 10:01:00',2,100,1,1,True,True),
            Event('2022-01-03 18:40:00',3,105,force=True)])
        grid=simulate_grid([day],parameter_grid([2,3],[10,20]))
        payload,tables=build_payload('RTS',[day],grid,[],{},10,[0,2,4,8],4,'2026-01-01',1,2)
        self.assertEqual(len(payload['curves']),1)
        self.assertEqual(payload['summary'][0]['holdout_trades'],0)
        self.assertEqual(payload['summary'][0]['development_pnl'],-35)
        with tempfile.TemporaryDirectory() as folder:
            save_tables(Path(folder),grid,payload,tables)
            self.assertTrue((Path(folder)/'results.sqlite3').is_file())
            self.assertTrue((Path(folder)/'grid_daily.npz').is_file())
            self.assertTrue((Path(folder)/'trades.csv').is_file())

    def test_zero_cost_diagnostic_heatmap(self):
        """Отчёт позволяет увидеть прибыль без издержек на всей сетке."""
        day=Day('2022-01-03',[Event('2022-01-03 10:01:00',2,100,1,1,True,True),
            Event('2022-01-03 18:40:00',3,105,force=True)])
        grid=simulate_grid([day],parameter_grid([2],[10]))
        payload,_=build_payload('RTS',[day],grid,[],{},10,[0,2,4,8],4,'2026-01-01',1,1)
        maps=[m for m in payload['heatmaps'] if '0 шаг' in m['title']]
        self.assertEqual(len(maps),1)
        self.assertEqual(maps[0]['z'],[[5]])


if __name__=='__main__':
    unittest.main()
