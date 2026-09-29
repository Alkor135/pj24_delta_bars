"""Проверки отбора параметров и календарного разделения истории.

Запуск: python -m unittest -v tests.test_duration_analysis
"""
import importlib.util
import unittest
import numpy as np
import pandas as pd
from source.duration_engine import GridResult,parameter_grid

if importlib.util.find_spec('source.duration_analysis'):
    from source.duration_analysis import grid_statistics,rank_parameters,comparison_indices,walk_forward,trade_metrics


class AnalysisTests(unittest.TestCase):
    """Не допускает влияния будущей прибыли на выбор порогов."""

    def setUp(self):
        """Создаёт сетку с изменяемой будущей частью."""
        self.assertIsNotNone(importlib.util.find_spec('source.duration_analysis'),'Нужен duration_analysis')
        self.params=parameter_grid([1,2,3],[10,15,20])
        self.dates=pd.bdate_range('2022-01-03','2026-06-30').strftime('%Y-%m-%d').tolist()
        rng=np.random.default_rng(77)
        self.gross=rng.normal(10,5,(len(self.dates),len(self.params)))
        self.grid=GridResult(self.dates,self.params,self.gross,np.ones_like(self.gross,dtype=int))

    def test_holdout_not_in_parameter_selection(self):
        """Изменение 2026 года не меняет рейтинг на 2022–2025."""
        mask=np.array(self.dates)<'2026-01-01'
        first=rank_parameters(grid_statistics(self.grid,mask,2),self.params,min_trades=100)
        self.grid.gross[~mask,0]=1e9
        second=rank_parameters(grid_statistics(self.grid,mask,2),self.params,min_trades=100)
        pd.testing.assert_frame_equal(first,second)
        self.assertEqual(comparison_indices(first,self.params,3),comparison_indices(second,self.params,3))

    def test_walk_forward_train_before_test(self):
        """Обучение каждого окна заканчивается строго до проверяемого квартала."""
        result=walk_forward(self.grid,2,holdout_start='2026-01-01',min_trades=100)
        self.assertTrue(result['folds'])
        self.assertTrue(all(f['train_end']<f['test_start'] for f in result['folds']))
        self.assertTrue(all(d<'2026-01-01' for d in result['dates']))
        first=result['folds'][0]
        changed=self.grid.gross.copy()
        changed[np.array(self.dates)>=first['test_start'],0]=1e9
        alternative=walk_forward(GridResult(self.dates,self.params,changed,self.grid.count),2,
                                 holdout_start='2026-01-01',min_trades=100)
        self.assertEqual(first['parameter_index'],alternative['folds'][0]['parameter_index'])

    def test_losses_stay_flat_and_drawdown_starts_at_zero(self):
        """Не выбирает убыточные параметры и учитывает убыток самого первого дня."""
        grid=GridResult(self.dates,self.params,-abs(self.gross),self.grid.count)
        stats=grid_statistics(grid,np.ones(len(self.dates),dtype=bool),0)
        self.assertTrue((stats.max_drawdown>0).all())
        self.assertTrue((rank_parameters(stats,self.params).eligible==False).all())
        result=walk_forward(grid,0,holdout_start='2026-01-01')
        self.assertTrue(all(f['parameter_index'] is None for f in result['folds']))
        self.assertTrue(all(x==0 for x in result['net']))

    def test_trade_metrics_and_empty_case(self):
        """Проверяет profit factor, долю выигрышей и нулевое число сделок."""
        rows=[dict(net_pnl=10,side='Long'),dict(net_pnl=-5,side='Short'),dict(net_pnl=0,side='Long')]
        result=trade_metrics(rows)
        self.assertEqual(result['profit_factor'],2)
        self.assertAlmostEqual(result['win_rate'],1/3)
        self.assertEqual(result['long_pnl'],10)
        self.assertIsNone(trade_metrics([])['profit_factor'])


if __name__=='__main__':
    unittest.main()
