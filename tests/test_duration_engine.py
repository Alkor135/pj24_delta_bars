"""Проверки исполнения стратегии длительности без знания будущих баров.

Запуск: python -m unittest -v tests.test_duration_engine
"""

import importlib.util
import unittest
import numpy as np

if importlib.util.find_spec('source.duration_engine'):
    from source.duration_engine import Event, Day, parameter_grid, simulate_grid, simulate_one


class EngineTests(unittest.TestCase):
    """Сравнивает расчёт сделок с заранее известными исходами."""

    def setUp(self):
        """Даёт понятное падение до появления реализации."""
        self.assertIsNotNone(importlib.util.find_spec('source.duration_engine'), 'Нужен duration_engine')

    def event(self, row, price, duration=-1, direction=0, enter=False, exit=False, force=False):
        """Создаёт событие с различимой меткой времени."""
        return Event(f'2022-01-03 10:00:{row:02d}', row, price, duration, direction, enter, exit, force, row-1)

    def test_strict_thresholds_next_tick_and_cost(self):
        """Не входит при равенстве, ждёт следующий сигнал и учитывает обе стороны издержек."""
        events = [self.event(1,100,10,1,True,True), self.event(2,120,9,1,True,True),
                  self.event(3,80,30,-1,True,True), self.event(4,90,31,-1,True,True),
                  self.event(5,100,force=True)]
        day = Day('2022-01-03',events)
        result = simulate_one([day],10,30,cost_points=4)
        self.assertEqual(len(result.trades),1)
        self.assertEqual(result.trades[0]['entry_price'],120)
        self.assertEqual(result.trades[0]['exit_price'],90)
        self.assertEqual(result.trades[0]['net_pnl'],-34)
        self.assertEqual(result.equity[-1]['net_pnl'],-34)
        self.assertLessEqual(min(x['net_pnl'] for x in result.equity),-42)

    def test_short_doji_partial_and_forced_close(self):
        """Пропускает doji и неподтверждённый сигнал, закрывает short по времени."""
        day=Day('2022-01-03',[self.event(1,100,1,0,True,True),
            self.event(2,110,1,1,False,False),self.event(3,120,1,-1,True,True),
            self.event(4,115,100,1,False,False),self.event(5,90,force=True)])
        result=simulate_one([day],5,30,0)
        self.assertEqual(len(result.trades),1)
        self.assertEqual(result.trades[0]['side'],'Short')
        self.assertEqual(result.trades[0]['gross_pnl'],30)
        self.assertEqual(result.trades[0]['reason'],'time')

    def test_grid_matches_scalar_random_days(self):
        """Проверяет разные последовательности независимо для каждой пары параметров."""
        rng=np.random.default_rng(812)
        days=[]
        for daynum in range(8):
            prices=100+np.cumsum(rng.integers(-10,11,size=40))
            events=[self.event(i,float(p),int(rng.integers(0,80)),int(rng.integers(-1,2)),
                      bool(rng.integers(0,2)),bool(rng.integers(0,2))) for i,p in enumerate(prices)]
            events.append(self.event(41,100,force=True))
            days.append(Day(f'2022-01-{daynum+3:02d}',events))
        params=parameter_grid([1,5,10,20],[5,15,30,60])
        grid=simulate_grid(days,params)
        for i,(entry,exit) in enumerate(params):
            scalar=simulate_one(days,int(entry),int(exit),3)
            expected=np.array([x['gross_pnl'] for x in scalar.daily])
            np.testing.assert_allclose(grid.gross[:,i],expected)
            np.testing.assert_array_equal(grid.count[:,i],[x['trades'] for x in scalar.daily])
            self.assertAlmostEqual(sum(x['net_pnl'] for x in scalar.trades),
                                   (grid.gross[:,i]-3*grid.count[:,i]).sum())

    def test_no_carry_or_lookahead(self):
        """Последующие цены не меняют ранний вход; позиция не переносится между днями."""
        a=Day('2022-01-03',[self.event(1,100,1,1,True,True),self.event(2,105,force=True)])
        b=Day('2022-01-04',[self.event(1,1000,force=True)])
        result=simulate_one([a,b],5,30,0)
        self.assertEqual(sum(x['gross_pnl'] for x in result.daily),5)
        self.assertEqual(result.daily[1]['trades'],0)
        self.assertEqual(result.trades[0]['entry_price'],100)

    def test_grid_bounds(self):
        """Сетка включает заданные границы и исключает выход не больше входа."""
        params=parameter_grid(range(1,46),range(5,301,5))
        self.assertTrue(np.all(params[:,1]>params[:,0]))
        self.assertIn([45,300],params.tolist())
        self.assertIn([1,5],params.tolist())
        with self.assertRaises(ValueError): parameter_grid([0],[5])


if __name__=='__main__':
    unittest.main()
