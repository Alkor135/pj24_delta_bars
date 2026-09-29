"""Проверки временных меток и привязки исполнения к исходным сделкам.

Запуск: python -m unittest -v tests.test_duration_data
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

if importlib.util.find_spec('source.duration_data'):
    from source.duration_data import Session, build_day, read_tick_zip


class DataTests(unittest.TestCase):
    """Проверяет границы дня, порядок одинаковых секунд и повреждённые источники."""

    def setUp(self):
        """Создаёт короткий день с тремя барами и отдельным тиком закрытия."""
        self.assertIsNotNone(importlib.util.find_spec('source.duration_data'),'Нужен duration_data')
        self.ticks=pd.DataFrame({'datetime':['2022-01-03 10:00:00.001','2022-01-03 10:00:00.002',
            '2022-01-03 10:00:01.000','2022-01-03 10:00:40.000',
            '2022-01-03 10:00:41.000','2022-01-03 10:01:02.000'],
            'last':[100,110,115,90,95,105],'volume':[1]*6})
        self.bars=pd.DataFrame([dict(bar_index=i,start_row=a,end_row=b,
            start_time=self.ticks.iloc[a-1]['datetime'],end_time=self.ticks.iloc[b-1]['datetime'],
            open=self.ticks.iloc[a-1]['last'],close=self.ticks.iloc[b-1]['last'],is_complete=int(i<2))
            for i,(a,b) in enumerate([(1,2),(3,4),(5,6)])])
        self.session=Session('10:00:00','10:00:50','10:01:00')

    def test_source_second_precision_and_next_row(self):
        """Игнорирует искусственные доли и исполняет после последней строки сигнала."""
        day,audit=build_day('2022-01-03',self.bars,self.ticks,self.session)
        signal=[e for e in day.events if e.enter][0]
        self.assertEqual(signal.row,3)
        self.assertEqual(signal.price,115)
        self.assertEqual(signal.duration,0)
        self.assertEqual(day.events[-1].row,6)
        self.assertEqual(day.events[-1].price,105)
        self.assertEqual(audit['close_delay_seconds'],2)

    def test_absent_close_tick_excludes_day(self):
        """Не придумывает закрытие по последнему тику до заданного времени."""
        ticks=self.ticks.copy()
        ticks.loc[5,'datetime']='2022-01-03 10:00:59.000'
        bars=self.bars.copy()
        bars.loc[2,'end_time']=ticks.loc[5,'datetime']
        day,audit=build_day('2022-01-03',bars,ticks,self.session)
        self.assertIsNone(day)
        self.assertEqual(audit['status'],'excluded')

    def test_changed_price_and_bad_session_rejected(self):
        """Обнаруживает несовпадение базы и архива и неверные часы торговли."""
        ticks=self.ticks.copy()
        ticks.loc[2,'last']=999
        with self.assertRaisesRegex(ValueError,'цен'):
            build_day('2022-01-03',self.bars,ticks,self.session)
        with self.assertRaises(ValueError): Session('18:00','10:00','18:40')

    def test_hash_mismatch_rejected(self):
        """Не использует ZIP, который изменился после построения баров."""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ticks.zip'
            with ZipFile(path,'w') as archive:
                archive.writestr('day.csv',self.ticks.to_csv(index=False))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                read_tick_zip(path,'0'*64)


if __name__=='__main__':
    unittest.main()
