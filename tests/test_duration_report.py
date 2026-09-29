"""Проверки автономного отчёта по длительности дельта-баров.

Пример запуска из корня проекта:
    python -m unittest -v tests.test_duration_report
"""

from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from source.duration_report import write_report


class ReportParser(HTMLParser):
    """Собирает встроенные данные и внешние ресурсы итогового HTML."""

    def __init__(self):
        """Создаёт хранилища содержимого и атрибутов HTML."""
        super().__init__()
        self.scripts = {}
        self.sources = []
        self.hrefs = []
        self.active_script = None

    def handle_starttag(self, tag, attrs):
        """Запоминает скрипты, ссылки и внешние источники."""
        attrs = dict(attrs)
        if tag == "script":
            self.active_script = attrs.get("id")
            if self.active_script:
                self.scripts[self.active_script] = ""
            if attrs.get("src"):
                self.sources.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.sources.append(attrs.get("href"))
        if tag == "a":
            self.hrefs.append(attrs.get("href"))

    def handle_endtag(self, tag):
        """Закрывает накопление содержимого скрипта."""
        if tag == "script":
            self.active_script = None

    def handle_data(self, data):
        """Сохраняет текст встроенного блока данных."""
        if self.active_script:
            self.scripts[self.active_script] += data


class DurationReportTests(unittest.TestCase):
    """Проверяет переносимость отчёта и сохранность исследовательских данных."""

    def render(self, payload):
        """Создаёт временный отчёт и возвращает его текст и разобранный HTML."""
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "nested"
            path = write_report(target, payload)
            self.assertEqual(path, target / "report.html")
            text = path.read_text(encoding="utf-8")
        parser = ReportParser()
        parser.feed(text)
        return text, parser

    def test_empty_payload_is_self_contained(self):
        """Пустой набор создаёт автономный документ с понятными состояниями."""
        text, parser = self.render({"symbol": "Si", "meta": {}})
        self.assertIn('<html lang="ru">', text)
        self.assertIn("Нет данных", text)
        self.assertIn("2022–2025", text)
        self.assertIn("2026", text)
        self.assertIn("пункты котировки на 1 контракт", text)
        self.assertIn("не рубли", text)
        self.assertIn("plotly.js", text.lower())
        self.assertEqual(parser.sources, [])
        self.assertEqual(json.loads(parser.scripts["report-data"])["symbol"], "Si")

    def test_entry_rule_is_visible_and_escaped(self):
        """Показывает режим ALF в основном описании отчёта с экранированием условий."""
        text, _ = self.render({'symbol': 'RTS', 'meta': {
            'entry_rule': 'Long: close > ALF; Short: close < ALF; α=0.4',
            'entry_filter': {'mode': 'alf', 'alpha': .4}}})
        self.assertTrue('<strong>Условия входа</strong>' in text, 'В отчёте не показаны условия входа')
        self.assertTrue('close &gt; ALF' in text, 'Не показано экранированное условие Long')
        self.assertTrue('close &lt; ALF' in text, 'Не показано экранированное условие Short')

    def test_holdout_curve_resets_development_profit(self):
        """Контрольный график начинается без накопленной прибыли периода подбора."""
        _,parser=self.render({'symbol':'RTS','meta':{'holdout_start':'2026-01-01'},
            'curves':[{'label':'10 / 100','dates':['2025-12-30','2026-01-05','2026-01-06'],
                       'net':[1000,990,1015],'gross':[1500,1505,1540],'drawdown':[0,-10,0]}]})
        figures=json.loads(parser.scripts['figure-data'])
        self.assertIn('holdout-equity',figures)
        self.assertEqual(figures['holdout-equity']['data'][0]['y'],[0,-10,15])

    def test_custom_period_and_metric_labels(self):
        """Периоды и единицы тепловой карты не подменяются фиксированными надписями."""
        text,parser=self.render({'symbol':'RTS','meta':{'development_start':'2023-01-01',
            'development_end':'2023-12-31','holdout_start':'2024-01-01'},
            'heatmaps':[{'title':'Число сделок','entries':[1],'exits':[10],'z':[[7]],'value_label':'Сделки','kind':'count'}]})
        figures=json.loads(parser.scripts['figure-data'])
        self.assertEqual(figures['heatmap-0']['data'][0]['colorbar']['title']['text'],'Сделки')
        self.assertIn('2023-01-01',text)
        self.assertNotIn('Отложенный контроль · 2026',text)

    def test_labels_and_metadata_cannot_break_html_or_json(self):
        """Метки и метаданные с HTML остаются текстом, опасные ссылки исключены."""
        hostile = '</script><img src=x onerror="alert(1)"> & тест'
        text, parser = self.render({
            "symbol": hostile,
            "meta": {"disclaimer": hostile},
            "summary": [{"label": hostile, "holdout_pnl": -8.5}],
            "curves": [{"label": hostile, "dates": ["2026-01-05"], "net": [-8.5],
                        "gross": [-7], "drawdown": [-8.5]}],
            "links": [{"title": hostile, "href": "../Si/report.html"},
                      {"title": "опасная", "href": "javascript:alert(1)"}],
        })
        self.assertNotIn(hostile, text)
        self.assertIn("&lt;/script&gt;&lt;img", text)
        self.assertEqual(json.loads(parser.scripts["report-data"])["symbol"], hostile)
        self.assertIn("../Si/report.html", parser.hrefs)
        self.assertNotIn("javascript:alert(1)", parser.hrefs)

    def test_nonfinite_metrics_become_null_and_negative_values_survive(self):
        """NaN и бесконечность не ломают JSON, отрицательные значения сохраняются."""
        _, parser = self.render({
            "symbol": "NG",
            "summary": [{"label": "10 / 30", "holdout_pnl": -125.5,
                         "holdout_profit_factor": float("inf"),
                         "holdout_win_rate": float("nan"), "holdout_max_dd": None}],
            "curves": [{"label": "10 / 30", "dates": ["2025-12-30", "2026-01-05"],
                        "net": [-5, -125.5], "gross": [-3, -110],
                        "drawdown": [-5, -125.5], "intraday_drawdown": [-12, -180]}],
            "heatmaps": [{"title": "Разработка", "entries": [10, 20], "exits": [30],
                          "z": [[None, -125.5]]}],
        })
        raw = parser.scripts["report-data"]
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        payload = json.loads(raw)
        self.assertEqual(payload["summary"][0]["holdout_pnl"], -125.5)
        self.assertIsNone(payload["summary"][0]["holdout_profit_factor"])
        figures = json.loads(parser.scripts["figure-data"])
        equity = figures["equity"]["data"][0]
        self.assertEqual(equity["y"], [-5, -125.5])
        self.assertIsInstance(equity["y"], list)
        self.assertEqual(figures["heatmap-0"]["data"][0]["z"], [[None, -125.5]])
        self.assertEqual(figures["intraday-drawdown"]["data"][0]["y"], [-12, -180])

    def test_curves_keep_development_order_with_zero_and_holdout_references(self):
        """Пять исходных серий видимы, остальные доступны в легенде; границы заданы."""
        curves = [{"label": f"{index} / 30", "dates": ["2025-12-30", "2026-02-01"],
                   "net": [-index, -index * 2], "gross": [0, -index],
                   "drawdown": [-index, -index * 2]} for index in range(8)]
        _, parser = self.render({"symbol": "Si", "meta": {"holdout_start": "2026-02-01"},
                                 "curves": curves})
        fig = json.loads(parser.scripts["figure-data"])["equity"]
        self.assertEqual([trace["name"] for trace in fig["data"]],
                         [curve["label"] for curve in curves])
        self.assertTrue(all(trace.get("visible", True) is True for trace in fig["data"][:5]))
        self.assertTrue(all(trace["visible"] == "legendonly" for trace in fig["data"][5:]))
        self.assertTrue(fig["layout"]["yaxis"]["zeroline"])
        self.assertIn("tozero", fig["layout"]["yaxis"]["rangemode"])
        self.assertTrue(any(shape.get("x0") == "2026-02-01"
                            for shape in fig["layout"]["shapes"]))

    def test_monthly_cost_and_walk_forward_data_are_preserved(self):
        """Месяцы, стоимость исполнения и окна без сделок доступны в отчёте."""
        text, parser = self.render({
            "symbol": "Si", "meta": {"base_cost_ticks": 2},
            "monthly": [{"label": "10 / 30", "months": ["2026-01", "2026-02"],
                         "pnl": [-10, 0]}],
            "costs": [{"label": "10 / 30", "cost_ticks": [0, 1, 2],
                       "development_pnl": [5, 2, -1], "holdout_pnl": [0, -5, -10]}],
            "walk_forward": {"dates": ["2024-01-04", "2025-01-03"], "net": [1, -1],
                             "gross": [3, 5], "drawdown": [0, -2],
                             "folds": [{"year": 2024, "label": "без сделок", "trades": 0}]},
            "coverage": [{"date": "2026-01-05", "status": "пропуск", "reason": "нет файла"}],
            "trades": [{"date": "2026-01-06", "net_pnl": -10}],
        })
        figures = json.loads(parser.scripts["figure-data"])
        self.assertEqual(figures["monthly"]["data"][0]["y"], [-10, 0])
        self.assertEqual(figures["costs"]["data"][1]["y"], [0, -5, -10])
        self.assertEqual(figures["walk-forward"]["data"][0]["y"], [1, -1])
        self.assertIn("без сделок", text)
        self.assertIn("нет файла", text)
        self.assertIn("открытой позиции", text)


if __name__ == "__main__":
    unittest.main()
