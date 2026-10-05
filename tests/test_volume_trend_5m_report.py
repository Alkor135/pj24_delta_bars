"""Проверяет статистические выводы и автономность пятиминутного HTML-отчёта.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_volume_trend_5m_report

Синтетические примеры проверяют динамические выводы без значимых результатов,
отсутствие ложного подтверждения при NaN,
различие отрицательного эффекта и статистической значимости, размер семьи
и соответствие доверительного интервала центрированному bootstrap-тесту.
"""

import importlib.util
import unittest

import numpy as np
import pandas as pd


class FiveMinuteReportTests(unittest.TestCase):
    """Проверяет значимые ошибки вывода и самодостаточность отчёта."""

    def test_report_module_exists(self):
        """Требует отдельный воспроизводимый генератор пятиминутного отчёта."""
        self.assertIsNotNone(importlib.util.find_spec("research.volume_trend_5m_report"))

    def test_verdict_distinguishes_support_opposite_and_uncertainty(self):
        """Направление эффекта без значимого p не подтверждает и не опровергает гипотезу."""
        from research.volume_trend_5m_report import verdict
        self.assertEqual(verdict(-1., .01), "Поддерживается")
        self.assertEqual(verdict(1., .01), "Обратная связь")
        self.assertEqual(verdict(-1., .3), "Не подтверждена")
        self.assertEqual(verdict(1., np.nan), "Недостаточно данных")

    def test_summary_does_not_call_partial_support_universal(self):
        """Четыре значимых снижения из 16 не превращаются в общее подтверждение."""
        from research.volume_trend_5m_report import overall_verdict
        frame = pd.DataFrame({"difference": [-1.]*16, "p_holm": [.01]*4+[.3]*12})
        result = overall_verdict(frame)
        self.assertIn("Частичная поддержка", result)
        self.assertIn("4 из 16", result)
        frame["p_holm"] = np.nan
        self.assertIn("Недостаточно данных", overall_verdict(frame))

    def test_indicator_conclusions_follow_statistics_without_support(self):
        """Отсутствие значимости в frame не оставляет в тексте утверждений о поддержке."""
        from research.volume_trend_5m_report import indicator_conclusion
        frame = pd.DataFrame({"indicator": ["sma", "st"], "difference": [-1., -1.], "p_holm": [1., 1.]})
        for indicator in ("sma", "st"):
            self.assertIn("значимых различий нет", indicator_conclusion(frame, indicator))
            self.assertNotIn("значимое снижение обнаружено", indicator_conclusion(frame, indicator))
        frame.loc[frame.indicator.eq("st"), "p_holm"] = .01
        self.assertIn("значимое снижение обнаружено", indicator_conclusion(frame, "st"))
        self.assertIn("значимых различий нет", indicator_conclusion(frame, "sma"))

    def test_summary_explicitly_reports_mixed_directions(self):
        """При значимом снижении и росте общий текст сообщает оба направления."""
        from research.volume_trend_5m_report import overall_verdict
        frame = pd.DataFrame({"difference": [-1., 1.], "p_holm": [.01, .01]})
        self.assertIn("Смешанные результаты", overall_verdict(frame))

    def test_symmetric_interval_and_p_use_the_same_bootstrap_errors(self):
        """Симметричный ДИ и двустороннее p основаны на одинаковых центрированных ошибках."""
        from research.volume_trend_5m_report import bootstrap_summary
        values = np.tile([1., 3., 4., 6.], 50)
        ratios = np.tile([2., 2., .5, .5], 50)
        result = bootstrap_summary(values, ratios, np.zeros(200), repetitions=300, block=20)
        self.assertAlmostEqual(result["difference"], -3)
        self.assertAlmostEqual(result["difference"]-result["ci_low"], result["ci_high"]-result["difference"])
        self.assertLess(result["ci_high"], 0)
        self.assertLess(result["p"], .05)
        self.assertEqual(result, bootstrap_summary(values, ratios, np.zeros(200), repetitions=300, block=20))

    def test_insufficient_blocks_do_not_get_a_confidence_interval(self):
        """Неподвижное повторение двух коротких сегментов не даёт уверенного вывода."""
        from research.volume_trend_5m_report import bootstrap_summary
        result = bootstrap_summary(np.r_[np.zeros(20), np.ones(20)*10], np.r_[np.ones(20)*2, np.ones(20)*.5], np.repeat([0,1],20), repetitions=200, block=20)
        self.assertTrue(np.isnan(result["p"]))
        self.assertTrue(np.isnan(result["ci_low"]))

    def test_holm_family_is_sixteen_per_period_and_block(self):
        """Поправка охватывает оба окна и обе базы, не только отдельную красивую таблицу."""
        from research.volume_trend_5m_report import adjust_families
        rows = [{"period": "2026", "block": 20, "difference": -1., "p": .001 if i==0 else .5} for i in range(16)]
        data = adjust_families(pd.DataFrame(rows))
        self.assertAlmostEqual(data.p_holm.iloc[0], .016)


if __name__ == "__main__":
    unittest.main()
