"""Проверки расположения, способов запуска и отчётов скриптов из backtest/.

Запуск из корня проекта:
    python -m unittest -v tests.test_backtest_launch
    python -m unittest discover -s tests -v

Справка проверяется из корня, папки backtest и посторонней рабочей папки.
Синтетические SQLite и ZIP позволяют выполнить все четыре скрипта полностью,
проверить SHA256 исходников и относительные пути результатов без реальных котировок.
"""

from contextlib import closing
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("backtest_duration", "backtest_duration_reversed",
           "backtest_stochastic", "backtest_heikin_ashi")


def create_database(folder):
    """Создаёт в folder SQLite/ZIP одного дня и возвращает путь к базе.

    folder — существующий временный каталог. Шесть тиков и три бара покрывают
    вход, изменение цены и закрытие в 18:40; цены кратны шагу RTS.
    """
    day = "2022-01-03"
    stamps = [f"{day} {value}" for value in
              ("10:00:00", "10:00:01", "10:00:02", "10:01:02", "10:01:03", "18:40:00")]
    prices = [100, 110, 120, 90, 100, 110]
    ticks = pd.DataFrame(dict(datetime=stamps, last=prices, volume=1))
    archive = folder / "ticks.zip"
    with ZipFile(archive, "w") as target:
        target.writestr("ticks.csv", ticks.to_csv(index=False))
    bars = [dict(dataset_id="test", symbol="RTS", day=day, bar_index=index,
                 start_row=first + 1, end_row=last + 1, start_time=stamps[first],
                 end_time=stamps[last], open=prices[first], close=prices[last],
                 high=max(prices[first:last + 1]), low=min(prices[first:last + 1]),
                 is_complete=int(index < 2))
            for index, (first, last) in enumerate(((0, 1), (2, 3), (4, 5)))]
    database = folder / "bars.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        pd.DataFrame(bars).to_sql("bars", connection, index=False)
        pd.DataFrame([dict(dataset_id="test", symbol="RTS", day=day, status="ready",
                           file_path=str(archive), sha256=sha256(archive.read_bytes()).hexdigest(),
                           tick_count=6, bar_count=3)]).to_sql("days", connection, index=False)
        pd.DataFrame([dict(dataset_id="test", config_json="{}")]).to_sql(
            "datasets", connection, index=False)
    return database


class BacktestLaunchTests(unittest.TestCase):
    """Проверяет реальный запуск перенесённых скриптов без настройки PYTHONPATH."""

    def run_python(self, arguments, cwd):
        """Запускает Python с arguments из cwd и возвращает stdout при успехе.

        arguments — список аргументов, cwd — рабочая папка. Код ошибки или
        превышение минуты завершает проверку с диагностикой процесса.
        """
        environment = dict(os.environ, PYTHONUTF8="1")
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-B", *map(str, arguments)],
                                cwd=cwd, env=environment, capture_output=True,
                                text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_scripts_are_in_backtest_directory(self):
        """Все четыре скрипта находятся в backtest, а в корне таких файлов нет."""
        self.assertEqual(list(PROJECT_ROOT.glob("backtest*.py")), [])
        for name in SCRIPTS:
            with self.subTest(script=name):
                self.assertTrue((PROJECT_ROOT / "backtest" / (name + ".py")).is_file())

    def test_help_with_direct_module_and_external_launch(self):
        """Справка доступна по новому пути, через пакет и из другой папки."""
        with tempfile.TemporaryDirectory() as directory:
            for name in SCRIPTS:
                script = PROJECT_ROOT / "backtest" / (name + ".py")
                for arguments, cwd in (
                    ([Path("backtest") / script.name, "--help"], PROJECT_ROOT),
                    (["-m", "backtest." + name, "--help"], PROJECT_ROOT),
                    ([script, "--help"], Path(directory)),
                    ([script.name, "--help"], PROJECT_ROOT / "backtest"),
                ):
                    with self.subTest(script=name, arguments=arguments, cwd=cwd):
                        self.assertIn("--symbols", self.run_python(arguments, cwd))

    def test_reports_and_source_hashes_from_external_directory(self):
        """Четыре полных запуска создают отчёты и верные SHA256 из другой папки."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            database = create_database(folder)
            working = folder / "рабочая папка"
            working.mkdir()
            for name in SCRIPTS:
                with self.subTest(script=name):
                    script = PROJECT_ROOT / "backtest" / (name + ".py")
                    relative_output = Path("reports") / name
                    arguments = [script, "--db", database, "--symbols", "RTS",
                                 "--start", "2022-01-01", "--end", "2022-01-03"]
                    if name.startswith("backtest_duration"):
                        arguments += ["--entry-grid", "2:2:1", "--exit-grid", "5:5:1",
                                      "--min-trades", "1", "--cache-dir", folder / "cache",
                                      "--output-dir", relative_output]
                    else:
                        arguments += ["--output", relative_output]
                    self.run_python(arguments, working)
                    output = working / relative_output
                    self.assertEqual(len(list(output.rglob("report.html"))), 1)
                    metadata_files = list(output.rglob("metadata.json"))
                    self.assertEqual(len(metadata_files), 1)
                    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
                    hashes = metadata["implementation_sha256"]
                    self.assertIn("backtest/" + script.name, hashes)
                    for path, digest in hashes.items():
                        self.assertEqual(digest, sha256((PROJECT_ROOT / path).read_bytes()).hexdigest())

    def test_duration_cache_stays_in_project_root(self):
        """Прямой и обратный тесты сохраняют стандартный кэш в корне проекта."""
        runner = importlib.import_module("backtest.backtest_duration")
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.assertEqual(runner.arguments([], reverse=reverse).cache_dir,
                                 PROJECT_ROOT / ".duration_cache")


if __name__ == "__main__":
    unittest.main()
