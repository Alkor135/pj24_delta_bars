r"""Общий сборщик сделок QUIK для одновременно работающих графиков RTS и MIX.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe realtime_quik_collector.py
    .\.venv\Scripts\python.exe realtime_quik_collector.py --config realtime_quik.json

Обычно запускается графиками автоматически в скрытом процессе. Ручной запуск
показывает адрес API и ошибки настройки. Путь QUIK, порты, контракты и отдельная
папка журнала задаются в JSON. Соединение с QuikSharp общее для двух окон;
при разрыве выполняются повторное подключение и сверка сделок текущего дня.
Через idle_seconds без обращений процесс завершает работу. Ctrl+C останавливает
ручной запуск. Исторические базы не открываются и не изменяются.
"""

import argparse
from pathlib import Path
from threading import Thread
import time

from source.realtime_feed import DEFAULT_CONFIG, CollectorState, collect_quik, load_config, make_http_server
from tick_to_delta_bars import RunLock


def main(argv=None):
    """Разбирает argv и обслуживает общий API до простоя/остановки; возвращает код 0/2."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="общая конфигурация обоих графиков")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        folder = Path(config["cache_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        with RunLock(folder / "collector.lock"):
            state = CollectorState(config)
            server, worker = None, None
            try:
                server = make_http_server(state)
                server.timeout = 0.5
                worker = Thread(target=collect_quik, args=(state,), name="QUIK-сборщик")
                worker.start()
                print(f"Сборщик: http://127.0.0.1:{server.server_port}; журнал: {folder}", flush=True)
                while time.monotonic() - state.last_client < config["idle_seconds"]:
                    server.handle_request()
            except KeyboardInterrupt:
                print("Остановка сборщика…", flush=True)
            finally:
                state.stop.set()
                if server is not None:
                    server.server_close()
                if worker is not None:
                    worker.join()
                state.close()
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Ошибка сборщика: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
