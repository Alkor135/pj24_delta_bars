"""Проверяет реальный JSON/TCP-мост QUIK и общий сборщик двух графиков.

Запуск: .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_realtime_feed
Используются локальные тестовые TCP/HTTP-серверы без брокера и торговых операций.
"""

from datetime import date
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen


def raw_trade(number=1, sec_code="RIZ6"):
    """Возвращает таблицу обезличенной сделки с точным номером и биржевым временем."""
    return dict(trade_num=number, class_code="SPBFUT", sec_code=sec_code, price=100, qty=2,
                datetime=dict(year=2026, month=10, day=2, hour=10, min=0, sec=number % 60, ms=12, mcs=12345))


class FeedTests(unittest.TestCase):
    """Проверяет настоящие сообщения и независимые курсоры двух инструментов."""

    def setUp(self):
        """Требует реализацию транспорта и общего сборщика до выполнения проверки."""
        for module in ("source.quik_bridge", "source.realtime_feed"):
            self.assertIsNotNone(importlib.util.find_spec(module), f"Ещё не реализован {module}")

    def test_bridge_reads_fragmented_json_and_receives_callback(self):
        """Разделённый пакет и callback во время запроса не теряют сделку/ответ."""
        from source.quik_bridge import QuikBridge
        with socket.socket() as requests, socket.socket() as callbacks:
            requests.bind(("127.0.0.1", 0))
            callbacks.bind(("127.0.0.1", 0))
            requests.listen()
            callbacks.listen()

            def serve():
                """Обслуживает один реальный запрос и посылает callback отдельным сокетом."""
                with requests.accept()[0] as response, callbacks.accept()[0] as callback:
                    message = json.loads(response.makefile("rb").readline())
                    callback.sendall((json.dumps(dict(cmd="OnAllTrade", data=raw_trade())) + "\n").encode())
                    result = (json.dumps(dict(message, data=1)) + "\n").encode()
                    response.sendall(result[:7])
                    response.sendall(result[7:])

            server = Thread(target=serve)
            server.start()
            bridge = QuikBridge("127.0.0.1", requests.getsockname()[1], callbacks.getsockname()[1], timeout=2)
            try:
                self.assertEqual(bridge.request("isConnected"), 1)
                self.assertEqual(bridge.events.get(timeout=2)["data"]["trade_num"], 1)
                with self.assertRaises(ValueError):
                    bridge.request("sendTransaction", "опасный запрос")
            finally:
                bridge.close()
                server.join(3)

    def test_contract_selects_nearest_unexpired_and_honors_explicit(self):
        """Автовыбор исключает старую серию; явный код сохраняет выбор пользователя."""
        from source.quik_bridge import select_contract

        class ReferenceBridge:
            """Предоставляет фиксированный биржевой справочник без сетевого QUIK."""

            def request(self, cmd, data=""):
                """Возвращает реальные формы списка инструментов и getSecurityInfo."""
                if cmd == "getClassSecurities":
                    return "RIU6,RIZ6,RIH7,MXZ6,RIZ6C100000,"
                code = data.split("|")[1]
                return dict(sec_code=code, mat_date={"RIU6": 20260917, "RIZ6": 20261217, "RIH7": 20270318, "MXZ6": 20261217}[code])

        info = dict(class_code="SPBFUT", prefix="RI", sec_code=None)
        self.assertEqual(select_contract(ReferenceBridge(), "RTS", info, date(2026, 10, 2))["sec_code"], "RIZ6")
        self.assertEqual(select_contract(ReferenceBridge(), "RTS", dict(info, sec_code="RIH7"), date(2026, 10, 2))["sec_code"], "RIH7")

    def test_two_http_clients_keep_independent_cursors_and_journal(self):
        """Оба графика читают один сборщик; поздняя новая запись остаётся доступной."""
        from source.realtime_feed import CollectorState, make_http_server, load_config
        with tempfile.TemporaryDirectory() as folder:
            config = load_config()
            config["cache_dir"] = folder
            state = CollectorState(config)
            state.ingest("RTS", [raw_trade(2), raw_trade(2)])
            state.ingest("MIX", [raw_trade(1, "MXZ6")])
            server = make_http_server(state, port=0)
            thread = Thread(target=server.serve_forever)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/trades?symbol=RTS&after=0") as response:
                    rts = json.load(response)
                with urlopen(url + "/trades?symbol=MIX&after=0") as response:
                    mix = json.load(response)
                self.assertEqual(len(rts["trades"]), 1)
                self.assertEqual(mix["trades"][0]["sec_code"], "MXZ6")
                state.ingest("RTS", [raw_trade(1)])
                with urlopen(url + f"/trades?symbol=RTS&after={rts['cursor']}") as response:
                    late = json.load(response)
                self.assertEqual([t["trade_num"] for t in late["trades"]], ["1"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(3)
                state.close()

    def test_callback_before_disconnect_is_written_before_recovery(self):
        """Сделка перед разрывом сохраняется, даже если последующего снимка пока нет."""
        from source.realtime_feed import CollectorState, collect_quik, load_config, moscow_day
        with tempfile.TemporaryDirectory() as folder, socket.socket() as requests, socket.socket() as callbacks:
            requests.bind(("127.0.0.1", 0))
            callbacks.bind(("127.0.0.1", 0))
            requests.listen()
            callbacks.listen()
            config = load_config()
            config.update(cache_dir=folder, requests_port=requests.getsockname()[1],
                          callbacks_port=callbacks.getsockname()[1], request_timeout=2)
            config["instruments"]["RTS"]["sec_code"] = "RIZ6"
            config["instruments"]["MIX"]["sec_code"] = "MXZ6"
            trade = raw_trade(2)
            year, month, day = map(int, moscow_day().split("-"))
            trade["datetime"].update(year=year, month=month, day=day)

            def serve():
                """Пустой снимок сопровождается новой сделкой и немедленным разрывом событий."""
                with requests.accept()[0] as response, callbacks.accept()[0] as callback:
                    with response.makefile("rb") as stream:
                        for line in stream:
                            msg = json.loads(line)
                            cmd = msg["cmd"]
                            result = 1 if cmd == "isConnected" else (dict(sec_code=msg["data"].split("|")[-1]) if cmd == "getSecurityInfo" else [])
                            response.sendall((json.dumps(dict(msg, data=result)) + "\n").encode())
                            if cmd == "get_all_trades" and msg["data"].endswith("MXZ6"):
                                callback.sendall((json.dumps(dict(cmd="OnAllTrade", data=trade)) + "\n" +
                                                  json.dumps(dict(cmd="OnDisconnected", data="")) + "\n").encode())

            state = CollectorState(config)
            server = Thread(target=serve)
            worker = Thread(target=collect_quik, args=(state,))
            server.start()
            worker.start()
            try:
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    if "недоступен" in state.snapshot()["message"]:
                        break
                    time.sleep(0.02)
                self.assertEqual([row["trade_num"] for row in state.journal.read("RTS")["trades"]], ["2"])
            finally:
                state.stop.set()
                worker.join(4)
                server.join(4)
                state.close()

    def test_pending_callbacks_survive_snapshot_failure_and_midnight(self):
        """Ошибка снимка и полночь сохраняют хвост очереди, в том числе поступивший при закрытии."""
        from queue import Queue
        from source.realtime_feed import CollectorState, collect_quik, load_config
        for failure in ("snapshot", "midnight"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                config = load_config()
                config["cache_dir"] = folder
                state = CollectorState(config)
                bridge_events = Queue()
                bridge_events.put(dict(cmd="OnAllTrade", data=raw_trade(1)))

                class PendingBridge:
                    """Оставляет callback в очереди при ошибке запроса или смене даты."""

                    def __init__(self, *args):
                        """Предоставляет заранее принятую очередь events без реальных сокетов."""
                        self.events = bridge_events

                    def request(self, cmd, data=""):
                        """Возвращает связь либо моделирует ошибку снимка с непустой очередью."""
                        if cmd == "get_all_trades":
                            state.stop.set()
                            raise TimeoutError("Таймаут снимка")
                        return 1

                    def close(self):
                        """Добавляет последний принятый callback до остановки читающего потока."""
                        self.events.put(dict(cmd="OnAllTrade", data=raw_trade(2)))
                        state.stop.set()

                def contract(bridge, symbol, settings, day):
                    """Возвращает фиксированный контракт инструмента без обращения к справочнику."""
                    return dict(settings, sec_code="RIZ6" if symbol == "RTS" else "MXZ6")

                dates = iter(["2026-10-02", "2026-10-03"] if failure == "midnight" else ["2026-10-02"] * 2)
                with patch("source.realtime_feed.QuikBridge", PendingBridge), \
                     patch("source.realtime_feed.select_contract", contract), \
                     patch("source.realtime_feed.moscow_day", side_effect=lambda: next(dates)):
                    try:
                        collect_quik(state)
                        self.assertEqual([row["trade_num"] for row in state.journal.read("RTS")["trades"]], ["1", "2"])
                    finally:
                        state.close()


if __name__ == "__main__":
    unittest.main()
