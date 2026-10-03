"""Читает рыночные данные установленного QuikSharp по двум JSON/TCP-соединениям.

Примеры из корня: python realtime_quik_collector.py --config realtime_quik.json
Проверки: python -m unittest -v tests.test_realtime_feed
Запросы ограничены состоянием связи, справочником и обезличенными сделками.
Callback читается в отдельном потоке; при ошибке соединение закрывается и
пересоздаётся сборщиком. Торговые запросы этим модулем не поддерживаются.
"""

from datetime import date
import json
from queue import Queue
import re
import socket
from threading import Event, Lock, Thread

READ_COMMANDS = frozenset(("ping", "isConnected", "getClassSecurities", "getSecurityInfo", "get_all_trades"))


class QuikBridge:
    """Передаёт разрешённые запросы и независимо принимает события QuikSharp."""

    def __init__(self, host="127.0.0.1", requests_port=34130, callbacks_port=34131, timeout=10):
        """Подключает два порта host с timeout секунд; публикует события в events."""
        self.events, self.stopping, self.lock = Queue(), Event(), Lock()
        self._sockets, self._files, self._counter = [], [], 0
        self.thread = None
        try:
            response = socket.create_connection((host, requests_port), timeout)
            self._sockets.append(response)
            callback = socket.create_connection((host, callbacks_port), timeout)
            self._sockets.append(callback)
            callback.settimeout(None)
            self._response_file, self._callback_file = response.makefile("rb"), callback.makefile("rb")
            self._files.extend((self._response_file, self._callback_file))
            self.thread = Thread(target=self._read_callbacks, name="QUIK-callback", daemon=True)
            self.thread.start()
        except Exception:
            self.close()
            raise

    def _read_callbacks(self):
        """Читает JSON-строки callback до остановки, включая сообщения о разрыве связи."""
        try:
            while not self.stopping.is_set():
                line = self._callback_file.readline()
                if not line:
                    raise ConnectionError("QUIK закрыл соединение событий")
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("QUIK прислал неверное сообщение события")
                self.events.put(message)
        except (OSError, ValueError) as exc:
            if not self.stopping.is_set():
                self.events.put(dict(cmd="bridge_error", data=str(exc)))

    def request(self, cmd, data=""):
        """Возвращает data ответа на разрешённый cmd; ошибки и несовпадение id возбуждают исключение."""
        if cmd not in READ_COMMANDS:
            raise ValueError(f"Запрос {cmd} не разрешён рыночным клиентом")
        with self.lock:
            self._counter += 1
            payload = json.dumps(dict(cmd=cmd, data=data, id=self._counter, t=""), ensure_ascii=False)
            self._sockets[0].sendall((payload + "\n").encode("utf-8"))
            line = self._response_file.readline()
            if not line:
                raise ConnectionError("QUIK закрыл соединение запросов")
            message = json.loads(line)
            if message.get("cmd") in ("lua_error", "error") or message.get("error"):
                raise ValueError(f"Ошибка Lua: {message.get('data', message.get('error'))}")
            if message.get("id") != self._counter:
                raise ValueError("QUIK прислал ответ на другой запрос")
            return message.get("data")

    def close(self):
        """Останавливает приём событий и закрывает оба сокета; повторное закрытие допустимо."""
        self.stopping.set()
        for connection in self._sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(3)
        for stream in self._files:
            stream.close()
        self._sockets.clear()
        self._files.clear()


def select_contract(bridge, symbol, settings, today):
    """Возвращает контракт symbol по settings и справочнику bridge для даты today.

    Явный sec_code проверяется существованием. Без него выбирается ближайшая
    неистёкшая серия prefix по mat_date; опционы и неизвестные даты исключены.
    """
    today = date.fromisoformat(str(today))
    class_code, explicit = settings["class_code"], settings.get("sec_code")
    if explicit:
        info = bridge.request("getSecurityInfo", f"{class_code}|{explicit}")
        if not isinstance(info, dict) or not info:
            raise ValueError(f"Нет контракта {class_code}/{explicit} для {symbol}")
        return dict(settings, sec_code=explicit)
    prefix = settings["prefix"]
    codes = bridge.request("getClassSecurities", class_code)
    if not isinstance(codes, str):
        raise ValueError(f"QUIK не загрузил справочник класса {class_code}")
    candidates = []
    for code in codes.split(","):
        if not re.fullmatch(re.escape(prefix) + r"[FGHJKMNQUVXZ]\d{1,2}", code):
            continue
        info = bridge.request("getSecurityInfo", f"{class_code}|{code}")
        try:
            expiry = date.fromisoformat(str(int(info["mat_date"])))
        except (KeyError, TypeError, ValueError):
            continue
        if expiry >= today:
            candidates.append((expiry, code))
    if not candidates:
        raise ValueError(f"Нет действующего контракта {prefix} для {symbol}; задайте sec_code в конфигурации")
    return dict(settings, sec_code=min(candidates)[1])
