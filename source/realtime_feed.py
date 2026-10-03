"""Общий сборщик рыночных сделок и локальный API двух графиков RTS/MIX.

Примеры из корня: python realtime_quik_collector.py --config realtime_quik.json
Проверки: python -m unittest -v tests.test_realtime_feed
Сборщик — единственный клиент QuikSharp. HTTP на localhost выдаёт отдельные
курсоры журналов; графики автоматически запускают процесс без видимого терминала.
После 90 секунд без обращений он завершается. Исторические БД не открываются.
Перед закрытием моста сохраняется хвост callback, включая ошибку снимка и полночь.
"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from queue import Empty
import socket
import subprocess
import sys
from threading import Event, RLock
import time
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

from source.quik_bridge import QuikBridge, select_contract
from source.realtime_core import TradeJournal, parse_trade, positive_integer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "realtime_quik.json"
PROTOCOL = "delta-quik-v1"
MOSCOW = timezone(timedelta(hours=3))


def moscow_day():
    """Возвращает текущую календарную дату по Москве в формате ISO."""
    return datetime.now(MOSCOW).date().isoformat()


def load_config(path=DEFAULT_CONFIG):
    """Читает общий JSON path, проверяет значения и разрешает cache_dir относительно файла."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("Эта версия рассчитана на локальный терминал QUIK")
    for key in ("requests_port", "callbacks_port", "collector_port"):
        config[key] = positive_integer(config[key], key)
        if config[key] > 65535:
            raise ValueError(f"Недопустимый порт {key}")
    if len({config[key] for key in ("requests_port", "callbacks_port", "collector_port")}) != 3:
        raise ValueError("Порты запросов, событий и сборщика должны различаться")
    for key in ("request_timeout", "reconcile_seconds", "idle_seconds"):
        config[key] = positive_integer(config.get(key, {"request_timeout": 15, "reconcile_seconds": 60, "idle_seconds": 90}[key]), key)
    for symbol in ("RTS", "MIX"):
        settings = config["instruments"][symbol]
        if not settings.get("class_code") or not settings.get("prefix"):
            raise ValueError(f"Не заданы класс и префикс {symbol}")
        scale = float(settings.get("price_scale", 1))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Неверный масштаб цены {symbol}")
        settings["price_scale"] = scale
    cache = Path(config["cache_dir"])
    config["cache_dir"] = str((cache if cache.is_absolute() else path.parent / cache).resolve())
    return config


def config_identity(config):
    """Возвращает идентификатор конфигурации config для совместного подключения окон."""
    return sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class CollectorState:
    """Согласует состояние связи и отдельный журнал между сборщиком и HTTP-потоками."""

    def __init__(self, config):
        """Создаёт журнал в cache_dir конфигурации config и начальный статус."""
        self.config, self.lock, self.stop = config, RLock(), Event()
        self.journal = TradeJournal(Path(config["cache_dir"]) / "quik_trades.sqlite3")
        self.last_client = time.monotonic()
        self.status = dict(protocol=PROTOCOL, config_id=config_identity(config), day=moscow_day(),
                           feed_connected=False, message="Ожидание QUIK", contracts={}, revision=0,
                           rejected_trades=0, last_trades={}, full_day_confirmed=False)

    def update(self, **values):
        """Атомарно заменяет переданные поля values статуса сборщика."""
        with self.lock:
            self.status.update(values)

    def snapshot(self):
        """Возвращает независимую копию текущего статуса для HTTP-клиента."""
        with self.lock:
            return json.loads(json.dumps(self.status, ensure_ascii=False))

    def ingest(self, symbol, payloads):
        """Нормализует payloads инструмента symbol и записывает новые сделки в журнал."""
        trades = []
        for payload in payloads:
            try:
                trades.append(parse_trade(payload, self.config["instruments"][symbol]["price_scale"]))
            except ValueError as exc:
                with self.lock:
                    self.status["rejected_trades"] += 1
                    self.status["message"] = f"Ошибка сделки {symbol}: {exc}"
        added = self.journal.append(symbol, trades)
        if added:
            with self.lock:
                self.status["revision"] += added
                self.status["last_trades"][symbol] = max(trades, key=lambda trade: trade.order).to_dict()
        return added

    def remembered_contract(self, symbol, day):
        """Возвращает контракт уже записанного дня symbol/day, запрещая смешанные серии."""
        with self.journal.lock:
            rows = self.journal.connection.execute(
                "SELECT DISTINCT class_code,sec_code FROM trades WHERE symbol=? AND day=?", (symbol, day)).fetchall()
        if len(rows) > 1:
            raise ValueError(f"В журнале {symbol} за {day} смешаны контракты")
        return dict(class_code=rows[0][0], sec_code=rows[0][1]) if rows else None

    def close(self):
        """Останавливает сбор и закрывает журнал после завершения обслуживающих потоков."""
        self.stop.set()
        self.journal.close()


def drain_callbacks(state, bridge, contracts, day, limit=20000):
    """Сохраняет события bridge для contracts/day и возвращает причину разрыва или None.

    state принимает сделки в журнал. limit ограничивает обычный пакет;
    limit=None полностью опустошает очередь после остановки приёмного потока.
    Сообщение о разрыве не мешает сохранить следующие уже принятые сделки.
    """
    batches, disconnected_reason, processed = {symbol: [] for symbol in contracts}, None, 0
    expected_date = tuple(map(int, day.split("-")))
    while limit is None or processed < limit:
        try:
            event = bridge.events.get_nowait()
        except Empty:
            break
        processed += 1
        cmd = event.get("cmd")
        if cmd in ("bridge_error", "OnDisconnected", "OnClose", "OnStop"):
            disconnected_reason = str(event.get("data") or "QUIK отключён")
        elif cmd == "OnAllTrade":
            trade = event.get("data", {})
            for symbol, contract in contracts.items():
                if (trade.get("class_code"), trade.get("sec_code")) == (contract["class_code"], contract["sec_code"]):
                    stamp = trade.get("datetime", {})
                    if (stamp.get("year"), stamp.get("month"), stamp.get("day")) == expected_date:
                        batches[symbol].append(trade)
    for symbol, trades in batches.items():
        if trades:
            state.ingest(symbol, trades)
    return disconnected_reason


def collect_quik(state):
    """Собирает сделки для state до stop; после остановки моста сохраняет весь хвост очереди."""
    bridge = None
    try:
        while not state.stop.is_set():
            contracts = {}
            try:
                day = moscow_day()
                state.update(day=day, feed_connected=False, message="Подключение к QUIK…")
                bridge = QuikBridge(state.config["host"], state.config["requests_port"],
                                    state.config["callbacks_port"], state.config["request_timeout"])
                if bridge.request("isConnected") != 1:
                    raise ConnectionError("QUIK не подключён к серверу брокера")
                for symbol, settings in state.config["instruments"].items():
                    remembered = state.remembered_contract(symbol, day)
                    if remembered and settings.get("sec_code") and settings["sec_code"] != remembered["sec_code"]:
                        raise ValueError(f"{symbol}: в журнале дня уже другой контракт; не меняйте серию внутри дня")
                    contracts[symbol] = select_contract(bridge, symbol, dict(settings, **(remembered or {})), day)
                state.update(contracts=contracts)
                reconcile_at = 0
                while not state.stop.is_set():
                    if moscow_day() != day:
                        break
                    if time.monotonic() >= reconcile_at:
                        if bridge.request("isConnected") != 1:
                            raise ConnectionError("Потеряно соединение QUIK с брокером")
                        state.update(feed_connected=False, message="Сверка сделок текущего дня…")
                        for symbol, contract in contracts.items():
                            trades = bridge.request("get_all_trades", f"{contract['class_code']}|{contract['sec_code']}")
                            if not isinstance(trades, list):
                                raise ValueError("QUIK вернул неверный снимок обезличенных сделок")
                            selected = []
                            for trade in trades:
                                stamp = trade.get("datetime", {})
                                if (stamp.get("year"), stamp.get("month"), stamp.get("day")) == tuple(map(int, day.split("-"))):
                                    selected.append(trade)
                            state.ingest(symbol, selected)
                        state.update(feed_connected=True, message="QUIK подключён; полнота дня не подтверждена")
                        reconcile_at = time.monotonic() + state.config["reconcile_seconds"]
                    disconnected_reason = drain_callbacks(state, bridge, contracts, day)
                    if disconnected_reason is not None:
                        raise ConnectionError(disconnected_reason)
                    state.stop.wait(0.05)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                state.update(feed_connected=False, message=f"QUIK недоступен: {exc}")
                state.stop.wait(3)
            finally:
                if bridge is not None:
                    bridge.close()
                    drain_callbacks(state, bridge, contracts, day, limit=None)
                    bridge = None
    finally:
        state.update(feed_connected=False, message="Сборщик остановлен")


def make_http_server(state, port=None):
    """Возвращает HTTP-сервер localhost для state; port=0 выбирает свободный тестовый порт."""
    class Handler(BaseHTTPRequestHandler):
        """Выдаёт состояние и сделки без доступа к торговым функциям или файлам."""

        def do_GET(self):
            """Обрабатывает /status и /trades; параметры задают инструмент, дату и курсор."""
            state.last_client = time.monotonic()
            route = urlparse(self.path)
            try:
                if route.path == "/status":
                    result = state.snapshot()
                elif route.path == "/trades":
                    params = parse_qs(route.query)
                    symbol = params.get("symbol", [""])[0]
                    if symbol not in state.config["instruments"]:
                        raise ValueError("Неизвестный инструмент")
                    day = params.get("day", [None])[0]
                    result = state.journal.read(symbol, params.get("after", [0])[0], day=day)
                    result["status"] = state.snapshot()
                else:
                    self.send_error(404)
                    return
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError) as exc:
                self.send_error(400, str(exc).encode("ascii", "replace").decode("ascii"))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            """Отключает стандартный журнал HTTP; format/args — параметры базового сервера."""

    return ThreadingHTTPServer(("127.0.0.1", state.config["collector_port"] if port is None else port), Handler)


def feed_request(config, route, **params):
    """Возвращает JSON локального API route с params, не используя системные HTTP-прокси."""
    from urllib.request import ProxyHandler, build_opener
    query = "?" + urlencode(params) if params else ""
    url = f"http://127.0.0.1:{config['collector_port']}/{route}{query}"
    with build_opener(ProxyHandler({})).open(url, timeout=3) as response:
        result = json.load(response)
    status = result.get("status", result)
    if status.get("protocol") != PROTOCOL or status.get("config_id") != config_identity(config):
        raise ValueError("На порту сборщика другой процесс или другая конфигурация")
    return result


def ensure_collector(config, config_path=DEFAULT_CONFIG):
    """Возвращает статус общего процесса; при отсутствии запускает его скрыто и ждёт API."""
    try:
        return feed_request(config, "status")
    except (URLError, OSError):
        pass
    kwargs = dict(cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen([sys.executable, str(PROJECT_ROOT / "realtime_quik_collector.py"), "--config", str(Path(config_path).resolve())], **kwargs)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            return feed_request(config, "status")
        except (URLError, OSError):
            time.sleep(0.1)
    raise ConnectionError("Сборщик не запустился; запустите realtime_quik_collector.py для диагностики")
