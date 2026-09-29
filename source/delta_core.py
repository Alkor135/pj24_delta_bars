"""Чтение тиков Финама, подбор порога и построение адаптивных дельта-баров.

Самостоятельный запуск не нужен. Пример использования через конвертер:
    python tick_to_delta_bars.py --symbols RTS MIX --start 2022-01-01
Проверки: python -m unittest -v tests.test_delta_bars.CoreTests
Требуется только стандартная библиотека Python 3.10+.
"""

from array import array
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
import csv
import io
import math
from pathlib import Path
import re
from zipfile import ZipFile

NS_SECOND = 1_000_000_000
LIMIT = (1 << 62) - 1
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?")


class DeltaPath:
    """Ищет первое пересечение порога через дерево экстремумов монотонных участков."""

    def __init__(self, values):
        """Строит компактный индекс кумулятивной дельты без изменения последовательности тиков."""
        self.values = values
        self.run_ends = array("q")
        previous, direction = 0, 0
        for i, value in enumerate(values):
            change = value - previous
            sign = (change > 0) - (change < 0)
            if sign:
                if direction and direction != sign:
                    self.run_ends.append(i - 1)
                direction = sign
            previous = value
        if values:
            self.run_ends.append(len(values) - 1)
        self.size = 1 << max(0, (len(self.run_ends) - 1).bit_length())
        self.minimum = array("q", [LIMIT]) * (2 * self.size)
        self.maximum = array("q", [-LIMIT]) * (2 * self.size)
        for i, end in enumerate(self.run_ends):
            self.minimum[self.size + i] = values[end]
            self.maximum[self.size + i] = values[end]
        for i in range(self.size - 1, 0, -1):
            self.minimum[i] = min(self.minimum[2 * i], self.minimum[2 * i + 1])
            self.maximum[i] = max(self.maximum[2 * i], self.maximum[2 * i + 1])

    def _outside_run(self, start, lower, upper):
        """Находит первый последующий участок, чей конец достигает одной из границ."""
        stack = [(1, 0, self.size)]
        minimum, maximum = self.minimum, self.maximum
        while stack:
            node, left, right = stack.pop()
            if right <= start or left >= len(self.run_ends):
                continue
            if minimum[node] > lower and maximum[node] < upper:
                continue
            if right - left == 1:
                return left
            middle = (left + right) // 2
            stack.append((node * 2 + 1, middle, right))
            stack.append((node * 2, left, middle))
        return None

    def crossing(self, start, threshold):
        """Возвращает индекс первого тика, закрывающего бар от позиции start, либо None."""
        if start >= len(self.values):
            return None
        baseline = self.values[start - 1] if start else 0
        lower, upper = baseline - threshold, baseline + threshold
        run = bisect_left(self.run_ends, start)
        endpoint = self.values[self.run_ends[run]]
        if lower < endpoint < upper:
            run = self._outside_run(run + 1, lower, upper)
            if run is None:
                return None
        left = max(start, self.run_ends[run - 1] + 1 if run else 0)
        right = self.run_ends[run]
        rising = self.values[right] >= upper
        while left < right:
            middle = (left + right) // 2
            value = self.values[middle]
            if (rising and value >= upper) or (not rising and value <= lower):
                right = middle
            else:
                left = middle + 1
        return left

    def ends(self, threshold):
        """Перечисляет концы всех баров, включая остаток календарного дня."""
        if threshold < 1:
            raise ValueError("Порог дельты должен быть положительным целым числом")
        start, length = 0, len(self.values)
        while start < length:
            end = self.crossing(start, threshold)
            if end is None:
                yield length - 1
                return
            yield end
            start = end + 1

    def count(self, threshold, limit=None):
        """Считает бары; при заданном ограничении возвращает не более limit + 1."""
        count = 0
        for _ in self.ends(threshold):
            count += 1
            if limit is not None and count > limit:
                break
        return count


@dataclass
class HistoryDay:
    """Хранит только данные, необходимые для повторной калибровки порога."""

    day: date
    path: DeltaPath
    active_intervals: int
    volume: int


@dataclass
class TickDay:
    """Представляет один проверенный дневной файл с исходным порядком сделок."""

    day: date
    times: array
    prices: array
    volumes: array
    path: DeltaPath
    active_intervals: int
    volume: int

    def history(self):
        """Возвращает представление для окна обучения без цен и времён отдельных тиков."""
        return HistoryDay(self.day, self.path, self.active_intervals, self.volume)


@dataclass(frozen=True)
class Calibration:
    """Сохраняет выбранный порог и наблюдавшуюся точность подбора."""

    threshold: int
    target_count: int
    actual_count: int
    evaluations: int

    @property
    def relative_error(self):
        """Возвращает относительную ошибку числа баров на обучающем окне."""
        return (self.actual_count - self.target_count) / self.target_count


def parse_time(value, expected_day):
    """Проверяет московскую дату и преобразует время в наносекунды от полуночи."""
    match = STAMP.fullmatch(value.strip())
    if match is None or match[1] != expected_day.isoformat():
        raise ValueError(f"Некорректная дата/время: {value!r}; ожидается {expected_day}")
    hour, minute, second = map(int, (match[2], match[3], match[4]))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"Некорректное время: {value!r}")
    fraction = int((match[5] or "").ljust(9, "0"))
    return (hour * 3600 + minute * 60 + second) * NS_SECOND + fraction


def format_time(day, value):
    """Форматирует московскую метку времени, сохраняя девять знаков долей секунды."""
    seconds, fraction = divmod(value, NS_SECOND)
    hour, seconds = divmod(seconds, 3600)
    minute, second = divmod(seconds, 60)
    return f"{day} {hour:02d}:{minute:02d}:{second:02d}.{fraction:09d}"


def read_day(path, day, minutes=5):
    """Проверяет дневной ZIP и вычисляет знаки сделок с ежедневным сбросом tick rule."""
    if minutes < 1:
        raise ValueError("Размер временного интервала должен быть положительным")
    path = Path(path)
    times, prices, volumes, cumulative = array("q"), array("d"), array("q"), array("q")
    previous_price, direction, delta, total_volume = None, 0, 0, 0
    active = set()
    with ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise ValueError(f"{path}: ожидается ровно один CSV внутри ZIP")
        with archive.open(members[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != ["datetime", "last", "volume"]:
                raise ValueError(f"{path}: ожидается заголовок datetime,last,volume")
            for row_number, row in enumerate(reader, 1):
                try:
                    if len(row) != 3:
                        raise ValueError("ожидаются три столбца")
                    stamp = parse_time(row[0], day)
                    price, volume = float(row[1]), int(row[2])
                    if not math.isfinite(price) or price <= 0 or volume <= 0:
                        raise ValueError("цена и целый объём должны быть положительными")
                    if times and stamp < times[-1]:
                        raise ValueError("нарушен порядок времени")
                    total_volume += volume
                    if total_volume >= LIMIT:
                        raise ValueError("объём превышает поддерживаемый диапазон int64")
                    if previous_price is not None and price != previous_price:
                        direction = 1 if price > previous_price else -1
                    delta += direction * volume
                    times.append(stamp)
                    prices.append(price)
                    volumes.append(volume)
                    cumulative.append(delta)
                    active.add(stamp // (minutes * 60 * NS_SECOND))
                    previous_price = price
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"{path.name}, строка данных {row_number}: {exc}") from exc
    return TickDay(day, times, prices, volumes, DeltaPath(cumulative), len(active), total_volume)


def calibrate(history, previous_threshold=None):
    """Подбирает целый порог по реальному числу баров прошлого окна без текущего дня."""
    if not history or any(not day.path.values for day in history):
        raise ValueError("Для калибровки нужны непустые дни")
    target = sum(day.active_intervals for day in history)
    if target <= 0:
        raise ValueError("В обучающем окне нет непустых временных интервалов")
    evaluated = {}
    cap = 2 * target

    def evaluate(threshold):
        """Считает результат кандидата с ранней остановкой заведомо слишком частых баров."""
        threshold = max(1, int(threshold))
        if threshold not in evaluated:
            count = 0
            for day in history:
                count += day.path.count(threshold, cap - count)
                if count > cap:
                    break
            evaluated[threshold] = count
        return evaluated[threshold]

    seed = max(1, previous_threshold or sum(day.volume for day in history) // target)
    low = high = seed
    result = evaluate(seed)
    if result < target:
        while low > 1 and evaluate(low) < target:
            high, low = low, max(1, low // 2)
    elif result > target:
        while evaluate(high) > target:
            low, high = high, high * 2
    for _ in range(24):
        if high - low <= 1 or target in evaluated.values():
            break
        middle = (low + high) // 2
        if evaluate(middle) >= target:
            low = middle
        else:
            high = middle
    best = min(evaluated, key=lambda value: (abs(evaluated[value] - target), value))
    # Локальные проверки смягчают немонотонность, возникающую из-за неделимых сделок.
    for distance in sorted({1, max(1, best // 50), max(1, best // 20), max(1, best // 10)}):
        evaluate(best - distance)
        evaluate(best + distance)
    best = min(evaluated, key=lambda value: (abs(evaluated[value] - target), value))
    actual = sum(day.path.count(best) for day in history)
    return Calibration(best, target, actual, len(evaluated))


def build_bars(ticks, threshold):
    """Формирует OHLC и объёмы баров, включая последний незавершённый бар дня."""
    bars, start = [], 0
    values, prices, volumes = ticks.path.values, ticks.prices, ticks.volumes
    for index, end in enumerate(ticks.path.ends(threshold)):
        baseline = values[start - 1] if start else 0
        delta = values[end] - baseline
        up, down, neutral = 0, 0, 0
        high = low = prices[start]
        previous = baseline
        for i in range(start, end + 1):
            signed = values[i] - previous
            if signed > 0:
                up += volumes[i]
            elif signed < 0:
                down += volumes[i]
            else:
                neutral += volumes[i]
            high = max(high, prices[i])
            low = min(low, prices[i])
            previous = values[i]
        complete = int(abs(delta) >= threshold)
        bars.append({"bar_index": index, "start_time": format_time(ticks.day, ticks.times[start]),
                     "end_time": format_time(ticks.day, ticks.times[end]),
                     "start_row": start + 1, "end_row": end + 1,
                     "open": prices[start], "high": high, "low": low, "close": prices[end],
                     "volume": up + down + neutral, "delta": delta,
                     "up_volume": up, "down_volume": down, "neutral_volume": neutral,
                     "tick_count": end - start + 1, "threshold": threshold,
                     "is_complete": complete, "close_reason": "threshold" if complete else "day_end"})
        start = end + 1
    return bars
