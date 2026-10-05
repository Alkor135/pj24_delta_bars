r"""Считает SuperTrend/SMA, внутридневные события и блочные сравнения групп объёма.

Запуск: .\.venv\Scripts\python.exe -m research.volume_trend --stage analyze
Проверки: .\.venv\Scripts\python.exe -m unittest -v tests.test_volume_trend
Индикаторы причинны; равенства SMA не добавляют пересечений; день является
единицей статистики. Bootstrap сохраняет последовательные блоки наблюдений.
"""

import numpy as np
import pandas as pd


def indicators(data, atr_period=10, multiplier=3., warmup=100, include_weekends=False):
    """Возвращает копию OHLC data с направлениями ST/SMA и флагом прогрева warmup."""
    result = data.copy()
    if "weekday" in result and not include_weekends:
        result = result[result.weekday < 5].copy()
    high, low, close = (result[name].to_numpy(float) for name in ("high", "low", "close"))
    size = len(result)
    direction = np.zeros(size, dtype=np.int8)
    if size >= atr_period:
        previous = np.r_[close[0], close[:-1]]
        tr = np.maximum.reduce((high-low, np.abs(high-previous), np.abs(low-previous)))
        first = atr_period-1
        atr = tr[:atr_period].mean()
        middle = (high[first]+low[first])/2
        upper, lower = middle+multiplier*atr, middle-multiplier*atr
        direction[first] = -1
        for index in range(atr_period, size):
            atr += (tr[index]-atr)/atr_period
            middle = (high[index]+low[index])/2
            next_upper, next_lower = middle+multiplier*atr, middle-multiplier*atr
            if next_upper < upper or close[index-1] > upper:
                upper = next_upper
            if next_lower > lower or close[index-1] < lower:
                lower = next_lower
            direction[index] = (1 if close[index] > upper else -1) if direction[index-1] == -1 else (-1 if close[index] < lower else 1)
    result["st_sign"] = direction
    result["sma_sign"] = np.sign(result.close.rolling(3).mean()-result.close.rolling(34).mean())
    result["warmed"] = np.arange(size) >= warmup
    return result


def count_switches(signs, eligible):
    """Возвращает число смен ненулевого signs и число соседних пригодных переходов."""
    signs, eligible = np.asarray(signs, dtype=float), np.asarray(eligible, dtype=bool)
    count, opportunities, previous, connected = 0, 0, 0, False
    for sign, valid in zip(signs, eligible):
        if not valid or not np.isfinite(sign):
            previous, connected = 0, False
            continue
        if connected:
            opportunities += 1
        if sign:
            if previous and sign != previous:
                count += 1
            previous = sign
        connected = True
    return (float(count) if opportunities else np.nan), opportunities


def daily_metrics(bars, start, end, drop_partial=False):
    """Возвращает события, покрытие и эффективность bars внутри включительного окна."""
    eligible = (bars.start_ns >= start) & (bars.end_ns <= end) & bars.warmed
    if drop_partial:
        eligible &= bars.is_complete.eq(1)
    st_eligible = eligible & bars.st_sign.ne(0)
    st_count, st_n = count_switches(bars.st_sign, st_eligible)
    sma_count, sma_n = count_switches(bars.sma_sign, eligible)
    selected = bars[eligible]
    hours = (end-start)/3.6e12
    result = {"st": st_count, "sma": sma_count, "bars": len(selected), "st_transitions": st_n, "sma_transitions": sma_n,
              "st_hour": st_count/hours if hours else np.nan, "sma_hour": sma_count/hours if hours else np.nan,
              "st_100": 100*st_count/st_n if st_n else np.nan, "sma_100": 100*sma_count/sma_n if sma_n else np.nan,
              "empty_fraction": float(selected.volume.eq(0).mean()) if len(selected) else np.nan}
    closes = selected.close.to_numpy()
    path = np.abs(np.diff(closes)).sum()
    result["er"] = abs(closes[-1]-closes[0])/path if len(closes)>1 and path>0 else np.nan
    result["range"] = float(selected.high.max()-selected.low.min()) if len(selected) else np.nan
    result["included_volume"] = int(selected.volume.sum())
    result["excluded_bars"] = int((~eligible & (bars.end_ns >= start) & (bars.start_ns <= end)).sum())
    result["span_fraction"] = min(1., float((selected.end_ns.max()-selected.start_ns.min())/(end-start))) if len(selected) and end>start else 0.
    return result


def group_effect(values, ratios):
    """Сравнивает средние/медианы values при ratios>1 и ratios<=1, возвращает словарь."""
    values, ratios = np.asarray(values, float), np.asarray(ratios, float)
    good = np.isfinite(values) & np.isfinite(ratios)
    high, low = values[good & (ratios>1)], values[good & (ratios<=1)]
    result = {"n_high": len(high), "n_low": len(low), "high_mean": float(high.mean()) if len(high) else np.nan, "low_mean": float(low.mean()) if len(low) else np.nan,
              "high_median": float(np.median(high)) if len(high) else np.nan, "low_median": float(np.median(low)) if len(low) else np.nan}
    result["difference"] = result["high_mean"]-result["low_mean"]
    result["change_percent"] = 100*result["difference"]/result["low_mean"] if result["low_mean"] else np.nan
    return result


def bootstrap_effect(values, ratios, repetitions=5000, block=20, seed=20261005, segments=None):
    """Оценивает эффект values по ratios, выполняя repetitions повторов с блоком block.

    seed фиксирует генератор; segments запрещает блокам пересекать разрывы.
    Возвращает средние, эффект, индивидуальный 95%-ный ДИ и p. Если все сегменты
    не длиннее блока, неопределённость не оценена: ДИ/p остаются NaN. При наличии
    длинного сегмента короткие части фиксированы; их неопределённость не учтена.
    """
    values, ratios = np.asarray(values, float), np.asarray(ratios, float)
    result = group_effect(values, ratios)
    result.update(ci_low=np.nan, ci_high=np.nan, p=np.nan)
    if min(result["n_high"], result["n_low"]) < 5:
        return result
    size = len(values)
    segments = np.zeros(size, dtype=int) if segments is None else np.asarray(segments)
    groups = [np.flatnonzero(segments == label) for label in np.unique(segments)]
    if not any(len(group)>block for group in groups):
        return result
    rng = np.random.default_rng(seed)
    distributions = []
    # Каждый сегмент сохраняет исходное число дат, непрерывные блоки не выходят за него.
    for _ in range(repetitions):
        pieces = []
        for group in groups:
            width = min(block, len(group))
            starts = rng.integers(0, len(group)-width+1, size=int(np.ceil(len(group)/width)))
            pieces.append(group[(starts[:, None]+np.arange(width)).ravel()[:len(group)]])
        index = np.concatenate(pieces)
        difference = group_effect(values[index], ratios[index])["difference"]
        if np.isfinite(difference):
            distributions.append(difference)
    distribution = np.asarray(distributions)
    if len(distribution):
        result["ci_low"], result["ci_high"] = np.quantile(distribution, [.025, .975]).tolist()
        result["p"] = float((np.count_nonzero(np.abs(distribution-result["difference"]) >= abs(result["difference"]))+1)/(len(distribution)+1))
    return result


def holm(pvalues):
    """Корректирует pvalues методом Holm; вся длина семьи сохраняется, включая NaN."""
    values = np.asarray(pvalues, dtype=float)
    result = np.full(len(values), np.nan)
    order = np.flatnonzero(np.isfinite(values))
    order = order[np.argsort(values[order])]
    adjusted = np.maximum.accumulate(values[order]*(len(values)-np.arange(len(order))))
    result[order] = np.minimum(1., adjusted)
    return result
