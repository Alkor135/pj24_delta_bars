"""Исправляет принудительный выход finplot 1.9.7 при русской локали Windows.

Запуск после установки зависимостей: python patch_finplot.py
Удаляется только известный блок locale/os._exit из локального окружения Python.
Повторный запуск безопасен. Настройки Windows и алгоритмы графиков не меняются.
Исходник finplot распространяется по лицензии MIT; автор Jonas Byström.
"""

from importlib import metadata, util
from pathlib import Path

BLOCK = """import locale
code,_ = locale.getdefaultlocale()
if code is not None and \\
    (any(sanctioned in code.lower() for sanctioned in '_ru _by ru_ by_'.split()) or \\
     any(sanctioned in code.lower() for sanctioned in 'ru be'.split())):
    import os
    os._exit(1)
    assert False
"""
MARKER = "# Local compatibility fix: locale-triggered process exit removed (pj24_delta_bars).\n"


def patch_installed_finplot():
    """Применяет точечное исправление к известной версии, отказываясь менять иной код."""
    if metadata.version("finplot") != "1.9.7":
        raise RuntimeError("Исправление рассчитано на finplot==1.9.7; установите requirements-chart.txt")
    spec = util.find_spec("finplot")
    if spec is None or spec.origin is None:
        raise RuntimeError("finplot не установлен")
    path = Path(spec.origin)
    source = path.read_text(encoding="utf-8")
    if MARKER in source and BLOCK not in source:
        return False
    if source.count(BLOCK) != 1:
        raise RuntimeError("Код finplot отличается от ожидаемого; автоматическое исправление остановлено")
    revised = source.replace(BLOCK, MARKER)
    compile(revised, str(path), "exec")
    path.write_text(revised, encoding="utf-8", newline="\n")
    return True


if __name__ == "__main__":
    changed = patch_installed_finplot()
    print("Исправление finplot применено." if changed else "Исправление finplot уже установлено.")
