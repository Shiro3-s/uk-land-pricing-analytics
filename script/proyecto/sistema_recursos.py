"""Utilidades multiplataforma (macOS, Linux y Windows) de recursos, RAM y entorno.

Garantiza que todo el proyecto corra sin errores en Windows, macOS (Darwin) y Linux
(incluidos servidores sin monitor / headless) sin requerir dependencias externas:
  - Deteccion nativa de RAM total/disponible y memoria fisica real del proceso (RSS MB).
  - Configuracion segura de UTF-8 en consola (evita fallos cp1252 en Windows).
  - Backend 'Agg' de matplotlib automatico fuera de Jupyter (evita TclError en Linux headless).
  - Rutas portables en formato POSIX ('/') para reportes, CSV y contratos JSON.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def configurar_matplotlib_portable() -> str:
    """Activa backend 'Agg' solo fuera de Jupyter si MPLBACKEND no esta fijado."""
    import matplotlib

    if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass
    return str(matplotlib.get_backend())


def configurar_entorno_portable() -> dict[str, str]:
    """Fija salida UTF-8 segura en consola y backend matplotlib portable."""
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            try:
                flujo.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    backend = configurar_matplotlib_portable()
    return {
        "sistema": platform.system(),
        "python": platform.python_version(),
        "matplotlib_backend": backend,
    }


def ruta_posix(ruta: Path | str, base: Path | None = None) -> str:
    """Devuelve una ruta con separadores '/' portables entre macOS, Linux y Windows."""
    p = Path(ruta)
    if base is not None:
        try:
            p = p.resolve().relative_to(Path(base).resolve())
        except ValueError:
            pass
    return p.as_posix()


def _ram_windows_bytes() -> tuple[int, int]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))  # type: ignore[attr-defined]
    return int(ms.ullTotalPhys), int(ms.ullAvailPhys)


def _ram_linux_bytes(ruta_meminfo: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    total_kb = 0
    avail_kb = 0
    free_kb = 0
    for linea in ruta_meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if linea.startswith("MemTotal:"):
            total_kb = int(linea.split()[1])
        elif linea.startswith("MemAvailable:"):
            avail_kb = int(linea.split()[1])
        elif linea.startswith("MemFree:"):
            free_kb = int(linea.split()[1])
    return total_kb * 1024, (avail_kb or free_kb) * 1024


def _ram_macos_bytes() -> tuple[int, int]:
    total = 0
    if hasattr(os, "sysconf"):
        try:
            page_sz = int(os.sysconf("SC_PAGE_SIZE"))
            n_pages = int(os.sysconf("SC_PHYS_PAGES"))
            total = page_sz * n_pages
        except (ValueError, OSError):
            total = 0
    if total <= 0:
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            total = int(out)
        except Exception:
            total = 0
    # En macOS, sin psutil no hay MemAvailable directo; estimamos conservadoramente 50 % si se desconoce
    return total, total // 2 if total > 0 else 0


def info_sistema_ram(so_forzado: str | None = None) -> dict[str, Any]:
    """Reporta SO, CPUs, RAM total y RAM disponible (GB) en Windows, Linux y macOS."""
    sistema = (so_forzado or platform.system()).strip()
    total_b, avail_b = 0, 0
    try:
        if sistema == "Windows":
            total_b, avail_b = _ram_windows_bytes()
        elif sistema == "Linux":
            total_b, avail_b = _ram_linux_bytes()
        elif sistema == "Darwin":
            total_b, avail_b = _ram_macos_bytes()
    except Exception:
        total_b, avail_b = 0, 0

    total_gb = round(total_b / (1024.0**3), 2) if total_b > 0 else 0.0
    avail_gb = round(avail_b / (1024.0**3), 2) if avail_b > 0 else 0.0
    uso_pct = round(100.0 * (1.0 - avail_b / total_b), 1) if total_b > 0 else 0.0

    return {
        "sistema_operativo": "macOS" if sistema == "Darwin" else sistema,
        "plataforma": platform.platform(),
        "cpus": int(os.cpu_count() or 1),
        "ram_total_gb": total_gb,
        "ram_disponible_gb": avail_gb,
        "ram_uso_pct": uso_pct,
    }


def convertir_ru_maxrss_a_mb(ru_maxrss: float, sistema: str) -> float:
    """Convierte resource.getrusage().ru_maxrss a MB respetando la unidad del SO:
    macOS (Darwin) reporta bytes; Linux reporta kilobytes."""
    if sistema == "Darwin":
        return float(ru_maxrss) / (1024.0**2)
    return float(ru_maxrss) / 1024.0


def _rss_windows_mb() -> float:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = k32.GetCurrentProcess()
    fn = getattr(k32, "K32GetProcessMemoryInfo", None)
    if fn is None:
        fn = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    fn.restype = ctypes.c_int
    ok = fn(handle, ctypes.byref(pmc), pmc.cb)
    if ok:
        return float(pmc.WorkingSetSize) / (1024.0**2)
    return 0.0


def rss_proceso_mb() -> float:
    """Devuelve la memoria fisica residente (RSS) actual del proceso en MB (Win/Linux/macOS)."""
    sistema = platform.system()
    try:
        if sistema == "Windows":
            return round(_rss_windows_mb(), 2)
        if sistema == "Linux":
            status_p = Path("/proc/self/status")
            if status_p.exists():
                for linea in status_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if linea.startswith("VmRSS:"):
                        return round(float(linea.split()[1]) / 1024.0, 2)
        import resource  # POSIX (Linux / macOS)

        ru = resource.getrusage(resource.RUSAGE_SELF)
        return round(convertir_ru_maxrss_a_mb(ru.ru_maxrss, sistema), 2)
    except Exception:
        return 0.0


if __name__ == "__main__":
    ent = configurar_entorno_portable()
    hw = info_sistema_ram()
    print(f"Entorno : {ent}")
    print(f"Hardware: {hw}")
    print(f"RSS act.: {rss_proceso_mb():.2f} MB")
