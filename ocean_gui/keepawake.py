import os
import shutil
import signal
import subprocess
import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_execution_state(flags: int) -> bool:
    import ctypes
    fn = ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32
    return bool(fn(flags))


class KeepAwake:
    """Blocks idle sleep between start() and stop(); does nothing if the OS can't.

    Closing a laptop lid or choosing Sleep still sleeps the computer.
    """

    def __init__(self) -> None:
        self._proc = None
        self._active = False

    def start(self) -> bool:
        if self._active:
            return True
        try:
            if sys.platform.startswith("win"):
                self._active = _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            elif sys.platform == "darwin" and shutil.which("caffeinate"):
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
                self._active = True
            elif shutil.which("systemd-inhibit"):
                self._proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep", "--who=Ocean Spectrometer GUI",
                     "--why=Acquisition in progress", "--mode=block",
                     "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                self._active = True
        except Exception:
            self._active = False
        return self._active

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if sys.platform.startswith("win"):
            try:
                _set_execution_state(_ES_CONTINUOUS)
            except Exception:
                pass
        if self._proc is not None:
            try:
                if sys.platform == "darwin":
                    self._proc.terminate()
                else:
                    os.killpg(self._proc.pid, signal.SIGTERM)
                self._proc.wait(2)
            except Exception:
                pass
            self._proc = None
