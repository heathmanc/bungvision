#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
# BungVision is a PySide6 GUI. Ultralytics imports Matplotlib during model
# loading, and Matplotlib may otherwise auto-select the QtAgg backend after
# QApplication exists. On some Jetson/PySide6/Matplotlib combinations that
# crashes in backend_qt.py while converting Qt.KeyboardModifier enums. Force
# Matplotlib to a non-GUI backend before anything can import pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_API", "pyside6")
# Keep CPU helper-thread fan-out low for predictable industrial HMI behavior.
# TensorRT does the heavy inference work; helper libraries do not need many
# CPU worker threads for this runtime.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BUNGVISION_TORCH_THREADS", "1")
import datetime as dt
import sys
import subprocess
import time
import traceback
from contextlib import contextmanager
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect, QEvent
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QBrush, QPixmap, QIntValidator, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QProxyStyle, QScrollArea, QSpinBox, QSizePolicy, QStatusBar, QStyle, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget
)

from camera_backend import BaslerPylonCamera, create_camera_backend, list_basler_cameras

APP_TITLE = "BungVision Python Line-Side HMI v0.9.90 Custom Reject Classes"
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
FAIL_DIR = ROOT / "fail_snapshots"
PASS_DIR = ROOT / "pass_snapshots"
TRAINING_REVIEW_DIR = ROOT / "training_review_captures"
CONFIG_DIR = ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
for d in (LOG_DIR, FAIL_DIR, PASS_DIR, TRAINING_REVIEW_DIR, CONFIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEBUG_LOG_FILE = LOG_DIR / "bungvision_debug.log"
OUT_OF_BAND_STOP_FILE = ROOT / "runtime_stop.flag"
OUT_OF_BAND_STOP_ACK_FILE = ROOT / "runtime_stop_ack.txt"
# Persistent production history for the operator summary dashboard. Operational
# record only; never read or written by the inference/grading/PLC/camera paths.
PRODUCTION_SUMMARY_FILE = LOG_DIR / "production_summary.json"
class VisibleCheckBoxStyle(QProxyStyle):
    """Draw a high-contrast checkbox indicator without stylesheet images.

    v0.9.79 used a PNG in QSS for the check mark. On some Linux/Jetson Qt
    styles, image-based checkbox indicators do not render consistently. This
    proxy style draws a plain white outlined square and a black X directly with
    QPainter, so checked state remains visible without relying on an external
    asset or theme-specific gray check marks.
    """

    def pixelMetric(self, metric, option=None, widget=None):
        if metric == QStyle.PM_IndicatorWidth:
            return 20
        if metric == QStyle.PM_IndicatorHeight:
            return 20
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element, option, painter, widget=None):
        if element == QStyle.PE_IndicatorCheckBox:
            state = option.state
            enabled = bool(state & QStyle.State_Enabled)
            checked = bool(state & QStyle.State_On)
            partial = bool(state & QStyle.State_NoChange)
            hot = bool(state & QStyle.State_MouseOver)

            rect = option.rect.adjusted(1, 1, -1, -1)
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)

            border = QColor("#38bdf8") if enabled and (checked or hot) else QColor("#cbd5e1")
            fill = QColor("#f8fafc") if enabled else QColor("#334155")
            if not enabled:
                border = QColor("#64748b")

            painter.setPen(QPen(border, 2))
            painter.setBrush(QBrush(fill))
            painter.drawRoundedRect(rect, 4, 4)

            if checked:
                mark = QColor("#000000") if enabled else QColor("#111827")
                pen = QPen(mark, 3)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                x_rect = rect.adjusted(5, 5, -5, -5)
                painter.drawLine(x_rect.topLeft(), x_rect.bottomRight())
                painter.drawLine(x_rect.topRight(), x_rect.bottomLeft())
            elif partial:
                mark = QColor("#000000") if enabled else QColor("#111827")
                pen = QPen(mark, 3)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                dash = rect.adjusted(5, 0, -5, 0)
                painter.drawLine(dash.left(), dash.center().y(), dash.right(), dash.center().y())

            painter.restore()
            return

        return super().drawPrimitive(element, option, painter, widget)


def ensure_visible_checkbox_style(app: QApplication) -> None:
    """Install the checkbox painter once; avoid stacking proxy styles."""
    if getattr(app, "_bungvision_visible_checkbox_style", None) is None:
        app._bungvision_visible_checkbox_style = VisibleCheckBoxStyle(app.style())
        app.setStyle(app._bungvision_visible_checkbox_style)


def high_contrast_checkbox_qss() -> str:
    """Shared checkbox label styling; the indicator itself is drawn by Qt code."""
    return """
        QCheckBox {
            background:transparent;
            color:#f8fafc;
            spacing:10px;
            padding:2px;
            font-weight:800;
        }
        QCheckBox:disabled {
            color:#94a3b8;
        }
    """


def readable_popup_qss() -> str:
    """Shared readable styling for QMessageBox and other small dialogs."""
    return """
        QMessageBox, QDialog {
            background:#020617;
            color:#f8fafc;
        }
        QMessageBox QLabel {
            background:transparent;
            color:#f8fafc;
            font-size:13px;
            font-weight:700;
            padding:4px;
        }
        QMessageBox QPushButton, QDialogButtonBox QPushButton {
            background:#2563eb;
            color:#ffffff;
            border:1px solid #60a5fa;
            border-radius:10px;
            padding:8px 14px;
            min-width:88px;
            font-weight:900;
        }
        QMessageBox QPushButton:hover, QDialogButtonBox QPushButton:hover {
            background:#3b82f6;
        }
        QMessageBox QPushButton:pressed, QDialogButtonBox QPushButton:pressed {
            background:#1d4ed8;
        }
        QFileDialog, QFileDialog QWidget {
            background:#020617;
            color:#f8fafc;
        }
        QFileDialog QLineEdit, QFileDialog QComboBox, QFileDialog QListView, QFileDialog QTreeView {
            background:#0f172a;
            color:#f8fafc;
            border:1px solid #475569;
            border-radius:7px;
            selection-background-color:#2563eb;
        }
    """


def common_readability_qss() -> str:
    return high_contrast_checkbox_qss() + readable_popup_qss()


def apply_global_readability_style() -> None:
    """Apply app-level readability rules so standalone popups are not gray-on-gray."""
    app = QApplication.instance()
    if app is not None:
        ensure_visible_checkbox_style(app)
        app.setStyleSheet(common_readability_qss())


def _should_write_debug_log(msg: str) -> bool:
    """Return True only for production file-log entries.

    v0.9.77: the file log records only errors/failures/exceptions and a
    confirmed MODEL_LOADED event. Normal runtime chatter, PROFILE lines,
    FPS/cadence messages, model-load requests, and diagnostics are suppressed.
    """
    text = str(msg or "")
    low = text.lower()
    if any(token in low for token in (
        "error", "failed", "failure", "exception", "traceback",
        "unhandled", "critical", "cannot", "could not", "timed out",
    )):
        return True
    if low.startswith("model_loaded") or low.startswith("model loaded") or "model_loaded" in low:
        return True
    return False


def _write_debug_log(msg: str) -> None:
    """Write the quiet production diagnostic log without relying on Qt.

    The operator-facing file log should contain only model load events and
    errors. This function remains best-effort and must never crash the HMI.
    """
    try:
        if not _should_write_debug_log(str(msg)):
            return
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _configure_cv2_threading() -> None:
    """Keep OpenCV preview helpers from monopolizing Jetson CPU cores."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


_configure_cv2_threading()


def _configure_torch_threading() -> None:
    """Limit PyTorch CPU helper threads used by Ultralytics preprocessing.

    TensorRT engines still run on GPU/DLA as before. This only prevents CPU
    helper libraries from spawning enough worker threads to freeze the desktop.
    """
    try:
        import torch
        n = max(1, min(4, int(os.environ.get("BUNGVISION_TORCH_THREADS", "1"))))
        try:
            torch.set_num_threads(n)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        _write_debug_log(f"Torch CPU threads limited: intraop={n}, interop=1")
    except Exception:
        pass


def _argv_string_list(values=None):
    """Return a sys.argv-safe list containing only plain strings.

    PySide6/Qt can expose flag/enum objects such as KeyboardModifier. Some
    third-party libraries, including Ultralytics helpers, assume every argv
    entry is string-like and may call int()/str parsing on those objects during
    import or model load. Keep the app argv clean before YOLO is imported.
    """
    src = list(sys.argv if values is None else values)
    out = []
    for i, item in enumerate(src):
        if isinstance(item, bytes):
            try:
                out.append(item.decode())
            except Exception:
                out.append("")
        elif isinstance(item, str):
            out.append(item)
        else:
            # Drop Qt enums/flags and any other non-string command-line values.
            # Use a harmless placeholder for argv[0] so QApplication still has
            # an application name if the first item was bad.
            if i == 0:
                out.append(str(ROOT / "main.py"))
    if not out:
        out = [str(ROOT / "main.py")]
    return out


@contextmanager
def _clean_sys_argv_for_ultralytics():
    """Keep sys.argv permanently safe while Ultralytics/TensorRT runs.

    Earlier builds temporarily sanitized argv and then restored the original
    list. On PySide6, that can restore enum/flag objects such as
    KeyboardModifier before lazy Ultralytics/TensorRT model properties are
    evaluated. This version never restores a non-string argv list.
    """
    before = list(sys.argv)
    cleaned = _argv_string_list(before)
    if cleaned != before:
        _write_debug_log(f"Sanitized sys.argv: before={[type(a).__name__ + ':' + repr(a) for a in before]} after={cleaned!r}")
    sys.argv = cleaned
    try:
        yield
    finally:
        # Do not restore PySide6 enum/flag objects. Keep argv string-only.
        sys.argv = _argv_string_list(sys.argv)


# Keep command-line args plain from startup onward. QApplication and
# Ultralytics both tolerate normal string argv, while Qt enum objects can break
# third-party parsers that call int() or similar.
sys.argv = _argv_string_list(sys.argv)
_write_debug_log(f"Startup argv={sys.argv!r}")


def _install_global_excepthook() -> None:
    old_hook = sys.excepthook
    def _hook(exc_type, exc, tb):
        _write_debug_log("UNHANDLED EXCEPTION:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        try:
            old_hook(exc_type, exc, tb)
        except Exception:
            pass
    sys.excepthook = _hook

_install_global_excepthook()


def _force_matplotlib_agg_backend(context: str = "") -> None:
    """Force Matplotlib to a non-GUI backend before Ultralytics import/use.

    Ultralytics imports modules that import matplotlib.pyplot even for normal
    inference. If a QApplication already exists, Matplotlib can pick QtAgg and
    hit a known bad PySide6/system-Matplotlib enum conversion path on Jetson.
    Agg is enough for YOLO inference and avoids loading backend_qt.py.

    v0.9.54: keep stable preview path, use low-cost display scaling, and avoid consuming inference cadence while YOLO is busy. Previous v0.9.53 note: avoid importing/using/
    logging Matplotlib before every prediction. The backend is forced at startup
    and model load; per-predict calls become a no-op to prevent log spam and
    unnecessary UI/disk churn.
    """
    try:
        os.environ["MPLBACKEND"] = "Agg"
        if str(context or "").strip().lower() == "before predict":
            return
        import matplotlib
        try:
            if str(matplotlib.get_backend()).lower() != "agg":
                matplotlib.use("Agg", force=True)
        except Exception:
            matplotlib.use("Agg", force=True)
        _write_debug_log(f"Matplotlib backend forced to Agg {context}".strip())
    except Exception:
        _write_debug_log("Matplotlib Agg force failed " + str(context) + ":\n" + traceback.format_exc())


# Apply once at startup, before QApplication. It is also applied again before
# Ultralytics model loading because user environments may import Matplotlib in
# different orders.
_force_matplotlib_agg_backend("at startup")


def _normalize_model_task(task: Any) -> str:
    """Normalize user/setting model task to Ultralytics task names."""
    text = str(task or "auto").strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "obb": "obb",
        "rotated": "obb",
        "oriented": "obb",
        "detect": "detect",
        "detection": "detect",
        "box": "detect",
        "boxes": "detect",
        "segment": "segment",
        "seg": "segment",
        "classify": "classify",
        "classification": "classify",
        "pose": "pose",
    }
    return aliases.get(text, "auto")


def _parse_class_names_override(value: Any) -> Dict[int, str]:
    """Parse optional class names as 'battery,bung' or '0:battery,1:bung'."""
    text = str(value or "").strip()
    out: Dict[int, str] = {}
    if not text:
        return out
    # Allow either comma or newline separated values.
    parts = []
    for line in text.replace(";", ",").splitlines():
        parts.extend([p.strip() for p in line.split(",") if p.strip()])
    next_idx = 0
    for part in parts:
        if ":" in part:
            left, right = part.split(":", 1)
            try:
                idx = int(left.strip())
            except Exception:
                idx = next_idx
            name = right.strip()
        elif "=" in part:
            left, right = part.split("=", 1)
            try:
                idx = int(left.strip())
            except Exception:
                idx = next_idx
            name = right.strip()
        else:
            idx = next_idx
            name = part.strip()
        if name:
            out[int(idx)] = name.lower()
            next_idx = max(next_idx, int(idx) + 1)
    return out


def _names_are_generic_numeric(names: Dict[int, str]) -> bool:
    if not names:
        return True
    for k, v in names.items():
        txt = str(v).strip().lower()
        if txt and txt not in {str(k), f"class{k}", f"class_{k}", f"cls{k}", f"cls_{k}"}:
            return False
    return True


Box = Tuple[int, int, int, int]

@dataclass
class Detection:
    label: str
    conf: float
    box: Box
    # For YOLO OBB models this holds the four rotated box corners in image pixels.
    # For standard detection models it remains None and the axis-aligned box is used.
    obb_points: Optional[List[Tuple[float, float]]] = None
    source_task: str = "detect"


@dataclass
class BatteryGrade:
    track_id: int
    box: Box
    confidence: float
    bung_count: int
    expected_bungs: int
    bung_boxes: List[Box]
    status: str
    reason: str
    stable_count: int = 0
    logged: bool = False
    obb_points: Optional[List[Tuple[float, float]]] = None
    assigned_bung_indices: Optional[List[int]] = None
    pattern_name: str = ""
    pattern_ok: Optional[bool] = None
    pattern_reason: str = ""



@dataclass
class InspectionResult:
    status: str
    reason: str
    battery_count: int
    bung_count: int
    expected_bungs: int
    detections: List[Detection]
    fps: float
    battery_grades: List[BatteryGrade]


class ProductionStats:
    """Persistent aggregate production history for the operator dashboard.

    This is an operational reporting record only. It is updated at the single
    battery commit point (the same place pass/fail/total counters increment)
    and never participates in inference, grading, tracking, PLC, or camera
    behavior. In-memory updates are immediate; the tiny JSON write is small and
    infrequent (once per committed physical battery) and is queued to the save
    worker by the caller so the live inspection path is never paused.

    History is kept as per-day rollups so a "today" / "last 7 days" view and a
    by-hour throughput view survive HMI restarts and shift changes.
    """

    SCHEMA_VERSION = 1
    MAX_DAYS = 180

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._days: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _empty_day() -> Dict[str, Any]:
        return {
            "total": 0,
            "pass": 0,
            "fail": 0,
            "first_ts": "",
            "last_ts": "",
            "fail_categories": {},
            "hours": {},
        }

    def load(self) -> None:
        """Load history from disk. A missing or corrupt file starts fresh."""
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            days = data.get("days", {}) if isinstance(data, dict) else {}
            clean: Dict[str, Dict[str, Any]] = {}
            for date_key, day in (days or {}).items():
                if not isinstance(day, dict):
                    continue
                entry = self._empty_day()
                entry["total"] = int(day.get("total", 0) or 0)
                entry["pass"] = int(day.get("pass", 0) or 0)
                entry["fail"] = int(day.get("fail", 0) or 0)
                entry["first_ts"] = str(day.get("first_ts", "") or "")
                entry["last_ts"] = str(day.get("last_ts", "") or "")
                fc = day.get("fail_categories", {})
                if isinstance(fc, dict):
                    entry["fail_categories"] = {str(k): int(v or 0) for k, v in fc.items()}
                hrs = day.get("hours", {})
                if isinstance(hrs, dict):
                    for hk, hv in hrs.items():
                        if isinstance(hv, dict):
                            entry["hours"][str(hk)] = {
                                "total": int(hv.get("total", 0) or 0),
                                "pass": int(hv.get("pass", 0) or 0),
                                "fail": int(hv.get("fail", 0) or 0),
                            }
                clean[str(date_key)] = entry
            with self._lock:
                self._days = clean
        except Exception:
            # Never let a damaged history file stop the HMI from starting.
            with self._lock:
                self._days = {}

    def record(self, status: str, category: str = "", when: Optional["dt.datetime"] = None) -> None:
        """Record one committed PASS/FAIL result. Ignores non-terminal states."""
        status = str(status or "").upper()
        if status not in ("PASS", "FAIL"):
            return
        when = when or dt.datetime.now()
        date_key = when.strftime("%Y-%m-%d")
        hour_key = when.strftime("%H")
        ts = when.isoformat(timespec="seconds")
        with self._lock:
            day = self._days.get(date_key)
            if day is None:
                day = self._empty_day()
                self._days[date_key] = day
            day["total"] += 1
            if not day["first_ts"]:
                day["first_ts"] = ts
            day["last_ts"] = ts
            hour = day["hours"].get(hour_key)
            if hour is None:
                hour = {"total": 0, "pass": 0, "fail": 0}
                day["hours"][hour_key] = hour
            hour["total"] += 1
            if status == "PASS":
                day["pass"] += 1
                hour["pass"] += 1
            else:
                day["fail"] += 1
                hour["fail"] += 1
                cat = str(category or "Other").strip() or "Other"
                day["fail_categories"][cat] = int(day["fail_categories"].get(cat, 0)) + 1

    def save(self) -> None:
        """Persist history atomically. Safe to call from the save worker."""
        try:
            with self._lock:
                # Bound the file size by keeping only the most recent days.
                if len(self._days) > self.MAX_DAYS:
                    for old in sorted(self._days.keys())[: -self.MAX_DAYS]:
                        self._days.pop(old, None)
                payload = {
                    "version": self.SCHEMA_VERSION,
                    "updated": dt.datetime.now().isoformat(timespec="seconds"),
                    "days": json.loads(json.dumps(self._days)),
                }
            text = json.dumps(payload, indent=2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(str(tmp), str(self.path))
        except Exception:
            pass

    def day_summary(self, date_key: str) -> Dict[str, Any]:
        with self._lock:
            day = self._days.get(date_key)
            if day is None:
                return self._empty_day()
            return json.loads(json.dumps(day))

    def recent_days(self, n: int = 7) -> List[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            keys = sorted(self._days.keys(), reverse=True)[: max(1, int(n))]
            return [(k, json.loads(json.dumps(self._days[k]))) for k in keys]

    def all_fail_categories(self) -> List[str]:
        cats: set = set()
        with self._lock:
            for day in self._days.values():
                cats.update(day.get("fail_categories", {}).keys())
        return sorted(cats)

    def export_csv(self, path: Path) -> None:
        cats = self.all_fail_categories()
        with self._lock:
            keys = sorted(self._days.keys(), reverse=True)
            rows: List[list] = []
            for k in keys:
                day = self._days[k]
                total = int(day.get("total", 0) or 0)
                passed = int(day.get("pass", 0) or 0)
                failed = int(day.get("fail", 0) or 0)
                rate = (100.0 * passed / total) if total else 0.0
                fc = day.get("fail_categories", {})
                row = [k, total, passed, failed, f"{rate:.1f}", day.get("first_ts", ""), day.get("last_ts", "")]
                row += [int(fc.get(c, 0) or 0) for c in cats]
                rows.append(row)
        header = ["date", "total", "pass", "fail", "pass_rate_percent", "first_ts", "last_ts"] + [f"fail_{c}" for c in cats]
        with Path(path).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


PLC_TAG_DEFAULTS = {
    "running": (
        "BungVision_Running",
        "BOOL output. True when the vision runtime is running OR when HMI bypass is enabled.",
    ),
    "bypass": (
        "BungVision_Bypass",
        "BOOL output. True when the operator has enabled bypass from the HMI.",
    ),
    "stop_request": (
        "BungVision_StopRequest",
        "BOOL output. True when inspection is FAIL and bypass is not active. Use this to stop the conveyor.",
    ),
    "alarm": (
        "BungVision_Alarm",
        "BOOL output. Mirrors StopRequest for stack light, horn, or alarm logic.",
    ),
    "ready": (
        "BungVision_Ready",
        "BOOL output. True only when the vision system is healthy: camera open, recent frame, model loaded, and no active stop request.",
    ),
    "heartbeat": (
        "BungVision_Heartbeat",
        "BOOL output. Toggles while PLC writes are enabled so the PLC can prove the HMI is alive.",
    ),
    "reset": (
        "BungVision_Reset",
        "BOOL output. Momentary pulse when Reset Counts is pressed from the HMI.",
    ),
}
PLC_TAG_LONG_DESCRIPTIONS = {
    "running": (
        "Vision runtime running / active",
        "TRUE when BungVision is actively running inspection or when Bypass is active. "
        "Use this as a general 'vision system active' signal."
    ),
    "bypass": (
        "Vision bypass active",
        "TRUE when the operator has enabled Bypass. PLC logic may allow conveyor operation "
        "even if vision would normally request a stop. This should be clearly indicated and controlled."
    ),
    "stop_request": (
        "Conveyor stop request",
        "TRUE when BungVision has a committed reject/fault and Bypass is not active. "
        "Use this bit to stop or inhibit the conveyor."
    ),
    "alarm": (
        "Vision alarm active",
        "TRUE when BungVision wants the operator/PLC to treat the condition as an alarm. "
        "Typically mirrors Stop Request for stack light, horn, or HMI alarm logic."
    ),
    "ready": (
        "Vision ready / permissive",
        "TRUE only when BungVision is genuinely able to inspect: runtime running, camera open, recent good frame, model loaded, prediction not faulted, and no stop request. "
        "Bypass is reported with its own Bypass bit and does not by itself make Ready TRUE."
    ),
    "heartbeat": (
        "Communication heartbeat",
        "BOOL that toggles while PLC writes are enabled. The PLC should monitor that this bit "
        "continues changing to prove the HMI/vision app is alive."
    ),
    "reset": (
        "Reset acknowledgement pulse",
        "Momentary TRUE pulse when Reset is pressed in BungVision. Use it to clear or acknowledge "
        "PLC-side vision stop/alarm logic if desired."
    ),
}



class PLCInterface:
    """Small pylogix wrapper used by the HMI.

    The vision loop must never depend on PLC availability. If pylogix is not
    installed, the PLC is disabled, or the write fails, this class returns a
    status string and the HMI continues running in SIM/offline mode.
    """

    def __init__(self):
        self.enabled = False
        self.ip_address = ""
        self.tags = {key: value[0] for key, value in PLC_TAG_DEFAULTS.items()}
        self._PLC = None
        self._comm = None
        self._last_error = ""
        self._last_status = "SIM"

    def configure(self, enabled: bool, ip_address: str, tags: dict) -> None:
        self.enabled = bool(enabled)
        self.ip_address = str(ip_address or "").strip()
        merged = {key: value[0] for key, value in PLC_TAG_DEFAULTS.items()}
        for key, value in (tags or {}).items():
            if key in merged and str(value).strip():
                merged[key] = str(value).strip()
        self.tags = merged
        if not self.enabled:
            self.close()
            self._last_status = "SIM"

    def _ensure_comm(self):
        if not self.enabled:
            return None, "PLC disabled"
        if not self.ip_address:
            return None, "PLC IP missing"
        if self._PLC is None:
            try:
                from pylogix import PLC  # type: ignore
                self._PLC = PLC
            except Exception as e:
                self._last_error = str(e)
                return None, "pylogix not installed"
        if self._comm is None:
            try:
                self._comm = self._PLC()
                self._comm.IPAddress = self.ip_address
                if hasattr(self._comm, "SocketTimeout"):
                    self._comm.SocketTimeout = 1.0
                if hasattr(self._comm, "Micro800"):
                    self._comm.Micro800 = False
            except Exception as e:
                self._last_error = str(e)
                self._comm = None
                return None, str(e)
        return self._comm, "OK"

    def _response_error(self, response, tag: str) -> str:
        """Return a pylogix response error string, or blank when the response looks OK."""
        if response is None:
            return "no response"
        if isinstance(response, (list, tuple)):
            for item in response:
                err = self._response_error(item, tag)
                if err:
                    return err
            return ""
        status = getattr(response, "Status", None)
        if status is None:
            # Older/mocked pylogix responses may not expose Status; treat the
            # absence of an exception as OK.
            return ""
        status_text = str(status).strip()
        if status_text.lower() in ("success", "ok", "0"):
            return ""
        return status_text or "unknown status"

    def write_states(self, states: dict) -> str:
        if not self.enabled:
            self._last_status = "SIM"
            return self._last_status
        comm, msg = self._ensure_comm()
        if comm is None:
            self._last_status = msg
            return msg
        try:
            for key, value in states.items():
                tag = self.tags.get(key, "")
                if tag:
                    response = comm.Write(tag, bool(value))
                    err = self._response_error(response, tag)
                    if err:
                        self._last_error = f"{key} / {tag}: {err}"
                        self._last_status = f"WRITE ERROR: {self._last_error}"
                        self.close()
                        return self._last_status
            self._last_status = "CONNECTED"
            self._last_error = ""
            return self._last_status
        except Exception as e:
            self._last_error = str(e)
            self._last_status = f"WRITE ERROR: {e}"
            self.close()
            return self._last_status

    def validate_tags(self, keys: Optional[List[str]] = None) -> str:
        """Read configured PLC tags to prove they exist without forcing outputs."""
        if not self.enabled:
            self._last_status = "SIM"
            return self._last_status
        comm, msg = self._ensure_comm()
        if comm is None:
            self._last_status = msg
            return msg
        try:
            keys_to_test = keys or list(PLC_TAG_DEFAULTS.keys())
            tested = 0
            for key in keys_to_test:
                tag = self.tags.get(key, "")
                if not tag:
                    continue
                tested += 1
                response = comm.Read(tag)
                err = self._response_error(response, tag)
                if err:
                    self._last_error = f"{key} / {tag}: {err}"
                    self._last_status = f"TAG ERROR: {self._last_error}"
                    return self._last_status
            self._last_error = ""
            self._last_status = f"CONNECTED - {tested} TAGS OK"
            return self._last_status
        except Exception as e:
            self._last_error = str(e)
            self._last_status = f"TAG TEST ERROR: {e}"
            self.close()
            return self._last_status

    def close(self) -> None:
        if self._comm is not None:
            try:
                self._comm.Close()
            except Exception:
                pass
        self._comm = None



class AsyncPLCWriter:
    """Background PLC writer so pylogix never blocks the camera/inference timer.

    MainWindow submits desired PLC states from the UI/vision loop. This worker owns
    the PLCInterface instance and performs Ethernet/IP writes on a daemon thread at
    a controlled cadence. That keeps pylogix network latency out of on_timer().
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._plc = PLCInterface()
        self._enabled = False
        self._ip_address = ""
        self._tags = {key: value[0] for key, value in PLC_TAG_DEFAULTS.items()}
        self._heartbeat_interval_ms = 500
        self._min_write_interval_s = 0.10
        self._states: Dict[str, bool] = {
            "running": False,
            "bypass": False,
            "stop_request": False,
            "alarm": False,
            "ready": False,
            "heartbeat": False,
            "reset": False,
        }
        self._reset_pulse_until = 0.0
        self._heartbeat = False
        self._last_heartbeat_t = 0.0
        self._last_write_t = 0.0
        self._last_status = "SIM"
        self._last_error = ""
        self._config_dirty = True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BungVisionPLCWriter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        try:
            self._plc.close()
        except Exception:
            pass

    def configure(self, enabled: bool, ip_address: str, tags: dict, heartbeat_interval_ms: int) -> None:
        with self._lock:
            merged = {key: value[0] for key, value in PLC_TAG_DEFAULTS.items()}
            for key, value in (tags or {}).items():
                if key in merged and str(value).strip():
                    merged[key] = str(value).strip()
            new_enabled = bool(enabled)
            new_ip = str(ip_address or "").strip()
            new_hb = max(100, min(10000, int(heartbeat_interval_ms or 500)))
            if (
                new_enabled != self._enabled
                or new_ip != self._ip_address
                or merged != self._tags
                or new_hb != self._heartbeat_interval_ms
            ):
                self._enabled = new_enabled
                self._ip_address = new_ip
                self._tags = merged
                self._heartbeat_interval_ms = new_hb
                self._config_dirty = True
                self._last_write_t = 0.0
        self.start()
        self._wake.set()

    def submit(self, states: dict, reset_pulse: bool = False) -> None:
        with self._lock:
            for key in ("running", "bypass", "stop_request", "alarm", "ready"):
                if key in states:
                    self._states[key] = bool(states[key])
            if reset_pulse or bool(states.get("reset", False)):
                # Hold reset long enough that the slower PLC writer cannot miss it.
                self._reset_pulse_until = max(self._reset_pulse_until, time.perf_counter() + 0.30)
        self.start()
        self._wake.set()

    def status(self) -> Tuple[str, str, bool]:
        with self._lock:
            return self._last_status, self._last_error, bool(self._heartbeat)

    def _snapshot(self) -> Tuple[bool, str, dict, int, Dict[str, bool]]:
        now = time.perf_counter()
        with self._lock:
            states = dict(self._states)
            states["reset"] = now < self._reset_pulse_until
            return self._enabled, self._ip_address, dict(self._tags), int(self._heartbeat_interval_ms), states

    def _apply_config_if_needed(self, enabled: bool, ip_address: str, tags: dict) -> None:
        with self._lock:
            dirty = self._config_dirty
            self._config_dirty = False
        if dirty:
            self._plc.configure(enabled=enabled, ip_address=ip_address, tags=tags)
            if not enabled:
                with self._lock:
                    self._last_status = "SIM"
                    self._last_error = ""

    def _run(self) -> None:
        while not self._stop.is_set():
            enabled, ip_address, tags, heartbeat_interval_ms, states = self._snapshot()
            try:
                self._apply_config_if_needed(enabled, ip_address, tags)
            except Exception as e:
                with self._lock:
                    self._last_status = "CONFIG ERROR"
                    self._last_error = str(e)

            if not enabled:
                self._wake.wait(0.25)
                self._wake.clear()
                continue

            now = time.perf_counter()
            hb_interval_s = max(0.1, min(10.0, float(heartbeat_interval_ms) / 1000.0))
            if self._last_heartbeat_t <= 0.0 or (now - self._last_heartbeat_t) >= hb_interval_s:
                self._heartbeat = not bool(self._heartbeat)
                self._last_heartbeat_t = now

            states["heartbeat"] = bool(self._heartbeat)
            # Writes are rate-limited. This prevents a 30 FPS camera loop from
            # causing 30 complete pylogix write batches per second.
            if (now - self._last_write_t) >= max(0.05, self._min_write_interval_s):
                status = self._plc.write_states(states)
                with self._lock:
                    self._last_status = status
                    self._last_error = getattr(self._plc, "_last_error", "")
                self._last_write_t = now

            # Wake sooner for reset/config changes, otherwise tick lightly.
            self._wake.wait(0.05)
            self._wake.clear()


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def parse_source(text: str):
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return text

def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detection_center(det: "Detection") -> Tuple[float, float]:
    if getattr(det, "obb_points", None):
        pts = det.obb_points or []
        if pts:
            return (sum(float(x) for x, _ in pts) / len(pts), sum(float(y) for _, y in pts) / len(pts))
    return box_center(det.box)


def polygon_to_box(points: List[Tuple[float, float]], w: int, h: int) -> Box:
    if not points:
        return (0, 0, 0, 0)
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    return clamp_box((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), w, h)


def clamp_polygon(points: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        out.append((max(0.0, min(float(w - 1), float(x))), max(0.0, min(float(h - 1), float(y)))))
    return out


def polygon_signed_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace signed area in image coordinates. Positive = clockwise on screen."""
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return area / 2.0


def normalize_obb_points_clockwise(points: Optional[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Return 4 OBB corners in Bung Labeler convention.

    Convention: point 1 is the top-left-ish corner, then points proceed
    clockwise in image/screen coordinates. This keeps BungVision runtime JSON,
    label-tool import, and YOLO OBB candidate export using the same ordering.
    """
    if not points:
        return []
    pts = [(float(x), float(y)) for x, y in points[:4]]
    if len(pts) < 4:
        return pts

    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)

    # In image coordinates where Y increases downward, sorting by atan2 from
    # low to high gives a screen-clockwise contour for normal boxes.
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

    # Force positive shoelace area, which is clockwise in image coordinates.
    if polygon_signed_area(pts) < 0:
        pts.reverse()

    # Rotate the list so point 1 is the top-left-ish corner. For rotated boxes
    # this is the corner closest to the image origin, matching the label-tool
    # practical convention better than leaving Ultralytics' arbitrary first point.
    start_idx = min(range(len(pts)), key=lambda i: (pts[i][0] + pts[i][1], pts[i][1], pts[i][0]))
    pts = pts[start_idx:] + pts[:start_idx]
    return pts


def point_in_polygon(pt: Tuple[float, float], points: Optional[List[Tuple[float, float]]], margin: int = 0) -> bool:
    if not points or len(points) < 3:
        return False
    contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    # pointPolygonTest returns positive inside, zero on edge, negative outside.
    # With a small margin, allow points slightly outside the rotated edge.
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), True) >= -float(margin)


def polygon_distance_to_center(pt: Tuple[float, float], points: Optional[List[Tuple[float, float]]]) -> float:
    if not points:
        return 1e9
    cx = sum(float(x) for x, _ in points) / len(points)
    cy = sum(float(y) for _, y in points) / len(points)
    return ((float(pt[0]) - cx) ** 2 + (float(pt[1]) - cy) ** 2) ** 0.5


def obb_geometry(points: Optional[List[Tuple[float, float]]]) -> dict:
    if not points or len(points) < 4:
        return {}
    pts = normalize_obb_points_clockwise([(float(x), float(y)) for x, y in points[:4]])
    cx = sum(x for x, _ in pts) / 4.0
    cy = sum(y for _, y in pts) / 4.0
    w1 = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
    h1 = ((pts[2][0] - pts[1][0]) ** 2 + (pts[2][1] - pts[1][1]) ** 2) ** 0.5
    angle = np.degrees(np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    return {
        "corners": [[round(x, 2), round(y, 2)] for x, y in pts],
        "center": [round(cx, 2), round(cy, 2)],
        "width": round(float(w1), 2),
        "height": round(float(h1), 2),
        "angle_degrees": round(float(angle), 2),
    }


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = box_area(a)
    area_b = box_area(b)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def center_distance(a: Box, b: Box) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def point_in_box(pt: Tuple[float, float], box: Box, margin: int = 0) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)

def detection_contains_point(det: "Detection", pt: Tuple[float, float], margin: int = 10) -> bool:
    if getattr(det, "obb_points", None):
        return point_in_polygon(pt, det.obb_points, margin=margin)
    return point_in_box(pt, det.box, margin=margin)


def detection_fully_inside_frame(det: "Detection", frame_w: int, frame_h: int, margin_percent: float = 3.0) -> bool:
    """True when the battery footprint is safely inside the camera frame.

    This prevents a battery that is only partly entering the infeed side from
    being graded FAIL before the whole lid/inspection region can be seen. OBB
    models use the rotated points; standard detect models use the rectangle.
    """
    try:
        margin = max(0.0, min(float(frame_w), float(frame_h)) * max(0.0, float(margin_percent)) / 100.0)
    except Exception:
        margin = 0.0

    if getattr(det, "obb_points", None):
        pts = det.obb_points or []
        if not pts:
            return False
        xs = [float(x) for x, _ in pts]
        ys = [float(y) for _, y in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        x1, y1, x2, y2 = det.box
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    return (
        x1 >= margin and
        y1 >= margin and
        x2 <= (float(frame_w) - 1.0 - margin) and
        y2 <= (float(frame_h) - 1.0 - margin)
    )


def normalized_point_in_detection(det: "Detection", pt: Tuple[float, float]) -> Tuple[float, float]:
    """Return point location in battery-local coordinates.

    For OBB batteries, x/y are expressed relative to the rotated battery
    footprint. For fallback detect boxes, x/y are relative to the normal box.
    Values are not clipped; slightly negative/>1 values are useful for debug.
    """
    if getattr(det, "obb_points", None):
        pts = normalize_obb_points_clockwise(det.obb_points or [])
        if len(pts) >= 4:
            p0 = np.array(pts[0], dtype=np.float32)
            vx = np.array(pts[1], dtype=np.float32) - p0
            vy = np.array(pts[3], dtype=np.float32) - p0
            px = np.array([float(pt[0]), float(pt[1])], dtype=np.float32) - p0
            lx = float(np.dot(vx, vx))
            ly = float(np.dot(vy, vy))
            if lx > 1e-6 and ly > 1e-6:
                return float(np.dot(px, vx) / lx), float(np.dot(px, vy) / ly)
    x1, y1, x2, y2 = det.box
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    return (float(pt[0]) - float(x1)) / bw, (float(pt[1]) - float(y1)) / bh


def _spacing_cv(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float32))
    if values.size < 2:
        return 999.0
    diffs = np.diff(values)
    mean = float(np.mean(diffs))
    if mean <= 1e-6:
        return 999.0
    return float(np.std(diffs) / mean)


def validate_six_bung_pattern(local_points: List[Tuple[float, float]], tolerance_percent: float = 25.0) -> Dict[str, Any]:
    """Validate recipe-less bung geometry for the two accepted layouts.

    Accepted patterns:
      * six bungs in one row
      * two rows of three bungs

    This is intentionally geometry-based, not zone-based. It only examines the
    six assigned bung centers in the battery's local coordinate space.
    """
    if len(local_points) != 6:
        return {"ok": False, "pattern": "", "score": 0.0, "reason": f"Need exactly 6 assigned bungs, got {len(local_points)}"}

    pts = np.asarray(local_points, dtype=np.float32)
    tol = max(0.05, min(0.75, float(tolerance_percent) / 100.0))

    # Pattern A: six in one row. In battery-local coordinates this should look
    # like one mostly horizontal row with fairly even spacing.
    xs = pts[:, 0]
    ys = pts[:, 1]
    row_y_span = float(np.max(ys) - np.min(ys))
    row_spacing_cv = _spacing_cv(xs)
    row_min_gap = float(np.min(np.diff(np.sort(xs)))) if xs.size >= 2 else 0.0
    row_ok = (row_y_span <= tol) and (row_spacing_cv <= max(0.20, tol * 1.15)) and (row_min_gap >= 0.035)
    row_score = max(0.0, 1.0 - (row_y_span / max(tol, 1e-6)) * 0.55 - (row_spacing_cv / max(tol, 1e-6)) * 0.45)

    # Pattern B: two rows of three. Split by local Y, sort each row by X, then
    # check row straightness, row separation, column alignment, and spacing.
    order_y = np.argsort(ys)
    top = pts[order_y[:3]]
    bottom = pts[order_y[3:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    top_y_span = float(np.max(top[:, 1]) - np.min(top[:, 1]))
    bottom_y_span = float(np.max(bottom[:, 1]) - np.min(bottom[:, 1]))
    row_sep = float(abs(np.mean(bottom[:, 1]) - np.mean(top[:, 1])))
    col_align = float(np.mean(np.abs(top[:, 0] - bottom[:, 0])))
    top_spacing_cv = _spacing_cv(top[:, 0])
    bottom_spacing_cv = _spacing_cv(bottom[:, 0])
    grid_spacing_cv = max(top_spacing_cv, bottom_spacing_cv)
    grid_ok = (
        top_y_span <= tol and
        bottom_y_span <= tol and
        row_sep >= max(0.08, tol * 0.45) and
        col_align <= max(0.10, tol * 0.90) and
        grid_spacing_cv <= max(0.25, tol * 1.20)
    )
    grid_penalty = (
        max(top_y_span, bottom_y_span) / max(tol, 1e-6) * 0.30 +
        col_align / max(tol, 1e-6) * 0.30 +
        grid_spacing_cv / max(tol, 1e-6) * 0.25 +
        (0.0 if row_sep >= max(0.08, tol * 0.45) else 0.40)
    )
    grid_score = max(0.0, 1.0 - grid_penalty)

    if row_ok and (not grid_ok or row_score >= grid_score):
        return {"ok": True, "pattern": "6-row", "score": round(row_score, 3), "reason": f"6-row geometry OK (row span {row_y_span:.2f}, spacing CV {row_spacing_cv:.2f})"}
    if grid_ok:
        return {"ok": True, "pattern": "2x3", "score": round(grid_score, 3), "reason": f"2x3 geometry OK (row span {max(top_y_span, bottom_y_span):.2f}, col align {col_align:.2f})"}

    # Return the closest miss so the operator/debug overlay gives a useful clue.
    if row_score >= grid_score:
        reason = f"Pattern invalid: not a clean 6-row (row span {row_y_span:.2f}, spacing CV {row_spacing_cv:.2f})"
        pattern = "6-row?"
        score = row_score
    else:
        reason = f"Pattern invalid: not a clean 2x3 (row span {max(top_y_span, bottom_y_span):.2f}, col align {col_align:.2f}, spacing CV {grid_spacing_cv:.2f})"
        pattern = "2x3?"
        score = grid_score
    return {"ok": False, "pattern": pattern, "score": round(float(score), 3), "reason": reason}

def clamp_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

def detection_kind(label: str) -> str:
    """Map generic or model-specific class names to runtime kind."""
    label = (label or "").lower()
    if label == "battery" or label.startswith("battery_"):
        return "battery"
    if label == "bung" or label.startswith("bung_"):
        return "bung"
    if label == "retainer" or label.startswith("retainer_"):
        return "retainer"
    return label


def detection_suffix(label: str) -> str:
    """Return model-specific suffix after the first underscore, if any."""
    label = (label or "").lower()
    if "_" not in label:
        return ""
    return label.split("_", 1)[1]




def bung_matches_battery(bung_label: str, battery_label: str) -> bool:
    """Allow generic bungs or model-specific bungs matching the battery suffix."""
    if detection_kind(bung_label) != "bung":
        return False

    bung_suffix = detection_suffix(bung_label)
    battery_suffix = detection_suffix(battery_label)

    # Generic bung matches any battery.
    if not bung_suffix:
        return True

    # If battery is generic, accept any bung.
    if not battery_suffix:
        return True

    return bung_suffix == battery_suffix


def build_preview_rgb(frame_bgr: np.ndarray, max_w: int, max_h: int) -> Optional[np.ndarray]:
    """Aspect-fit downscale + BGR->RGB for the operator preview.

    This is the expensive part of preparing a preview frame (full-resolution
    cv2.resize + color convert). Running it off the Qt UI thread (in the
    inference worker) keeps the operator screen responsive. The returned array
    is contiguous so it can back a QImage directly. The full-resolution frame
    is untouched and still used for YOLO, tracking, saves, and PASS/FAIL logic.
    """
    try:
        src_h, src_w = frame_bgr.shape[:2]
    except Exception:
        return None
    if src_w <= 0 or src_h <= 0 or max_w <= 0 or max_h <= 0:
        return None
    scale = min(max_w / float(src_w), max_h / float(src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    try:
        if dst_w != src_w or dst_h != src_h:
            resized = cv2.resize(frame_bgr, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_bgr
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)
    except Exception:
        return None


class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(680, 300)
        self.frame_bgr: Optional[np.ndarray] = None
        self.result: Optional[InspectionResult] = None
        self.overlay_enabled = True
        self.show_detection_boxes = True
        self.show_detection_labels = True
        self.show_grade_badges = True
        self.show_fail_banner = True
        # Honest preview metrics. set_frame() is only a submit to the widget;
        # paintEvent() is the actual Qt paint. Track both so the operator screen
        # does not report the timer-loop rate as video FPS.
        self._paint_fps = 0.0
        self._paint_last_t = 0.0
        self._paint_count = 0
        self._paint_total = 0
        self._paint_window_t = time.perf_counter()
        self._paint_ms = 0.0
        # In v0.9.50, the preview pixmap is pre-scaled in set_frame() instead
        # of converting a full 5MP frame to QImage and scaling it inside
        # paintEvent(). This keeps full-resolution frames available for
        # inference/inspection while making the operator preview much cheaper.
        self._qimage_ms = 0.0
        self._scale_ms = 0.0
        self._overlay_ms = 0.0
        self._preview_pixmap: Optional[QPixmap] = None
        # Identity of the frame the cached pixmap was built from. set_frame()
        # is called on every UI timer tick, but during the overlay-sync hold
        # window the same frame is re-submitted many times. These let set_frame()
        # skip the expensive full-frame resize/QImage conversion when the
        # underlying frame has not actually changed.
        self._preview_seq: Optional[int] = None
        self._preview_frame_id: int = 0
        self._preview_content_size: Tuple[int, int] = (0, 0)
        self._preview_source_shape: Tuple[int, int] = (0, 0)
        self._preview_render_size: Tuple[int, int] = (0, 0)
        # Overlay pixmap cache: the detection/badge drawing loops are expensive
        # Python work that produces identical output every repaint while the
        # inference result is stable. Pre-render it once into a transparent
        # QPixmap keyed on (result_id, target_w, target_h); paintEvent composites
        # it with a single drawPixmap instead of re-running the loops each time.
        self._overlay_pixmap: Optional[QPixmap] = None
        self._overlay_result_id: int = 0
        self._overlay_target_key: Tuple[int, int] = (0, 0)
        self.setStyleSheet("background:#020617; border-radius:18px;")

    def set_overlay_options(
        self,
        enabled: bool = True,
        boxes: bool = True,
        labels: bool = True,
        grades: bool = True,
        fail_banner: bool = True,
    ) -> None:
        self.overlay_enabled = bool(enabled)
        self.show_detection_boxes = bool(boxes)
        self.show_detection_labels = bool(labels)
        self.show_grade_badges = bool(grades)
        self.show_fail_banner = bool(fail_banner)
        self._overlay_result_id = 0  # invalidate overlay cache
        self.update()

    def set_frame(self, frame_bgr: Optional[np.ndarray], result: Optional[InspectionResult], seq: Optional[int] = None, preview_rgb: Optional[np.ndarray] = None) -> None:
        self.frame_bgr = frame_bgr
        self.result = result
        # Rebuild the display pixmap only when the underlying frame actually
        # changed. The UI timer re-submits the same held frame across many
        # ticks during the overlay-sync hold window; without this guard a full
        # cv2.resize + cvtColor + QImage conversion would run every tick on the
        # Qt UI thread. Overlays/badges are drawn from self.result in
        # paintEvent(), so a stable frame can safely reuse the cached pixmap.
        # A widget resize is still handled lazily in paintEvent().
        #
        # preview_rgb, when provided, is the display-sized RGB buffer already
        # scaled off the UI thread by the inference worker; the UI thread then
        # only wraps it in a QImage/QPixmap instead of resizing a full frame.
        frame_id = id(frame_bgr) if frame_bgr is not None else 0
        if (
            frame_bgr is None
            or self._preview_pixmap is None
            or seq is None
            or seq != self._preview_seq
            or frame_id != self._preview_frame_id
        ):
            self._preview_seq = seq
            self._preview_frame_id = frame_id
            self._build_preview_pixmap(preview_rgb)
        self.update()

    def preview_target_size_hint(self) -> Tuple[int, int]:
        """Current preview content area (max width/height) for off-thread scaling."""
        content = self.rect().adjusted(8, 8, -8, -8)
        return max(1, int(content.width())), max(1, int(content.height()))

    def _preview_target_size(self, frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """Return source and preview-render sizes preserving aspect ratio.

        The returned render size is the pixel size actually converted to QImage
        for the Qt preview. It is intentionally bounded by the current widget
        content area, so a 2592x1944 camera frame is not converted/scaled in
        full every paint event.
        """
        try:
            src_h, src_w = frame_bgr.shape[:2]
        except Exception:
            return 0, 0, 0, 0
        content = self.rect().adjusted(8, 8, -8, -8)
        max_w = max(1, int(content.width()))
        max_h = max(1, int(content.height()))
        if src_w <= 0 or src_h <= 0:
            return 0, 0, 0, 0
        scale = min(max_w / float(src_w), max_h / float(src_h))
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        return int(src_w), int(src_h), int(dst_w), int(dst_h)

    def _build_preview_pixmap(self, preview_rgb: Optional[np.ndarray] = None) -> None:
        """Build/cache a display-sized QPixmap for paintEvent().

        This is display-only optimization. It does not alter the full-resolution
        frame used by YOLO, tracking, PASS/FAIL, image saves, or PLC logic.

        When preview_rgb is supplied (already scaled off the UI thread by the
        inference worker), the costly cv2.resize + cvtColor is skipped and the
        UI thread only wraps the buffer in a QImage/QPixmap.
        """
        frame = self.frame_bgr
        if frame is None and preview_rgb is None:
            self._preview_pixmap = None
            self._preview_content_size = (0, 0)
            self._preview_source_shape = (0, 0)
            self._preview_render_size = (0, 0)
            self._qimage_ms = 0.0
            self._scale_ms = 0.0
            return
        if preview_rgb is not None:
            try:
                ph, pw = preview_rgb.shape[:2]
            except Exception:
                ph, pw = 0, 0
            if pw > 0 and ph > 0:
                content = self.rect().adjusted(8, 8, -8, -8)
                self._preview_content_size = (int(content.width()), int(content.height()))
                if frame is not None:
                    try:
                        fh, fw = frame.shape[:2]
                        self._preview_source_shape = (int(fw), int(fh))
                    except Exception:
                        pass
                self._preview_render_size = (int(pw), int(ph))
                try:
                    qimg = QImage(preview_rgb.data, pw, ph, preview_rgb.strides[0], QImage.Format_RGB888)
                    self._preview_pixmap = QPixmap.fromImage(qimg)
                    self._scale_ms = 0.0
                    self._qimage_ms = 0.0
                    return
                except Exception:
                    pass
            # Fall back to UI-thread scaling if the supplied buffer was unusable.
            if frame is None:
                self._preview_pixmap = None
                return
        src_w, src_h, dst_w, dst_h = self._preview_target_size(frame)
        if dst_w <= 0 or dst_h <= 0:
            self._preview_pixmap = None
            return
        content = self.rect().adjusted(8, 8, -8, -8)
        self._preview_content_size = (int(content.width()), int(content.height()))
        self._preview_source_shape = (src_w, src_h)
        self._preview_render_size = (dst_w, dst_h)
        try:
            t0 = time.perf_counter()
            if src_w != dst_w or src_h != dst_h:
                # Display-only preview downscale. INTER_AREA can be very expensive
                # on Jetson when reducing a 5MP frame to the HMI preview while
                # TensorRT/Ultralytics is active. Use INTER_LINEAR for the live
                # operator preview; the full-resolution frame is still preserved
                # for YOLO, tracking, saves, and PASS/FAIL logic.
                interp = cv2.INTER_LINEAR
                display_bgr = cv2.resize(frame, (dst_w, dst_h), interpolation=interp)
            else:
                display_bgr = frame
            t1 = time.perf_counter()
            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = display_rgb.shape
            # QPixmap.fromImage() deep-copies the pixels into the pixmap's native
            # format, so the extra QImage.copy() of the numpy buffer is redundant.
            # display_rgb stays referenced until fromImage() returns, so the view
            # is valid for the duration of the conversion.
            qimg = QImage(display_rgb.data, w, h, display_rgb.strides[0], QImage.Format_RGB888)
            self._preview_pixmap = QPixmap.fromImage(qimg)
            t2 = time.perf_counter()
            self._scale_ms = (t1 - t0) * 1000.0
            self._qimage_ms = (t2 - t1) * 1000.0
        except Exception:
            self._preview_pixmap = None
            self._scale_ms = 0.0
            self._qimage_ms = 0.0

    def _qimage_from_bgr(self, frame_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def paint_fps(self) -> float:
        try:
            return float(self._paint_fps)
        except Exception:
            return 0.0

    def paint_total(self) -> int:
        try:
            return int(self._paint_total)
        except Exception:
            return 0

    def paint_metrics(self) -> Dict[str, float]:
        try:
            return {
                "paint_fps": float(self._paint_fps),
                "paint_ms": float(self._paint_ms),
                "qimage_ms": float(self._qimage_ms),
                "scale_ms": float(self._scale_ms),
                "overlay_ms": float(self._overlay_ms),
            }
        except Exception:
            return {"paint_fps": 0.0, "paint_ms": 0.0, "qimage_ms": 0.0, "scale_ms": 0.0, "overlay_ms": 0.0}

    def _build_overlay_pixmap(self, target: "QRect") -> "QPixmap":
        """Render detection boxes, labels, and grade badges into a transparent QPixmap.

        Coordinates are target-relative so the caller can drawPixmap(target, pixmap)
        without any further offset. Rebuilds only when the result or target size changes.
        """
        tw, th = target.width(), target.height()
        pix = QPixmap(tw, th)
        pix.fill(Qt.transparent)
        if self.result is None or self.frame_bgr is None:
            return pix
        p2 = QPainter(pix)
        try:
            p2.setRenderHint(QPainter.Antialiasing)
            h, w = self.frame_bgr.shape[:2]
            sx = tw / max(1, w)
            sy = th / max(1, h)

            if self.show_detection_boxes or self.show_detection_labels:
                for det in self.result.detections:
                    x1, y1, x2, y2 = det.box
                    rx1 = int(x1 * sx)
                    ry1 = int(y1 * sy)
                    rx2 = int(x2 * sx)
                    ry2 = int(y2 * sy)
                    kind = detection_kind(det.label)
                    if kind == "battery":
                        color, width_pen = QColor("#38bdf8"), 4
                    elif kind == "bung":
                        color, width_pen = QColor("#22c55e"), 3
                    elif kind == "retainer":
                        color, width_pen = QColor("#f59e0b"), 2
                    else:
                        color, width_pen = QColor("#94a3b8"), 2
                    draw_pts = []
                    if getattr(det, "obb_points", None):
                        draw_pts = [(int(px * sx), int(py * sy)) for px, py in (det.obb_points or [])]
                    if self.show_detection_boxes:
                        p2.setPen(QPen(color, width_pen))
                        if len(draw_pts) >= 3:
                            for i in range(len(draw_pts)):
                                x_a, y_a = draw_pts[i]
                                x_b, y_b = draw_pts[(i + 1) % len(draw_pts)]
                                p2.drawLine(x_a, y_a, x_b, y_b)
                        else:
                            p2.drawRoundedRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1), 8, 8)
                    if self.show_detection_labels:
                        label = f"{det.label} {det.conf:.2f}"
                        if getattr(det, "obb_points", None):
                            label += " OBB"
                        p2.setFont(QFont("Arial", 10, QFont.Bold))
                        tw_lbl = p2.fontMetrics().horizontalAdvance(label) + 12
                        lx = min([p0 for p0, _ in draw_pts], default=rx1)
                        ly = min([p1 for _, p1 in draw_pts], default=ry1)
                        p2.fillRect(QRect(lx, max(0, ly - 24), tw_lbl, 22), QColor(2, 6, 23, 210))
                        p2.setPen(color)
                        p2.drawText(lx + 6, max(16, ly - 8), label)

            if self.show_grade_badges:
                for grade in getattr(self.result, "battery_grades", []):
                    x1, y1, x2, y2 = grade.box
                    rx1 = int(x1 * sx)
                    ry1 = int(y1 * sy)
                    rx2 = int(x2 * sx)
                    ry2 = int(y2 * sy)
                    ok = grade.status == "PASS"
                    waiting = grade.status == "WAIT"
                    color = QColor("#22c55e") if ok else (QColor("#f59e0b") if waiting else QColor("#ef4444"))
                    bg = QColor(6, 78, 59, 220) if ok else (QColor(120, 53, 15, 225) if waiting else QColor(127, 29, 29, 230))
                    p2.setPen(QPen(color, 5))
                    grade_pts = []
                    if getattr(grade, "obb_points", None):
                        grade_pts = [(int(px * sx), int(py * sy)) for px, py in (grade.obb_points or [])]
                    if len(grade_pts) >= 3:
                        for i in range(len(grade_pts)):
                            x_a, y_a = grade_pts[i]
                            x_b, y_b = grade_pts[(i + 1) % len(grade_pts)]
                            p2.drawLine(x_a, y_a, x_b, y_b)
                    else:
                        p2.drawRoundedRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1), 10, 10)
                    id_text = f"ID {grade.track_id}" if grade.track_id > 0 else "CAND"
                    line1 = f"{id_text}  {grade.status}"
                    line2 = f"{grade.bung_count}/{grade.expected_bungs}"

                    bx1 = min([p0 for p0, _ in grade_pts], default=rx1)
                    by1 = min([p1 for _, p1 in grade_pts], default=ry1)
                    bx2 = max([p0 for p0, _ in grade_pts], default=rx2)
                    by2 = max([p1 for _, p1 in grade_pts], default=ry2)

                    font1 = QFont("Arial", 11, QFont.Bold)
                    font2 = QFont("Arial", 10, QFont.Bold)
                    p2.setFont(font1)
                    fm1 = p2.fontMetrics()
                    line1_w = fm1.horizontalAdvance(line1)
                    line1_h = fm1.height()
                    p2.setFont(font2)
                    fm2 = p2.fontMetrics()
                    line2_w = fm2.horizontalAdvance(line2)
                    line2_h = fm2.height()

                    badge_w = max(line1_w, line2_w) + 18
                    badge_h = line1_h + line2_h + 10
                    preferred_x = bx1 + 8
                    preferred_y = by1 + 8
                    max_x_inside_battery = bx2 - badge_w - 6
                    max_y_inside_battery = by2 - badge_h - 6
                    if max_x_inside_battery >= bx1 + 4:
                        badge_x = max(bx1 + 4, min(preferred_x, max_x_inside_battery))
                    else:
                        badge_x = preferred_x
                    if max_y_inside_battery >= by1 + 4:
                        badge_y = max(by1 + 4, min(preferred_y, max_y_inside_battery))
                    else:
                        badge_y = preferred_y

                    badge_x = max(4, min(badge_x, tw - badge_w - 4))
                    badge_y = max(38, min(badge_y, th - badge_h - 4))

                    badge_rect = QRect(int(badge_x), int(badge_y), int(badge_w), int(badge_h))
                    p2.fillRect(badge_rect, bg)
                    p2.setPen(QPen(color, 2))
                    p2.drawRoundedRect(badge_rect, 5, 5)
                    p2.setPen(QColor("#ffffff"))
                    p2.setFont(font1)
                    p2.drawText(badge_rect.x() + 9, badge_rect.y() + line1_h, line1)
                    p2.setFont(font2)
                    p2.drawText(badge_rect.x() + 9, badge_rect.y() + line1_h + line2_h + 3, line2)

            if self.show_fail_banner and self.result.status == "FAIL":
                msg = self.result.reason
                p2.fillRect(QRect(20, th - 72, tw - 40, 52), QColor(127, 29, 29, 225))
                p2.setPen(QColor("#fecaca"))
                p2.setFont(QFont("Arial", 18, QFont.Bold))
                p2.drawText(QRect(34, th - 66, tw - 68, 42), Qt.AlignVCenter, f"FAIL: {msg}")
        finally:
            p2.end()
        return pix

    def paintEvent(self, event):
        paint_start = time.perf_counter()
        overlay_start = 0.0
        now_paint = paint_start
        self._paint_count += 1
        self._paint_total += 1
        if now_paint - self._paint_window_t >= 1.0:
            self._paint_fps = self._paint_count / max(1e-6, now_paint - self._paint_window_t)
            self._paint_count = 0
            self._paint_window_t = now_paint
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            p.fillRect(self.rect(), QColor("#020617"))
            content_rect = self.rect().adjusted(8, 8, -8, -8)

            if self.frame_bgr is None:
                p.setPen(QColor("#64748b"))
                p.setFont(QFont("Arial", 22, QFont.Bold))
                p.drawText(content_rect, Qt.AlignCenter, "NO CAMERA FRAME\nOpen camera and load model")
                return

            # Rebuild the preview cache only if the widget size changed since
            # the last submitted frame. Normal paint events draw the cached
            # display-sized pixmap directly; no full-frame QImage conversion or
            # Qt scaling is performed here.
            current_content_size = (int(content_rect.width()), int(content_rect.height()))
            if self._preview_pixmap is None or self._preview_content_size != current_content_size:
                self._build_preview_pixmap()
            pix = self._preview_pixmap
            if pix is None:
                p.setPen(QColor("#64748b"))
                p.setFont(QFont("Arial", 18, QFont.Bold))
                p.drawText(content_rect, Qt.AlignCenter, "PREVIEW NOT AVAILABLE")
                return
            x = content_rect.x() + (content_rect.width() - pix.width()) // 2
            y = content_rect.y() + (content_rect.height() - pix.height()) // 2
            target = QRect(x, y, pix.width(), pix.height())
            p.drawPixmap(target, pix)

            if not self.overlay_enabled:
                return

            p.fillRect(QRect(target.x(), target.y(), target.width(), 34), QColor(2, 6, 23, 180))
            p.setPen(QColor("#e2e8f0"))
            p.setFont(QFont("Arial", 11, QFont.Bold))
            p.drawText(target.x() + 12, target.y() + 23, "LIVE INSPECTION VIEW")

            if not self.result:
                self._overlay_ms = 0.0
                return

            overlay_start = time.perf_counter()
            # Use cached overlay pixmap when result and target size are unchanged.
            # The Python coordinate loops only re-run when a new InspectionResult
            # arrives or the display area is resized; every other repaint (OS
            # expose, focus events, etc.) is just two drawPixmap calls.
            overlay_key = (target.width(), target.height())
            result_id = id(self.result)
            if result_id != self._overlay_result_id or overlay_key != self._overlay_target_key:
                self._overlay_pixmap = self._build_overlay_pixmap(target)
                self._overlay_result_id = result_id
                self._overlay_target_key = overlay_key
            if self._overlay_pixmap is not None:
                p.drawPixmap(target.x(), target.y(), self._overlay_pixmap)
        finally:
            try:
                if overlay_start:
                    self._overlay_ms = (time.perf_counter() - overlay_start) * 1000.0
                self._paint_ms = (time.perf_counter() - paint_start) * 1000.0
            except Exception:
                pass
            p.end()

class Pill(QLabel):
    def __init__(self, text: str, tone: str = "neutral"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(28)
        self.set_tone(tone)

    def set_tone(self, tone: str):
        colors = {
            "pass": ("#064e3b", "#6ee7b7", "#10b981"),
            "fail": ("#7f1d1d", "#fecaca", "#ef4444"),
            "warn": ("#78350f", "#fde68a", "#f59e0b"),
            "info": ("#0c4a6e", "#bae6fd", "#38bdf8"),
            "neutral": ("#1e293b", "#cbd5e1", "#475569"),
        }
        bg, fg, border = colors.get(tone, colors["neutral"])
        self.setStyleSheet(f"QLabel {{ background:{bg}; color:{fg}; border:1px solid {border}; border-radius:14px; padding:4px 10px; font-weight:700; }}")

class MetricCard(QFrame):
    def __init__(self, title: str, value: str, sub: str = "", tone: str = "neutral"):
        super().__init__()
        # Name the frame so the border/background rule below targets only the
        # card itself. A plain "QFrame {...}" rule would cascade onto the child
        # QLabels (a QLabel is a QFrame) and box every line of text.
        self.setObjectName("MetricCard")
        self.title = QLabel(title)
        self.value = QLabel(value)
        self.sub = QLabel(sub)
        self.title.setStyleSheet("color:#94a3b8; font-size:11px; font-weight:800; letter-spacing:1px; background:transparent; border:none;")
        self.value.setStyleSheet("color:white; font-size:21px; font-weight:900; background:transparent; border:none;")
        self.sub.setStyleSheet("color:#64748b; font-size:12px; background:transparent; border:none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.sub)
        border = {"pass": "#10b981", "fail": "#ef4444", "warn": "#f59e0b", "info": "#38bdf8", "neutral": "#334155"}.get(tone, "#334155")
        self.setStyleSheet(f"QFrame#MetricCard {{ background:#0f172a; border:1px solid {border}; border-radius:18px; }}")

    def set_value(self, value: str, sub: str = ""):
        self.value.setText(value)
        if sub:
            self.sub.setText(sub)

class ModelRunner:
    def __init__(self):
        self.model = None
        self.names: Dict[int, str] = {}
        self.path = ""
        self.task = "detect"
        self.requested_task = "auto"
        self.class_names_override = ""
        # Passive performance diagnostics only. These are updated by predict()
        # and displayed/logged by the HMI; they do not change runtime behavior.
        self.last_backend_ms = 0.0
        self.last_parse_ms = 0.0
        self.last_total_predict_ms = 0.0
        self.last_warmup_ms = 0.0

    def _coerce_names_dict(self, names: Any) -> Dict[int, str]:
        """Return class names using only plain Python int keys."""
        out: Dict[int, str] = {}
        if names is None:
            return out
        if hasattr(names, "items"):
            for k, v in names.items():
                try:
                    out[int(k)] = str(v).strip().lower()
                except Exception:
                    continue
            return out
        if isinstance(names, (list, tuple)):
            return {i: str(v).strip().lower() for i, v in enumerate(names)}
        return out

    def _model_names(self, model: Any) -> Dict[int, str]:
        """Return class names from model metadata, then optional override.

        TensorRT .engine files often do not carry the Python model's class-name
        metadata. When names are missing/generic, BungVision can use the
        operator-provided Class Names Override instead of drawing raw numeric
        IDs such as 0/1 on the live view.
        """
        names = self._coerce_names_dict(getattr(model, "names", {}) or {})
        override = _parse_class_names_override(getattr(self, "class_names_override", ""))
        if override and _names_are_generic_numeric(names):
            names = dict(override)
        # If the model provides real names, do not replace them. The override is
        # specifically for TensorRT engines that lost metadata, not for normal
        # .pt models that already know their classes.
        return names

    def _result_names(self, result: Any) -> Dict[int, str]:
        """Some Ultralytics backends expose names on each Results object."""
        names = self._coerce_names_dict(getattr(result, "names", {}) or {})
        override = _parse_class_names_override(getattr(self, "class_names_override", ""))
        if override and _names_are_generic_numeric(names):
            return dict(override)
        return names

    def _label_for_class(self, cls_id: int, names: Optional[Dict[int, str]] = None) -> str:
        names = names or self.names or {}
        label = str(names.get(int(cls_id), "")).strip().lower()
        if not label:
            label = str(int(cls_id))
        return label

    def load(self, model_path: str, task: str = "auto", class_names_override: str = "", device: str = "", imgsz: int = 0) -> str:
        """Load an Ultralytics model/engine with explicit task support.

        Plain .pt models normally load with YOLO(path). TensorRT engines may not
        retain enough metadata for Ultralytics to infer the task, and an OBB
        engine interpreted as detect can produce nonsense boxes/artifacts.
        BungVision therefore defaults .engine files to task=obb unless the
        operator chooses Detect/Auto differently in Runtime settings.
        """
        model_path = str(Path(str(model_path)).expanduser())
        self.model = None
        self.names = {}
        self.path = model_path
        self.requested_task = _normalize_model_task(task)
        self.class_names_override = str(class_names_override or "")

        suffix = Path(model_path).suffix.lower()
        effective_task = self.requested_task
        if effective_task == "auto":
            # BungVision runtime engines are expected to be YOLO-OBB engines.
            # This avoids Ultralytics' engine default of assuming detect.
            effective_task = "obb" if suffix == ".engine" else ""

        with _clean_sys_argv_for_ultralytics():
            _write_debug_log(
                f"ModelRunner.load start path={model_path!r} requested_task={self.requested_task!r} "
                f"effective_task={effective_task!r} class_override={self.class_names_override!r} argv={sys.argv!r}"
            )
            _force_matplotlib_agg_backend("before ultralytics import")
            from ultralytics import YOLO
            _configure_torch_threading()
            if effective_task:
                self.model = YOLO(model_path, task=effective_task)
            else:
                self.model = YOLO(model_path)
            self.names = self._model_names(self.model)
            self.task = str(getattr(self.model, "task", "") or effective_task or "detect").lower()
            if self.task == "detect" and suffix == ".engine" and effective_task == "obb":
                self.task = "obb"
            _write_debug_log(f"MODEL_LOADED task={self.task!r} names={self.names!r}")

        # Warm up once on the target device while still on the background load
        # thread. Ultralytics initializes its inference backend lazily on the
        # first predict(); without this the model effectively initializes a
        # second time on the first live frame (the operator-visible "loads
        # twice" behavior, and a first-frame UI stall at run start). Doing it
        # here pays that cost once, off the Qt UI thread, during "Loading...".
        self.last_warmup_ms = 0.0
        device = str(device or "").strip()
        try:
            warm_imgsz = int(imgsz) if int(imgsz or 0) > 0 else 736
        except Exception:
            warm_imgsz = 736
        try:
            dummy = np.zeros((warm_imgsz, warm_imgsz, 3), dtype=np.uint8)
            t_warm0 = time.perf_counter()
            self.predict(dummy, conf=0.25, iou=0.45, imgsz=warm_imgsz, device=device)
            self.last_warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
            _write_debug_log(
                f"MODEL_WARMUP ok device={device!r} imgsz={warm_imgsz} ms={self.last_warmup_ms:.1f}"
            )
        except Exception:
            _write_debug_log("MODEL_WARMUP failed:\n" + traceback.format_exc())

        warm_note = f"\nWarmup: {self.last_warmup_ms:.0f} ms (device={device or 'auto'})" if self.last_warmup_ms > 0 else ""
        return f"Loaded model: {model_path}\nRequested Task: {self.requested_task}\nRuntime Task: {self.task}\nClasses: {self.names}{warm_note}"

    def _scalar_value(self, value: Any, default: float = 0.0) -> float:
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "item"):
                return float(value.item())
            if isinstance(value, (list, tuple)) and value:
                return self._scalar_value(value[0], default)
            return float(value)
        except Exception:
            return float(default)

    def _array_value(self, value: Any) -> Optional[np.ndarray]:
        """Convert an Ultralytics tensor attribute to a CPU numpy array once.

        Parsing detections box-by-box previously triggered a separate
        .detach().cpu().numpy() (a GPU->CPU sync) per detection. Pulling the
        whole tensor a single time avoids those repeated device syncs.
        """
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
        except Exception:
            return None

    def predict(self, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int, device: str = "") -> List[Detection]:
        if self.model is None:
            return []
        try:
            conf_v = max(0.01, min(0.99, float(conf)))
        except Exception:
            conf_v = 0.25
        try:
            iou_v = max(0.01, min(0.99, float(iou)))
        except Exception:
            iou_v = 0.45
        try:
            imgsz_v = int(imgsz)
        except Exception:
            imgsz_v = 736
        kwargs = {"conf": conf_v, "iou": iou_v, "imgsz": imgsz_v, "verbose": False}
        device = str(device or "").strip()
        if device:
            kwargs["device"] = device
        t_backend0 = time.perf_counter()
        with _clean_sys_argv_for_ultralytics():
            _force_matplotlib_agg_backend("before predict")
            results = self.model.predict(frame_bgr, **kwargs)
        t_backend1 = time.perf_counter()
        self.last_backend_ms = (t_backend1 - t_backend0) * 1000.0
        if not results:
            self.last_parse_ms = 0.0
            self.last_total_predict_ms = self.last_backend_ms
            return []
        t_parse0 = time.perf_counter()
        result = results[0]
        result_names = self._result_names(result)
        if result_names and (_names_are_generic_numeric(self.names) or result_names != self.names):
            self.names = result_names
        names = self.names or result_names or {}
        detections: List[Detection] = []
        h, w = frame_bgr.shape[:2]

        # YOLO OBB models expose rotated boxes through result.obb. Use those
        # corners when present so battery/bung ownership can be based on the
        # rotated footprint instead of a large axis-aligned rectangle.
        obb = getattr(result, "obb", None)
        if obb is not None:
            # Pull all rotated boxes/confs/classes once instead of per detection.
            polys_all = self._array_value(getattr(obb, "xyxyxyxy", None))
            confs_all = self._array_value(getattr(obb, "conf", None))
            cls_all = self._array_value(getattr(obb, "cls", None))
            n_obb = int(polys_all.shape[0]) if polys_all is not None and polys_all.ndim >= 1 else 0
            for i in range(n_obb):
                try:
                    det_conf = float(confs_all[i]) if confs_all is not None and i < len(confs_all) else 0.0
                    if det_conf < conf_v:
                        continue
                    cls_id = int(cls_all[i]) if cls_all is not None and i < len(cls_all) else 0
                    raw_pts = polys_all[i].reshape(4, 2).tolist()
                    pts = clamp_polygon([(float(x), float(y)) for x, y in raw_pts], w, h)
                    if len(pts) < 4:
                        continue
                    pts = normalize_obb_points_clockwise(pts)
                    box = polygon_to_box(pts, w, h)
                    # Reject malformed/near-full-frame TensorRT decode artifacts.
                    bw = max(0, box[2] - box[0]); bh = max(0, box[3] - box[1])
                    if bw <= 2 or bh <= 2 or bw * bh > w * h * 0.98:
                        continue
                    label = self._label_for_class(cls_id, names)
                    detections.append(Detection(label=label, conf=det_conf, box=box, obb_points=pts, source_task="obb"))
                except Exception:
                    continue
            if detections:
                self.last_parse_ms = (time.perf_counter() - t_parse0) * 1000.0
                self.last_total_predict_ms = self.last_backend_ms + self.last_parse_ms
                return detections

        # Standard YOLO detect fallback. This keeps older detection-only models
        # usable while the OBB dataset/model is being developed. For an OBB
        # engine forced to task=obb, this fallback should normally not run; if
        # it does, malformed boxes are filtered before drawing.
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            self.last_parse_ms = (time.perf_counter() - t_parse0) * 1000.0
            self.last_total_predict_ms = self.last_backend_ms + self.last_parse_ms
            return detections
        # Pull all axis-aligned boxes/confs/classes once instead of per detection.
        xyxy_all = self._array_value(getattr(boxes, "xyxy", None))
        confs_all = self._array_value(getattr(boxes, "conf", None))
        cls_all = self._array_value(getattr(boxes, "cls", None))
        n_box = int(xyxy_all.shape[0]) if xyxy_all is not None and xyxy_all.ndim >= 1 else 0
        for i in range(n_box):
            try:
                det_conf = float(confs_all[i]) if confs_all is not None and i < len(confs_all) else 0.0
                if det_conf < conf_v:
                    continue
                cls_id = int(cls_all[i]) if cls_all is not None and i < len(cls_all) else 0
                xyxy = xyxy_all[i].astype(int).tolist()
                box = clamp_box(tuple(xyxy), w, h)
                bw = max(0, box[2] - box[0]); bh = max(0, box[3] - box[1])
                if bw <= 2 or bh <= 2 or bw * bh > w * h * 0.98:
                    continue
                label = self._label_for_class(cls_id, names)
                detections.append(Detection(label=label, conf=det_conf, box=box, source_task="detect"))
            except Exception:
                continue
        self.last_parse_ms = (time.perf_counter() - t_parse0) * 1000.0
        self.last_total_predict_ms = self.last_backend_ms + self.last_parse_ms
        return detections



@dataclass
class CameraFramePacket:
    seq: int
    frame: np.ndarray
    timestamp: float
    fps: float = 0.0
    error: str = ""
    read_ms: float = 0.0
    interval_ms: float = 0.0


@dataclass
class InferencePacket:
    seq: int
    frame: np.ndarray
    detections: List[Detection]
    infer_ms: float
    timestamp: float
    error: str = ""
    fps: float = 0.0
    backend_ms: float = 0.0
    parse_ms: float = 0.0
    cycle_ms: float = 0.0
    idle_ms: float = 0.0
    preview_rgb: Optional[np.ndarray] = None


class CameraCaptureWorker:
    """Continuously read the newest camera frame away from the Qt UI thread.

    The worker keeps only the latest frame. This is intentional for conveyor
    inspection: stale queued frames are worse than skipped frames.
    """

    def __init__(self, cap: Any, log_cb=None):
        self.cap = cap
        self.log_cb = log_cb
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[CameraFramePacket] = None
        self._seq = 0
        self._fps = 0.0
        self._last_t = 0.0
        self._last_error = ""
        self._read_ms = 0.0
        self._interval_ms = 0.0
        self._grab_wait_ms = 0.0
        self._convert_ms = 0.0
        self._array_ms = 0.0
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BungVisionCameraWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def get_latest(self) -> Optional[CameraFramePacket]:
        with self._lock:
            return self._latest

    def status(self) -> Tuple[float, str, int]:
        with self._lock:
            return float(self._fps), str(self._last_error), int(self._seq)

    def metrics(self) -> Dict[str, float]:
        """Passive camera timing diagnostics. Does not alter capture behavior."""
        with self._lock:
            now = time.perf_counter()
            age_ms = (now - self._last_t) * 1000.0 if self._last_t > 0 else 0.0
            return {
                "fps": float(self._fps),
                "seq": float(self._seq),
                "read_ms": float(self._read_ms),
                "grab_wait_ms": float(self._grab_wait_ms),
                "convert_ms": float(self._convert_ms),
                "array_ms": float(self._array_ms),
                "interval_ms": float(self._interval_ms),
                "age_ms": float(age_ms),
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                t_read0 = time.perf_counter()
                ok, frame = self.cap.read() if self.cap is not None else (False, None)
                t_read1 = time.perf_counter()
                read_ms = (t_read1 - t_read0) * 1000.0
                grab_wait_ms = float(getattr(self.cap, "last_grab_wait_ms", 0.0) or 0.0)
                convert_ms = float(getattr(self.cap, "last_convert_ms", 0.0) or 0.0)
                array_ms = float(getattr(self.cap, "last_array_ms", 0.0) or 0.0)
            except Exception as exc:
                ok, frame = False, None
                read_ms = 0.0
                grab_wait_ms = 0.0
                convert_ms = 0.0
                array_ms = 0.0
                err = f"Camera read exception: {exc}"
                with self._lock:
                    self._last_error = err
                if self.log_cb:
                    self.log_cb(err)
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            if ok and frame is not None:
                with self._lock:
                    interval_ms = 0.0
                    if self._last_t > 0:
                        dt_frame = max(1e-6, now - self._last_t)
                        interval_ms = dt_frame * 1000.0
                        inst = 1.0 / dt_frame
                        self._fps = inst if self._fps <= 0 else (self._fps * 0.90 + inst * 0.10)
                    self._last_t = now
                    self._seq += 1
                    self._last_error = ""
                    self._read_ms = float(read_ms)
                    self._grab_wait_ms = float(grab_wait_ms)
                    self._convert_ms = float(convert_ms)
                    self._array_ms = float(array_ms)
                    self._interval_ms = float(interval_ms)
                    self._latest = CameraFramePacket(seq=self._seq, frame=frame, timestamp=now, fps=self._fps, read_ms=float(read_ms), interval_ms=float(interval_ms))
            else:
                with self._lock:
                    self._last_error = "Camera read failed"
                time.sleep(0.01)



class InferenceWorker:
    """Run YOLO prediction in a worker thread using latest-frame semantics."""

    def __init__(self, model_runner: ModelRunner, log_cb=None):
        self.model_runner = model_runner
        self.log_cb = log_cb
        self._lock = threading.RLock()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[InferencePacket] = None
        # Pull-based scheduling: the worker fetches the newest camera frame
        # itself via this source callback as soon as it finishes a prediction,
        # instead of waiting for the UI timer to push one (which inserted up to
        # one UI-tick of idle time between inferences). _enabled mirrors the
        # operator running/model-loaded state; _last_input_seq tracks the last
        # camera frame actually inferred so newer frames are picked up and
        # already-seen / duplicate frames are skipped.
        self._frame_source = None
        self._enabled = False
        self._last_input_seq = 0
        self._conf = 0.25
        self._iou = 0.45
        self._imgsz = 736
        self._device = ""
        # Operator preview target size, published by the UI thread. When set,
        # the worker pre-scales the displayed frame off the UI thread.
        self._preview_w = 0
        self._preview_h = 0
        self._fps = 0.0
        self._last_done_t = 0.0
        self._dropped = 0
        self._skipped_busy = 0
        self._busy = False
        self._last_error = ""
        self._cycle_ms = 0.0
        self._idle_ms = 0.0
        self._backend_ms = 0.0
        self._parse_ms = 0.0
        self._done_count = 0
        # Stall diagnostics. These do not alter normal inference speed; they let
        # the HMI identify when a running inference job is blocking fresh results
        # long enough to make the operator display look frozen.
        self._busy_start_t = 0.0
        self._current_seq = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BungVisionInferenceWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._last_error = ""
            self._dropped = 0
            self._skipped_busy = 0
            self._busy = False
            self._cycle_ms = 0.0
            self._idle_ms = 0.0
            self._backend_ms = 0.0
            self._parse_ms = 0.0
            self._busy_start_t = 0.0
            self._current_seq = 0
            self._last_input_seq = 0
        self._event.clear()

    def set_frame_source(self, source) -> None:
        """Provide a callable returning the newest CameraFramePacket.

        Setting (or replacing) the source resets the last-inferred sequence so a
        freshly opened camera, whose sequence counter restarts at zero, is not
        mistaken for already-seen frames.
        """
        with self._lock:
            self._frame_source = source
            self._last_input_seq = 0
        self._event.set()

    def set_enabled(self, enabled: bool) -> None:
        """Allow/halt autonomous inference (mirrors running + model-loaded)."""
        with self._lock:
            self._enabled = bool(enabled)
        if enabled:
            self._event.set()

    def update_config(self, conf: float, iou: float, imgsz: int, device: str = "", preview_w: int = 0, preview_h: int = 0) -> None:
        with self._lock:
            self._conf = float(conf)
            self._iou = float(iou)
            self._imgsz = int(imgsz)
            self._device = str(device or "").strip()
            self._preview_w = max(0, int(preview_w))
            self._preview_h = max(0, int(preview_h))

    def get_latest(self) -> Optional[InferencePacket]:
        with self._lock:
            return self._latest

    def status(self) -> Tuple[float, int, str]:
        with self._lock:
            # Keep the existing UI label name (drop) but count frames skipped
            # because inference was intentionally busy/backpressured.
            return float(self._fps), int(self._dropped + self._skipped_busy), str(self._last_error)

    def metrics(self) -> Dict[str, float]:
        """Passive inference timing diagnostics. Does not alter scheduling."""
        with self._lock:
            return {
                "fps": float(self._fps),
                "cycle_ms": float(self._cycle_ms),
                "idle_ms": float(self._idle_ms),
                "backend_ms": float(self._backend_ms),
                "parse_ms": float(self._parse_ms),
                "done_count": float(self._done_count),
                "skipped_busy": float(self._skipped_busy),
                "dropped": float(self._dropped),
                "busy": 1.0 if self._busy else 0.0,
                "busy_age_ms": ((time.perf_counter() - self._busy_start_t) * 1000.0) if self._busy and self._busy_start_t > 0 else 0.0,
                "current_seq": float(self._current_seq),
            }

    def _next_input_packet(self) -> Optional[CameraFramePacket]:
        """Pull the newest source frame if it has not been inferred yet.

        Returns None when inference is disabled, no model/source is set, or the
        newest frame was already processed. When it returns a packet it has
        marked the worker busy and counted any camera frames skipped (never
        inferred) since the previous prediction.
        """
        with self._lock:
            enabled = self._enabled
            source = self._frame_source
        if not enabled or source is None:
            return None
        if getattr(self.model_runner, "model", None) is None:
            return None
        try:
            packet = source()
        except Exception:
            packet = None
        if packet is None or getattr(packet, "frame", None) is None:
            return None
        seq = int(getattr(packet, "seq", 0) or 0)
        with self._lock:
            if seq == self._last_input_seq:
                return None
            if self._last_input_seq > 0 and seq > self._last_input_seq + 1:
                self._skipped_busy += (seq - self._last_input_seq - 1)
            self._last_input_seq = seq
            self._busy = True
            self._busy_start_t = time.perf_counter()
            self._current_seq = seq
        return packet

    def _run(self) -> None:
        while not self._stop.is_set():
            packet = self._next_input_packet()
            if packet is None:
                # No new frame to process (idle, paused, no model, or a frame we
                # already inferred). Wake immediately on enable/source/stop
                # changes; otherwise poll at a short interval so a fresh camera
                # frame starts inference with minimal latency instead of waiting
                # for a UI timer tick.
                self._event.wait(0.005)
                self._event.clear()
                continue
            with self._lock:
                conf = self._conf
                iou = self._iou
                imgsz = self._imgsz
                device = self._device
                preview_w = self._preview_w
                preview_h = self._preview_h
            t0 = time.perf_counter()
            idle_ms = ((t0 - self._last_done_t) * 1000.0) if self._last_done_t > 0 else 0.0
            detections: List[Detection] = []
            error = ""
            try:
                detections = self.model_runner.predict(packet.frame, conf, iou, imgsz, device)
            except Exception:
                error = traceback.format_exc()
                if self.log_cb:
                    self.log_cb("Prediction failed:\n" + error)
            done = time.perf_counter()
            infer_ms = (done - t0) * 1000.0
            cycle_ms = ((done - self._last_done_t) * 1000.0) if self._last_done_t > 0 else infer_ms
            backend_ms = float(getattr(self.model_runner, "last_backend_ms", 0.0) or 0.0)
            parse_ms = float(getattr(self.model_runner, "last_parse_ms", 0.0) or 0.0)
            if self._last_done_t > 0:
                inst = 1.0 / max(1e-6, done - self._last_done_t)
                self._fps = inst if self._fps <= 0 else (self._fps * 0.85 + inst * 0.15)
            self._last_done_t = done
            # Pre-scale the displayed frame here, off the Qt UI thread. This is
            # the costly cv2.resize + BGR->RGB conversion that previously ran in
            # CameraWidget.set_frame() on every displayed frame.
            preview_rgb = None
            if preview_w > 0 and preview_h > 0:
                preview_rgb = build_preview_rgb(packet.frame, preview_w, preview_h)
            result = InferencePacket(
                seq=packet.seq,
                frame=packet.frame,
                detections=detections,
                infer_ms=infer_ms,
                timestamp=done,
                error=error,
                fps=self._fps,
                backend_ms=backend_ms,
                parse_ms=parse_ms,
                cycle_ms=cycle_ms,
                idle_ms=idle_ms,
                preview_rgb=preview_rgb,
            )
            with self._lock:
                self._latest = result
                self._last_error = error
                self._cycle_ms = float(cycle_ms)
                self._idle_ms = float(idle_ms)
                self._backend_ms = float(backend_ms)
                self._parse_ms = float(parse_ms)
                self._done_count += 1
                self._busy = False
                self._busy_start_t = 0.0
                self._current_seq = 0
            # Loop straight back to pull the newest available frame with no
            # UI-timer dependency. When inference is GPU-bound a fresh frame is
            # already waiting, so the next prediction starts immediately.


class SaveWorker:
    """Serialize disk writes on a background thread so inspection never waits on cv2.imwrite."""

    def __init__(self, log_cb=None, max_jobs: int = 200):
        self.log_cb = log_cb
        self._queue: queue.Queue = queue.Queue(maxsize=max_jobs)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BungVisionSaveWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        # Queue the sentinel behind existing save jobs so normal shutdown flushes
        # already-committed PASS/FAIL records instead of dropping them.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self._stop.set()
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(None)
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def enqueue(self, fn, *args, **kwargs) -> bool:
        try:
            self._queue.put_nowait((fn, args, kwargs))
            return True
        except queue.Full:
            if self.log_cb:
                self.log_cb("WARNING: Save worker queue is full; dropping save job.")
            return False

    def qsize(self) -> int:
        try:
            return int(self._queue.qsize())
        except Exception:
            return 0

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None or self._stop.is_set():
                self._queue.task_done()
                break
            fn, args, kwargs = job
            try:
                fn(*args, **kwargs)
            except Exception:
                if self.log_cb:
                    self.log_cb("Background save job failed:\n" + traceback.format_exc())
            finally:
                self._queue.task_done()



class CompactIntEdit(QLineEdit):
    """Plain numeric entry with QSpinBox-like value()/setValue() helpers, no up/down arrows."""
    def __init__(self, value: int, lo: int, hi: int, step: int = 1, parent=None):
        super().__init__(parent)
        self.lo = int(lo)
        self.hi = int(hi)
        self.step = int(step)
        self.setValidator(QIntValidator(self.lo, self.hi, self))
        self.setText(str(int(value)))
        self.setAlignment(Qt.AlignCenter)
        self.setMaximumWidth(78)
        self.setMinimumWidth(58)
        self.setFixedHeight(25)
        self.setToolTip(f"Enter {self.lo} to {self.hi}")

    def value(self) -> int:
        txt = self.text().strip()
        try:
            v = int(txt)
        except Exception:
            v = self.lo
        return max(self.lo, min(self.hi, v))

    def setValue(self, value: int) -> None:
        self.setText(str(max(self.lo, min(self.hi, int(value)))))

    def focusOutEvent(self, event):
        self.setValue(self.value())
        super().focusOutEvent(event)


class SettingsDialog(QDialog):
    """Compact settings popup with temporary widgets copied back to MainWindow."""

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.parent_hmi = parent
        self.setWindowTitle("BungVision Settings")
        self.resize(820, 520)
        self.setMinimumSize(720, 460)
        self.setMaximumSize(16777215, 16777215)
        self.build_ui()
        self.load_from_parent()
        self.setStyleSheet("""
            QDialog { background:#020617; }
            QWidget { background:#020617; color:#e2e8f0; font-family:Arial; font-size:12px; }
            QTabWidget::pane { border:1px solid #334155; border-radius:10px; background:#0f172a; padding:4px; }
            QTabBar::tab {
                background:#1e293b; color:#cbd5e1; border:1px solid #334155;
                padding:5px 10px; border-top-left-radius:8px; border-top-right-radius:8px; font-weight:800;
            }
            QTabBar::tab:selected { background:#2563eb; color:white; border-color:#60a5fa; }
            QLabel { background:transparent; color:#cbd5e1; font-weight:700; }
            QLineEdit, QComboBox {
                background:#020617; color:#f8fafc; border:1px solid #475569;
                border-radius:7px; padding:3px 6px; selection-background-color:#2563eb;
            }
            QLineEdit:focus, QComboBox:focus { border:1px solid #38bdf8; }
            QPushButton {
                background:#2563eb; color:white; border:none; border-radius:9px;
                padding:6px 10px; font-weight:900;
            }
            QPushButton:hover { background:#3b82f6; }
            QPushButton#CancelButton { background:#475569; }
            QPushButton#CancelButton:hover { background:#64748b; }
        """ + common_readability_qss())

    def _spin(self, value: int, lo: int, hi: int, step: int = 1) -> CompactIntEdit:
        # Plain numeric entry, intentionally not QSpinBox, so there are no tickers/arrows.
        return CompactIntEdit(value, lo, hi, step, self)

    def _scroll_tab(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _setup_form(self, form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)

    def _plc_tag_label(self, key: str) -> QWidget:
        title, desc = PLC_TAG_LONG_DESCRIPTIONS.get(key, (key, ""))
        short_desc = desc.split(". ")[0].strip().rstrip(".") + "." if desc else "PLC tag name."
        label = self._setting_label(title, short_desc)
        label.setToolTip(f"{key}: {desc}")
        label.setMinimumWidth(150)
        label.setMaximumWidth(210)
        return label

    def _two_col_tab(self) -> tuple[QWidget, QGridLayout]:
        w = QWidget()
        grid = QGridLayout(w)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return w, grid

    def _desc_label(self, text: str) -> QLabel:
        desc = QLabel(text)
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#94a3b8; background:transparent; font-size:10px; font-weight:500; line-height:11px;")
        return desc

    def _setting_label(self, title: str, desc: str) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        title_lab = QLabel(title)
        title_lab.setStyleSheet("color:#dbeafe; background:transparent; font-weight:800;")
        lay.addWidget(title_lab)
        lay.addWidget(self._desc_label(desc))
        return box

    def _add_grid_row(self, grid: QGridLayout, row: int, col_pair: int, label: str, widget: QWidget, desc: str = "") -> None:
        c = 0 if col_pair == 0 else 2
        lab = self._setting_label(label, desc) if desc else QLabel(label)
        lab.setMinimumWidth(150)
        lab.setMaximumWidth(210)
        grid.addWidget(lab, row, c)
        grid.addWidget(widget, row, c + 1)

    def _check_item(self, checkbox: QCheckBox, desc: str) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        lay.addWidget(checkbox)
        lay.addWidget(self._desc_label(desc))
        return box

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        tabs = QTabWidget()

        runtime, rgrid = self._two_col_tab()
        self.camera_backend_local = QComboBox()
        self.camera_backend_local.addItems(["OpenCV", "Basler/Pylon"])
        self.source_edit_local = QLineEdit(); self.source_edit_local.setMaximumWidth(140)
        self.opencv_api_local = QComboBox()
        self.opencv_api_local.addItems(["Auto", "DirectShow", "MSMF", "V4L2", "GStreamer"])
        self.basler_serial_local = QLineEdit(); self.basler_serial_local.setMaximumWidth(180)
        self.camera_width_local = self._spin(2592, 320, 8192, 1)
        self.camera_height_local = self._spin(1944, 240, 8192, 1)
        self.camera_fps_local = self._spin(30, 1, 240, 1)
        self.basler_exposure_mode_local = QComboBox()
        self.basler_exposure_mode_local.addItems(["Manual", "Auto"])
        self.basler_exposure_local = self._spin(5000, 1, 1000000, 1)
        self.basler_gain_local = self._spin(0, 0, 48, 1)
        self.basler_roi_enabled_local = QCheckBox("Enable Basler ROI")
        self.basler_roi_offset_x_local = self._spin(0, 0, 8192, 1)
        self.basler_roi_offset_y_local = self._spin(0, 0, 8192, 1)
        self.basler_roi_width_local = self._spin(2592, 1, 8192, 1)
        self.basler_roi_height_local = self._spin(1944, 1, 8192, 1)
        self.model_edit_local = QLineEdit(); self.model_edit_local.setMinimumWidth(320)
        model_row_widget = QWidget(); model_row = QHBoxLayout(model_row_widget)
        model_row.setContentsMargins(0, 0, 0, 0); model_row.setSpacing(5)
        model_row.addWidget(self.model_edit_local, 1)
        browse = QPushButton("Browse"); browse.clicked.connect(lambda _checked=False: self.browse_model())
        model_row.addWidget(browse)
        self.model_task_local = QComboBox()
        self.model_task_local.addItems(["Auto", "OBB", "Detect"])
        self.model_class_names_local = QLineEdit(); self.model_class_names_local.setMinimumWidth(260)
        self.model_class_names_local.setPlaceholderText("battery,bung or 0:battery,1:bung")
        self.device_edit_local = QLineEdit(); self.device_edit_local.setMaximumWidth(140)
        self.imgsz_spin_local = self._spin(736, 320, 1920, 1)
        self.conf_spin_local = self._spin(25, 1, 99, 1)
        self.yolo_iou_spin_local = self._spin(45, 1, 99, 1)
        self._add_grid_row(rgrid, 0, 0, "Backend", self.camera_backend_local, "OpenCV for USB/UVC/video paths. Basler/Pylon uses the native pypylon grab loop.")
        self._add_grid_row(rgrid, 0, 1, "OpenCV Src", self.source_edit_local, "Camera index or video path. Used only by the OpenCV backend.")
        self._add_grid_row(rgrid, 1, 0, "OpenCV API", self.opencv_api_local, "Auto tries the best platform backend. Windows can use DirectShow/MSMF; Jetson/Linux normally uses V4L2. Use GStreamer for pipeline strings.")
        self._add_grid_row(rgrid, 1, 1, "Basler SN", self.basler_serial_local, "Optional Basler serial number. Leave blank to use the first detected Basler camera.")
        self._add_grid_row(rgrid, 2, 0, "Cam FPS", self.camera_fps_local, "Requested camera frame rate. Actual value depends on camera and exposure.")
        self._add_grid_row(rgrid, 2, 1, "Width", self.camera_width_local, "Requested camera width in pixels.")
        self._add_grid_row(rgrid, 3, 0, "Height", self.camera_height_local, "Requested camera height in pixels.")
        self._add_grid_row(rgrid, 3, 1, "Exposure", self.basler_exposure_mode_local, "Basler exposure mode. Auto lets the camera adjust exposure; Manual uses the time field.")
        self._add_grid_row(rgrid, 4, 0, "Manual us", self.basler_exposure_local, "Manual Basler exposure time in microseconds. Used only when Exposure is Manual.")
        self._add_grid_row(rgrid, 4, 1, "Gain", self.basler_gain_local, "Basler manual gain. Use 0 to leave the camera/default setting alone.")
        rgrid.addWidget(self._check_item(self.basler_roi_enabled_local, "Basler only. When enabled, ROI Width/Height define the camera crop/output size and Offset X/Y shifts that crop on the sensor. Offset only moves when the ROI is smaller than the full sensor."), 5, 0, 1, 4)
        self._add_grid_row(rgrid, 6, 0, "ROI Offset X", self.basler_roi_offset_x_local, "Basler ROI starting X pixel. The backend clamps to the camera's valid increment.")
        self._add_grid_row(rgrid, 6, 1, "ROI Offset Y", self.basler_roi_offset_y_local, "Basler ROI starting Y pixel. The backend clamps to the camera's valid increment.")
        self._add_grid_row(rgrid, 7, 0, "ROI Width", self.basler_roi_width_local, "Basler ROI/crop width. Used only when Basler ROI is enabled. Set smaller than the full sensor if you want Offset X to move.")
        self._add_grid_row(rgrid, 7, 1, "ROI Height", self.basler_roi_height_local, "Basler ROI/crop height. Used only when Basler ROI is enabled. Set smaller than the full sensor if you want Offset Y to move.")
        self._add_grid_row(rgrid, 8, 0, "Model Task", self.model_task_local, ".pt models can usually use Auto. BungVision TensorRT OBB .engine files should use OBB so Ultralytics decodes rotated boxes correctly.")
        self._add_grid_row(rgrid, 8, 1, "Device", self.device_edit_local, "Inference device: cpu, cuda, 0, or blank for auto.")
        self._add_grid_row(rgrid, 9, 0, "Class Names", self.model_class_names_local, "Optional class mapping for TensorRT engines that do not retain labels. Use battery,bung or 0:battery,1:bung.")
        self._add_grid_row(rgrid, 9, 1, "Image Size", self.imgsz_spin_local, "YOLO inference size. Must match the TensorRT engine export size unless the engine was exported dynamic.")
        self._add_grid_row(rgrid, 10, 0, "Confidence %", self.conf_spin_local, "Minimum detection confidence required before inspection logic uses a box.")
        self._add_grid_row(rgrid, 10, 1, "YOLO IoU %", self.yolo_iou_spin_local, "Non-max suppression IoU threshold. Default is 45%; tune for duplicate/overlap behavior.")
        test_basler_btn = QPushButton("List Basler")
        test_basler_btn.clicked.connect(lambda _checked=False: self.list_basler_from_dialog())
        rgrid.addWidget(test_basler_btn, 11, 3)
        rgrid.addWidget(self._setting_label("Model", "Path to the trained YOLO OBB/detection model or engine."), 12, 0)
        rgrid.addWidget(model_row_widget, 12, 1, 1, 3)
        rgrid.setRowStretch(13, 1)

        inspect, igrid = self._two_col_tab()
        self.expected_spin_local = self._spin(6, 1, 24, 1)
        self.debounce_spin_local = self._spin(6, 1, 60, 1)
        self.entry_grace_spin_local = self._spin(12, 0, 300, 1)
        self.clear_spin_local = self._spin(10, 1, 120, 1)
        self.match_distance_spin_local = self._spin(180, 20, 1000, 1)
        self.track_match_iou_spin_local = self._spin(5, 0, 100, 1)
        self.committed_track_iou_spin_local = self._spin(25, 0, 100, 1)
        self.require_full_view_local = QCheckBox("Require full view before grading")
        self.full_view_margin_spin_local = self._spin(3, 0, 25, 1)
        self.pattern_validation_local = QCheckBox("Validate bung pattern")
        self.pattern_tolerance_spin_local = self._spin(25, 5, 75, 1)
        self._add_grid_row(igrid, 0, 0, "Bungs", self.expected_spin_local, "Expected bung count per battery before a PASS can commit.")
        self._add_grid_row(igrid, 0, 1, "Debounce", self.debounce_spin_local, "Frames a PASS/FAIL must stay stable before it is counted.")
        self._add_grid_row(igrid, 1, 0, "Entry Grace", self.entry_grace_spin_local, "WAIT frames after a new battery appears so bungs can enter view.")
        self._add_grid_row(igrid, 1, 1, "Clear Frames", self.clear_spin_local, "Frames with no match before a tracked battery is cleared.")
        self._add_grid_row(igrid, 2, 0, "Match px", self.match_distance_spin_local, "Maximum center movement allowed when matching a battery track.")
        self._add_grid_row(igrid, 2, 1, "Track IoU %", self.track_match_iou_spin_local, "Overlap threshold used to keep moving batteries tied to the same ID.")
        self._add_grid_row(igrid, 3, 0, "Locked IoU %", self.committed_track_iou_spin_local, "Overlap required to keep a committed PASS/FAIL locked while visible.")
        igrid.addWidget(self._check_item(self.require_full_view_local, "Hold a partial infeed/edge battery in WAIT instead of grading it FAIL before the full lid is visible."), 3, 2, 1, 2)
        self._add_grid_row(igrid, 4, 0, "Full View Margin %", self.full_view_margin_spin_local, "Battery must be this far from the frame edge before PASS/FAIL grading can commit.")
        igrid.addWidget(self._check_item(self.pattern_validation_local, "Require the 6 assigned bungs to form either a clean 6-in-row pattern or a 2x3 pattern before PASS can commit."), 4, 2, 1, 2)
        self._add_grid_row(igrid, 5, 0, "Pattern Tol %", self.pattern_tolerance_spin_local, "Geometry tolerance for row straightness, spacing consistency, and 2x3 alignment.")
        igrid.setRowStretch(6, 1)

        capture = QWidget()
        cgrid = QGridLayout(capture)
        cgrid.setContentsMargins(8, 8, 8, 8)
        cgrid.setHorizontalSpacing(10)
        cgrid.setVerticalSpacing(4)
        self.save_pass_images_local = QCheckBox("Save PASS examples")
        self.save_fail_images_local = QCheckBox("Save FAIL examples")
        self.save_annotated_images_local = QCheckBox("Save annotated images")
        self.save_detection_json_local = QCheckBox("Save detection JSON")
        self.save_yolo_txt_local = QCheckBox("Save YOLO OBB .txt candidate")
        self.pass_sample_spin_local = self._spin(1, 1, 1000, 1)
        cgrid.addWidget(self._check_item(self.save_pass_images_local, "Periodically save passing examples for retraining review."), 0, 0)
        cgrid.addWidget(self._check_item(self.save_fail_images_local, "Save failed inspections immediately for review and correction."), 0, 1)
        cgrid.addWidget(self._check_item(self.save_annotated_images_local, "Also save an overlay image with OBB/box detections drawn."), 1, 0)
        cgrid.addWidget(self._check_item(self.save_detection_json_local, "Write label-tool JSON with OBB shapes/corners when available."), 1, 1)
        cgrid.addWidget(self._check_item(self.save_yolo_txt_local, "Write YOLO OBB corner labels when OBB detections are available; otherwise box labels."), 2, 0)
        sample = QWidget(); sample_row = QHBoxLayout(sample)
        sample_row.setContentsMargins(0, 0, 0, 0); sample_row.setSpacing(5)
        sample_row.addWidget(self._setting_label("PASS sample: 1 /", "Save one PASS image every N committed PASS results.")); sample_row.addWidget(self.pass_sample_spin_local); sample_row.addStretch()
        cgrid.addWidget(sample, 2, 1)
        open_all = QPushButton("Open Captures"); open_all.clicked.connect(lambda: self.parent_hmi.open_folder(TRAINING_REVIEW_DIR))
        open_fail = QPushButton("Open Fail"); open_fail.clicked.connect(lambda: self.parent_hmi.open_folder(TRAINING_REVIEW_DIR / "fail"))
        open_pass = QPushButton("Open Pass"); open_pass.clicked.connect(lambda: self.parent_hmi.open_folder(TRAINING_REVIEW_DIR / "pass"))
        cgrid.addWidget(open_all, 3, 0); cgrid.addWidget(open_fail, 3, 1); cgrid.addWidget(open_pass, 3, 2)
        note = QLabel("Review/correct saved examples in the labeling tool before retraining.")
        note.setStyleSheet("color:#94a3b8; background:transparent; padding-top:4px;")
        cgrid.addWidget(note, 4, 0, 1, 3)
        cgrid.setRowStretch(5, 1)

        plc_tab = QWidget()
        pgrid = QGridLayout(plc_tab)
        pgrid.setContentsMargins(8, 8, 8, 8)
        pgrid.setHorizontalSpacing(10)
        pgrid.setVerticalSpacing(4)
        self.plc_enabled_local = QCheckBox("Enable Writes")
        self.plc_ip_local = QLineEdit(); self.plc_ip_local.setMaximumWidth(180)
        self.plc_heartbeat_interval_local = self._spin(500, 100, 10000, 100)
        hb_label = QLabel("Heartbeat ms")
        hb_label.setToolTip("Recommended: 500 ms; PLC timeout usually 2–3 seconds.")
        # Keep the enable checkbox out of the far-right grid column so it does not get clipped
        # on narrower screens or when the settings dialog is opened at its compact size.
        pgrid.setColumnStretch(1, 1)
        pgrid.setColumnStretch(3, 1)
        pgrid.addWidget(self._setting_label("PLC IP", "Controller address for pylogix writes."), 0, 0)
        pgrid.addWidget(self.plc_ip_local, 0, 1)
        pgrid.addWidget(self._setting_label("Heartbeat ms", "How often the heartbeat bit toggles."), 0, 2)
        pgrid.addWidget(self.plc_heartbeat_interval_local, 0, 3)
        pgrid.addWidget(self._check_item(self.plc_enabled_local, "Turn on only after tags are verified."), 1, 0, 1, 4)
        self.plc_tag_local = {}
        row = 2
        col = 0
        for idx, (key, (default_tag, desc)) in enumerate(PLC_TAG_DEFAULTS.items()):
            edit = QLineEdit(default_tag)
            edit.setToolTip(desc)
            edit.setMinimumWidth(190)
            self.plc_tag_local[key] = edit
            pgrid.addWidget(self._plc_tag_label(key), row, col)
            pgrid.addWidget(edit, row, col + 1)
            col += 2
            if col >= 4:
                col = 0
                row += 1
        test_conn_btn = QPushButton("Test Conn"); test_conn_btn.clicked.connect(lambda _checked=False: self.test_plc_connection())
        test_hb_btn = QPushButton("Test HB"); test_hb_btn.clicked.connect(lambda _checked=False: self.test_plc_heartbeat())
        test_stop_btn = QPushButton("Test Stop"); test_stop_btn.clicked.connect(lambda _checked=False: self.test_plc_stop_request())
        pgrid.addWidget(test_conn_btn, row + 1, 0)
        pgrid.addWidget(test_hb_btn, row + 1, 1)
        pgrid.addWidget(test_stop_btn, row + 1, 2)
        pgrid.setRowStretch(row + 2, 1)

        tabs.addTab(self._scroll_tab(runtime), "Runtime")
        tabs.addTab(self._scroll_tab(inspect), "Inspection")
        tabs.addTab(self._scroll_tab(capture), "Capture")
        tabs.addTab(self._scroll_tab(plc_tab), "PLC")

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        apply_btn = QPushButton("Apply"); apply_btn.clicked.connect(lambda _checked=False: self.apply_settings())
        save_btn = QPushButton("Save"); save_btn.clicked.connect(lambda _checked=False: self.save_only())
        save_close_btn = QPushButton("Save + Close"); save_close_btn.clicked.connect(lambda _checked=False: self.save_and_close())
        load_model = QPushButton("Load Model"); load_model.clicked.connect(lambda _checked=False: self.load_model_from_dialog())
        cancel = QPushButton("Cancel"); cancel.setObjectName("CancelButton"); cancel.clicked.connect(lambda _checked=False: self.reject())
        buttons.addWidget(apply_btn); buttons.addWidget(save_btn); buttons.addWidget(save_close_btn); buttons.addWidget(load_model)
        buttons.addStretch(); buttons.addWidget(cancel)

        layout.addWidget(tabs, 1)
        layout.addLayout(buttons)

    def load_from_parent(self):
        p = self.parent_hmi
        backend = str(getattr(p, "camera_backend", "opencv")).lower()
        self.camera_backend_local.setCurrentIndex(1 if backend in ("basler", "pylon", "basler/pylon") else 0)
        self.source_edit_local.setText(p.source_edit.text() if hasattr(p, "source_edit") else "0")
        api_map = {"auto": 0, "dshow": 1, "directshow": 1, "msmf": 2, "v4l2": 3, "gstreamer": 4, "gst": 4}
        self.opencv_api_local.setCurrentIndex(api_map.get(str(getattr(p, "opencv_api", "auto")).strip().lower(), 0))
        self.basler_serial_local.setText(str(getattr(p, "basler_serial", "")))
        cam_w = int(getattr(p, "camera_width", 2592))
        cam_h = int(getattr(p, "camera_height", 1944))
        self.camera_width_local.setValue(cam_w)
        self.camera_height_local.setValue(cam_h)
        self.camera_fps_local.setValue(int(float(getattr(p, "camera_fps", 30))))
        self.basler_exposure_mode_local.setCurrentIndex(1 if bool(getattr(p, "basler_exposure_auto", False)) else 0)
        self.basler_exposure_local.setValue(int(float(getattr(p, "basler_exposure_us", 5000))))
        self.basler_gain_local.setValue(int(float(getattr(p, "basler_gain", 0))))
        self.basler_roi_enabled_local.setChecked(bool(getattr(p, "basler_roi_enabled", False)))
        self.basler_roi_offset_x_local.setValue(int(getattr(p, "basler_roi_offset_x", 0)))
        self.basler_roi_offset_y_local.setValue(int(getattr(p, "basler_roi_offset_y", 0)))
        self.basler_roi_width_local.setValue(int(getattr(p, "basler_roi_width", cam_w) or cam_w))
        self.basler_roi_height_local.setValue(int(getattr(p, "basler_roi_height", cam_h) or cam_h))
        self.model_edit_local.setText(p.model_edit.text() if hasattr(p, "model_edit") else "")
        task_map = {"auto": 0, "obb": 1, "detect": 2}
        self.model_task_local.setCurrentIndex(task_map.get(_normalize_model_task(getattr(p, "model_task", "auto")), 0))
        self.model_class_names_local.setText(str(getattr(p, "model_class_names_override", "battery,bung")))
        self.device_edit_local.setText(p.device_edit.text() if hasattr(p, "device_edit") else "")
        self.imgsz_spin_local.setValue(p._spin_value("imgsz_spin", 736))
        self.conf_spin_local.setValue(p._spin_value("conf_spin", 25))
        self.yolo_iou_spin_local.setValue(int(float(getattr(p, "yolo_iou", 0.45)) * 100))
        self.expected_spin_local.setValue(p._spin_value("expected_spin", 6))
        self.debounce_spin_local.setValue(p._spin_value("debounce_spin", 6))
        self.entry_grace_spin_local.setValue(p._spin_value("entry_grace_spin", 12))
        self.clear_spin_local.setValue(p._spin_value("clear_spin", 10))
        self.match_distance_spin_local.setValue(p._spin_value("match_distance_spin", 180))
        self.track_match_iou_spin_local.setValue(int(float(getattr(p, "track_match_iou", 0.05)) * 100))
        self.committed_track_iou_spin_local.setValue(int(float(getattr(p, "committed_track_iou", 0.25)) * 100))
        self.require_full_view_local.setChecked(bool(getattr(p, "require_full_view_before_grade", True)))
        self.full_view_margin_spin_local.setValue(int(float(getattr(p, "full_view_margin_percent", 3.0))))
        self.pattern_validation_local.setChecked(bool(getattr(p, "enable_pattern_validation", True)))
        self.pattern_tolerance_spin_local.setValue(int(float(getattr(p, "pattern_tolerance_percent", 25.0))))

        self.save_pass_images_local.setChecked(bool(getattr(p, "save_pass_training_images", True)))
        self.save_fail_images_local.setChecked(bool(getattr(p, "save_fail_training_images", True)))
        self.save_annotated_images_local.setChecked(bool(getattr(p, "save_training_annotated", True)))
        self.save_detection_json_local.setChecked(bool(getattr(p, "save_training_json", True)))
        self.save_yolo_txt_local.setChecked(bool(getattr(p, "save_training_yolo_txt", False)))
        self.pass_sample_spin_local.setValue(int(getattr(p, "pass_training_sample_rate", 1)))
        self.plc_enabled_local.setChecked(p._checkbox_checked("plc_enabled_check", False))
        self.plc_ip_local.setText(p.plc_ip_edit.text() if hasattr(p, "plc_ip_edit") else "")
        self.plc_heartbeat_interval_local.setValue(int(getattr(p, "plc_heartbeat_interval_ms", 500)))
        for key, edit in self.plc_tag_local.items():
            if hasattr(p, "plc_tag_edits") and key in p.plc_tag_edits:
                edit.setText(p.plc_tag_edits[key].text())

    def copy_to_parent(self):
        p = self.parent_hmi
        old_camera_settings = (
            str(getattr(p, "camera_backend", "opencv")),
            p.source_edit.text().strip() if hasattr(p, "source_edit") else "0",
            str(getattr(p, "opencv_api", "auto")),
            str(getattr(p, "basler_serial", "")),
            int(getattr(p, "camera_width", 0)),
            int(getattr(p, "camera_height", 0)),
            float(getattr(p, "camera_fps", 0.0)),
            bool(getattr(p, "basler_exposure_auto", False)),
            float(getattr(p, "basler_exposure_us", 0.0)),
            float(getattr(p, "basler_gain", 0.0)),
            bool(getattr(p, "basler_roi_enabled", False)),
            int(getattr(p, "basler_roi_offset_x", 0)),
            int(getattr(p, "basler_roi_offset_y", 0)),
            int(getattr(p, "basler_roi_width", 0)),
            int(getattr(p, "basler_roi_height", 0)),
        )
        p.camera_backend = "basler" if self.camera_backend_local.currentIndex() == 1 else "opencv"
        p.source_edit.setText(self.source_edit_local.text().strip())
        api_values = ["auto", "dshow", "msmf", "v4l2", "gstreamer"]
        p.opencv_api = api_values[self.opencv_api_local.currentIndex()] if self.opencv_api_local.currentIndex() < len(api_values) else "auto"
        p.basler_serial = self.basler_serial_local.text().strip()
        p.camera_width = int(self.camera_width_local.value())
        p.camera_height = int(self.camera_height_local.value())
        p.camera_fps = float(self.camera_fps_local.value())
        p.basler_exposure_auto = self.basler_exposure_mode_local.currentIndex() == 1
        p.basler_exposure_us = float(self.basler_exposure_local.value())
        p.basler_gain = float(self.basler_gain_local.value())
        p.basler_roi_enabled = self.basler_roi_enabled_local.isChecked()
        p.basler_roi_offset_x = int(self.basler_roi_offset_x_local.value())
        p.basler_roi_offset_y = int(self.basler_roi_offset_y_local.value())
        # Basler ROI is a real four-value crop: OffsetX, OffsetY, Width, Height.
        # Main Width/Height remain the non-ROI/default camera request.
        p.basler_roi_width = int(self.basler_roi_width_local.value())
        p.basler_roi_height = int(self.basler_roi_height_local.value())
        p.model_edit.setText(self.model_edit_local.text().strip())
        task_values = ["auto", "obb", "detect"]
        p.model_task = task_values[self.model_task_local.currentIndex()] if self.model_task_local.currentIndex() < len(task_values) else "auto"
        p.model_class_names_override = self.model_class_names_local.text().strip()
        p.device_edit.setText(self.device_edit_local.text().strip())
        p.imgsz_spin.setValue(self.imgsz_spin_local.value())
        p.conf_spin.setValue(self.conf_spin_local.value())
        p.yolo_iou = self.yolo_iou_spin_local.value() / 100.0
        p.expected_spin.setValue(self.expected_spin_local.value())
        p.debounce_spin.setValue(self.debounce_spin_local.value())
        p.entry_grace_spin.setValue(self.entry_grace_spin_local.value())
        p.clear_spin.setValue(self.clear_spin_local.value())
        p.match_distance_spin.setValue(self.match_distance_spin_local.value())
        p.track_match_iou = self.track_match_iou_spin_local.value() / 100.0
        p.committed_track_iou = self.committed_track_iou_spin_local.value() / 100.0
        p.require_full_view_before_grade = self.require_full_view_local.isChecked()
        p.full_view_margin_percent = float(self.full_view_margin_spin_local.value())
        p.enable_pattern_validation = self.pattern_validation_local.isChecked()
        p.pattern_tolerance_percent = float(self.pattern_tolerance_spin_local.value())

        p.save_pass_training_images = self.save_pass_images_local.isChecked()
        p.save_fail_training_images = self.save_fail_images_local.isChecked()
        p.save_training_annotated = self.save_annotated_images_local.isChecked()
        p.save_training_json = self.save_detection_json_local.isChecked()
        p.save_training_yolo_txt = self.save_yolo_txt_local.isChecked()
        p.pass_training_sample_rate = self.pass_sample_spin_local.value()
        if hasattr(p, "plc_enabled_check"):
            p.plc_enabled_check.setChecked(self.plc_enabled_local.isChecked())
        if hasattr(p, "plc_ip_edit"):
            p.plc_ip_edit.setText(self.plc_ip_local.text().strip())
        p.plc_heartbeat_interval_ms = int(self.plc_heartbeat_interval_local.value())
        if hasattr(p, "plc_tag_edits"):
            for key, edit in self.plc_tag_local.items():
                if key in p.plc_tag_edits:
                    p.plc_tag_edits[key].setText(edit.text().strip())

        new_camera_settings = (
            str(getattr(p, "camera_backend", "opencv")),
            p.source_edit.text().strip() if hasattr(p, "source_edit") else "0",
            str(getattr(p, "opencv_api", "auto")),
            str(getattr(p, "basler_serial", "")),
            int(getattr(p, "camera_width", 0)),
            int(getattr(p, "camera_height", 0)),
            float(getattr(p, "camera_fps", 0.0)),
            bool(getattr(p, "basler_exposure_auto", False)),
            float(getattr(p, "basler_exposure_us", 0.0)),
            float(getattr(p, "basler_gain", 0.0)),
            bool(getattr(p, "basler_roi_enabled", False)),
            int(getattr(p, "basler_roi_offset_x", 0)),
            int(getattr(p, "basler_roi_offset_y", 0)),
            int(getattr(p, "basler_roi_width", 0)),
            int(getattr(p, "basler_roi_height", 0)),
        )
        p._camera_settings_changed = old_camera_settings != new_camera_settings

        p.apply_runtime_settings()

    def test_plc_connection(self):
        self.copy_to_parent()
        self.parent_hmi.test_plc_connection()

    def test_plc_heartbeat(self):
        self.copy_to_parent()
        self.parent_hmi.test_plc_write("heartbeat")

    def test_plc_stop_request(self):
        self.copy_to_parent()
        self.parent_hmi.test_plc_write("stop_request")

    def list_basler_from_dialog(self, *args):
        ok, msg = BaslerPylonCamera.available()
        if not ok:
            QMessageBox.warning(self, "Basler/Pylon", "pypylon is not available. Install Basler Pylon SDK and run: pip install pypylon\n\n" + msg)
            return
        cams = list_basler_cameras()
        if not cams:
            QMessageBox.warning(self, "Basler/Pylon", "pypylon loaded, but no Basler cameras were found.")
            return
        lines = []
        for cam in cams:
            lines.append(f"Model: {cam.get('model','')}\nSerial: {cam.get('serial','')}\nName: {cam.get('name','')}")
        QMessageBox.information(self, "Basler Cameras", "\n\n".join(lines))

    def browse_model(self, *args):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO model",
            self.model_edit_local.text().strip() or str(ROOT),
            "YOLO Models (*.pt *.engine);;All Files (*)",
        )
        if path:
            self.model_edit_local.setText(path)

    def apply_settings(self):
        self.copy_to_parent()
        if hasattr(self.parent_hmi, "save_settings"):
            if self.parent_hmi.save_settings(silent=True):
                self.load_from_parent()
                self.parent_hmi.log("Settings applied and saved to disk.")
            else:
                QMessageBox.warning(self, "Settings", "Settings were applied, but could not be saved to disk.")
        else:
            self.parent_hmi.log("Settings applied, but save_settings() is missing.")

    def save_only(self):
        self.copy_to_parent()
        if hasattr(self.parent_hmi, "save_settings"):
            if self.parent_hmi.save_settings(silent=False):
                self.load_from_parent()
                self.parent_hmi.log("Capture/runtime settings saved and refreshed in dialog.")
        else:
            QMessageBox.warning(self, "Settings", "Internal error: save_settings() is missing on MainWindow.")

    def save_and_close(self):
        self.copy_to_parent()
        if hasattr(self.parent_hmi, "save_settings"):
            if self.parent_hmi.save_settings(silent=False):
                self.accept()
        else:
            QMessageBox.warning(self, "Settings", "Internal error: save_settings() is missing on MainWindow.")

    def load_model_from_dialog(self, *args):
        self.copy_to_parent()
        self.parent_hmi.load_model()


class ProductionDashboardDialog(QDialog):
    """Read-only production summary for operators and supervisors.

    Aggregates committed PASS/FAIL results for the current session (since the
    last Reset Counts, which maps to a shift when reset at shift start), for
    today, for the last seven days, and by reject reason and hour. This is a
    pure reporting view: it reads the live counters and the persistent
    ProductionStats history and never changes inspection, grading, or PLC
    behavior.
    """

    REFRESH_MS = 2000

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.hmi = parent
        self.setWindowTitle("BungVision Production Summary")
        self.setMinimumSize(720, 660)
        self._build_ui()
        self.refresh()
        # Live-refresh while open so a supervisor can leave it up during a run.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.REFRESH_MS)

    # ---- construction helpers -------------------------------------------------
    def _metric_card(self, title: str, tone: str = "neutral") -> MetricCard:
        return MetricCard(title, "--", "", tone)

    def _make_table(self, headers: List[str], cap_height: int = 300) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(30)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(Qt.NoFocus)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # The inherited theme leaves alternate rows a pale default, which renders
        # as faint, low-contrast text. Pin a dark alternate color and bright text.
        table.setStyleSheet(
            "QTableWidget { background:#0b1220; alternate-background-color:#111c33; color:#e2e8f0; gridline-color:#1e293b; border:1px solid #334155; border-radius:12px; }"
            "QTableWidget::item { color:#e2e8f0; padding:3px 8px; }"
            "QHeaderView::section { background:#1e293b; color:#cbd5e1; border:none; padding:6px; font-weight:800; }"
        )
        table._cap_height = int(cap_height)
        header = table.horizontalHeader()
        try:
            header.setStretchLastSection(True)
        except Exception:
            pass
        return table

    def _fit_table_height(self, table: QTableWidget) -> None:
        """Size a populated table to its rows so nothing is clipped or overlapped.

        Height is capped so a long table scrolls internally instead of dominating
        the dialog; the surrounding scroll area handles any remaining overflow.
        """
        header_h = table.horizontalHeader().height() or table.horizontalHeader().sizeHint().height()
        rows_h = sum(table.rowHeight(r) for r in range(table.rowCount()))
        total = header_h + rows_h + 2 * table.frameWidth() + 2
        cap = int(getattr(table, "_cap_height", 300))
        table.setFixedHeight(max(60, min(total, cap)))

    def _group(self, title: str) -> Tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        box.setStyleSheet(
            "QGroupBox { background:#0f172a; border:1px solid #334155; border-radius:12px;"
            " margin-top:22px; padding:6px; font-weight:800; color:#cbd5e1; }"
            "QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top left;"
            " left:14px; top:4px; padding:0 6px; color:#93c5fd; }"
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(6)
        return box, lay

    def _build_ui(self):
        self.setStyleSheet(
            "QDialog { background:#060d1a; }"
            "QScrollArea { background:#060d1a; border:none; }"
            "QWidget#dash_content { background:#060d1a; }"
            "QLabel { background:transparent; border:none; color:#e2e8f0; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        content = QWidget()
        content.setObjectName("dash_content")
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # Current session (since last Reset Counts).
        sess_box, sess_lay = self._group("Current Session (since last Reset Counts)")
        sgrid = QGridLayout()
        self.sess_total = self._metric_card("PARTS", "info")
        self.sess_pass = self._metric_card("PASS", "pass")
        self.sess_fail = self._metric_card("REJECTS", "fail")
        self.sess_rate = self._metric_card("PASS RATE", "info")
        sgrid.addWidget(self.sess_total, 0, 0)
        sgrid.addWidget(self.sess_pass, 0, 1)
        sgrid.addWidget(self.sess_fail, 0, 2)
        sgrid.addWidget(self.sess_rate, 0, 3)
        sess_lay.addLayout(sgrid)
        self.sess_meta = QLabel("")
        self.sess_meta.setStyleSheet("color:#94a3b8; font-size:12px; font-weight:700;")
        sess_lay.addWidget(self.sess_meta)
        root.addWidget(sess_box)

        # Today (all sessions on this calendar day).
        today_box, today_lay = self._group("Today")
        tgrid = QGridLayout()
        self.today_total = self._metric_card("PARTS", "info")
        self.today_pass = self._metric_card("PASS", "pass")
        self.today_fail = self._metric_card("REJECTS", "fail")
        self.today_rate = self._metric_card("PASS RATE", "info")
        tgrid.addWidget(self.today_total, 0, 0)
        tgrid.addWidget(self.today_pass, 0, 1)
        tgrid.addWidget(self.today_fail, 0, 2)
        tgrid.addWidget(self.today_rate, 0, 3)
        today_lay.addLayout(tgrid)
        root.addWidget(today_box)

        # Today's reject breakdown by reason.
        fail_box, fail_lay = self._group("Today — Reject Breakdown")
        self.fail_table = self._make_table(["Reject Reason", "Count", "% of Rejects"], cap_height=260)
        fail_lay.addWidget(self.fail_table)
        root.addWidget(fail_box)

        # Last seven days trend.
        trend_box, trend_lay = self._group("Last 7 Days")
        self.trend_table = self._make_table(["Date", "Parts", "PASS", "Rejects", "Pass Rate"], cap_height=320)
        trend_lay.addWidget(self.trend_table)
        root.addWidget(trend_box)

        # Today by hour.
        hour_box, hour_lay = self._group("Today — By Hour")
        self.hour_table = self._make_table(["Hour", "Parts", "PASS", "Rejects"], cap_height=320)
        hour_lay.addWidget(self.hour_table)
        root.addWidget(hour_box)
        root.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        btns = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda _checked=False: self.refresh())
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(lambda _checked=False: self.export_csv())
        open_logs_btn = QPushButton("Open Logs Folder")
        open_logs_btn.clicked.connect(lambda _checked=False: self.hmi.open_folder(LOG_DIR))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(refresh_btn)
        btns.addWidget(export_btn)
        btns.addWidget(open_logs_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        outer.addLayout(btns)

    # ---- formatting helpers ---------------------------------------------------
    @staticmethod
    def _rate_text(passed: int, total: int) -> str:
        if total <= 0:
            return "--"
        return f"{100.0 * passed / total:.1f}%"

    @staticmethod
    def _rate_sub(passed: int, total: int) -> str:
        if total <= 0:
            return "no parts yet"
        return f"{passed}/{total} PASS"

    @staticmethod
    def _fmt_duration(delta: "dt.timedelta") -> str:
        secs = max(0, int(delta.total_seconds()))
        hours, rem = divmod(secs, 3600)
        mins, _ = divmod(rem, 60)
        if hours:
            return f"{hours}h {mins:02d}m"
        return f"{mins}m"

    def _add_row(self, table: QTableWidget, values: List[str]) -> None:
        r = table.rowCount()
        table.insertRow(r)
        for c, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, c, item)

    # ---- refresh / actions ----------------------------------------------------
    def refresh(self):
        hmi = self.hmi
        stats = getattr(hmi, "production_stats", None)

        # Current session uses the live counters that the commit path maintains.
        s_total = int(getattr(hmi, "total_count", 0) or 0)
        s_pass = int(getattr(hmi, "pass_count", 0) or 0)
        s_fail = int(getattr(hmi, "fail_count", 0) or 0)
        self.sess_total.set_value(str(s_total), "committed")
        self.sess_pass.set_value(str(s_pass), "PASS")
        self.sess_fail.set_value(str(s_fail), "FAIL")
        self.sess_rate.set_value(self._rate_text(s_pass, s_total), self._rate_sub(s_pass, s_total))

        start = getattr(hmi, "session_start_t", None)
        if start is not None:
            now = dt.datetime.now()
            duration = now - start
            hours = max(duration.total_seconds() / 3600.0, 0.0)
            pph = (s_total / hours) if hours > 0 else 0.0
            self.sess_meta.setText(
                f"Since {start.strftime('%Y-%m-%d %H:%M')}  •  {self._fmt_duration(duration)} elapsed  •  {pph:.0f} parts/hr"
            )
        else:
            self.sess_meta.setText("")

        if stats is None:
            return

        today_key = dt.datetime.now().strftime("%Y-%m-%d")
        day = stats.day_summary(today_key)
        t_total = int(day.get("total", 0) or 0)
        t_pass = int(day.get("pass", 0) or 0)
        t_fail = int(day.get("fail", 0) or 0)
        self.today_total.set_value(str(t_total), "all sessions")
        self.today_pass.set_value(str(t_pass), "PASS")
        self.today_fail.set_value(str(t_fail), "FAIL")
        self.today_rate.set_value(self._rate_text(t_pass, t_total), self._rate_sub(t_pass, t_total))

        # Reject breakdown (today).
        self.fail_table.setRowCount(0)
        categories = sorted(day.get("fail_categories", {}).items(), key=lambda kv: kv[1], reverse=True)
        if categories:
            for cat, count in categories:
                pct = (100.0 * int(count) / t_fail) if t_fail else 0.0
                self._add_row(self.fail_table, [cat, str(int(count)), f"{pct:.0f}%"])
        else:
            self._add_row(self.fail_table, ["No rejects today", "0", "--"])
        self._fit_table_height(self.fail_table)

        # Last seven days.
        self.trend_table.setRowCount(0)
        recent = stats.recent_days(7)
        if recent:
            for date_key, d in recent:
                d_total = int(d.get("total", 0) or 0)
                d_pass = int(d.get("pass", 0) or 0)
                d_fail = int(d.get("fail", 0) or 0)
                self._add_row(
                    self.trend_table,
                    [date_key, str(d_total), str(d_pass), str(d_fail), self._rate_text(d_pass, d_total)],
                )
        else:
            self._add_row(self.trend_table, ["No data yet", "0", "0", "0", "--"])
        self._fit_table_height(self.trend_table)

        # Today by hour.
        self.hour_table.setRowCount(0)
        hours_map = day.get("hours", {})
        if hours_map:
            for hk in sorted(hours_map.keys()):
                hv = hours_map[hk]
                self._add_row(
                    self.hour_table,
                    [f"{hk}:00", str(int(hv.get("total", 0) or 0)), str(int(hv.get("pass", 0) or 0)), str(int(hv.get("fail", 0) or 0))],
                )
        else:
            self._add_row(self.hour_table, ["No data yet", "0", "0", "0"])
        self._fit_table_height(self.hour_table)

    def export_csv(self):
        stats = getattr(self.hmi, "production_stats", None)
        if stats is None:
            QMessageBox.warning(self, "Export Production Summary", "No production history is available.")
            return
        default = str(LOG_DIR / f"production_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export Production Summary", default, "CSV Files (*.csv)")
        if not path:
            return
        try:
            stats.export_csv(Path(path))
            self.hmi.log(f"Exported production summary to {path}")
            QMessageBox.information(self, "Export Production Summary", f"Saved:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Production Summary", f"Could not export production summary:\n{exc}")


class CustomRejectClassesDialog(QDialog):
    """Let operators define extra YOLO class labels that trigger an immediate PLC reject latch.

    Any detection whose label (case-insensitive) matches one of these names will
    latch the reject output as soon as it is seen, independently of the normal
    battery PASS/FAIL grading pipeline. Typical use: add 'battery_down' so a
    toppled battery trips the reject without needing bung-count inspection.
    """

    def __init__(self, parent=None, reject_classes: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Custom Reject Classes")
        self.setMinimumWidth(380)
        self.setStyleSheet(
            "QDialog { background:#060d1a; }"
            "QLabel { background:transparent; border:none; color:#e2e8f0; }"
            "QListWidget { background:#0b1220; color:#e2e8f0; border:1px solid #334155;"
            " border-radius:8px; font-size:13px; }"
            "QListWidget::item:selected { background:#1e3a5f; color:white; }"
            "QLineEdit { background:#0f172a; color:#e2e8f0; border:1px solid #334155;"
            " border-radius:6px; padding:4px 8px; font-size:13px; }"
            "QPushButton { background:#1e293b; color:#e2e8f0; border:1px solid #334155;"
            " border-radius:6px; padding:5px 14px; font-size:12px; }"
            "QPushButton:hover { background:#334155; }"
        )
        self._build_ui(reject_classes or [])

    def _build_ui(self, initial: List[str]):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        desc = QLabel(
            "Add YOLO class names below. When any detection matches one of these labels\n"
            "the PLC reject latch fires immediately, regardless of bung count.\n"
            "Names are case-insensitive. Example: <b>battery_down</b>"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#94a3b8; font-size:12px; background:transparent; border:none;")
        lay.addWidget(desc)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        for cls in initial:
            self.list_widget.addItem(cls.strip().lower())
        lay.addWidget(self.list_widget)

        add_row = QHBoxLayout()
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("e.g. battery_down")
        self.new_class_edit.returnPressed.connect(self._add_class)
        add_row.addWidget(self.new_class_edit)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_class)
        add_row.addWidget(add_btn)
        lay.addLayout(add_row)

        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        lay.addWidget(rm_btn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _add_class(self):
        text = self.new_class_edit.text().strip().lower()
        if not text:
            return
        existing = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]
        if text not in existing:
            self.list_widget.addItem(text)
        self.new_class_edit.clear()

    def _remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def get_classes(self) -> List[str]:
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setWindowFlags(Qt.Window)
        self.resize(1280, 720)
        self.cap: Optional[Any] = None
        self.camera_backend = "opencv"
        self.opencv_api = "auto"
        self.basler_serial = ""
        self.camera_width = 2592
        self.camera_height = 1944
        self.camera_fps = 30.0
        self.yolo_iou = 0.45
        self.model_task = "auto"
        self.model_class_names_override = "battery,bung"
        self._last_ui_lag_log_t = 0.0
        self._last_metrics_ui_update_t = 0.0
        self._last_stop_request_t = 0.0
        self._stop_cleanup_pending = False
        self._stop_request_source = ""
        self._operator_stop_count = 0
        # Out-of-band stop path for Jetson/X11 input stalls. This is intentionally
        # independent of Qt mouse/key events; it watches for ROOT/runtime_stop.flag
        # from a daemon thread and latches running=False immediately.
        self._external_stop_requested = False
        self._external_stop_count = 0
        self._external_stop_thread_stop = threading.Event()
        self._external_stop_thread = None
        self.camera_sleep_ms = 0.0
        # Model loading can take several seconds for TensorRT engines on Jetson.
        # Keep it off the Qt/UI thread so the HMI and mouse remain responsive.
        self._model_loading = False
        self._model_load_queue: queue.Queue = queue.Queue(maxsize=1)
        self._model_load_thread: Optional[threading.Thread] = None
        self.basler_exposure_auto = False
        self.basler_exposure_us = 5000.0
        self.basler_gain = 0.0
        self.basler_roi_enabled = False
        self.basler_roi_offset_x = 0
        self.basler_roi_offset_y = 0
        self.basler_roi_width = self.camera_width
        self.basler_roi_height = self.camera_height
        self.model_runner = ModelRunner()
        self.demo_mode = False
        self.running = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_preview_rgb: Optional[np.ndarray] = None
        self.last_result: Optional[InspectionResult] = None
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.on_timer)
        self.last_t = time.perf_counter()
        self.fps = 0.0
        self.total_count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.last_logged_status = None

        # Persistent production history for the operator summary dashboard.
        # Operational reporting only; not part of inference/grading/PLC. The
        # session start time tracks the current shift when the operator presses
        # Reset Counts at shift start.
        self.session_start_t = dt.datetime.now()
        self.production_stats = ProductionStats(PRODUCTION_SUMMARY_FILE)
        self.production_stats.load()

        # User-defined YOLO class names that trigger an immediate reject latch.
        self.custom_reject_classes: List[str] = []

        # Latched machine-control reject state. This is separate from counters.
        self.reject_latched = False
        self.reject_latch_id = 0
        self.reject_latch_reason = ""
        self.reject_latch_time = ""

        # Multi-object tracking polish.
        self.track_match_iou = 0.05
        self.committed_track_iou = 0.25

        # Infeed/full-view gate. A newly visible battery can be tracked, but it
        # stays WAIT while its footprint touches the frame edge so a partial
        # battery is not failed before the full lid is visible.
        self.require_full_view_before_grade = True
        self.full_view_margin_percent = 3.0

        # Recipe-less bung pattern validation. When enabled, a battery with six
        # assigned bungs must geometrically match either a six-in-row layout or
        # a 2x3 layout before PASS can commit. This replaces fixed zone logic.
        self.enable_pattern_validation = True
        self.pattern_tolerance_percent = 25.0

        # Multi-battery tracking state. Initialized here so it exists before any timer tick.
        self.stable_required = 6
        self.clear_required = 10
        self.match_distance_px = 180
        self.entry_grace_frames = 12
        self._tracks = {}
        self._next_track_id = 1
        self._next_inspection_id = 1
        self._accepted_track_ids = set()

        # PLC runtime state must exist before first update_status_pills().
        self.plc = PLCInterface()
        self.plc_writer = AsyncPLCWriter()
        self._plc_heartbeat = False
        self._plc_reset_pulse = False
        self._plc_last_status = "DISABLED"
        self._plc_last_error = ""
        self.plc_heartbeat_interval_ms = 500
        self._plc_last_heartbeat_toggle_t = 0.0

        self.actual_camera_width = 0
        self.actual_camera_height = 0
        self.actual_camera_fps = 0.0
        self.actual_camera_fourcc = ""
        self._camera_settings_changed = False
        self._last_frame_ok_t = 0.0
        self._last_prediction_ok_t = 0.0
        self._last_camera_error = ""
        self._last_prediction_error = ""
        self._inference_ms = 0.0
        self.camera_capture_fps = 0.0
        self.camera_read_ms = 0.0
        self.camera_interval_ms = 0.0
        self.camera_frame_age_ms = 0.0
        self.inference_fps = 0.0
        self.inference_cycle_ms = 0.0
        self.inference_idle_ms = 0.0
        self.inference_backend_ms = 0.0
        self.inference_parse_ms = 0.0
        self.paint_ms = 0.0
        self.qimage_ms = 0.0
        self.scale_ms = 0.0
        self.overlay_draw_ms = 0.0
        self.dropped_inference_frames = 0
        self.preview_fps = 0.0
        self.preview_skipped_frames = 0
        self.inference_skipped_frames = 0
        self._preview_count = 0
        self._preview_total = 0
        self._preview_window_t = time.perf_counter()
        self._last_camera_seq_seen = 0
        self._last_submitted_inference_seq = 0
        self._last_processed_inference_seq = 0
        self._last_result_seq = 0
        self._last_displayed_seq = 0
        self._last_inference_result_t = 0.0
        self._last_profiler_log_t = time.perf_counter()
        self._prof_last_cam_seq = 0
        self._prof_last_preview_total = 0
        self._prof_last_paint_total = 0
        self._prof_last_inf_done_count = 0
        self._ui_thread_ident = threading.get_ident()
        self._background_log_queue: queue.Queue = queue.Queue()
        self.camera_worker: Optional[CameraCaptureWorker] = None
        self.inference_worker = InferenceWorker(self.model_runner, log_cb=self.log_from_worker)
        self.inference_worker.start()
        self.save_worker = SaveWorker(log_cb=self.log_from_worker)
        self.save_worker.start()
        self._loading_settings = False
        self._bypass_change_guard = False

        # Training-review capture defaults. These are loaded/overridden by
        # config/settings.json when present.
        self.save_pass_training_images = True
        self.save_fail_training_images = True
        self.save_training_annotated = True
        self.save_training_json = True
        self.save_training_yolo_txt = False
        self.pass_training_sample_rate = 1

        # Ensure capture folders exist so the user can find them immediately.
        (TRAINING_REVIEW_DIR / "pass").mkdir(parents=True, exist_ok=True)
        (TRAINING_REVIEW_DIR / "fail").mkdir(parents=True, exist_ok=True)

        self.log_file = LOG_DIR / "inspection_log.csv"
        self.init_log()
        self.build_ui()
        self.load_settings()
        self.log(
            "Training-review captures: "
            f"PASS={'ON' if self.save_pass_training_images else 'OFF'}, "
            f"FAIL={'ON' if self.save_fail_training_images else 'OFF'}, "
            f"JSON={'ON' if self.save_training_json else 'OFF'}, "
            f"Annotated={'ON' if self.save_training_annotated else 'OFF'}, "
            f"folder={TRAINING_REVIEW_DIR}"
        )
        self.apply_theme()
        self.update_status_pills()
        self.start_external_stop_watchdog()
        self.frame_timer.start(30)

    def init_log(self):
        if not self.log_file.exists():
            with self.log_file.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "status", "reason", "battery_count", "bung_count", "expected_bungs", "fps", "snapshot"])


    def build_ui(self):
        central = QWidget()
        main = QVBoxLayout(central)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        header = QFrame()
        header.setObjectName("Header")
        header.setMaximumHeight(72)
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 6, 12, 6)
        title_box = QVBoxLayout()
        title = QLabel("BungVision Line-Side Inspection")
        title.setObjectName("MainTitle")
        subtitle = QLabel("Line-side runtime inspection  •  Camera + model + PASS/FAIL output")
        subtitle.setObjectName("SubTitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.plc_pill = Pill("PLC SIM", "warn")
        self.camera_pill = Pill("CAMERA OFF", "neutral")
        self.model_pill = Pill("NO MODEL", "warn")
        self.mode_pill = Pill("LIVE RUNTIME", "pass")
        h.addLayout(title_box, 1)
        h.addWidget(self.plc_pill)
        h.addWidget(self.camera_pill)
        h.addWidget(self.model_pill)
        h.addWidget(self.mode_pill)
        main.addWidget(header)

        body = QGridLayout()
        body.setSpacing(8)
        left = QVBoxLayout()

        self.decision_frame = QFrame()
        self.decision_frame.setObjectName("DecisionFrame")
        self.decision_frame.setMaximumHeight(86)
        dlay = QHBoxLayout(self.decision_frame)
        dlay.setContentsMargins(12, 5, 12, 5)
        self.decision_label = QLabel("READY")
        self.decision_label.setObjectName("DecisionLabel")
        self.reason_label = QLabel("Open camera and load model.")
        self.reason_label.setObjectName("ReasonLabel")
        dtext = QVBoxLayout()
        dtext.addWidget(self.decision_label)
        dtext.addWidget(self.reason_label)
        dlay.addLayout(dtext, 1)
        self.bung_big = QLabel("0/6")
        self.bung_big.setObjectName("BungBig")
        dlay.addWidget(self.bung_big)
        left.addWidget(self.decision_frame)

        self.camera_widget = CameraWidget()
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.camera_widget, 1)
        body.addLayout(left, 0, 0, 2, 10)

        side = QVBoxLayout()
        metric_grid = QGridLayout()
        self.pass_rate_card = MetricCard("PASS RATE", "--", "no inspections", "neutral")
        self.reject_card = MetricCard("REJECTS", "0", "current session", "fail")
        self.infer_card = MetricCard("INFERENCE", "-- FPS", "-- ms", "info")
        # v0.9.77: keep the operator screen clean. Preview/camera/paint/skip
        # performance diagnostics remain in the debug PROFILE log, but the visible
        # HMI only shows inference FPS and inference timing.
        metric_grid.addWidget(self.pass_rate_card, 0, 0)
        metric_grid.addWidget(self.reject_card, 0, 1)
        metric_grid.addWidget(self.infer_card, 1, 0, 1, 2)
        side.addLayout(metric_grid)

        # Runtime controls exist as widgets but live in the Settings popup, not on the main operator screen.
        self.source_edit = QLineEdit("0")
        self.model_edit = QLineEdit("")
        self.expected_spin = QSpinBox()
        self.expected_spin.setRange(1, 24)
        self.expected_spin.setValue(6)

        self.debounce_spin = QSpinBox()
        self.debounce_spin.setRange(1, 60)
        self.debounce_spin.setValue(6)

        self.clear_spin = QSpinBox()
        self.clear_spin.setRange(1, 120)
        self.clear_spin.setValue(10)

        self.match_distance_spin = QSpinBox()
        self.match_distance_spin.setRange(20, 1000)
        self.match_distance_spin.setValue(180)

        self.entry_grace_spin = QSpinBox()
        self.entry_grace_spin.setRange(0, 300)
        self.entry_grace_spin.setValue(12)

        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 99)
        self.conf_spin.setValue(25)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1920)
        self.imgsz_spin.setSingleStep(1)
        self.imgsz_spin.setValue(736)

        self.device_edit = QLineEdit("")

        # Hidden backing widgets used by SettingsDialog for PLC config persistence.
        self.plc_enabled_check = QCheckBox("Enable PLC Writes")
        self.plc_enabled_check.setChecked(False)
        self.plc_ip_edit = QLineEdit("192.168.1.10")
        self.plc_tag_edits = {}
        for key, (default_tag, _desc) in PLC_TAG_DEFAULTS.items():
            self.plc_tag_edits[key] = QLineEdit(default_tag)

        controls = QGroupBox("Operator Controls")
        cg = QGridLayout(controls)

        # Demo controls still exist internally for development, but are hidden from the operator screen.
        self.demo_check = QCheckBox("Demo Mode")
        self.demo_check.setChecked(False)
        self.demo_check.stateChanged.connect(lambda _state=None: self.toggle_demo())
        self.demo_check.hide()
        self.fail_demo_check = QCheckBox("Demo Fail")
        self.fail_demo_check.setChecked(False)
        self.fail_demo_check.hide()

        self.open_btn = QPushButton("Run")
        self.open_btn.clicked.connect(lambda _checked=False: self.open_camera())
        self.close_btn = QPushButton("Stop")
        self.close_btn.setAutoRepeat(False)
        # v0.9.73: do not wait for QPushButton.clicked(), which fires on mouse release.
        # On some desktops, the operator can see live preview continue while mouse
        # input/release events are delayed under TensorRT load. Intercept mouse press
        # directly with an event filter so Stop latches as soon as Qt receives the
        # first press event.
        self.close_btn.installEventFilter(self)
        self.close_btn.pressed.connect(lambda: self.request_operator_stop("button_pressed"))
        self.close_btn.clicked.connect(lambda _checked=False: self.request_operator_stop("button_clicked"))
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(lambda _checked=False: self.open_settings_dialog())
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(lambda _checked=False: self.load_model())
        self.reset_reject_btn = QPushButton("Reset Reject")
        self.reset_reject_btn.clicked.connect(lambda _checked=False: self.reset_reject_latch())
        self.reset_btn = QPushButton("Reset Counts")
        self.reset_btn.clicked.connect(lambda _checked=False: self.reset_counts())
        self.summary_btn = QPushButton("Production Summary")
        self.summary_btn.setToolTip("Open the read-only production summary: session, today, last 7 days, reject breakdown, and by-hour throughput.")
        self.summary_btn.clicked.connect(lambda _checked=False: self.open_production_dashboard())
        self.reject_classes_btn = QPushButton("Reject Classes")
        self.reject_classes_btn.setToolTip("Define YOLO class names that trigger an immediate PLC reject latch (e.g. 'battery_down').")
        self.reject_classes_btn.clicked.connect(lambda _checked=False: self.open_reject_classes_dialog())
        self.bypass_check = QCheckBox("Bypass (Supervisor)")
        self.bypass_check.setToolTip("Supervisor-only bypass. Bypass inhibits vision stop/alarm requests, but does not make the PLC Ready bit true.")
        self.bypass_check.clicked.connect(lambda checked=False: self.on_bypass_changed(bool(checked)))

        cg.addWidget(self.open_btn, 0, 0)
        cg.addWidget(self.close_btn, 0, 1)
        cg.addWidget(self.load_btn, 1, 0)
        cg.addWidget(self.settings_btn, 1, 1)
        cg.addWidget(self.reset_reject_btn, 2, 0)
        cg.addWidget(self.reset_btn, 2, 1)
        cg.addWidget(self.summary_btn, 3, 0, 1, 2)
        cg.addWidget(self.reject_classes_btn, 4, 0, 1, 2)
        cg.addWidget(self.bypass_check, 5, 0, 1, 2)
        side.addWidget(controls)

        overlay_box = QGroupBox("Camera Overlay")
        og = QGridLayout(overlay_box)
        og.setContentsMargins(6, 8, 6, 6)
        og.setSpacing(2)
        self.overlay_enable_check = QCheckBox("Overlay On")
        self.overlay_enable_check.setChecked(True)
        self.overlay_boxes_check = QCheckBox("Detection Boxes")
        self.overlay_boxes_check.setChecked(True)
        self.overlay_labels_check = QCheckBox("Class Labels")
        self.overlay_labels_check.setChecked(True)
        self.overlay_grades_check = QCheckBox("PASS/FAIL Badges")
        self.overlay_grades_check.setChecked(True)
        self.overlay_fail_banner_check = QCheckBox("Fail Banner")
        self.overlay_fail_banner_check.setChecked(True)
        for cb in (
            self.overlay_enable_check,
            self.overlay_boxes_check,
            self.overlay_labels_check,
            self.overlay_grades_check,
            self.overlay_fail_banner_check,
        ):
            cb.stateChanged.connect(lambda _state=None: self.on_overlay_control_changed())
        og.addWidget(self.overlay_enable_check, 0, 0)
        og.addWidget(self.overlay_boxes_check, 0, 1)
        og.addWidget(self.overlay_labels_check, 1, 0)
        og.addWidget(self.overlay_grades_check, 1, 1)
        og.addWidget(self.overlay_fail_banner_check, 2, 0, 1, 2)
        side.addWidget(overlay_box)

        saving_box = QGroupBox("Image Saving")
        sg = QGridLayout(saving_box)
        sg.setContentsMargins(6, 8, 6, 6)
        sg.setSpacing(4)
        self.save_pass_images_check = QCheckBox("Save PASS Images")
        self.save_pass_images_check.setChecked(True)
        self.save_fail_images_check = QCheckBox("Save FAIL Images")
        self.save_fail_images_check.setChecked(True)
        self.save_pass_images_check.stateChanged.connect(lambda _state=None: self.on_image_saving_changed())
        self.save_fail_images_check.stateChanged.connect(lambda _state=None: self.on_image_saving_changed())
        sg.addWidget(self.save_pass_images_check, 0, 0)
        sg.addWidget(self.save_fail_images_check, 0, 1)
        side.addWidget(saving_box)

        plc = QGroupBox("Machine Interface")
        pg = QGridLayout(plc)
        self.heartbeat_pill = Pill("Heartbeat SIM", "warn")
        self.stop_output_pill = Pill("Stop Output OFF", "neutral")
        self.alarm_pill = Pill("Alarm OFF", "neutral")
        self.ready_pill = Pill("Not Ready", "warn")
        self.camera_actual_pill = Pill("Actual Camera --", "neutral")
        self.plc_write_pill = Pill("PLC Writes OFF", "neutral")
        pg.addWidget(self.heartbeat_pill, 0, 0)
        pg.addWidget(self.stop_output_pill, 0, 1)
        pg.addWidget(self.alarm_pill, 1, 0)
        pg.addWidget(self.ready_pill, 1, 1)
        pg.addWidget(self.camera_actual_pill, 2, 0, 1, 2)
        pg.addWidget(self.plc_write_pill, 3, 0, 1, 2)
        side.addWidget(plc)

        # Keep the backing history table for CSV/export/internal calls, but do not show
        # the Previous/Recent Inspections panel on the operator screen.
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time", "Result", "Bungs", "Reason", "FPS"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.hide()
        side.addStretch(1)
        body.addLayout(side, 0, 10, 2, 2)
        main.addLayout(body, 1)

        # Hidden backing log: keep log() calls and export/debug routines safe without
        # consuming operator-screen real estate.
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.hide()
        self.log(f"{APP_TITLE} started. Confirm this title before testing.")

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        # v0.9.81: production operator screen does not use the top menu bar.
        # Keep emergency keyboard shortcuts, but remove the visible menu actions.
        self.menuBar().setVisible(False)
        self._esc_stop_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._esc_stop_shortcut.activated.connect(lambda: self.request_operator_stop("esc"))
        self._f12_stop_shortcut = QShortcut(QKeySequence("F12"), self)
        self._f12_stop_shortcut.activated.connect(lambda: self.request_operator_stop("f12"))

    def apply_theme(self):
        # App-level rules are needed for QMessageBox/QFileDialog because they are
        # standalone top-level dialogs, not children of the main window stylesheet.
        apply_global_readability_style()
        self.setStyleSheet("""
            QMainWindow { background:#020617; }
            QWidget { color:#e2e8f0; font-family:Arial; }
            QFrame#Header, QGroupBox { background:#0f172a; border:1px solid #334155; border-radius:18px; }
            QGroupBox { margin-top:8px; padding:7px; font-weight:800; color:#cbd5e1; }
            QGroupBox::title { subcontrol-origin: margin; left:14px; padding:0 6px; color:#93c5fd; }
            QLabel#MainTitle { color:white; font-size:20px; font-weight:900; }
            QLabel#SubTitle { color:#94a3b8; font-size:11px; }
            QFrame#DecisionFrame { background:#1e293b; border:1px solid #334155; border-radius:22px; }
            QLabel#DecisionLabel { color:white; font-size:38px; font-weight:900; }
            QLabel#ReasonLabel { color:#cbd5e1; font-size:12px; font-weight:700; }
            QLabel#BungBig { color:white; font-size:30px; font-weight:900; padding-right:6px; }
            QPushButton { background:#2563eb; color:white; border:none; border-radius:13px; padding:7px 9px; font-weight:800; }
            QPushButton:hover { background:#3b82f6; }
            QPushButton:pressed { background:#1d4ed8; }
            QLineEdit, QSpinBox, QComboBox { background:#020617; color:#e2e8f0; border:1px solid #334155; border-radius:9px; padding:7px; }
            QTableWidget { background:#020617; color:#e2e8f0; border:1px solid #334155; border-radius:12px; gridline-color:#1e293b; }
            QHeaderView::section { background:#1e293b; color:#cbd5e1; border:none; padding:6px; font-weight:800; }
            QTextEdit { background:#020617; border:1px solid #334155; border-radius:12px; color:#cbd5e1; padding:8px; }
        """ + common_readability_qss())

    def _bool_attr(self, name: str, default: bool) -> bool:
        return bool(getattr(self, name, default))

    def _int_attr(self, name: str, default: int) -> int:
        try:
            return int(getattr(self, name, default))
        except Exception:
            return int(default)

    def _capture_settings_payload(self) -> dict:
        return {
            "save_pass_training_images": self._bool_attr("save_pass_training_images", True),
            "save_fail_training_images": self._bool_attr("save_fail_training_images", True),
            "save_training_annotated": self._bool_attr("save_training_annotated", True),
            "save_training_json": self._bool_attr("save_training_json", True),
            "save_training_yolo_txt": self._bool_attr("save_training_yolo_txt", False),
            "pass_training_sample_rate": max(1, self._int_attr("pass_training_sample_rate", 1)),
        }

    def settings_payload(self) -> dict:
        """Current runtime settings for persistence."""
        return {
            "camera_backend": str(getattr(self, "camera_backend", "opencv")),
            "camera_source": self.source_edit.text().strip() if hasattr(self, "source_edit") else "0",
            "opencv_api": str(getattr(self, "opencv_api", "auto")),
            "basler_serial": str(getattr(self, "basler_serial", "")),
            "camera_width": int(getattr(self, "camera_width", 2592)),
            "camera_height": int(getattr(self, "camera_height", 1944)),
            "camera_fps": float(getattr(self, "camera_fps", 30.0)),
            "basler_exposure_auto": bool(getattr(self, "basler_exposure_auto", False)),
            "basler_exposure_us": float(getattr(self, "basler_exposure_us", 5000.0)),
            "basler_gain": float(getattr(self, "basler_gain", 0.0)),
            "basler_roi_enabled": bool(getattr(self, "basler_roi_enabled", False)),
            "basler_roi_offset_x": int(getattr(self, "basler_roi_offset_x", 0)),
            "basler_roi_offset_y": int(getattr(self, "basler_roi_offset_y", 0)),
            "basler_roi_width": int(getattr(self, "basler_roi_width", getattr(self, "camera_width", 2592))),
            "basler_roi_height": int(getattr(self, "basler_roi_height", getattr(self, "camera_height", 1944))),
            "model_path": self.model_edit.text().strip() if hasattr(self, "model_edit") else "",
            "model_task": str(getattr(self, "model_task", "auto")),
            "model_class_names_override": str(getattr(self, "model_class_names_override", "")),
            "expected_bungs": self._spin_value("expected_spin", 6),
            "confidence_percent": self._spin_value("conf_spin", 25),
            "yolo_iou_percent": int(round(float(getattr(self, "yolo_iou", 0.45)) * 100)),
            "yolo_image_size": self._spin_value("imgsz_spin", 736),
            "device": self.device_edit.text().strip() if hasattr(self, "device_edit") else "",
            "debounce_frames": self._spin_value("debounce_spin", 6),
            "entry_grace_frames": self._spin_value("entry_grace_spin", 12),
            "clear_frames": self._spin_value("clear_spin", 10),
            "track_match_px": self._spin_value("match_distance_spin", 180),
            "track_match_iou_percent": int(round(float(getattr(self, "track_match_iou", 0.05)) * 100)),
            "committed_track_iou_percent": int(round(float(getattr(self, "committed_track_iou", 0.25)) * 100)),
            "require_full_view_before_grade": bool(getattr(self, "require_full_view_before_grade", True)),
            "full_view_margin_percent": float(getattr(self, "full_view_margin_percent", 3.0)),
            "enable_pattern_validation": bool(getattr(self, "enable_pattern_validation", True)),
            "pattern_tolerance_percent": float(getattr(self, "pattern_tolerance_percent", 25.0)),
            "save_pass_training_images": bool(getattr(self, "save_pass_training_images", True)),
            "save_fail_training_images": bool(getattr(self, "save_fail_training_images", True)),
            "save_training_annotated": bool(getattr(self, "save_training_annotated", True)),
            "save_training_json": bool(getattr(self, "save_training_json", True)),
            "save_training_yolo_txt": bool(getattr(self, "save_training_yolo_txt", False)),
            "pass_training_sample_rate": int(getattr(self, "pass_training_sample_rate", 1)),
            "overlay_enabled": self._checkbox_checked("overlay_enable_check", True),
            "overlay_boxes": self._checkbox_checked("overlay_boxes_check", True),
            "overlay_labels": self._checkbox_checked("overlay_labels_check", True),
            "overlay_grades": self._checkbox_checked("overlay_grades_check", True),
            "overlay_fail_banner": self._checkbox_checked("overlay_fail_banner_check", True),
            "save_pass_images": self._checkbox_checked("save_pass_images_check", True),
            "save_fail_images": self._checkbox_checked("save_fail_images_check", True),
            "plc_enabled": self._checkbox_checked("plc_enabled_check", False),
            "plc_ip": self.plc_ip_edit.text().strip() if hasattr(self, "plc_ip_edit") else "",
            "plc_heartbeat_interval_ms": int(getattr(self, "plc_heartbeat_interval_ms", 500)),
            "plc_tags": {key: edit.text().strip() for key, edit in getattr(self, "plc_tag_edits", {}).items()},
            "custom_reject_classes": list(getattr(self, "custom_reject_classes", [])),
        }

    def save_settings(self, silent: bool = False):
        """Save runtime settings to config/settings.json."""
        if bool(getattr(self, "_loading_settings", False)):
            return True
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(self.settings_payload(), indent=2), encoding="utf-8")
            if not silent:
                self.log(f"Settings saved to {SETTINGS_FILE}")
            return True
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "Settings", f"Could not save settings:\n{e}")
            return False

    def load_settings(self):
        """Load runtime settings from config/settings.json if present."""
        if not SETTINGS_FILE.exists():
            return False
        try:
            self._loading_settings = True
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))

            self.camera_backend = str(data.get("camera_backend", getattr(self, "camera_backend", "opencv"))).lower()
            if self.camera_backend in ("basler/pylon", "pylon"):
                self.camera_backend = "basler"
            if self.camera_backend not in ("opencv", "basler"):
                self.camera_backend = "opencv"
            self.opencv_api = str(data.get("opencv_api", getattr(self, "opencv_api", "auto"))).strip().lower()
            if self.opencv_api in ("directshow", "direct_show"):
                self.opencv_api = "dshow"
            if self.opencv_api in ("gst",):
                self.opencv_api = "gstreamer"
            if self.opencv_api not in ("auto", "dshow", "msmf", "v4l2", "gstreamer"):
                self.opencv_api = "auto"
            self.basler_serial = str(data.get("basler_serial", getattr(self, "basler_serial", ""))).strip()
            self.camera_width = max(320, int(data.get("camera_width", getattr(self, "camera_width", 2592))))
            self.camera_height = max(240, int(data.get("camera_height", getattr(self, "camera_height", 1944))))
            self.camera_fps = max(1.0, float(data.get("camera_fps", getattr(self, "camera_fps", 30.0))))
            self.basler_exposure_auto = bool(data.get("basler_exposure_auto", getattr(self, "basler_exposure_auto", False)))
            self.basler_exposure_us = max(1.0, float(data.get("basler_exposure_us", getattr(self, "basler_exposure_us", 5000.0))))
            self.basler_gain = max(0.0, float(data.get("basler_gain", getattr(self, "basler_gain", 0.0))))
            self.basler_roi_enabled = bool(data.get("basler_roi_enabled", getattr(self, "basler_roi_enabled", False)))
            self.basler_roi_offset_x = max(0, int(data.get("basler_roi_offset_x", getattr(self, "basler_roi_offset_x", 0))))
            self.basler_roi_offset_y = max(0, int(data.get("basler_roi_offset_y", getattr(self, "basler_roi_offset_y", 0))))
            self.basler_roi_width = max(1, int(data.get("basler_roi_width", getattr(self, "basler_roi_width", self.camera_width))))
            self.basler_roi_height = max(1, int(data.get("basler_roi_height", getattr(self, "basler_roi_height", self.camera_height))))
            if hasattr(self, "source_edit"):
                self.source_edit.setText(str(data.get("camera_source", self.source_edit.text())))
            if hasattr(self, "model_edit"):
                self.model_edit.setText(str(data.get("model_path", self.model_edit.text())))
            self.model_task = _normalize_model_task(data.get("model_task", getattr(self, "model_task", "auto")))
            self.model_class_names_override = str(data.get("model_class_names_override", getattr(self, "model_class_names_override", "battery,bung"))).strip()
            if hasattr(self, "device_edit"):
                self.device_edit.setText(str(data.get("device", "")))
            self.yolo_iou = max(0.01, min(0.99, float(data.get("yolo_iou_percent", int(round(float(getattr(self, "yolo_iou", 0.45)) * 100)))) / 100.0))

            if hasattr(self, "expected_spin"):
                self.expected_spin.setValue(int(data.get("expected_bungs", self.expected_spin.value())))
            if hasattr(self, "conf_spin"):
                self.conf_spin.setValue(int(data.get("confidence_percent", self.conf_spin.value())))
            if hasattr(self, "imgsz_spin"):
                self.imgsz_spin.setValue(int(data.get("yolo_image_size", self.imgsz_spin.value())))
            if hasattr(self, "debounce_spin"):
                self.debounce_spin.setValue(int(data.get("debounce_frames", self.debounce_spin.value())))
            if hasattr(self, "entry_grace_spin"):
                self.entry_grace_spin.setValue(int(data.get("entry_grace_frames", self.entry_grace_spin.value())))
            if hasattr(self, "clear_spin"):
                self.clear_spin.setValue(int(data.get("clear_frames", self.clear_spin.value())))
            if hasattr(self, "match_distance_spin"):
                self.match_distance_spin.setValue(int(data.get("track_match_px", self.match_distance_spin.value())))
            self.track_match_iou = self._load_percent_setting(data, "track_match_iou_percent", "track_match_iou", getattr(self, "track_match_iou", 0.05), 0.0, 1.0)
            self.committed_track_iou = self._load_percent_setting(data, "committed_track_iou_percent", "committed_track_iou", getattr(self, "committed_track_iou", 0.25), 0.0, 1.0)
            self.require_full_view_before_grade = bool(data.get("require_full_view_before_grade", getattr(self, "require_full_view_before_grade", True)))
            self.full_view_margin_percent = max(0.0, min(25.0, float(data.get("full_view_margin_percent", getattr(self, "full_view_margin_percent", 3.0)))))
            self.enable_pattern_validation = bool(data.get("enable_pattern_validation", getattr(self, "enable_pattern_validation", True)))
            self.pattern_tolerance_percent = max(5.0, min(75.0, float(data.get("pattern_tolerance_percent", getattr(self, "pattern_tolerance_percent", 25.0)))))

            self.save_pass_training_images = bool(data.get("save_pass_training_images", getattr(self, "save_pass_training_images", True)))
            self.save_fail_training_images = bool(data.get("save_fail_training_images", getattr(self, "save_fail_training_images", True)))
            self.save_training_annotated = bool(data.get("save_training_annotated", getattr(self, "save_training_annotated", True)))
            self.save_training_json = bool(data.get("save_training_json", getattr(self, "save_training_json", True)))
            self.save_training_yolo_txt = bool(data.get("save_training_yolo_txt", getattr(self, "save_training_yolo_txt", False)))
            self.pass_training_sample_rate = int(data.get("pass_training_sample_rate", getattr(self, "pass_training_sample_rate", 1)))
            for name, key, default in (
                ("overlay_enable_check", "overlay_enabled", True),
                ("overlay_boxes_check", "overlay_boxes", True),
                ("overlay_labels_check", "overlay_labels", True),
                ("overlay_grades_check", "overlay_grades", True),
                ("overlay_fail_banner_check", "overlay_fail_banner", True),
                ("save_pass_images_check", "save_pass_images", True),
                ("save_fail_images_check", "save_fail_images", True),
                ("plc_enabled_check", "plc_enabled", False),
            ):
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setChecked(bool(data.get(key, default)))
            if hasattr(self, "bypass_check"):
                self.bypass_check.setChecked(False)
            if hasattr(self, "plc_ip_edit"):
                self.plc_ip_edit.setText(str(data.get("plc_ip", self.plc_ip_edit.text())))
            self.plc_heartbeat_interval_ms = int(data.get("plc_heartbeat_interval_ms", getattr(self, "plc_heartbeat_interval_ms", 500)))
            self.plc_heartbeat_interval_ms = max(100, min(10000, self.plc_heartbeat_interval_ms))
            tags = data.get("plc_tags", {}) or {}
            if hasattr(self, "plc_tag_edits"):
                for key, edit in self.plc_tag_edits.items():
                    if key in tags:
                        edit.setText(str(tags[key]))

            raw_classes = data.get("custom_reject_classes", [])
            if isinstance(raw_classes, list):
                self.custom_reject_classes = [str(c).strip().lower() for c in raw_classes if str(c).strip()]
            else:
                self.custom_reject_classes = []

            self.apply_runtime_settings()
            self.log(f"Settings loaded from {SETTINGS_FILE}; bypass forced OFF at startup.")
            return True
        except Exception as e:
            self.log(f"Could not load settings: {e}")
            return False
        finally:
            self._loading_settings = False

    def _load_percent_setting(self, data: dict, percent_key: str, legacy_float_key: str, default: float, lo: float, hi: float) -> float:
        """Load a setting normally stored as an integer percent, with decimal-key fallback."""
        try:
            if percent_key in data:
                value = float(data.get(percent_key, float(default) * 100.0)) / 100.0
            elif legacy_float_key in data:
                value = float(data.get(legacy_float_key, default))
            else:
                value = float(default)
        except Exception:
            value = float(default)
        return max(float(lo), min(float(hi), value))

    def _checkbox_checked(self, name: str, default: bool = False) -> bool:
        widget = getattr(self, name, None)
        if widget is None:
            return default
        try:
            return bool(widget.isChecked())
        except Exception:
            return default

    def on_overlay_control_changed(self):
        if hasattr(self, "camera_widget"):
            self.camera_widget.set_overlay_options(
                enabled=self._checkbox_checked("overlay_enable_check", True),
                boxes=self._checkbox_checked("overlay_boxes_check", True),
                labels=self._checkbox_checked("overlay_labels_check", True),
                grades=self._checkbox_checked("overlay_grades_check", True),
                fail_banner=self._checkbox_checked("overlay_fail_banner_check", True),
            )
        if not bool(getattr(self, "_loading_settings", False)):
            self.save_settings(silent=True)

    def on_image_saving_changed(self):
        if not bool(getattr(self, "_loading_settings", False)):
            self.save_settings(silent=True)

    def on_bypass_changed(self, requested_checked=None):
        if bool(getattr(self, "_bypass_change_guard", False)):
            return

        requested = self._checkbox_checked("bypass_check", False) if requested_checked is None else bool(requested_checked)

        if bool(getattr(self, "_loading_settings", False)):
            self.update_status_pills(self.last_result)
            self.update_plc_outputs(self.last_result)
            return

        if requested:
            # Do not leave Bypass visually or logically enabled while the confirmation
            # dialog is open. This prevents a cancel/no response from priming the next
            # click into an unconfirmed bypass state.
            self._bypass_change_guard = True
            try:
                self.bypass_check.setChecked(False)
            finally:
                self._bypass_change_guard = False

            reply = QMessageBox.question(
                self,
                "Enable Bypass",
                "Bypass will inhibit BungVision stop/alarm requests. Use this only with supervisor authorization.\n\nEnable bypass?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._bypass_change_guard = True
                try:
                    self.bypass_check.setChecked(True)
                finally:
                    self._bypass_change_guard = False
                self.log("Bypass enabled by operator confirmation.")
            else:
                self.log("Bypass enable cancelled by operator.")
        else:
            self._bypass_change_guard = True
            try:
                self.bypass_check.setChecked(False)
            finally:
                self._bypass_change_guard = False
            self.log("Bypass disabled by operator.")

        self.update_status_pills(self.last_result)
        self.update_plc_outputs(self.last_result)
        # Bypass is intentionally not persisted across restarts.

    def apply_plc_config(self):
        """Push PLC configuration to the asynchronous PLC writer.

        This intentionally does not open the PLC connection or write tags on the
        UI thread. The background worker applies the config on its next cycle.
        """
        if not hasattr(self, "plc_writer") or self.plc_writer is None:
            self.plc_writer = AsyncPLCWriter()
        tags = {key: edit.text().strip() for key, edit in getattr(self, "plc_tag_edits", {}).items()}
        try:
            hb_ms = max(100, min(10000, int(getattr(self, "plc_heartbeat_interval_ms", 500))))
            self.plc_writer.configure(
                enabled=self._checkbox_checked("plc_enabled_check", False),
                ip_address=self.plc_ip_edit.text().strip() if hasattr(self, "plc_ip_edit") else "",
                tags=tags,
                heartbeat_interval_ms=hb_ms,
            )
        except Exception as e:
            self._plc_last_status = "CONFIG ERROR"
            self._plc_last_error = str(e)
            if hasattr(self, "plc_write_pill"):
                self.plc_write_pill.setText("PLC Config Error")
                self.plc_write_pill.set_tone("fail")

    def _update_plc_heartbeat_bit(self) -> bool:
        """Compatibility shim; heartbeat is now owned by AsyncPLCWriter."""
        if hasattr(self, "plc_writer") and self.plc_writer is not None:
            status, error, heartbeat = self.plc_writer.status()
            self._plc_last_status = status
            self._plc_last_error = error
            self._plc_heartbeat = heartbeat
        return bool(getattr(self, "_plc_heartbeat", False))

    def vision_healthy(self) -> bool:
        """Fail-safe health gate for PLC Ready."""
        if bool(getattr(self, "demo_mode", False)):
            return bool(getattr(self, "running", False))
        if not bool(getattr(self, "running", False)):
            return False
        cap = getattr(self, "cap", None)
        if cap is None:
            return False
        try:
            if hasattr(cap, "is_opened") and not cap.is_opened():
                return False
        except Exception:
            return False
        if getattr(self, "camera_worker", None) is None:
            return False
        if getattr(self.model_runner, "model", None) is None:
            return False
        now = time.perf_counter()
        if now - float(getattr(self, "_last_frame_ok_t", 0.0)) > 1.5:
            return False
        if str(getattr(self, "_last_prediction_error", "")):
            return False
        return True

    def update_plc_outputs(self, result: Optional[InspectionResult] = None):
        if not hasattr(self, "plc_writer") or self.plc_writer is None:
            self.plc_writer = AsyncPLCWriter()

        try:
            self.apply_plc_config()
            bypass = self._checkbox_checked("bypass_check", False)
            fail_active = bool(getattr(self, "reject_latched", False))
            stop_request = fail_active and not bypass
            running_state = bool(self.running or bypass)
            ready_state = bool(self.vision_healthy() and not stop_request)
            reset_pulse = bool(getattr(self, "_plc_reset_pulse", False))

            states = {
                "running": running_state,
                "bypass": bypass,
                "stop_request": stop_request,
                "alarm": stop_request,
                "ready": ready_state,
                "reset": reset_pulse,
            }
            self.plc_writer.submit(states, reset_pulse=reset_pulse)
            self._plc_reset_pulse = False
            status, error, heartbeat = self.plc_writer.status()
            self._plc_last_status = status
            self._plc_last_error = error
            self._plc_heartbeat = heartbeat
        except Exception as e:
            status = "ERROR"
            self._plc_last_status = "ERROR"
            self._plc_last_error = str(e)
            self._plc_reset_pulse = False
            self.log(f"PLC async update error: {e}")

        if hasattr(self, "plc_write_pill"):
            if self._checkbox_checked("plc_enabled_check", False):
                short_status = status if len(str(status)) <= 28 else str(status)[:25] + "..."
                self.plc_write_pill.setText(f"PLC Writes {short_status}")
                self.plc_write_pill.set_tone("pass" if str(status).startswith("CONNECTED") else "fail")
            else:
                self.plc_write_pill.setText("PLC Writes OFF")
                self.plc_write_pill.set_tone("neutral")

    def pulse_plc_reset(self):
        self._plc_reset_pulse = True
        self.update_plc_outputs(self.last_result)

    def apply_runtime_settings(self):
        """Apply visible settings widgets to runtime state without necessarily saving to disk."""
        self.ensure_tracker_state()
        self.stable_required = self._spin_value("debounce_spin", 6)
        self.clear_required = self._spin_value("clear_spin", 10)
        self.match_distance_px = self._spin_value("match_distance_spin", 180)
        self.entry_grace_frames = self._spin_value("entry_grace_spin", 12)
        self.demo_mode = False
        if hasattr(self, "demo_check"):
            self.demo_check.setChecked(False)
        if hasattr(self, "fail_demo_check"):
            self.fail_demo_check.setChecked(False)

        try:
            self.pass_training_sample_rate = max(1, int(getattr(self, "pass_training_sample_rate", 1)))
        except Exception:
            self.pass_training_sample_rate = 1
        self.on_overlay_control_changed()
        try:
            self.plc_heartbeat_interval_ms = max(100, min(10000, int(getattr(self, "plc_heartbeat_interval_ms", 500))))
        except Exception:
            self.plc_heartbeat_interval_ms = 500
        try:
            self.track_match_iou = max(0.0, min(1.0, float(getattr(self, "track_match_iou", 0.05))))
            self.committed_track_iou = max(0.0, min(1.0, float(getattr(self, "committed_track_iou", 0.25))))
            self.yolo_iou = max(0.01, min(0.99, float(getattr(self, "yolo_iou", 0.45))))
        except Exception:
            self.track_match_iou = 0.05
            self.committed_track_iou = 0.25
            self.yolo_iou = 0.45
        self.apply_plc_config()
        if bool(getattr(self, "_camera_settings_changed", False)):
            was_open = self.cap is not None and self.cap.is_opened() if hasattr(self.cap, "is_opened") else self.cap is not None
            self._camera_settings_changed = False
            if was_open:
                self.log("Camera settings changed; reopening camera with the requested resolution/ROI.")
                self.open_camera()

    def open_settings_dialog(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self.apply_runtime_settings()
        self.update_status_pills()

    def test_plc_connection(self):
        self.copy_to_parent()
        self.parent_hmi.test_plc_connection()

    def test_plc_heartbeat(self):
        self.copy_to_parent()
        self.parent_hmi.test_plc_write("heartbeat")

    def test_plc_stop_request(self):
        self.copy_to_parent()
        self.parent_hmi.test_plc_write("stop_request")

    def list_basler_from_dialog(self, *args):
        ok, msg = BaslerPylonCamera.available()
        if not ok:
            QMessageBox.warning(self, "Basler/Pylon", "pypylon is not available. Install Basler Pylon SDK and run: pip install pypylon\n\n" + msg)
            return
        cams = list_basler_cameras()
        if not cams:
            QMessageBox.warning(self, "Basler/Pylon", "pypylon loaded, but no Basler cameras were found.")
            return
        lines = []
        for cam in cams:
            lines.append(f"Model: {cam.get('model','')}\nSerial: {cam.get('serial','')}\nName: {cam.get('name','')}")
        QMessageBox.information(self, "Basler Cameras", "\n\n".join(lines))

    def browse_model(self, *args):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", str(ROOT), "YOLO Models (*.pt *.engine);;All Files (*)")
        if path:
            self.model_edit.setText(path)

    def load_model(self, *args):
        """Start model loading on a background thread.

        Earlier builds loaded TensorRT/Ultralytics directly on the Qt main
        thread. On Jetson, deserializing a TensorRT engine can take several
        seconds and completely stalls the HMI event loop. The actual YOLO object
        is still the same ModelRunner/Ultralytics object; only the loading work
        is moved off the UI thread.
        """
        if bool(getattr(self, "_model_loading", False)):
            QMessageBox.information(self, "Model", "A model is already loading. Please wait for it to finish.")
            return
        path = self.model_edit.text().strip()
        if not path:
            QMessageBox.information(self, "Model", "Choose a .pt or .engine model first.")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "Model", f"Model does not exist:\n{path}")
            return

        # Pause prediction before replacing/initializing the YOLO object. This is
        # still done on the UI thread, but should be quick compared to engine load.
        try:
            if hasattr(self, "inference_worker") and self.inference_worker is not None:
                self.inference_worker.stop()
        except Exception:
            pass

        bad_argv = [(type(a).__name__, repr(a)) for a in sys.argv if not isinstance(a, (str, bytes))]
        if bad_argv:
            self.log(f"Sanitizing non-string sys.argv entries before YOLO load: {bad_argv}")
        task = _normalize_model_task(getattr(self, "model_task", "auto"))
        class_names_override = str(getattr(self, "model_class_names_override", "")).strip()
        # Capture the operator's device/image-size so the background load can
        # initialize and warm up the model on the real target device once,
        # instead of paying that cost again on the first live inspection frame.
        load_device = self.device_edit.text().strip() if hasattr(self, "device_edit") else ""
        try:
            load_imgsz = int(self.imgsz_spin.value()) if hasattr(self, "imgsz_spin") else 0
        except Exception:
            load_imgsz = 0
        self.log(f"MODEL_LOAD_REQUEST using YOLO(path) path={path!r}, task={task!r}, class_names_override={class_names_override!r}, device={load_device!r}, imgsz={load_imgsz}")
        _write_debug_log(f"MODEL_LOAD_REQUEST path={path!r} task={task!r} class_override={class_names_override!r}")

        self._model_loading = True
        try:
            self.load_btn.setEnabled(False)
            self.load_btn.setText("Loading...")
        except Exception:
            pass
        self.model_pill.setText("MODEL LOADING")
        self.model_pill.set_tone("warn")

        # Empty stale completion messages, if any.
        try:
            while True:
                self._model_load_queue.get_nowait()
        except Exception:
            pass

        def _worker():
            try:
                msg = self.model_runner.load(path, task=task, class_names_override=class_names_override, device=load_device, imgsz=load_imgsz)
                self._model_load_queue.put(("ok", msg, ""), timeout=0.1)
            except Exception:
                tb = traceback.format_exc()
                _write_debug_log("ASYNC MODEL LOAD FAILED FULL TRACEBACK:\n" + tb)
                try:
                    self._model_load_queue.put(("error", "", tb), timeout=0.1)
                except Exception:
                    pass

        self._model_load_thread = threading.Thread(target=_worker, name="BungVisionModelLoadWorker", daemon=True)
        self._model_load_thread.start()

    def _poll_model_load_result(self) -> None:
        """Complete async model loading from the Qt/UI thread."""
        if not bool(getattr(self, "_model_loading", False)):
            return
        try:
            status, msg, tb = self._model_load_queue.get_nowait()
        except queue.Empty:
            return
        except Exception:
            return

        self._model_loading = False
        try:
            self.load_btn.setEnabled(True)
            self.load_btn.setText("Load Model")
        except Exception:
            pass

        if status == "ok":
            try:
                if hasattr(self, "inference_worker") and self.inference_worker is not None:
                    self.inference_worker.clear()
                    self.inference_worker.start()
            except Exception:
                pass
            self.demo_check.setChecked(False)
            self.demo_mode = False
            self.model_pill.setText("MODEL LOADED")
            self.model_pill.set_tone("pass")
            self.log(str(msg))
            return

        try:
            if hasattr(self, "inference_worker") and self.inference_worker is not None:
                self.inference_worker.start()
        except Exception:
            pass
        self.model_pill.setText("MODEL ERROR")
        self.model_pill.set_tone("fail")
        self.log("Model load failed:\n" + str(tb))
        QMessageBox.warning(
            self,
            "Model Load Failed",
            "Could not load the model with the simple Ultralytics YOLO(path) loader.\n\n"
            "A plain text diagnostic file was written here:\n"
            f"{DEBUG_LOG_FILE}\n\n"
            "Short error:\n" + (str(tb).splitlines()[-1] if str(tb).strip() else "unknown error"),
        )

    def reset_runtime_session(self, reset_counts: bool = False):
        """Clear all per-run tracking/decision state.

        This must run whenever the runtime is stopped/started so a new battery
        cannot inherit a previous battery's PASS lock or inspection ID.
        """
        # Clear active tracks/PASS locks. Preserve the next public inspection ID
        # unless counts are explicitly reset.
        self._tracks = {}
        self._accepted_track_ids = set()
        if reset_counts:
            self._next_inspection_id = 1
        self.last_logged_status = None
        self.last_frame = None
        self.last_preview_rgb = None
        self.last_result = None
        self._last_camera_seq_seen = 0
        self._last_submitted_inference_seq = 0
        self._last_processed_inference_seq = 0
        self._last_result_seq = 0
        self._last_displayed_seq = 0
        self.preview_skipped_frames = 0
        self.inference_skipped_frames = 0
        self.dropped_inference_frames = 0
        self.camera_read_ms = 0.0
        self.camera_interval_ms = 0.0
        self.camera_frame_age_ms = 0.0
        self.inference_cycle_ms = 0.0
        self.inference_idle_ms = 0.0
        self.inference_backend_ms = 0.0
        self.inference_parse_ms = 0.0
        self.paint_ms = 0.0
        self.qimage_ms = 0.0
        self.scale_ms = 0.0
        self.overlay_draw_ms = 0.0
        self.preview_fps = 0.0
        self._preview_count = 0
        self._preview_total = 0
        self._preview_window_t = time.perf_counter()
        self._last_profiler_log_t = time.perf_counter()
        self._prof_last_cam_seq = 0
        self._prof_last_preview_total = 0
        self._prof_last_paint_total = 0
        self._prof_last_inf_done_count = 0
        self._last_inference_result_t = 0.0
        if hasattr(self, "inference_worker") and self.inference_worker is not None:
            self.inference_worker.clear()

        if reset_counts:
            self.total_count = 0
            self.pass_count = 0
            self.fail_count = 0
            if hasattr(self, "table"):
                self.table.setRowCount(0)

        if hasattr(self, "camera_widget"):
            self.camera_widget.set_frame(None, None)

        if hasattr(self, "decision_label"):
            self.decision_label.setText("READY")
        if hasattr(self, "reason_label"):
            self.reason_label.setText("Runtime session reset.")
        if hasattr(self, "bung_big"):
            self.bung_big.setText(f"0/{self._spin_value('expected_spin', 6) if hasattr(self, '_spin_value') else 6}")

        self.update_metrics()
        self.update_status_pills()

    def capture_camera_status(self):
        try:
            if self.cap is None:
                return
            self.actual_camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.actual_camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.actual_camera_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
            chars = [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
            self.actual_camera_fourcc = "".join(c for c in chars if c.isprintable()).strip()
            requested_w = int(getattr(self, "camera_width", 0))
            requested_h = int(getattr(self, "camera_height", 0))
            backend_raw = str(getattr(self, "camera_backend", "opencv")).lower()
            if backend_raw == "basler" and bool(getattr(self, "basler_roi_enabled", False)):
                requested_w = int(getattr(self, "basler_roi_width", requested_w))
                requested_h = int(getattr(self, "basler_roi_height", requested_h))
            roi_desc = ""
            try:
                roi_desc = self.cap.roi_description() if hasattr(self.cap, "roi_description") else ""
            except Exception:
                roi_desc = ""
            if hasattr(self, "camera_actual_pill"):
                txt = f"Actual {self.actual_camera_width}x{self.actual_camera_height}"
                if self.actual_camera_fps:
                    txt += f" @ {self.actual_camera_fps:.0f}"
                if roi_desc:
                    txt += f" {roi_desc}"
                self.camera_actual_pill.setText(txt)
                expected_ok = (self.actual_camera_width == requested_w and self.actual_camera_height == requested_h)
                self.camera_actual_pill.set_tone("pass" if expected_ok else "warn")
            backend = backend_raw.upper()
            req_desc = f"{getattr(self, 'camera_width', 0)}x{getattr(self, 'camera_height', 0)} @ {getattr(self, 'camera_fps', 0):.0f}"
            if backend_raw == "basler" and bool(getattr(self, "basler_roi_enabled", False)):
                req_desc += (
                    f" ROI X={getattr(self, 'basler_roi_offset_x', 0)}"
                    f" Y={getattr(self, 'basler_roi_offset_y', 0)}"
                    f" W={getattr(self, 'basler_roi_width', 0)}"
                    f" H={getattr(self, 'basler_roi_height', 0)}"
                )
            roi_log = f", {roi_desc}" if roi_desc else ""
            self.log(f"Camera [{backend}] requested {req_desc}. Actual {self.actual_camera_width}x{self.actual_camera_height}, FPS {self.actual_camera_fps:.1f}, FOURCC {self.actual_camera_fourcc or 'native/unknown'}{roi_log}.")
        except Exception as e:
            self.log(f"Camera status read failed: {e}")

    def open_camera(self):
        self.close_camera(reset_session=False)
        self.reset_runtime_session(reset_counts=False)
        backend = str(getattr(self, "camera_backend", "opencv")).lower()
        self.cap = create_camera_backend(
            backend=backend,
            source_text=self.source_edit.text() if hasattr(self, "source_edit") else "0",
            basler_serial=str(getattr(self, "basler_serial", "")),
            width=int(getattr(self, "camera_width", 2592)),
            height=int(getattr(self, "camera_height", 1944)),
            fps=float(getattr(self, "camera_fps", 30.0)),
            exposure_us=float(getattr(self, "basler_exposure_us", 5000.0)),
            gain=float(getattr(self, "basler_gain", 0.0)),
            exposure_auto=bool(getattr(self, "basler_exposure_auto", False)),
            opencv_api=str(getattr(self, "opencv_api", "auto")),
            basler_roi_enabled=bool(getattr(self, "basler_roi_enabled", False)),
            basler_roi_offset_x=int(getattr(self, "basler_roi_offset_x", 0)),
            basler_roi_offset_y=int(getattr(self, "basler_roi_offset_y", 0)),
            basler_roi_width=int(getattr(self, "basler_roi_width", getattr(self, "camera_width", 2592))),
            basler_roi_height=int(getattr(self, "basler_roi_height", getattr(self, "camera_height", 1944))),
        )
        open_result = self.cap.open()
        if not open_result.ok or not self.cap.is_opened():
            self.cap.release() if self.cap is not None else None
            self.cap = None
            self.camera_pill.setText("CAMERA ERROR")
            self.camera_pill.set_tone("fail")
            self._last_camera_error = open_result.message or "Camera open failed."
            QMessageBox.warning(self, "Camera", open_result.message or "Could not open camera.")
            self.log(open_result.message or "Camera open failed.")
            return
        self.capture_camera_status()
        self._last_camera_error = ""
        self._last_prediction_error = ""
        self._last_frame_ok_t = 0.0
        self._last_prediction_ok_t = 0.0
        self.running = True
        self._last_camera_seq_seen = 0
        self._last_submitted_inference_seq = 0
        self._last_processed_inference_seq = 0
        self.camera_worker = CameraCaptureWorker(self.cap, log_cb=self.log_from_worker)
        self.camera_worker.start()
        if hasattr(self, "inference_worker") and self.inference_worker is not None:
            self.inference_worker.clear()
            self.inference_worker.set_frame_source(self.camera_worker.get_latest)
            self.inference_worker.start()
        self.camera_pill.setText("BASLER ON" if backend == "basler" else "CAMERA ON")
        self.camera_pill.set_tone("pass")
        try:
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
            fourcc_val = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
            fourcc = "".join(chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)).strip()
            self.log(f"{open_result.message}  actual={actual_w}x{actual_h} {actual_fps:.1f} FPS FOURCC={fourcc or 'native'}")
        except Exception:
            self.log(open_result.message or f"Camera opened using {backend}")

    def eventFilter(self, obj, event):
        """Low-latency operator Stop event filter.

        QPushButton.clicked() is emitted on mouse release. If the Jetson/X11 input
        path is sluggish while TensorRT is active, the release/click can be delayed
        even though the video preview continues painting. Catch the Stop button's
        mouse-press event directly and latch Stop before waiting for a release.
        """
        try:
            if obj is getattr(self, "close_btn", None) and event is not None:
                etype = event.type()
                if etype == QEvent.Type.MouseButtonPress:
                    try:
                        if hasattr(event, "button") and event.button() != Qt.MouseButton.LeftButton:
                            return super().eventFilter(obj, event)
                    except Exception:
                        pass
                    try:
                        self.log(f"OPERATOR_STOP_MOUSE_PRESS_EVENT {self._operator_stop_snapshot()}")
                    except Exception:
                        pass
                    self.request_operator_stop("mouse_press")
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _operator_stop_snapshot(self) -> str:
        """Compact runtime state for delayed-input/stop diagnostics."""
        try:
            inf_metrics = self.inference_worker.metrics() if getattr(self, "inference_worker", None) is not None else {}
        except Exception:
            inf_metrics = {}
        try:
            cam_seq = int(getattr(self.camera_worker, "_seq", 0)) if getattr(self, "camera_worker", None) is not None else 0
        except Exception:
            cam_seq = 0
        return (
            f"running={bool(getattr(self, 'running', False))} "
            f"model_loaded={getattr(self.model_runner, 'model', None) is not None} "
            f"cam_seq={cam_seq} "
            f"inf_busy={bool(inf_metrics.get('busy', 0.0))} "
            f"busy_age_ms={float(inf_metrics.get('busy_age_ms', 0.0) or 0.0):.0f}"
        )

    def request_operator_stop(self, source: str = "button"):
        """Acknowledge Stop immediately, then perform cleanup without accepting repeated queued clicks.

        Previous builds let a delayed mouse/button event call close_camera() directly. If the
        Desktop or Qt input queue was briefly starved, several Stop clicks could arrive
        together and each one would run the full shutdown path. This method first flips the
        runtime state to stopped, clears queued inference work, updates operator/PLC state, and
        schedules one cleanup pass.
        """
        now = time.perf_counter()
        last = float(getattr(self, "_last_stop_request_t", 0.0) or 0.0)
        if now - last < 0.75:
            try:
                self.log(f"OPERATOR_STOP_DEBOUNCE ignored duplicate source={source} dt_ms={(now-last)*1000:.0f} {self._operator_stop_snapshot()}")
            except Exception:
                pass
            return
        self._last_stop_request_t = now
        self._operator_stop_count = int(getattr(self, "_operator_stop_count", 0) or 0) + 1
        self._stop_request_source = str(source or "button")
        try:
            self.log(f"OPERATOR_STOP_REQUEST source={source} count={self._operator_stop_count} {self._operator_stop_snapshot()}")
        except Exception:
            pass

        # Stop new camera/inference decisions immediately. Do not wait here for a
        # TensorRT call or Basler read to finish before giving the operator visual feedback.
        self.running = False
        self._last_prediction_error = "Operator stop requested."
        try:
            if getattr(self, "inference_worker", None) is not None:
                self.inference_worker.set_enabled(False)
        except Exception:
            pass
        try:
            if hasattr(self, "decision_label"):
                self.decision_label.setText("STOPPING")
            if hasattr(self, "reason_label"):
                self.reason_label.setText("Operator stop requested. Runtime cleanup pending.")
            if hasattr(self, "camera_pill"):
                self.camera_pill.setText("STOP REQUESTED")
                self.camera_pill.set_tone("warn")
            if hasattr(self, "statusBar") and self.statusBar() is not None:
                self.statusBar().showMessage("Stop requested — waiting for camera/inference cleanup", 3000)
        except Exception:
            pass
        try:
            self.update_plc_outputs(self.last_result)
        except Exception:
            pass

        if not bool(getattr(self, "_stop_cleanup_pending", False)):
            self._stop_cleanup_pending = True
            QTimer.singleShot(0, self._finish_operator_stop)

    def start_external_stop_watchdog(self):
        """Start a non-Qt stop-file watcher for cases where X11/Qt input stalls.

        To request Stop without using the GUI, create:
            /home/enersys/bungvision_env/runtime_stop.flag
        The watcher latches running=False immediately from a daemon thread, then
        the next Qt timer tick completes the normal operator stop cleanup.
        """
        try:
            if getattr(self, "_external_stop_thread", None) is not None and self._external_stop_thread.is_alive():
                return
            self._external_stop_thread_stop.clear()
            self._external_stop_thread = threading.Thread(target=self._external_stop_watchdog_loop, name="BungVisionExternalStopWatchdog", daemon=True)
            self._external_stop_thread.start()
            _write_debug_log(f"OUT_OF_BAND_STOP_WATCHDOG started file={OUT_OF_BAND_STOP_FILE}")
        except Exception as e:
            _write_debug_log(f"OUT_OF_BAND_STOP_WATCHDOG failed to start: {e}")

    def _external_stop_watchdog_loop(self):
        last_seen = 0.0
        while not getattr(self, "_external_stop_thread_stop", threading.Event()).wait(0.05):
            try:
                if not OUT_OF_BAND_STOP_FILE.exists():
                    continue
                now = time.perf_counter()
                # Coalesce accidental repeated touch/write operations.
                if now - last_seen < 0.25:
                    try:
                        OUT_OF_BAND_STOP_FILE.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue
                last_seen = now
                try:
                    OUT_OF_BAND_STOP_FILE.unlink(missing_ok=True)
                except Exception:
                    pass

                self._external_stop_count = int(getattr(self, "_external_stop_count", 0) or 0) + 1
                # This is the important part: do not wait for Qt input delivery.
                # Stop new inference submission immediately.
                self.running = False
                self._stop_request_source = "external_stop_file"
                self._external_stop_requested = True
                try:
                    if getattr(self, "inference_worker", None) is not None:
                        self.inference_worker.set_enabled(False)
                except Exception:
                    pass
                try:
                    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    OUT_OF_BAND_STOP_ACK_FILE.write_text(f"{ts} external stop accepted count={self._external_stop_count}\n", encoding="utf-8")
                except Exception:
                    pass
                _write_debug_log(
                    f"OUT_OF_BAND_STOP_FILE accepted count={self._external_stop_count} "
                    f"running_latched_false=True cleanup_pending={bool(getattr(self, '_stop_cleanup_pending', False))}"
                )
            except Exception as e:
                _write_debug_log(f"OUT_OF_BAND_STOP_WATCHDOG error: {e}")

    def _service_external_stop_request(self):
        if not bool(getattr(self, "_external_stop_requested", False)):
            return
        self._external_stop_requested = False
        # If the Qt loop is still alive enough to paint video, it should reach
        # this path quickly and show STOPPING. If it does not, the debug log
        # still proves the background watchdog already latched running=False.
        try:
            self.request_operator_stop("external_stop_file")
        except Exception as e:
            _write_debug_log(f"OUT_OF_BAND_STOP_SERVICE error: {e}")

    def _finish_operator_stop(self):
        """Run the actual cleanup exactly once after the Stop request has been acknowledged."""
        try:
            self.log(f"OPERATOR_STOP_CLEANUP_BEGIN source={getattr(self, '_stop_request_source', '')} {self._operator_stop_snapshot()}")
        except Exception:
            pass
        try:
            self.close_camera(reset_session=True)
        finally:
            self._stop_cleanup_pending = False
            try:
                self.log(f"OPERATOR_STOP_CLEANUP_END source={getattr(self, '_stop_request_source', '')} {self._operator_stop_snapshot()}")
            except Exception:
                pass

    def close_camera(self, reset_session: bool = True):
        self.running = False
        t_close0 = time.perf_counter()
        if getattr(self, "camera_worker", None) is not None:
            try:
                self.camera_worker.stop(timeout=0.35)
            except Exception:
                pass
            self.camera_worker = None
        if getattr(self, "inference_worker", None) is not None:
            try:
                # Do not join the inference thread on an operator Stop. If TensorRT is
                # mid-call, waiting here is exactly what makes the Stop button feel
                # frozen. Disable autonomous pulling and detach the frame source so
                # the worker stops fetching from the stopped camera; it stays alive
                # for the next Run and any in-flight call finishes naturally.
                self.inference_worker.set_enabled(False)
                self.inference_worker.set_frame_source(None)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.camera_pill.setText("CAMERA OFF")
        self.camera_pill.set_tone("neutral")
        self._last_frame_ok_t = 0.0
        self._last_prediction_ok_t = 0.0
        self._last_camera_error = ""
        self._last_prediction_error = ""
        if hasattr(self, "camera_actual_pill"):
            self.camera_actual_pill.setText("Actual Camera --")
            self.camera_actual_pill.set_tone("neutral")
        if reset_session:
            self.reset_runtime_session(reset_counts=False)
            try:
                close_ms = (time.perf_counter() - t_close0) * 1000.0
            except Exception:
                close_ms = 0.0
            self.log(f"Runtime stopped. Active tracks and PASS locks cleared. close_ms={close_ms:.0f}")
        self.update_plc_outputs(self.last_result)

    def diagnostic_preview_only(self) -> bool:
        return False

    def diagnostic_inference_paused(self) -> bool:
        return False

    def on_diagnostic_mode_changed(self):
        return None

    def toggle_demo(self):
        self.demo_mode = self.demo_check.isChecked()
        self.mode_pill.setText("DEV DEMO" if self.demo_mode else "LIVE RUNTIME")
        self.mode_pill.set_tone("warn" if self.demo_mode else "pass")

    def reset_reject_latch(self):
        self.reject_latched = False
        self.reject_latch_id = 0
        self.reject_latch_reason = ""
        self.reject_latch_time = ""
        self._plc_reset_pulse = True
        self.update_plc_outputs(self.last_result)
        self.update_status_pills(self.last_result)
        if hasattr(self, "decision_label") and (not self.last_result or self.last_result.status != "FAIL"):
            self.decision_label.setText("READY")
            self.reason_label.setText("Reject latch cleared.")
        self.log("Reject latch cleared; PLC reset pulse sent.")

    def latch_reject(self, grade: BatteryGrade):
        if self.reject_latched:
            return
        self.reject_latched = True
        self.reject_latch_id = int(grade.track_id)
        self.reject_latch_reason = str(grade.reason)
        self.reject_latch_time = dt.datetime.now().isoformat(timespec="seconds")
        self.log(f"REJECT LATCHED: ID {self.reject_latch_id} — {self.reject_latch_reason}")

    def latch_reject_for_class(self, label: str, conf: float):
        """Latch a reject because a custom reject class was detected (no battery grade)."""
        if self.reject_latched:
            return
        self.reject_latched = True
        self.reject_latch_id = 0
        self.reject_latch_reason = f"Custom class: {label} ({conf:.0%})"
        self.reject_latch_time = dt.datetime.now().isoformat(timespec="seconds")
        self.log(f"REJECT LATCHED (custom class): {label} conf={conf:.2f}")

    def reset_counts(self):
        self.total_count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.last_logged_status = None

        # Start a fresh session window for the production dashboard. Historical
        # per-day totals in production_stats are intentionally preserved.
        self.session_start_t = dt.datetime.now()

        # Latched machine-control reject state. This is separate from counters.
        self.reject_latched = False
        self.reject_latch_id = 0
        self.reject_latch_reason = ""
        self.reject_latch_time = ""

        # Reset all runtime tracking and both public/internal ID counters.
        # Do not overwrite tuned inspection settings such as Track IoU / Locked IoU.
        self._tracks = {}
        self._accepted_track_ids = set()
        self._next_inspection_id = 1
        self._next_track_id = 1

        if hasattr(self, "table"):
            self.table.setRowCount(0)
        if hasattr(self, "camera_widget"):
            self.camera_widget.set_frame(None, None)
        if hasattr(self, "decision_label"):
            self.decision_label.setText("READY")
        if hasattr(self, "reason_label"):
            self.reason_label.setText("Counts and inspection IDs reset.")
        if hasattr(self, "bung_big"):
            self.bung_big.setText(f"0/{self._spin_value('expected_spin', 6) if hasattr(self, '_spin_value') else 6}")

        self.update_metrics()
        self.update_status_pills()
        self.pulse_plc_reset()
        self.log("Counts, active tracks, PASS/FAIL locks, and inspection IDs reset to 1.")

    def log_from_worker(self, msg: str):
        try:
            self._background_log_queue.put_nowait(str(msg))
        except Exception:
            pass

    def drain_background_logs(self, limit: int = 25):
        if not hasattr(self, "_background_log_queue"):
            return
        for _ in range(max(1, int(limit))):
            try:
                msg = self._background_log_queue.get_nowait()
            except queue.Empty:
                break
            self.log(msg)

    def log(self, msg: str):
        text = str(msg)
        _write_debug_log(text)
        if hasattr(self, "_ui_thread_ident") and threading.get_ident() != self._ui_thread_ident:
            self.log_from_worker(text)
            return
        t = dt.datetime.now().strftime("%H:%M:%S")
        try:
            self.log_box.append(f"[{t}] {text}")
        except Exception:
            pass

    def demo_frame(self) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for y in range(frame.shape[0]):
            shade = int(20 + y / frame.shape[0] * 30)
            frame[y, :, :] = (shade, shade + 8, shade + 18)
        cv2.rectangle(frame, (160, 255), (1120, 475), (95, 105, 118), -1)
        cv2.rectangle(frame, (160, 255), (1120, 475), (240, 210, 160), 4)
        cv2.putText(frame, "BATTERY", (190, 305), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (210, 230, 240), 2)
        return frame

    def demo_detections(self, frame: np.ndarray) -> List[Detection]:
        fail = self.fail_demo_check.isChecked()
        detections = [Detection("battery", 0.98, (160, 255, 1120, 475))]
        positions = [(255, 345), (405, 330), (555, 326), (705, 326), (855, 335), (1010, 350)]
        for i, (cx, cy) in enumerate(positions):
            if fail and i == 4:
                continue
            detections.append(Detection("bung", 0.92 - i * 0.01, (cx - 36, cy - 30, cx + 36, cy + 30)))
        return detections


    def reset_tracker(self):
        self._tracks = {}
        self._next_track_id = 1
        self._next_inspection_id = 1
        self._accepted_track_ids = set()

    def ensure_tracker_state(self):
        """Guarantee multi-battery tracker attributes exist.

        This protects against packaging/version drift and prevents timer spam if
        a prior settings/UI path skipped initialization.
        """
        if not hasattr(self, "_tracks") or self._tracks is None:
            self._tracks = {}
        if not hasattr(self, "_next_track_id"):
            self._next_track_id = 1
        if not hasattr(self, "_next_inspection_id"):
            self._next_inspection_id = 1
        if not hasattr(self, "_accepted_track_ids"):
            self._accepted_track_ids = set()
        if not hasattr(self, "stable_required"):
            self.stable_required = 6
        if not hasattr(self, "clear_required"):
            self.clear_required = 10
        if not hasattr(self, "match_distance_px"):
            self.match_distance_px = 180
        if not hasattr(self, "entry_grace_frames"):
            self.entry_grace_frames = 12
        if not hasattr(self, "require_full_view_before_grade"):
            self.require_full_view_before_grade = True
        if not hasattr(self, "full_view_margin_percent"):
            self.full_view_margin_percent = 3.0
        if not hasattr(self, "enable_pattern_validation"):
            self.enable_pattern_validation = True
        if not hasattr(self, "pattern_tolerance_percent"):
            self.pattern_tolerance_percent = 25.0

    def _spin_value(self, name: str, default: int) -> int:
        widget = getattr(self, name, None)
        if widget is None:
            return default
        try:
            return int(widget.value())
        except Exception:
            return default

    def assign_track_ids(self, grades: List[BatteryGrade]) -> List[BatteryGrade]:
        """Assign persistent IDs to multiple visible batteries using distance + IoU matching."""
        self.ensure_tracker_state()
        self.stable_required = self._spin_value("debounce_spin", 6)
        self.clear_required = self._spin_value("clear_spin", 10)
        self.match_distance_px = self._spin_value("match_distance_spin", 180)
        self.entry_grace_frames = self._spin_value("entry_grace_spin", 12)
        track_match_iou = float(getattr(self, "track_match_iou", 0.05))
        committed_iou = float(getattr(self, "committed_track_iou", 0.25))

        for tid in list(self._tracks.keys()):
            self._tracks[tid]["missed"] = self._tracks[tid].get("missed", 0) + 1

        used_tracks = set()

        for grade in grades:
            gx, gy = box_center(grade.box)
            best_tid = None
            best_score = 1e18

            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                tx, ty = box_center(tr["box"])
                dist = ((gx - tx) ** 2 + (gy - ty) ** 2) ** 0.5
                iou = box_iou(grade.box, tr["box"])
                committed = bool(tr.get("committed", False))

                if committed:
                    if iou < committed_iou:
                        continue
                else:
                    if dist > self.match_distance_px and iou < track_match_iou:
                        continue

                score = dist - (iou * 1000.0)
                if score < best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is None:
                best_tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_tid] = {
                    "box": grade.box,
                    "status_key": None,
                    "stable": 0,
                    "logged": False,
                    "final_status": None,
                    "inspection_id": None,
                    "committed": False,
                    "committed_status": None,
                    "missed": 0,
                    "age": 0,
                }

            tr = self._tracks[best_tid]
            tr["box"] = grade.box
            tr["missed"] = 0
            tr["age"] = tr.get("age", 0) + 1

            if tr.get("committed", False):
                if tr.get("inspection_id") is None:
                    tr["inspection_id"] = self._next_inspection_id
                    self._next_inspection_id += 1
                grade.track_id = tr.get("inspection_id") or -best_tid
                grade.status = tr.get("committed_status") or tr.get("final_status") or grade.status
                grade.reason = f"{grade.status} locked for ID {grade.track_id}"
                if grade.status == "PASS":
                    grade.bung_count = grade.expected_bungs
                grade.stable_count = self.stable_required
                grade.logged = True
                used_tracks.add(best_tid)
                continue

            if best_tid in self._accepted_track_ids or tr.get("final_status") == "PASS":
                self._accepted_track_ids.add(best_tid)
                tr["final_status"] = "PASS"
                tr["logged"] = True
                if tr.get("inspection_id") is None:
                    tr["inspection_id"] = self._next_inspection_id
                    self._next_inspection_id += 1
                grade.status = "PASS"
                grade.reason = f"PASS locked for ID {tr['inspection_id']}"
                grade.bung_count = grade.expected_bungs
                grade.track_id = tr["inspection_id"]
                grade.stable_count = max(tr.get("stable", 0), self.stable_required)
                grade.logged = True
                used_tracks.add(best_tid)
                continue

            key = (grade.status, grade.bung_count, grade.reason)
            if tr.get("status_key") == key:
                tr["stable"] = tr.get("stable", 0) + 1
            else:
                tr["status_key"] = key
                tr["stable"] = 1
                if tr.get("final_status") is None:
                    tr["logged"] = False

            if grade.status == "FAIL" and tr.get("age", 0) <= self.entry_grace_frames:
                remaining = max(0, self.entry_grace_frames - tr.get("age", 0) + 1)
                grade.status = "WAIT"
                grade.reason = f"Waiting for bungs: {remaining} frames"
                tr["status_key"] = ("WAIT", grade.bung_count, grade.reason)
                tr["stable"] = 1
                tr["logged"] = False

            if grade.status == "PASS" and tr["stable"] >= self.stable_required:
                self._accepted_track_ids.add(best_tid)
                tr["final_status"] = "PASS"
                if tr.get("inspection_id") is None:
                    tr["inspection_id"] = self._next_inspection_id
                    self._next_inspection_id += 1
                tr["logged"] = True
                grade.status = "PASS"
                grade.reason = f"PASS locked for ID {tr['inspection_id']}"
                grade.bung_count = grade.expected_bungs
                grade.logged = True

            grade.track_id = tr.get("inspection_id") or -best_tid
            grade.stable_count = tr["stable"]
            grade.logged = tr.get("logged", False)
            used_tracks.add(best_tid)

        for tid in list(self._tracks.keys()):
            if self._tracks[tid].get("missed", 0) >= self.clear_required:
                del self._tracks[tid]
                self._accepted_track_ids.discard(tid)

        return grades

    def _should_capture_training_example(self, grade: BatteryGrade) -> bool:
        if grade.status == "FAIL":
            enabled = bool(getattr(self, "save_fail_training_images", True))
            if not enabled:
                self.log("Training-review capture skipped: FAIL captures are disabled in Settings -> Capture.")
            return enabled

        if grade.status == "PASS":
            if not bool(getattr(self, "save_pass_training_images", True)):
                self.log("Training-review capture skipped: PASS captures are disabled in Settings -> Capture.")
                return False
            rate = max(1, int(getattr(self, "pass_training_sample_rate", 1)))
            should_save = (int(self.pass_count) % rate) == 0
            if not should_save:
                self.log(f"Training-review capture skipped: PASS sample rate is 1/{rate}.")
            return should_save

        self.log(f"Training-review capture skipped: grade status {grade.status} is not PASS/FAIL.")
        return False

    def _runtime_class_names(self) -> list[str]:
        names = []
        if getattr(self.model_runner, "names", None):
            for _idx, name in sorted(self.model_runner.names.items(), key=lambda kv: str(kv[0])):
                names.append(str(name))
        return names

    def _editor_label_for_detection(self, det: Detection) -> tuple[str, int]:
        kind = detection_kind(det.label)
        if kind == "battery":
            if det.label in ("battery_6_row", "battery_row6", "battery_6inrow", "battery_6_in_row", "battery_2x3", "battery_2_x_3", "battery_grid2x3"):
                return det.label, self._source_class_id_for_detection(det, self._runtime_class_names())
            return "battery", 0
        if kind == "bung":
            return "bung", 1
        if kind == "retainer":
            return "retainer", 2
        return kind or det.label, -1

    def _source_class_id_for_detection(self, det: Detection, class_names: list[str]) -> int:
        try:
            return class_names.index(det.label)
        except ValueError:
            return -1

    def _normalized_points(self, points: list[list[float]] | list[tuple[float, float]], image_w: int = 0, image_h: int = 0) -> list[list[float]]:
        """Return YOLO/label-tool friendly normalized point coordinates."""
        iw = max(1, int(image_w or 1))
        ih = max(1, int(image_h or 1))
        out: list[list[float]] = []
        for x, y in points or []:
            out.append([
                round(max(0.0, min(1.0, float(x) / iw)), 6),
                round(max(0.0, min(1.0, float(y) / ih)), 6),
            ])
        return out

    def _label_tool_common_fields(self, det: Detection, class_names: list[str], index: int = 0) -> dict:
        editor_label, editor_class_id = self._editor_label_for_detection(det)
        source_class_id = self._source_class_id_for_detection(det, class_names)
        # The label tool import path expects a non-empty class identifier on the
        # annotation itself. Keep several aliases because older Bung Labeler
        # builds used different names while the OBB/seg/detect tools evolved.
        return {
            "identifier": editor_label,
            "class_identifier": editor_label,
            "label_identifier": editor_label,
            "class_name": editor_label,
            "name": editor_label,
            "display_name": editor_label,
            "label": editor_label,
            "class": editor_label,
            "category": editor_label,
            "class_id": int(editor_class_id),
            "source_label": det.label,
            "source_class_id": int(source_class_id),
            "confidence": float(det.conf),
            "reviewed": False,
            "needs_review": True,
            "from_bungvision": True,
            "annotation_source": "bungvision_runtime_capture",
        }

    def _label_tool_box(self, det: Detection, class_names: list[str], image_w: int = 0, image_h: int = 0, index: int = 0) -> dict:
        """Axis-aligned rectangle annotation for standard detect fallback."""
        x1, y1, x2, y2 = [int(v) for v in det.box]
        pts = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
        item = self._label_tool_common_fields(det, class_names, index)
        item.update({
            "kind": "box",
            "type": "box",
            "shape_type": "rectangle",
            "geometry_type": "rectangle",
            "x": int(x1),
            "y": int(y1),
            "w": int(max(0, x2 - x1)),
            "h": int(max(0, y2 - y1)),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "points": pts,
            "corners": pts,
            "points_norm": self._normalized_points(pts, image_w, image_h),
            "normalized_points": self._normalized_points(pts, image_w, image_h),
        })
        return item

    def _label_tool_obb_shape(self, det: Detection, class_names: list[str], image_w: int = 0, image_h: int = 0, index: int = 0) -> dict:
        """OBB annotation in the format expected by the BungVision label tool.

        Important compatibility note: current Bung Labeler sidecar import still
        scans the top-level boxes[] list first. Therefore OBB annotations must be
        valid items in boxes[] too, not only in shapes[] or obb_boxes[]. The
        fields below preserve the four rotated corners and explicitly mark the
        item as kind/type=obb so the label tool can import it as an OBB instead
        of flattening it to a standard detection rectangle.
        """
        x1, y1, x2, y2 = [int(v) for v in det.box]
        ordered_pts = normalize_obb_points_clockwise(det.obb_points or [])
        pts = [[round(float(x), 2), round(float(y), 2)] for x, y in ordered_pts[:4]]
        if len(pts) < 4:
            return self._label_tool_box(det, class_names, image_w, image_h, index)
        geom = obb_geometry([(float(x), float(y)) for x, y in pts])
        pts_norm = self._normalized_points(pts, image_w, image_h)
        item = self._label_tool_common_fields(det, class_names, index)
        item.update({
            "kind": "obb",
            "type": "obb",
            "shape_type": "rotated_box",
            "geometry_type": "obb",
            "tool_shape": "rotated_box",
            "closed": True,
            "rotated": True,
            "is_obb": True,
            "points": pts,
            "corners": pts,
            "obb_points": pts,
            "points_norm": pts_norm,
            "normalized_points": pts_norm,
            "obb_points_norm": pts_norm,
            "aabb_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "x": int(x1),
            "y": int(y1),
            "w": int(max(0, x2 - x1)),
            "h": int(max(0, y2 - y1)),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "geometry": geom,
            "obb": {
                "points": pts,
                "points_norm": pts_norm,
                **geom,
            },
        })
        item.update(geom)
        return item

    def _label_tool_annotation(self, det: Detection, class_names: list[str], image_w: int = 0, image_h: int = 0, index: int = 0) -> dict:
        if getattr(det, "obb_points", None):
            return self._label_tool_obb_shape(det, class_names, image_w, image_h, index)
        return self._label_tool_box(det, class_names, image_w, image_h, index)

    def _stable_annotation_identifier(self, ann: dict, index: int) -> str:
        """Return a non-empty class identifier for the label tool import path."""
        label = str(
            ann.get("identifier")
            or ann.get("label")
            or ann.get("class_name")
            or ann.get("source_label")
            or "object"
        ).strip()
        return label or "object"

    def _add_label_tool_identifiers(self, annotations: list[dict], image_stem: str = "") -> list[dict]:
        """Populate the exact identifier aliases used by different label-tool revisions."""
        prefix = image_stem or "runtime_capture"
        for idx, ann in enumerate(annotations, start=1):
            identifier = self._stable_annotation_identifier(ann, idx)
            safe_identifier = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in identifier)
            obj_id = f"{prefix}_{idx:03d}_{safe_identifier}"
            for key in ("identifier", "class_identifier", "label_identifier", "class_name", "name", "display_name", "label", "class", "category"):
                ann[key] = identifier
            ann["object_identifier"] = obj_id
            ann["object_id"] = obj_id
            ann["uid"] = obj_id
            ann["id"] = obj_id
        return annotations

    def _write_yolo_candidate_labels(self, txt_path: Path, detections: List[Detection], image_w: int, image_h: int):
        class_names = self._runtime_class_names()
        class_index = {name: i for i, name in enumerate(class_names)}
        lines = []
        has_obb = any(getattr(det, "obb_points", None) for det in detections)

        for det in detections:
            if det.label not in class_index:
                continue
            cls = class_index[det.label]

            # If an OBB model produced corners, write YOLO OBB candidate labels:
            # class x1 y1 x2 y2 x3 y3 x4 y4, normalized. When a non-OBB object
            # is mixed in, approximate it with the axis-aligned rectangle corners
            # so the file remains consistently OBB-shaped.
            if has_obb:
                if getattr(det, "obb_points", None):
                    pts = normalize_obb_points_clockwise(det.obb_points or [])
                else:
                    x1, y1, x2, y2 = det.box
                    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                vals = []
                for x, y in pts[:4]:
                    vals.append(max(0.0, min(1.0, float(x) / max(1, image_w))))
                    vals.append(max(0.0, min(1.0, float(y) / max(1, image_h))))
                if len(vals) == 8:
                    lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in vals))
                continue

            x1, y1, x2, y2 = det.box
            cx = ((x1 + x2) / 2.0) / max(1, image_w)
            cy = ((y1 + y2) / 2.0) / max(1, image_h)
            bw = (x2 - x1) / max(1, image_w)
            bh = (y2 - y1) / max(1, image_h)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        txt_path.write_text("\n".join(lines), encoding="utf-8")

    def append_training_manifest(self, grade: BatteryGrade, raw_path: Path, json_path: Path, annotated_path: Path):
        try:
            manifest = TRAINING_REVIEW_DIR / "manifest.csv"
            new_file = not manifest.exists()
            with manifest.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "status", "battery_id", "reason", "raw_image", "json_file", "annotated_image", "model_path"])
                w.writerow([
                    dt.datetime.now().isoformat(timespec="milliseconds"),
                    grade.status,
                    grade.track_id,
                    grade.reason,
                    str(raw_path),
                    str(json_path) if json_path else "",
                    str(annotated_path) if annotated_path else "",
                    getattr(self.model_runner, "path", ""),
                ])
        except Exception as e:
            self.log(f"Could not update training capture manifest: {e}")

    def save_training_review_capture(self, grade: BatteryGrade, result: InspectionResult, frame: np.ndarray):
        """Save runtime capture package for later human review/re-labeling."""
        if not self._should_capture_training_example(grade):
            return
        try:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            status_dir = TRAINING_REVIEW_DIR / grade.status.lower()
            status_dir.mkdir(parents=True, exist_ok=True)

            base = f"{stamp}_ID{grade.track_id}_{grade.status}"
            raw_path = status_dir / f"{base}.jpg"
            annotated_path = status_dir / f"{base}_annotated.jpg"
            json_path = status_dir / f"{base}.json"
            txt_path = status_dir / f"{base}.txt"

            ok_raw = cv2.imwrite(str(raw_path), frame)
            if not ok_raw:
                raise RuntimeError(f"cv2.imwrite failed for raw capture: {raw_path}")
            h, w = frame.shape[:2]
            class_names = self._runtime_class_names()

            if bool(getattr(self, "save_training_annotated", True)):
                annotated = frame.copy()
                self.draw_snapshot_overlay(annotated, result)
                ok_annotated = cv2.imwrite(str(annotated_path), annotated)
                if not ok_annotated:
                    self.log(f"WARNING: cv2.imwrite failed for annotated capture: {annotated_path}")

            if bool(getattr(self, "save_training_json", True)):
                annotations = [self._label_tool_annotation(det, class_names, w, h, idx) for idx, det in enumerate(result.detections, start=1)]
                annotations = self._add_label_tool_identifiers(annotations, raw_path.stem)
                obb_shapes = [ann for ann in annotations if ann.get("kind") == "obb" or ann.get("type") == "obb"]
                legacy_boxes = [ann for ann in annotations if ann.get("kind") == "box" or ann.get("type") == "box"]
                # Bung Labeler currently discovers sidecar annotations from boxes[] first.
                # Keep OBB annotations in boxes[] too, but with kind/type=obb and four
                # corner points so they import as rotated boxes rather than empty items.
                import_boxes = annotations
                payload = {
                    "image": raw_path.name,
                    "identifier": raw_path.stem,
                    "image_identifier": raw_path.stem,
                    "capture_identifier": raw_path.stem,
                    "width": int(w),
                    "height": int(h),
                    # Primary sidecar import list. Includes OBB items marked with
                    # kind/type=obb because Bung Labeler scans boxes[] first.
                    "boxes": import_boxes,
                    # Additional OBB aliases for newer import paths.
                    "shapes": obb_shapes,
                    "obb_boxes": obb_shapes,
                    "annotations": annotations,
                    "objects": annotations,
                    "inspection_grade": {
                        "battery_id": int(grade.track_id),
                        "status": grade.status,
                        "reason": grade.reason,
                        "expected_bungs": int(grade.expected_bungs),
                        "assigned_bung_count": int(grade.bung_count),
                        "battery_obb": obb_geometry(grade.obb_points),
                        "assigned_bung_indices": list(getattr(grade, "assigned_bung_indices", []) or []),
                        "pattern_validation": {
                            "enabled": bool(getattr(self, "enable_pattern_validation", True)),
                            "pattern": getattr(grade, "pattern_name", ""),
                            "ok": getattr(grade, "pattern_ok", None),
                            "reason": getattr(grade, "pattern_reason", ""),
                            "tolerance_percent": float(getattr(self, "pattern_tolerance_percent", 25.0)),
                        },
                    },
                    "annotation_format": "bungvision_labeler_v0_14_boxes_first_obb_clockwise",
                    "label_tool_import": {
                        "preferred_source": "boxes",
                        "preferred_obb_source": "boxes",
                        "secondary_obb_source": "shapes",
                        "fallback_box_source": "boxes",
                        "obb_shape_type": "rotated_box",
                        "note": "OBB detections are included in boxes[] with kind/type=obb, non-empty identifier fields, and point order normalized to point 1 then clockwise for Bung Labeler import.",
                    },
                    "metadata": {
                        "timestamp": dt.datetime.now().isoformat(timespec="milliseconds"),
                        "source": "BungVision runtime training-review capture",
                        "candidate_warning": "Review/correct these OBB/box candidates in the labeling tool before using them for training.",
                        "result": grade.status,
                        "battery_id": int(grade.track_id),
                        "expected_bungs": int(grade.expected_bungs),
                        "detected_bungs": int(grade.bung_count),
                        "reason": grade.reason,
                        "model_path": getattr(self.model_runner, "path", ""),
                        "model_task": getattr(self.model_runner, "task", "detect"),
                        "annotation_mode": "boxes_first_with_clockwise_obb_kind_and_points",
                        "class_names": class_names,
                        "obb_count": len(obb_shapes),
                        "box_count": len(legacy_boxes),
                        "import_box_count": len(import_boxes),
                        "annotated_image": annotated_path.name if bool(getattr(self, "save_training_annotated", True)) else "",
                    },
                }
                json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            if bool(getattr(self, "save_training_yolo_txt", False)):
                self._write_yolo_candidate_labels(txt_path, result.detections, w, h)

            self.append_training_manifest(
                grade,
                raw_path,
                json_path if bool(getattr(self, "save_training_json", True)) else Path(""),
                annotated_path if bool(getattr(self, "save_training_annotated", True)) else Path(""),
            )
            self.log(f"Saved training-review capture: {raw_path}")
        except Exception:
            self.log("Training-review capture failed:\n" + traceback.format_exc())

    def _fail_category(self, grade: BatteryGrade) -> str:
        """Bucket a committed reject into a coarse reason for the dashboard.

        Derived from the grade fields rather than the free-text reason so the
        breakdown stays stable across reason-string wording changes.
        """
        if grade.status != "FAIL":
            return ""
        try:
            if grade.pattern_ok is False:
                return "Pattern invalid"
            if int(grade.bung_count) < int(grade.expected_bungs):
                return "Missing bungs"
            if int(grade.bung_count) > int(grade.expected_bungs):
                return "Extra bungs"
        except Exception:
            pass
        return "Other"

    def open_production_dashboard(self):
        try:
            dlg = ProductionDashboardDialog(self)
            dlg.exec()
        except Exception as exc:
            self.log(f"Could not open production summary: {exc}")
            QMessageBox.warning(self, "Production Summary", f"Could not open production summary:\n{exc}")

    def open_reject_classes_dialog(self):
        dlg = CustomRejectClassesDialog(self, list(self.custom_reject_classes))
        if dlg.exec() == QDialog.Accepted:
            self.custom_reject_classes = dlg.get_classes()
            self.save_settings(silent=True)
            count = len(self.custom_reject_classes)
            summary = ", ".join(self.custom_reject_classes[:5])
            if count > 5:
                summary += f" … (+{count - 5} more)"
            self.log(f"Custom reject classes updated ({count}): {summary or 'none'}")

    def commit_battery_grade(self, internal_tid: int, grade: BatteryGrade, result: InspectionResult, frame: np.ndarray) -> bool:
        """Commit exactly one PASS/FAIL result for one tracked physical battery.

        This is the only function allowed to increment pass_count, fail_count,
        and total_count. Snapshot/log writing is intentionally separate and
        cannot affect the counter.
        """
        self.ensure_tracker_state()

        tr = self._tracks.get(internal_tid)
        if tr is None:
            return False

        if tr.get("committed", False):
            return False

        if grade.status not in ("PASS", "FAIL"):
            return False

        # If this physical track was previously accepted PASS, it can never
        # be committed as FAIL while it remains visible.
        if internal_tid in self._accepted_track_ids and grade.status != "PASS":
            return False

        if tr.get("inspection_id") is None:
            tr["inspection_id"] = self._next_inspection_id
            self._next_inspection_id += 1

        grade.track_id = tr["inspection_id"]

        if grade.status == "PASS":
            self.pass_count = int(self.pass_count) + 1
            self._accepted_track_ids.add(internal_tid)
        elif grade.status == "FAIL":
            self.fail_count = int(self.fail_count) + 1
            self.latch_reject(grade)

        self.total_count = int(self.total_count) + 1

        tr["committed"] = True
        tr["logged"] = True
        tr["final_status"] = grade.status
        tr["committed_status"] = grade.status
        grade.logged = True

        # Update the metric cards immediately, before any file I/O.
        self.update_metrics()

        self.add_history_row_for_grade(grade, result)

        # Production summary (operator dashboard). The in-memory aggregate is
        # updated immediately so the live counters and the dashboard agree; the
        # tiny JSON persist is queued to the save worker like the image/record
        # I/O below so it can never pause live inspection or PLC updates.
        try:
            self.production_stats.record(grade.status, self._fail_category(grade))
            if hasattr(self, "save_worker") and self.save_worker is not None:
                self.save_worker.enqueue(self.production_stats.save)
            else:
                self.production_stats.save()
        except Exception:
            pass

        # Disk I/O is intentionally queued after the counter commits so high-resolution
        # JPG/JSON writes cannot pause live inspection or PLC updates.
        save_frame = frame.copy()
        operator_snapshot_enabled = (grade.status == "PASS" and self._checkbox_checked("save_pass_images_check", True)) or (
            grade.status == "FAIL" and self._checkbox_checked("save_fail_images_check", True)
        )
        if hasattr(self, "save_worker") and self.save_worker is not None:
            queued = self.save_worker.enqueue(self._write_committed_outputs, grade, result, save_frame, operator_snapshot_enabled)
            if not queued:
                self.log(f"WARNING: Counter committed but save queue was full for {grade.status} ID {grade.track_id}")
        else:
            self._write_committed_outputs(grade, result, save_frame, operator_snapshot_enabled)
        self.update_metrics()

        self.log(
            f"COMMIT {grade.status} ID {grade.track_id}. "
            f"Pass={self.pass_count} Rejects={self.fail_count} Total={self.total_count}"
        )
        return True

    def _write_committed_outputs(self, grade: BatteryGrade, result: InspectionResult, frame: np.ndarray, operator_snapshot_enabled: bool = True):
        wrote = self.write_battery_record(grade, result, frame, save_image=operator_snapshot_enabled)
        if not wrote:
            self.log(f"WARNING: Counter committed but record write failed for {grade.status} ID {grade.track_id}")
        self.save_training_review_capture(grade, result, frame)

    def update_tracker(self, result: InspectionResult, frame: np.ndarray):
        """Update per-battery tracking and commit one result per physical battery."""
        self.ensure_tracker_state()

        if not result.battery_grades:
            self.assign_track_ids([])
            return

        for grade in result.battery_grades:
            # Public committed IDs are positive. Temporary candidates are negative internal IDs.
            if grade.track_id < 0:
                internal_tid = -grade.track_id
            else:
                internal_tid = None
                for tid, candidate in self._tracks.items():
                    if candidate.get("inspection_id") == grade.track_id:
                        internal_tid = tid
                        break

            if internal_tid is None:
                continue

            tr = self._tracks.get(internal_tid)
            if tr is None:
                continue

            # Already committed: never count the same physical track twice.
            if tr.get("committed", False):
                continue

            # PASS-accepted track: force/commit PASS, never FAIL.
            if internal_tid in self._accepted_track_ids or tr.get("final_status") == "PASS":
                grade.status = "PASS"
                grade.reason = f"PASS locked for ID {tr.get('inspection_id') or self._next_inspection_id}"
                grade.bung_count = grade.expected_bungs
                if grade.stable_count >= self.stable_required:
                    self.commit_battery_grade(internal_tid, grade, result, frame)
                continue

            # WAIT is still in entry grace and must not count as reject.
            if grade.status == "WAIT":
                continue

            # Explicit FAIL path. If the battery is no longer in entry grace and
            # FAIL has been stable for the configured debounce, commit a reject.
            if grade.status == "FAIL":
                if grade.stable_count >= self.stable_required:
                    self.commit_battery_grade(internal_tid, grade, result, frame)
                continue

            # Explicit PASS path.
            if grade.status == "PASS":
                if grade.stable_count >= self.stable_required:
                    self.commit_battery_grade(internal_tid, grade, result, frame)
                continue

    def inspect(self, detections: List[Detection], frame: np.ndarray) -> InspectionResult:
        expected = self.expected_spin.value()
        batteries = [d for d in detections if detection_kind(d.label) == "battery"]
        bungs = [d for d in detections if detection_kind(d.label) == "bung"]

        grades: List[BatteryGrade] = []

        # Sort left-to-right for stable operator display. Track IDs still come
        # from nearest-center matching. For OBB detections, use the rotated
        # polygon center instead of the axis-aligned bounding rectangle center.
        batteries = sorted(batteries, key=lambda d: detection_center(d)[0])

        # Assign each bung to exactly one battery. This is intentionally different
        # from the old per-battery "count every bung inside my box" method because
        # overlapping battery boxes can otherwise borrow the same bung.
        # OBB models use the rotated battery polygon for this ownership test;
        # standard detect models fall back to the axis-aligned box.
        battery_bung_indices: Dict[int, List[int]] = {i: [] for i in range(len(batteries))}
        for bung_idx, bung in enumerate(bungs):
            pt = detection_center(bung)
            candidates: List[Tuple[float, int]] = []
            for batt_idx, batt in enumerate(batteries):
                if not bung_matches_battery(bung.label, batt.label):
                    continue
                if detection_contains_point(batt, pt, margin=10):
                    bx, by = detection_center(batt)
                    dist = ((float(pt[0]) - bx) ** 2 + (float(pt[1]) - by) ** 2) ** 0.5
                    # Prefer true OBB containment over broad axis-aligned fallback
                    # when both are possible, then break ties by confidence/distance.
                    candidates.append((dist - (float(batt.conf) * 10.0), batt_idx))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                battery_bung_indices[candidates[0][1]].append(bung_idx)

        for batt_idx, batt in enumerate(batteries):
            assigned_indices = battery_bung_indices.get(batt_idx, [])
            inside: List[Box] = [bungs[i].box for i in assigned_indices]

            count = len(inside)
            batt_suffix = detection_suffix(batt.label)
            model_note = f" [{batt_suffix}]" if batt_suffix else ""
            ownership_note = " OBB" if getattr(batt, "obb_points", None) else ""

            h, w = frame.shape[:2]
            full_view_ok = True
            if bool(getattr(self, "require_full_view_before_grade", True)):
                full_view_ok = detection_fully_inside_frame(
                    batt,
                    frame_w=w,
                    frame_h=h,
                    margin_percent=float(getattr(self, "full_view_margin_percent", 3.0)),
                )

            pattern_info = {"ok": None, "pattern": "", "reason": "", "score": 0.0}
            if not full_view_ok:
                status = "WAIT"
                reason = f"Waiting for full battery view / infeed gate: {count}/{expected} bungs{model_note}{ownership_note}"
            elif count == expected:
                if bool(getattr(self, "enable_pattern_validation", True)) and int(expected) == 6:
                    local_points = [normalized_point_in_detection(batt, detection_center(bungs[i])) for i in assigned_indices]
                    pattern_info = validate_six_bung_pattern(
                        local_points,
                        tolerance_percent=float(getattr(self, "pattern_tolerance_percent", 25.0)),
                    )
                    if pattern_info.get("ok"):
                        status = "PASS"
                        reason = f"{count}/{expected} bungs, pattern {pattern_info.get('pattern')}{model_note}{ownership_note}"
                    else:
                        status = "FAIL"
                        reason = f"{count}/{expected} bungs, {pattern_info.get('reason', 'pattern invalid')}{model_note}{ownership_note}"
                else:
                    status = "PASS"
                    reason = f"{count}/{expected} bungs{model_note}{ownership_note}"
            else:
                status = "FAIL"
                reason = f"{count}/{expected} bungs{model_note}{ownership_note}"

            grades.append(
                BatteryGrade(
                    track_id=-1,
                    box=batt.box,
                    confidence=batt.conf,
                    bung_count=count,
                    expected_bungs=expected,
                    bung_boxes=inside,
                    status=status,
                    reason=reason,
                    obb_points=batt.obb_points,
                    assigned_bung_indices=list(assigned_indices),
                    pattern_name=str(pattern_info.get("pattern") or ""),
                    pattern_ok=pattern_info.get("ok"),
                    pattern_reason=str(pattern_info.get("reason") or ""),
                )
            )

        grades = self.assign_track_ids(grades)

        if not batteries:
            status = "NO BATTERY"
            reason = "No battery detected"
            total_bungs = len(bungs)
        elif any(g.status == "FAIL" for g in grades):
            bad = sum(1 for g in grades if g.status == "FAIL")
            status = "FAIL"
            reason = f"{bad}/{len(grades)} visible batteries failed"
            total_bungs = sum(g.bung_count for g in grades)
        elif any(g.status == "WAIT" for g in grades):
            waiting = sum(1 for g in grades if g.status == "WAIT")
            status = "WAIT"
            reason = f"{waiting}/{len(grades)} visible batteries waiting for full view"
            total_bungs = sum(g.bung_count for g in grades)
        else:
            status = "PASS"
            reason = f"All {len(grades)} visible batteries passed"
            total_bungs = sum(g.bung_count for g in grades)

        return InspectionResult(status, reason, len(batteries), total_bungs, expected, detections, self.fps, grades)

    def _record_preview_submit(self, display_seq: int = 0) -> None:
        """Record actual frames submitted to the preview widget.

        This is separate from the Qt timer rate. A frame is counted only when
        BungVision actually calls CameraWidget.set_frame(). The seq-based skip
        counter shows how many camera/inference frames were intentionally not
        displayed between preview updates.
        """
        now = time.perf_counter()
        try:
            display_seq = int(display_seq or 0)
        except Exception:
            display_seq = 0
        if display_seq > 0:
            prev = int(getattr(self, "_last_displayed_seq", 0) or 0)
            if prev > 0 and display_seq > prev + 1:
                self.preview_skipped_frames = int(getattr(self, "preview_skipped_frames", 0) or 0) + (display_seq - prev - 1)
            self._last_displayed_seq = display_seq
        self._preview_count = int(getattr(self, "_preview_count", 0) or 0) + 1
        self._preview_total = int(getattr(self, "_preview_total", 0) or 0) + 1
        window_t = float(getattr(self, "_preview_window_t", now) or now)
        if now - window_t >= 1.0:
            self.preview_fps = self._preview_count / max(1e-6, now - window_t)
            self._preview_count = 0
            self._preview_window_t = now

    def on_timer(self):
        self._service_external_stop_request()
        self.drain_background_logs()
        self._poll_model_load_result()
        now_loop = time.perf_counter()
        dt_s = max(1e-6, now_loop - self.last_t)
        self.last_t = now_loop
        inst_fps = 1.0 / dt_s
        # This is only the Qt timer-loop rate. Keep it internally, but do not
        # present it as preview/display FPS.
        self.fps = inst_fps if self.fps <= 0 else self.fps * 0.85 + inst_fps * 0.15


        latest_cam = None
        latest_cam_seq = 0
        frame_for_display = None
        overlay_result = None
        preview_for_display = None
        display_seq = 0

        if self.demo_mode:
            # Demo synthesizes its own detections; halt the autonomous inference
            # worker so it does not keep pulling/predicting real camera frames.
            if getattr(self, "inference_worker", None) is not None:
                self.inference_worker.set_enabled(False)
            frame = self.demo_frame()
            detections = self.demo_detections(frame)
            self._last_frame_ok_t = time.perf_counter()
            self._last_prediction_ok_t = time.perf_counter()
            self._last_camera_error = ""
            self._last_prediction_error = ""
            result = self.inspect(detections, frame)
            result.fps = self.fps
            self.update_tracker(result, frame)
            self.last_frame = frame
            self.last_result = result
            self._last_result_seq = int(getattr(self, "_last_result_seq", 0) or 0) + 1
            self.camera_widget.set_frame(frame, result, seq=self._last_result_seq)
            self._record_preview_submit(self._last_result_seq)
            self.update_decision(result)
            self.update_metrics(fps=self.fps)
            self.update_status_pills(result)
            self.update_plc_outputs(result)
            return

        if getattr(self, "camera_worker", None) is not None:
            latest_cam = self.camera_worker.get_latest()
            cam_fps, cam_err, cam_seq = self.camera_worker.status()
            self.camera_capture_fps = cam_fps
            try:
                cam_metrics = self.camera_worker.metrics()
                self.camera_read_ms = float(cam_metrics.get("read_ms", 0.0) or 0.0)
                self.camera_interval_ms = float(cam_metrics.get("interval_ms", 0.0) or 0.0)
                self.camera_frame_age_ms = float(cam_metrics.get("age_ms", 0.0) or 0.0)
                self.camera_sleep_ms = float(cam_metrics.get("sleep_ms", 0.0) or 0.0)
            except Exception:
                pass
            if latest_cam is not None:
                latest_cam_seq = int(latest_cam.seq)
                self._last_frame_ok_t = latest_cam.timestamp
                self._last_camera_error = ""
                try:
                    fh, fw = latest_cam.frame.shape[:2]
                    if fw and fh and (fw != getattr(self, "actual_camera_width", 0) or fh != getattr(self, "actual_camera_height", 0)):
                        self.actual_camera_width = int(fw)
                        self.actual_camera_height = int(fh)
                        if hasattr(self, "camera_actual_pill"):
                            roi_desc = ""
                            try:
                                roi_desc = self.cap.roi_description() if self.cap is not None and hasattr(self.cap, "roi_description") else ""
                            except Exception:
                                roi_desc = ""
                            txt = f"Actual {fw}x{fh}" + (f" {roi_desc}" if roi_desc else "")
                            self.camera_actual_pill.setText(txt)
                except Exception:
                    pass
            elif cam_err:
                self._last_camera_error = cam_err

        model_loaded = getattr(self.model_runner, "model", None) is not None
        preview_only_mode = False
        inference_paused_mode = False

        # Keep the inference worker pointed at the current operator settings.
        if getattr(self, "inference_worker", None) is not None:
            preview_w, preview_h = self.camera_widget.preview_target_size_hint()
            self.inference_worker.update_config(
                self.conf_spin.value() / 100.0,
                float(getattr(self, "yolo_iou", 0.45)),
                int(self.imgsz_spin.value()),
                self.device_edit.text().strip(),
                preview_w=preview_w,
                preview_h=preview_h,
            )
            # Pull-based inference (v0.9.86): the worker fetches the newest
            # camera frame on its own thread as soon as it is free, so it no
            # longer waits for this UI timer to push a frame (which inserted up
            # to one UI-tick of idle between predictions). The UI thread only
            # enables/disables the worker and reads the latest result below.
            self.inference_worker.set_enabled(bool(self.running and model_loaded))

            inf_fps, dropped, inf_err = self.inference_worker.status()
            self.inference_fps = inf_fps
            try:
                inf_metrics = self.inference_worker.metrics()
                self.inference_cycle_ms = float(inf_metrics.get("cycle_ms", 0.0) or 0.0)
                self.inference_idle_ms = float(inf_metrics.get("idle_ms", 0.0) or 0.0)
                self.inference_backend_ms = float(inf_metrics.get("backend_ms", 0.0) or 0.0)
                self.inference_parse_ms = float(inf_metrics.get("parse_ms", 0.0) or 0.0)
                # Inference stall watchdog: log when a running TensorRT call takes
                # long enough to explain an operator-visible freeze. This is a
                # diagnostic layer only; it does not kill a running TensorRT call.
                try:
                    max_stall_ms = float(getattr(self, "inference_stall_watchdog_ms", 1500.0) or 1500.0)
                    busy_age_ms = float(inf_metrics.get("busy_age_ms", 0.0) or 0.0)
                    now_stall_log = now_loop
                    if busy_age_ms >= max_stall_ms and now_stall_log - float(getattr(self, "_last_inference_stall_log_t", 0.0) or 0.0) > 1.0:
                        self._last_inference_stall_log_t = now_stall_log
                        current_seq = int(inf_metrics.get("current_seq", 0.0) or 0)
                        self.log(f"INFER_STALL active inference seq={current_seq} busy_age_ms={busy_age_ms:.0f} latest_cam_seq={latest_cam_seq}; showing live camera until result returns")
                except Exception:
                    pass
            except Exception:
                pass
            # status() already sums worker-side dropped + skipped (never-inferred)
            # camera frames, so this is the total frames not inspected.
            self.dropped_inference_frames = int(dropped)
            if inf_err:
                self._last_prediction_error = inf_err
            latest_inf = self.inference_worker.get_latest()
            if latest_inf is not None and latest_inf.seq != self._last_processed_inference_seq:
                # If TensorRT/driver scheduling hiccups and returns a very old
                # frame after the camera has moved far ahead, do not update the
                # overlay/decision with a stale image. This prevents a brief
                # inference stall from turning into a long frozen-looking HMI.
                try:
                    stale_frames = int(getattr(self, "inference_stale_result_max_frames", 8) or 8)
                    if latest_cam_seq > 0 and int(latest_inf.seq) < int(latest_cam_seq) - stale_frames:
                        self._last_processed_inference_seq = latest_inf.seq
                        now_stale_log = now_loop
                        if now_stale_log - float(getattr(self, "_last_stale_result_log_t", 0.0) or 0.0) > 1.0:
                            self._last_stale_result_log_t = now_stale_log
                            self.log(f"INFER_STALE_RESULT dropped result seq={latest_inf.seq} latest_cam_seq={latest_cam_seq} max_lag_frames={stale_frames}")
                        latest_inf = None
                except Exception:
                    pass
            if latest_inf is not None and latest_inf.seq != self._last_processed_inference_seq:
                self._last_processed_inference_seq = latest_inf.seq
                self._last_result_seq = latest_inf.seq
                self._inference_ms = latest_inf.infer_ms
                self.inference_cycle_ms = float(getattr(latest_inf, "cycle_ms", self.inference_cycle_ms) or 0.0)
                self.inference_idle_ms = float(getattr(latest_inf, "idle_ms", self.inference_idle_ms) or 0.0)
                self.inference_backend_ms = float(getattr(latest_inf, "backend_ms", self.inference_backend_ms) or 0.0)
                self.inference_parse_ms = float(getattr(latest_inf, "parse_ms", self.inference_parse_ms) or 0.0)
                self._last_inference_result_t = latest_inf.timestamp
                if latest_inf.error:
                    self._last_prediction_error = latest_inf.error
                    detections = []
                else:
                    self._last_prediction_ok_t = latest_inf.timestamp
                    self._last_prediction_error = ""
                    detections = latest_inf.detections
                    # Custom reject classes: latch immediately on first matching detection.
                    if self.custom_reject_classes and not self.reject_latched:
                        custom_lower = {c.lower() for c in self.custom_reject_classes}
                        for det in detections:
                            if det.label.lower() in custom_lower:
                                self.latch_reject_for_class(det.label, det.conf)
                                break
                result = self.inspect(detections, latest_inf.frame)
                result.fps = self.fps
                self.update_tracker(result, latest_inf.frame)
                self.last_frame = latest_inf.frame
                self.last_preview_rgb = getattr(latest_inf, "preview_rgb", None)
                self.last_result = result
                self.update_decision(result)

        # Overlay synchronization rule:
        # Never draw YOLO results from one frame on top of a newer moving camera
        # frame. When overlays are enabled and a model is loaded, the preview uses
        # the exact frame that produced the result. If no matching result exists
        # yet, show the newest camera frame without object overlays.
        overlay_enabled = bool(getattr(self.camera_widget, "overlay_enabled", True))
        result_age = now_loop - float(getattr(self, "_last_inference_result_t", 0.0) or 0.0)
        has_synced_result = self.last_frame is not None and self.last_result is not None and int(getattr(self, "_last_result_seq", 0) or 0) > 0
        if self.running and model_loaded and overlay_enabled and has_synced_result and result_age <= 0.75:
            frame_for_display = self.last_frame
            overlay_result = self.last_result
            preview_for_display = getattr(self, "last_preview_rgb", None)
            display_seq = int(getattr(self, "_last_result_seq", 0) or 0)
        elif latest_cam is not None:
            # Direct-camera path (no synced result yet): the worker did not
            # pre-scale this frame, so the UI thread scales it as a fallback.
            frame_for_display = latest_cam.frame
            overlay_result = None
            preview_for_display = None
            display_seq = int(latest_cam.seq)
        elif self.last_frame is not None:
            frame_for_display = self.last_frame
            overlay_result = self.last_result if overlay_enabled else None
            preview_for_display = getattr(self, "last_preview_rgb", None)
            display_seq = int(getattr(self, "_last_result_seq", 0) or 0)

        if frame_for_display is None:
            self.camera_widget.set_frame(None, None)
            self._record_preview_submit(0)
            self.update_status_pills(self.last_result)
            self.update_plc_outputs(self.last_result)
            self.update_metrics(fps=self.fps)
            return

        # v0.9.77: uncapped preview refresh. Update the camera widget on every UI
        # timer tick that has a frame available.
        self.camera_widget.set_frame(frame_for_display, overlay_result, seq=display_seq, preview_rgb=preview_for_display)
        self._record_preview_submit(display_seq)
        # PLC state submission and the expensive QLabel/card refreshes do not
        # need to run at the full 30 Hz timer rate. The PLC writer is async with
        # its own ~100 ms cadence and independent heartbeat, so submitting at
        # 10 Hz here avoids rebuilding PLC config and pill text every tick while
        # keeping outputs responsive. reset pulses are sent immediately via
        # pulse_plc_reset() and held by the writer, so they cannot be missed.
        if now_loop - float(getattr(self, "_last_metrics_ui_update_t", 0.0) or 0.0) >= 0.10:
            self._last_metrics_ui_update_t = now_loop
            self.update_plc_outputs(self.last_result)
            self.update_metrics(fps=self.fps)
            self.update_status_pills(self.last_result)

    def update_decision(self, result: InspectionResult):
        if getattr(self, "reject_latched", False):
            self.decision_label.setText("REJECT LATCHED")
            self.reason_label.setText(f"ID {self.reject_latch_id}: {self.reject_latch_reason} — Reset required")
        else:
            self.decision_label.setText(result.status)
            self.reason_label.setText(result.reason)
        self.bung_big.setText(f"{result.bung_count}/{result.expected_bungs}")
        if getattr(self, "reject_latched", False):
            self.decision_frame.setStyleSheet("QFrame#DecisionFrame { background:#7f1d1d; border:2px solid #f97316; border-radius:22px; }")
        elif result.status == "PASS":
            self.decision_frame.setStyleSheet("QFrame#DecisionFrame { background:#064e3b; border:1px solid #10b981; border-radius:22px; }")
        elif result.status == "FAIL":
            self.decision_frame.setStyleSheet("QFrame#DecisionFrame { background:#7f1d1d; border:1px solid #ef4444; border-radius:22px; }")
        elif result.status == "WAIT":
            self.decision_frame.setStyleSheet("QFrame#DecisionFrame { background:#78350f; border:1px solid #f59e0b; border-radius:22px; }")
        else:
            self.decision_frame.setStyleSheet("QFrame#DecisionFrame { background:#1e293b; border:1px solid #334155; border-radius:22px; }")

    def _log_profiler_sample(self) -> None:
        """Write passive performance telemetry every ~2 seconds.

        v0.9.47 used several one-second rolling/window values directly. That made
        first samples and some idle states misleading (for example preview_fps=0
        while paint/QImage work was active). v0.9.48 calculates the log rates from
        monotonic counters/sequence deltas over the profiler interval, which is
        measurement-only and does not alter runtime scheduling.
        """
        try:
            now = time.perf_counter()
            last = float(getattr(self, "_last_profiler_log_t", 0.0) or 0.0)
            if last <= 0.0:
                self._last_profiler_log_t = now
                return
            elapsed = now - last
            if elapsed < 2.0:
                return

            cam_seq = 0
            cam_read_ms = float(getattr(self, "camera_read_ms", 0.0) or 0.0)
            cam_grab_wait_ms = 0.0
            cam_convert_ms = 0.0
            cam_array_ms = 0.0
            cam_interval_ms = float(getattr(self, "camera_interval_ms", 0.0) or 0.0)
            cam_age_ms = float(getattr(self, "camera_frame_age_ms", 0.0) or 0.0)
            if getattr(self, "camera_worker", None) is not None:
                try:
                    cm = self.camera_worker.metrics()
                    cam_seq = int(cm.get("seq", 0) or 0)
                    cam_read_ms = float(cm.get("read_ms", cam_read_ms) or 0.0)
                    cam_grab_wait_ms = float(cm.get("grab_wait_ms", cam_grab_wait_ms) or 0.0)
                    cam_convert_ms = float(cm.get("convert_ms", cam_convert_ms) or 0.0)
                    cam_array_ms = float(cm.get("array_ms", cam_array_ms) or 0.0)
                    cam_interval_ms = float(cm.get("interval_ms", cam_interval_ms) or 0.0)
                    cam_age_ms = float(cm.get("age_ms", cam_age_ms) or 0.0)
                except Exception:
                    cam_seq = int(getattr(self, "_last_camera_seq_seen", 0) or 0)

            preview_total = int(getattr(self, "_preview_total", 0) or 0)
            paint_total = 0
            paint_fps_window = 0.0
            try:
                paint_total = int(self.camera_widget.paint_total()) if hasattr(self, "camera_widget") else 0
                paint_fps_window = float(self.camera_widget.paint_fps()) if hasattr(self, "camera_widget") else 0.0
            except Exception:
                paint_total = 0

            inf_done = int(getattr(self, "_prof_last_inf_done_count", 0) or 0)
            if getattr(self, "inference_worker", None) is not None:
                try:
                    im = self.inference_worker.metrics()
                    inf_done = int(im.get("done_count", inf_done) or 0)
                except Exception:
                    pass

            last_cam = int(getattr(self, "_prof_last_cam_seq", 0) or 0)
            last_prev = int(getattr(self, "_prof_last_preview_total", 0) or 0)
            last_paint = int(getattr(self, "_prof_last_paint_total", 0) or 0)
            last_inf = int(getattr(self, "_prof_last_inf_done_count", 0) or 0)

            cam_fps_delta = max(0, cam_seq - last_cam) / max(1e-6, elapsed)
            preview_fps_delta = max(0, preview_total - last_prev) / max(1e-6, elapsed)
            paint_fps_delta = max(0, paint_total - last_paint) / max(1e-6, elapsed)
            inf_fps_delta = max(0, inf_done - last_inf) / max(1e-6, elapsed)

            self._last_profiler_log_t = now
            self._prof_last_cam_seq = cam_seq
            self._prof_last_preview_total = preview_total
            self._prof_last_paint_total = paint_total
            self._prof_last_inf_done_count = inf_done

            # Keep the cards aligned with the interval-based profile rates.
            self.camera_capture_fps = cam_fps_delta
            self.preview_fps = preview_fps_delta
            if inf_done > 0:
                self.inference_fps = inf_fps_delta

            save_q = self.save_worker.qsize() if hasattr(self, "save_worker") and self.save_worker is not None else 0
            try:
                prev_src = getattr(self.camera_widget, "_preview_source_shape", (0, 0))
                prev_render = getattr(self.camera_widget, "_preview_render_size", (0, 0))
                prev_src_txt = f"{int(prev_src[0])}x{int(prev_src[1])}"
                prev_render_txt = f"{int(prev_render[0])}x{int(prev_render[1])}"
            except Exception:
                prev_src_txt = "0x0"
                prev_render_txt = "0x0"
            _write_debug_log(
                "PROFILE76 "
                f"elapsed_s={elapsed:.2f} "
                f"cam_fps={cam_fps_delta:.2f} cam_seq={cam_seq} "
                f"cam_read_ms={cam_read_ms:.2f} cam_grab_wait_ms={cam_grab_wait_ms:.2f} "
                f"cam_convert_ms={cam_convert_ms:.2f} cam_array_ms={cam_array_ms:.2f} "
                f"cam_sleep_ms={float(getattr(self, 'camera_sleep_ms', 0.0) or 0.0):.2f} "
                f"cam_interval_ms={cam_interval_ms:.2f} cam_age_ms={cam_age_ms:.2f} "
                f"preview_fps={preview_fps_delta:.2f} preview_total={preview_total} "
                f"preview_src={prev_src_txt} preview_render={prev_render_txt} "
                f"paint_fps={paint_fps_delta:.2f} paint_window_fps={paint_fps_window:.2f} paint_total={paint_total} "
                f"paint_ms={float(getattr(self, 'paint_ms', 0.0) or 0.0):.2f} "
                f"qimage_ms={float(getattr(self, 'qimage_ms', 0.0) or 0.0):.2f} "
                f"scale_ms={float(getattr(self, 'scale_ms', 0.0) or 0.0):.2f} "
                f"overlay_ms={float(getattr(self, 'overlay_draw_ms', 0.0) or 0.0):.2f} "
                f"inf_fps={inf_fps_delta:.2f} inf_done={inf_done} "
                f"infer_ms={float(getattr(self, '_inference_ms', 0.0) or 0.0):.2f} "
                f"trt_py_ms={float(getattr(self, 'inference_backend_ms', 0.0) or 0.0):.2f} "
                f"parse_ms={float(getattr(self, 'inference_parse_ms', 0.0) or 0.0):.2f} "
                f"cycle_ms={float(getattr(self, 'inference_cycle_ms', 0.0) or 0.0):.2f} "
                f"idle_ms={float(getattr(self, 'inference_idle_ms', 0.0) or 0.0):.2f} "
                f"inf_skip={int(getattr(self, 'dropped_inference_frames', 0) or 0)} "
                f"prev_skip={int(getattr(self, 'preview_skipped_frames', 0) or 0)} "
                f"save_q={int(save_q)}"
            )
        except Exception:
            try:
                _write_debug_log("PROFILE76 logging failed:\n" + traceback.format_exc())
            except Exception:
                pass

    def update_metrics(self, fps=None, **_ignored):
        """Update the main operator metric cards with honest throughput numbers.

        UI loop FPS is not the same as camera, preview, paint, or inference FPS.
        The operator screen now reports actual preview submissions and paint rate
        instead of the 30 Hz Qt timer rate.
        """
        if hasattr(self, "total_card"):
            self.total_card.set_value(str(int(getattr(self, "total_count", 0))), "committed results")
        if hasattr(self, "pass_card"):
            self.pass_card.set_value(str(int(getattr(self, "pass_count", 0))), "committed PASS results")
        if hasattr(self, "reject_card"):
            self.reject_card.set_value(str(int(getattr(self, "fail_count", 0))), "committed FAIL results")
        if hasattr(self, "pass_rate_card"):
            total = int(getattr(self, "total_count", 0))
            passed = int(getattr(self, "pass_count", 0))
            if total <= 0:
                self.pass_rate_card.set_value("--", "no inspections")
            else:
                rate = 100.0 * passed / max(1, total)
                self.pass_rate_card.set_value(f"{rate:.1f}%", f"{passed}/{total} PASS")
        if hasattr(self, "infer_card"):
            infer_ms = float(getattr(self, "_inference_ms", 0.0) or 0.0)
            inf_fps = float(getattr(self, "inference_fps", 0.0) or 0.0)
            if infer_ms > 0.0 or inf_fps > 0.0:
                self.infer_card.set_value(f"{inf_fps:.1f} FPS", f"{infer_ms:.1f} ms")
            else:
                self.infer_card.set_value("-- FPS", "-- ms")

        # Keep paint/preview timing fields refreshed for the profiler log without
        # showing them on the operator screen. This keeps troubleshooting data
        # available while removing the extra visible performance indicators.
        try:
            pm = self.camera_widget.paint_metrics()
            self.paint_ms = float(pm.get("paint_ms", 0.0) or 0.0)
            self.qimage_ms = float(pm.get("qimage_ms", 0.0) or 0.0)
            self.scale_ms = float(pm.get("scale_ms", 0.0) or 0.0)
            self.overlay_draw_ms = float(pm.get("overlay_ms", 0.0) or 0.0)
        except Exception:
            pass

        self._log_profiler_sample()

    def update_status_pills(self, result: Optional[InspectionResult] = None):
        self.ensure_tracker_state()
        plc_enabled = self._checkbox_checked("plc_enabled_check", False)
        self.plc_pill.setText("PLC ON" if plc_enabled else "PLC SIM")
        self.plc_pill.set_tone("pass" if plc_enabled else "warn")
        # The heartbeat BOOL toggles for the PLC. Do not show the raw bit state
        # in the UI or the pill will flicker ON/OFF.
        if plc_enabled:
            plc_status = getattr(self, "_plc_last_status", "DISABLED")
            if plc_status in ("CONNECTED", "SIM", "DISABLED") or str(plc_status).startswith("CONNECTED"):
                self.heartbeat_pill.setText(f"Heartbeat {int(getattr(self, 'plc_heartbeat_interval_ms', 500))} ms")
                self.heartbeat_pill.set_tone("pass")
            else:
                self.heartbeat_pill.setText("Heartbeat Error")
                self.heartbeat_pill.set_tone("fail")
        else:
            self.heartbeat_pill.setText("Heartbeat SIM")
            self.heartbeat_pill.set_tone("warn")

        bypass = self._checkbox_checked("bypass_check", False)
        self.mode_pill.setText("BYPASS" if bypass else ("DEV DEMO" if self.demo_mode else "LIVE RUNTIME"))
        self.mode_pill.set_tone("warn" if (bypass or self.demo_mode) else "pass")

        stop_active = bool(getattr(self, "reject_latched", False) and not bypass)
        if stop_active:
            self.stop_output_pill.setText("Stop Output ON")
            self.stop_output_pill.set_tone("fail")
            self.alarm_pill.setText("Alarm ON")
            self.alarm_pill.set_tone("fail")
        else:
            self.stop_output_pill.setText("Stop Output OFF")
            self.stop_output_pill.set_tone("neutral")
            self.alarm_pill.setText("Alarm OFF" if not bypass else "Alarm BYPASS")
            self.alarm_pill.set_tone("neutral" if not bypass else "warn")

        active_tracks = len(self._tracks)
        unlogged = sum(1 for tr in self._tracks.values() if not tr.get("logged", False))
        healthy = self.vision_healthy() if hasattr(self, "vision_healthy") else False
        if getattr(self, "reject_latched", False) and not bypass:
            self.ready_pill.setText("Not Ready - Reject")
            self.ready_pill.set_tone("fail")
        elif not healthy:
            if not getattr(self, "running", False):
                reason = "Stopped"
            elif getattr(self.model_runner, "model", None) is None:
                reason = "No Model"
            elif getattr(self, "cap", None) is None:
                reason = "No Camera"
            elif str(getattr(self, "_last_prediction_error", "")):
                reason = "Prediction Error"
            else:
                reason = "No Recent Frame"
            self.ready_pill.setText(f"Not Ready - {reason}")
            self.ready_pill.set_tone("warn")
        elif active_tracks:
            self.ready_pill.setText(f"Ready - Tracking {active_tracks} / New {unlogged}")
            self.ready_pill.set_tone("warn" if unlogged else "pass")
        else:
            self.ready_pill.setText("Ready")
            self.ready_pill.set_tone("pass")

        if hasattr(self, "plc_write_pill"):
            if plc_enabled:
                status = getattr(self, "_plc_last_status", getattr(self.plc, "_last_status", "PLC")) if hasattr(self, "plc") else "PLC"
                short_status = status if len(status) <= 28 else status[:25] + "..."
                self.plc_write_pill.setText(f"PLC Writes {short_status}")
                self.plc_write_pill.set_tone("pass" if str(status).startswith("CONNECTED") else "fail")
            else:
                self.plc_write_pill.setText("PLC Writes OFF")
                self.plc_write_pill.set_tone("neutral")
        if hasattr(self, "footer_label"):
            model_txt = Path(getattr(self.model_runner, "path", "")).name if getattr(self.model_runner, "path", "") else "none"
            cam_txt = f"{getattr(self, 'actual_camera_width', 0)}x{getattr(self, 'actual_camera_height', 0)}" if getattr(self, "cap", None) is not None else "off"
            plc_txt = getattr(self, "_plc_last_status", "SIM")
            self.footer_label.setText(f"{APP_TITLE} | Model: {model_txt} | Camera: {cam_txt} | PLC: {plc_txt}")


    def record_inspection(self, result: InspectionResult, frame: np.ndarray):
        # Kept for compatibility. Normal runtime uses update_tracker().
        self.update_tracker(result, frame)

    def write_battery_record(self, grade: BatteryGrade, result: InspectionResult, frame: np.ndarray, save_image: Optional[bool] = None) -> bool:
        if grade.track_id <= 0:
            return False
        # Tracker state belongs to the UI thread. The same PASS-never-becomes-FAIL
        # rule is enforced before the save job is queued, so background disk writes
        # must not iterate live UI tracking dictionaries.
        if threading.get_ident() == getattr(self, "_ui_thread_ident", threading.get_ident()):
            tr = None
            for candidate in self._tracks.values():
                if candidate.get("inspection_id") == grade.track_id:
                    tr = candidate
                    break
            if (tr is not None and tr.get("final_status") == "PASS") and grade.status != "PASS":
                return False

        if grade.status == "PASS":
            folder = PASS_DIR
        elif grade.status == "FAIL":
            folder = FAIL_DIR
        else:
            return False

        if save_image is None:
            save_image = (grade.status == "PASS" and self._checkbox_checked("save_pass_images_check", True)) or (
                grade.status == "FAIL" and self._checkbox_checked("save_fail_images_check", True)
            )
        save_image = bool(save_image)
        snapshot_text = ""
        if save_image:
            stamp = now_stamp()
            snapshot = folder / f"{stamp}_ID{grade.track_id}_{grade.status}.jpg"
            annotated = frame.copy()
            self.draw_snapshot_overlay(annotated, result)
            if cv2.imwrite(str(snapshot), annotated):
                snapshot_text = str(snapshot)
            else:
                self.log(f"WARNING: Could not write {grade.status} image for ID {grade.track_id}.")

        with self.log_file.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                dt.datetime.now().isoformat(timespec="milliseconds"),
                grade.status,
                f"Battery ID {grade.track_id}: {grade.reason}",
                1,
                grade.bung_count,
                grade.expected_bungs,
                f"{result.fps:.2f}",
                snapshot_text,
            ])

        if snapshot_text:
            self.log(f"Wrote {grade.status} record and image for ID {grade.track_id}.")
        else:
            self.log(f"Wrote {grade.status} record for ID {grade.track_id}; image saving is off.")
        return True

    def draw_snapshot_overlay(self, frame: np.ndarray, result: InspectionResult):
        for d in result.detections:
            kind = detection_kind(d.label)
            if kind == "battery":
                color = (255, 180, 40)
            elif kind == "bung":
                color = (60, 230, 80)
            elif kind == "retainer":
                color = (0, 180, 255)
            else:
                color = (180, 180, 180)
            x1, y1, x2, y2 = d.box
            if getattr(d, "obb_points", None):
                pts = np.array(d.obb_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, 3)
                lx = int(min(p[0] for p in d.obb_points))
                ly = int(min(p[1] for p in d.obb_points))
                label = f"{d.label} {d.conf:.2f} OBB"
                cv2.putText(frame, label, (lx, max(20, ly - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{d.label} {d.conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw committed/candidate battery grade outlines over the raw detections so
        # the saved review image shows which rotated battery footprint owned the bungs.
        for grade in getattr(result, "battery_grades", []):
            color = (60, 230, 80) if grade.status == "PASS" else ((0, 190, 255) if grade.status == "WAIT" else (40, 40, 230))
            if getattr(grade, "obb_points", None):
                pts = np.array(grade.obb_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, 5)
                lx = int(min(p[0] for p in grade.obb_points))
                ly = int(max(p[1] for p in grade.obb_points))
            else:
                x1, y1, x2, y2 = grade.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                lx, ly = x1, y2
            id_text = f"ID {grade.track_id}" if grade.track_id > 0 else "CAND"
            line1 = f"{id_text} {grade.status}"
            line2 = f"{grade.bung_count}/{grade.expected_bungs}"

            if getattr(grade, "obb_points", None):
                gx1 = int(min(p[0] for p in grade.obb_points))
                gy1 = int(min(p[1] for p in grade.obb_points))
                gx2 = int(max(p[0] for p in grade.obb_points))
                gy2 = int(max(p[1] for p in grade.obb_points))
            else:
                gx1, gy1, gx2, gy2 = grade.box

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale1 = 0.75
            scale2 = 0.65
            thick = 2
            (line1_w, line1_h), base1 = cv2.getTextSize(line1, font, scale1, thick)
            (line2_w, line2_h), base2 = cv2.getTextSize(line2, font, scale2, thick)
            badge_w = max(line1_w, line2_w) + 18
            badge_h = line1_h + line2_h + base1 + base2 + 18
            bx = gx1 + 10
            by = gy1 + 10
            if gx2 - badge_w - 6 >= gx1 + 4:
                bx = max(gx1 + 4, min(bx, gx2 - badge_w - 6))
            if gy2 - badge_h - 6 >= gy1 + 4:
                by = max(gy1 + 4, min(by, gy2 - badge_h - 6))
            bx = max(4, min(int(bx), frame.shape[1] - badge_w - 4))
            by = max(4, min(int(by), frame.shape[0] - badge_h - 4))

            bg_color = (30, 110, 70) if grade.status == "PASS" else ((30, 90, 150) if grade.status == "WAIT" else (40, 40, 170))
            cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), bg_color, -1)
            cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), color, 2)
            cv2.putText(frame, line1, (bx + 9, by + line1_h + 7), font, scale1, (255, 255, 255), thick)
            cv2.putText(frame, line2, (bx + 9, by + line1_h + line2_h + base1 + 12), font, scale2, (255, 255, 255), thick)

        banner_color = (40, 160, 80) if result.status == "PASS" else (40, 40, 220)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), banner_color, -1)
        cv2.putText(frame, f"{result.status}: {result.reason}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    def add_history_row(self, result: InspectionResult):
        for grade in result.battery_grades:
            self.add_history_row_for_grade(grade, result)

    def add_history_row_for_grade(self, grade: BatteryGrade, result: InspectionResult):
        self.table.insertRow(0)
        values = [
            dt.datetime.now().strftime("%H:%M:%S"),
            f"ID {grade.track_id} {grade.status}",
            f"{grade.bung_count}/{grade.expected_bungs}",
            grade.reason,
            f"{result.fps:.1f}",
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col == 1:
                item.setForeground(QColor("#86efac" if grade.status == "PASS" else "#fca5a5"))
                item.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(0, col, item)
        while self.table.rowCount() > 20:
            self.table.removeRow(self.table.rowCount() - 1)

    def open_folder(self, path: Path):
        try:
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                self.log(f"Open folder: {path}")
        except Exception as e:
            self.log(f"Could not open folder {path}: {e}")

    def run_preflight_check(self):
        checks = []
        checks.append(("Camera source", self.source_edit.text().strip() or "0"))
        checks.append(("Model loaded", "YES" if self.model_runner.model is not None else "NO"))
        checks.append(("Model path", getattr(self.model_runner, "path", "") or self.model_edit.text().strip() or "none"))
        checks.append(("Expected bungs", str(self._spin_value("expected_spin", 6))))
        checks.append(("Confidence", f"{self._spin_value('conf_spin', 25)}%"))
        checks.append(("YOLO image size", str(self._spin_value("imgsz_spin", 736))))
        checks.append(("YOLO IoU", f"{float(getattr(self, 'yolo_iou', 0.45)):.2f}"))
        checks.append(("Device", self.device_edit.text().strip() if hasattr(self, "device_edit") and self.device_edit.text().strip() else "auto"))
        requested_camera = f"{int(getattr(self, 'camera_width', 0))}x{int(getattr(self, 'camera_height', 0))} @ {float(getattr(self, 'camera_fps', 0.0)):.0f} FPS"
        if str(getattr(self, "camera_backend", "opencv")).lower() == "basler" and bool(getattr(self, "basler_roi_enabled", False)):
            requested_camera += f"; ROI offset X={getattr(self, 'basler_roi_offset_x', 0)} Y={getattr(self, 'basler_roi_offset_y', 0)}"
        checks.append(("Camera requested", requested_camera))
        checks.append(("Camera actual", f"{getattr(self, 'actual_camera_width', 0)}x{getattr(self, 'actual_camera_height', 0)}"))
        checks.append(("PLC writes", "ENABLED" if self._checkbox_checked("plc_enabled_check", False) else "OFF/SIM"))
        checks.append(("PLC IP", self.plc_ip_edit.text().strip() if hasattr(self, "plc_ip_edit") else ""))
        checks.append(("Heartbeat interval", f"{int(getattr(self, 'plc_heartbeat_interval_ms', 500))} ms"))
        checks.append(("Capture folder", str(TRAINING_REVIEW_DIR)))
        checks.append(("Capture folder writable", "YES" if os.access(str(TRAINING_REVIEW_DIR), os.W_OK) else "NO"))
        checks.append(("Save PASS captures", "YES" if getattr(self, "save_pass_training_images", True) else "NO"))
        checks.append(("Save FAIL captures", "YES" if getattr(self, "save_fail_training_images", True) else "NO"))
        checks.append(("PASS sample rate", f"1/{max(1, int(getattr(self, 'pass_training_sample_rate', 1)))}"))
        text = "\n".join(f"{k}: {v}" for k, v in checks)
        self.log("Preflight check:\\n" + text)
        QMessageBox.information(self, "BungVision Preflight Check", text)

    def _configure_sync_plc_for_test(self) -> None:
        """Configure the synchronous PLCInterface used only by test buttons."""
        if not hasattr(self, "plc") or self.plc is None:
            self.plc = PLCInterface()
        tags = {key: edit.text().strip() for key, edit in getattr(self, "plc_tag_edits", {}).items()}
        self.plc.configure(
            enabled=self._checkbox_checked("plc_enabled_check", False),
            ip_address=self.plc_ip_edit.text().strip() if hasattr(self, "plc_ip_edit") else "",
            tags=tags,
        )

    def test_plc_connection(self):
        # Test buttons are manual actions, so it is acceptable for them to do a
        # direct pylogix call. Normal runtime writes remain asynchronous.
        self.apply_plc_config()
        self._configure_sync_plc_for_test()
        if not self._checkbox_checked("plc_enabled_check", False):
            QMessageBox.information(self, "PLC Test", "PLC writes are disabled. Enable PLC Writes first.")
            return
        status = self.plc.validate_tags()
        self._plc_last_status = status
        self._plc_last_error = getattr(self.plc, "_last_error", "")
        self.update_status_pills(self.last_result)
        detail = f"PLC status: {status}"
        if self._plc_last_error:
            detail += f"\n\nDetail: {self._plc_last_error}"
        QMessageBox.information(self, "PLC Tag Test", detail)

    def test_plc_write(self, key: str):
        # Test buttons are manual actions, so it is acceptable for them to do a
        # direct pylogix call. Normal runtime writes remain asynchronous.
        self.apply_plc_config()
        self._configure_sync_plc_for_test()
        if not self._checkbox_checked("plc_enabled_check", False):
            QMessageBox.information(self, "PLC Test", "PLC writes are disabled. Enable PLC Writes first.")
            return
        tag = self.plc.tags.get(key, "")
        if not tag:
            QMessageBox.warning(self, "PLC Test", f"No tag configured for {key}.")
            return
        try:
            status = self.plc.write_states({key: True})
            time.sleep(0.05)
            status2 = self.plc.write_states({key: False})
            self._plc_last_status = status2
            QMessageBox.information(self, "PLC Write Test", f"{key} / {tag}\nON status: {status}\nOFF status: {status2}")
        except Exception as e:
            QMessageBox.warning(self, "PLC Write Test", f"{key} / {tag}\nFailed:\n{e}")

    def export_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export BungVision config", str(ROOT / "bungvision_settings_export.json"), "JSON Files (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.settings_payload(), indent=2), encoding="utf-8")
            self.log(f"Exported config to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Config", f"Could not export config:\\n{e}")

    def import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import BungVision config", str(ROOT), "JSON Files (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.load_settings()
            self.log(f"Imported config from {path}")
        except Exception as e:
            QMessageBox.warning(self, "Import Config", f"Could not import config:\\n{e}")

    def factory_defaults(self):
        if QMessageBox.question(self, "Factory Defaults", "Reset runtime settings to defaults? Counts/history are not changed.") != QMessageBox.Yes:
            return
        try:
            if SETTINGS_FILE.exists():
                SETTINGS_FILE.unlink()
            self.camera_backend = "opencv"
            self.opencv_api = "auto"
            self.basler_serial = ""
            self.camera_width = 2592
            self.camera_height = 1944
            self.camera_fps = 30.0
            self.basler_exposure_auto = False
            self.basler_exposure_us = 5000.0
            self.basler_gain = 0.0
            self.basler_roi_enabled = False
            self.basler_roi_offset_x = 0
            self.basler_roi_offset_y = 0
            self.basler_roi_width = self.camera_width
            self.basler_roi_height = self.camera_height
            self.source_edit.setText("0")
            self.model_edit.setText("")
            self.expected_spin.setValue(6)
            self.debounce_spin.setValue(6)
            self.entry_grace_spin.setValue(12)
            self.clear_spin.setValue(10)
            self.match_distance_spin.setValue(180)
            self.conf_spin.setValue(25)
            self.yolo_iou = 0.45
            self.imgsz_spin.setValue(736)
            self.device_edit.setText("")
            self.plc_enabled_check.setChecked(False)
            self.plc_ip_edit.setText("192.168.1.10")
            self.save_pass_training_images = True
            self.save_fail_training_images = True
            self.save_training_annotated = True
            self.save_training_json = True
            self.save_training_yolo_txt = False
            self.pass_training_sample_rate = 1
            self.plc_heartbeat_interval_ms = 500
            self.track_match_iou = 0.05
            self.committed_track_iou = 0.25
            self.require_full_view_before_grade = True
            self.full_view_margin_percent = 3.0
            self.enable_pattern_validation = True
            self.pattern_tolerance_percent = 25.0
            self.apply_runtime_settings()
            self.save_settings(silent=True)
            self.log("Factory defaults restored.")
        except Exception as e:
            QMessageBox.warning(self, "Factory Defaults", f"Could not restore defaults:\\n{e}")

    def closeEvent(self, event):
        if hasattr(self, "save_settings"):
            try:
                self.save_settings(silent=True)
            except Exception:
                pass

        # Persist the production summary synchronously here: a save queued to the
        # save worker by the final commit may not run once the worker is stopped.
        if getattr(self, "production_stats", None) is not None:
            try:
                self.production_stats.save()
            except Exception:
                pass

        self.running = False
        try:
            if getattr(self, "_external_stop_thread_stop", None) is not None:
                self._external_stop_thread_stop.set()
        except Exception:
            pass
        if getattr(self, "camera_worker", None) is not None:
            try:
                self.camera_worker.stop()
            except Exception:
                pass
            self.camera_worker = None
        if getattr(self, "inference_worker", None) is not None:
            try:
                self.inference_worker.stop()
            except Exception:
                pass
        if getattr(self, "save_worker", None) is not None:
            try:
                self.save_worker.stop()
            except Exception:
                pass
        self.drain_background_logs(limit=200)
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

        if hasattr(self, "plc_writer") and self.plc_writer is not None:
            try:
                self.plc_writer.submit({
                    "running": False,
                    "bypass": False,
                    "stop_request": False,
                    "alarm": False,
                    "ready": False,
                    "reset": False,
                })
                # Give the worker one short opportunity to publish the safe
                # final state, then close the socket. This happens only during
                # application shutdown, not during live frame processing.
                time.sleep(0.12)
                self.plc_writer.stop()
            except Exception:
                pass

        if hasattr(self, "plc") and self.plc is not None:
            try:
                self.plc.close()
            except Exception:
                pass

        event.accept()


def main():
    sys.argv = _argv_string_list(sys.argv)
    _write_debug_log(f"main() QApplication argv={sys.argv!r}")
    app = QApplication(sys.argv)
    apply_global_readability_style()
    sys.argv = _argv_string_list(sys.argv)
    _write_debug_log(f"main() post-QApplication argv={sys.argv!r}")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
