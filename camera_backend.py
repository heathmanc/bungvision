#!/usr/bin/env python3
"""Camera backends for BungVision.

OpenCV stays the default. Basler/Pylon support is optional and only activates
when pypylon is installed and the selected backend is Basler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import sys

import cv2
import numpy as np


def parse_source(text: str) -> Any:
    text = str(text or "").strip()
    try:
        return int(text)
    except ValueError:
        return text


def normalize_opencv_api(api: str = "auto") -> str:
    """Normalize the OpenCV capture API setting for cross-platform camera use."""
    text = str(api or "auto").strip().lower().replace(" ", "").replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "any": "auto",
        "cap_any": "auto",
        "auto": "auto",
        "dshow": "dshow",
        "directshow": "dshow",
        "direct_show": "dshow",
        "msmf": "msmf",
        "mediafoundation": "msmf",
        "media_foundation": "msmf",
        "v4l2": "v4l2",
        "video4linux": "v4l2",
        "gstreamer": "gstreamer",
        "gst": "gstreamer",
    }
    return aliases.get(text, "auto")


def _opencv_api_code(api: str) -> Optional[int]:
    api = normalize_opencv_api(api)
    if api == "auto":
        return getattr(cv2, "CAP_ANY", 0)
    if api == "dshow":
        return getattr(cv2, "CAP_DSHOW", None)
    if api == "msmf":
        return getattr(cv2, "CAP_MSMF", None)
    if api == "v4l2":
        return getattr(cv2, "CAP_V4L2", None)
    if api == "gstreamer":
        return getattr(cv2, "CAP_GSTREAMER", None)
    return getattr(cv2, "CAP_ANY", 0)


def _opencv_api_label(api: str) -> str:
    return {
        "auto": "Auto",
        "dshow": "DirectShow",
        "msmf": "MSMF",
        "v4l2": "V4L2",
        "gstreamer": "GStreamer",
    }.get(normalize_opencv_api(api), "Auto")


def _auto_opencv_candidates(source: Any) -> list[tuple[str, int]]:
    """Backend candidates that work well on Windows and Jetson/Linux."""
    # For string sources such as video files, RTSP URLs, or full GStreamer pipelines,
    # OpenCV's automatic backend is normally the safest default.
    if not isinstance(source, int):
        return [("auto", getattr(cv2, "CAP_ANY", 0))]

    candidates: list[tuple[str, Optional[int]]]
    if sys.platform.startswith("win"):
        # Windows Camera may work while OpenCV Auto fails; DirectShow/MSMF are the
        # two practical choices for UVC cameras on Windows.
        candidates = [
            ("dshow", getattr(cv2, "CAP_DSHOW", None)),
            ("msmf", getattr(cv2, "CAP_MSMF", None)),
            ("auto", getattr(cv2, "CAP_ANY", 0)),
        ]
    elif sys.platform.startswith("linux"):
        # Jetson/Linux UVC cameras normally use V4L2. GStreamer can still be used
        # explicitly by selecting GStreamer and providing a pipeline string.
        candidates = [
            ("v4l2", getattr(cv2, "CAP_V4L2", None)),
            ("auto", getattr(cv2, "CAP_ANY", 0)),
        ]
    else:
        candidates = [("auto", getattr(cv2, "CAP_ANY", 0))]
    return [(name, int(code)) for name, code in candidates if code is not None]


@dataclass
class CameraOpenResult:
    ok: bool
    message: str = ""


class BaseCamera:
    backend_name = "base"

    def open(self) -> CameraOpenResult:
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def is_opened(self) -> bool:
        return False

    def get(self, prop: int) -> float:
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return False

    def description(self) -> str:
        return self.backend_name

    def roi_description(self) -> str:
        return ""


class OpenCVCamera(BaseCamera):
    backend_name = "opencv"

    def __init__(self, source: Any, width: int = 2592, height: int = 1944, fps: float = 30.0, api: str = "auto"):
        self.source = source
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.fps = float(fps or 0.0)
        self.api = normalize_opencv_api(api)
        self.actual_api = ""
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_fourcc = ""
        self.cap: Optional[cv2.VideoCapture] = None

    def _fourcc_str(self, code: float) -> str:
        try:
            c = int(code)
            return "".join(chr((c >> (i * 8)) & 0xFF) for i in range(4)).strip("\x00")
        except Exception:
            return ""

    def _configure_capture(self) -> None:
        if self.cap is None:
            return
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Only request MJPG if the camera reports it can do the requested FPS
        # at the requested resolution. Forcing MJPG on cameras that don't
        # support it properly causes dark/corrupt frames. Try MJPG first; if
        # the resulting FPS drops significantly below the request, fall back to
        # the camera's native format.
        if self.width > 0:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            except Exception:
                pass
        if self.height > 0:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            except Exception:
                pass
        if self.fps > 0:
            try:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            except Exception:
                pass
        # Try MJPG to get higher FPS; revert if the camera doesn't support it
        # (indicated by a large FPS drop after setting the fourcc).
        try:
            fps_before = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            fps_after = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            # If FPS dropped to nearly zero or halved after forcing MJPG,
            # revert to the camera's native format (YUYV/BGR3/etc.).
            if fps_after > 0 and fps_before > 0 and fps_after < fps_before * 0.5:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        except Exception:
            pass
        # Enable auto-exposure so the camera isn't dark on first open.
        # CAP_PROP_AUTO_EXPOSURE: 0.25 = manual, 0.75 = auto (V4L2 convention).
        # Silently ignored on cameras/drivers that don't support it.
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass
        # Read back what the driver actually accepted.
        try:
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.actual_fourcc = self._fourcc_str(self.cap.get(cv2.CAP_PROP_FOURCC))
        except Exception:
            pass

    def _build_gstreamer_pipeline(self) -> str:
        """Build a Jetson-optimised GStreamer pipeline that uses nvjpegdec for
        hardware MJPG decode instead of OpenCV's software libjpeg path.

        This is the recommended backend for USB cameras on Jetson when the
        camera streams MJPG at high resolution (e.g. 2592x1944 @ 30fps),
        because software MJPG decode on ARM cores limits throughput to ~15fps.
        nvjpegdec offloads decode to the Jetson media engine and removes the
        CPU bottleneck entirely.

        Select backend=GStreamer and set the OpenCV source field to "gstreamer"
        (or leave it empty) to activate this pipeline. The device node is
        derived from the numeric camera index (default /dev/video0).
        """
        dev = f"/dev/video{self.source}" if isinstance(self.source, int) else str(self.source or "/dev/video0")
        w = int(self.width or 2592)
        h = int(self.height or 1944)
        fps = int(self.fps or 30)
        return (
            f"v4l2src device={dev} ! "
            f"image/jpeg,width={w},height={h},framerate={fps}/1 ! "
            f"nvjpegdec ! "
            f"video/x-raw ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink max-buffers=1 drop=true sync=false"
        )

    def open(self) -> CameraOpenResult:
        try:
            # GStreamer backend: build a hardware-decode pipeline for Jetson.
            # When the api is 'gstreamer', ignore _auto_opencv_candidates and
            # use nvjpegdec to offload MJPG decode from the ARM cores.
            if self.api == "gstreamer":
                pipeline = self._build_gstreamer_pipeline()
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.actual_api = "gstreamer"
                    try:
                        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
                        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
                        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
                        self.actual_fourcc = "MJPG→BGR"
                    except Exception:
                        pass
                    return CameraOpenResult(True, f"GStreamer (nvjpegdec) pipeline opened {self.actual_width}x{self.actual_height}@{self.actual_fps:.0f}fps")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                return CameraOpenResult(False, f"GStreamer pipeline failed to open. Ensure JetPack GStreamer and nvjpegdec are installed.\nPipeline: {pipeline}")

            if self.api == "auto":
                candidates = _auto_opencv_candidates(self.source)
            else:
                code = _opencv_api_code(self.api)
                if code is None:
                    return CameraOpenResult(False, f"OpenCV API {_opencv_api_label(self.api)} is not available in this OpenCV build.")
                candidates = [(self.api, int(code))]

            tried: list[str] = []
            for api_name, api_code in candidates:
                tried.append(_opencv_api_label(api_name))
                cap = cv2.VideoCapture(self.source, api_code)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.actual_api = api_name
                    self._configure_capture()
                    diag = (
                        f" actual={self.actual_width}x{self.actual_height}"
                        f"@{self.actual_fps:.0f}fps"
                        f" fmt={self.actual_fourcc or 'unknown'}"
                    )
                    return CameraOpenResult(True, f"OpenCV camera opened: {self.source} via {_opencv_api_label(api_name)}.{diag}")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
            self.cap = None
            return CameraOpenResult(False, f"Could not open OpenCV source: {self.source}. Tried: {', '.join(tried) or 'none'}")
        except Exception as exc:
            self.cap = None
            return CameraOpenResult(False, f"OpenCV camera error: {exc}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def is_opened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    def get(self, prop: int) -> float:
        if self.cap is None:
            return 0.0
        try:
            return float(self.cap.get(prop) or 0.0)
        except Exception:
            return 0.0

    def set(self, prop: int, value: float) -> bool:
        if self.cap is None:
            return False
        try:
            return bool(self.cap.set(prop, value))
        except Exception:
            return False

    def description(self) -> str:
        api = _opencv_api_label(self.actual_api or self.api)
        if self.actual_width and self.actual_height:
            return f"OpenCV:{self.source} ({api}) {self.actual_width}x{self.actual_height}@{self.actual_fps:.0f}fps {self.actual_fourcc or ''}".strip()
        return f"OpenCV:{self.source} ({api})"


class BaslerPylonCamera(BaseCamera):
    backend_name = "basler"

    def __init__(
        self,
        serial: str = "",
        width: int = 2592,
        height: int = 1944,
        fps: float = 30.0,
        exposure_us: float = 0.0,
        gain: float = 0.0,
        exposure_auto: bool = False,
        roi_enabled: bool = False,
        roi_offset_x: int = 0,
        roi_offset_y: int = 0,
        roi_width: int = 0,
        roi_height: int = 0,
    ):
        self.serial = str(serial or "").strip()
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.fps = float(fps or 0.0)
        self.exposure_us = float(exposure_us or 0.0)
        self.gain = float(gain or 0.0)
        self.exposure_auto = bool(exposure_auto)
        self.roi_enabled = bool(roi_enabled)
        self.roi_offset_x = int(roi_offset_x or 0)
        self.roi_offset_y = int(roi_offset_y or 0)
        self.roi_width = int(roi_width or 0)
        self.roi_height = int(roi_height or 0)
        self._pylon = None
        self.camera = None
        self.converter = None
        self._actual_width = 0
        self._actual_height = 0
        self._actual_offset_x = 0
        self._actual_offset_y = 0
        self._actual_fps = self.fps
        self._model = ""
        self._serial_opened = ""
        self.last_grab_wait_ms = 0.0
        self.last_convert_ms = 0.0
        self.last_array_ms = 0.0
        self.last_read_total_ms = 0.0
        self._actual_exposure_us = 0.0
        self._actual_exposure_auto = ""
        self._actual_pixel_format = ""
        self._resulting_fps = 0.0
        self._throughput_limit = 0.0
        self._consecutive_grab_failures = 0
        self._grab_restart_count = 0
        self.last_grab_error = ""

    @staticmethod
    def available() -> Tuple[bool, str]:
        try:
            from pypylon import pylon  # noqa: F401
            return True, "pypylon available"
        except Exception as exc:
            return False, str(exc)

    def _node(self, name: str) -> Any:
        cam = self.camera
        if cam is None or not hasattr(cam, name):
            return None
        try:
            return getattr(cam, name)
        except Exception:
            return None

    def _set_node(self, name: str, value: Any) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            if hasattr(node, "SetValue"):
                node.SetValue(value)
                return True
        except Exception:
            return False
        return False

    def _get_node(self, name: str, default: Any = 0) -> Any:
        node = self._node(name)
        if node is None:
            return default
        try:
            if hasattr(node, "GetValue"):
                return node.GetValue()
        except Exception:
            pass
        return default

    def _node_min(self, name: str, default: int = 0) -> int:
        node = self._node(name)
        try:
            if node is not None and hasattr(node, "GetMin"):
                return int(node.GetMin())
        except Exception:
            pass
        return int(default)

    def _node_max(self, name: str, default: int = 0) -> int:
        node = self._node(name)
        try:
            if node is not None and hasattr(node, "GetMax"):
                return int(node.GetMax())
        except Exception:
            pass
        return int(default)

    def _node_inc(self, name: str, default: int = 1) -> int:
        node = self._node(name)
        try:
            if node is not None and hasattr(node, "GetInc"):
                inc = int(node.GetInc())
                return max(1, inc)
        except Exception:
            pass
        return max(1, int(default))

    def _coerce_int_node(self, name: str, value: int, fallback: int = 0) -> int:
        """Clamp an integer GenICam node value to its Min/Max/Inc requirements."""
        try:
            v = int(value)
        except Exception:
            v = int(fallback)
        mn = self._node_min(name, int(fallback))
        mx = self._node_max(name, max(mn, int(fallback)))
        inc = self._node_inc(name, 1)
        v = max(mn, min(mx, v))
        if inc > 1:
            # Align downward to a valid increment from the node minimum. This avoids
            # asking Basler for an invalid ROI value such as 1279 on a camera that
            # requires width increments of 2, 4, 8, etc.
            v = mn + ((v - mn) // inc) * inc
            v = max(mn, min(mx, v))
        return int(v)

    def _set_int_node(self, name: str, value: int, fallback: int = 0) -> int:
        coerced = self._coerce_int_node(name, value, fallback)
        self._set_node(name, coerced)
        try:
            return int(self._get_node(name, coerced))
        except Exception:
            return coerced

    def _configure_roi(self) -> None:
        """Apply Basler Width/Height plus OffsetX/OffsetY safely.

        Basler ROI is a real GenICam sensor crop:
        - Width/Height define the grabbed/output image size.
        - OffsetX/OffsetY move that crop window on the sensor.

        When ROI is disabled, the normal camera Width/Height settings are used
        and offsets are reset to minimum. When ROI is enabled, the dedicated ROI
        Width/Height and Offset fields are used. Offset will only have room to
        move if the ROI size is smaller than the sensor maximum.
        """
        if self.camera is None:
            return

        min_x = self._node_min("OffsetX", 0)
        min_y = self._node_min("OffsetY", 0)

        # Basler/GenICam requires offsets to be reduced before changing Width or
        # Height. Otherwise a smaller/larger ROI can be rejected because the old
        # offset makes the requested rectangle extend beyond the sensor.
        self._set_int_node("OffsetX", min_x, min_x)
        self._set_int_node("OffsetY", min_y, min_y)

        if self.roi_enabled:
            target_w = int(self.roi_width or self.width or 0)
            target_h = int(self.roi_height or self.height or 0)
        else:
            target_w = int(self.width or 0)
            target_h = int(self.height or 0)

        if target_w <= 0:
            target_w = self._node_max("Width", self.width or 0)
        if target_h <= 0:
            target_h = self._node_max("Height", self.height or 0)

        actual_w = self._set_int_node("Width", int(target_w), self.width or target_w)
        actual_h = self._set_int_node("Height", int(target_h), self.height or target_h)

        target_x = self.roi_offset_x if self.roi_enabled else min_x
        target_y = self.roi_offset_y if self.roi_enabled else min_y
        actual_x = self._set_int_node("OffsetX", int(target_x), min_x)
        actual_y = self._set_int_node("OffsetY", int(target_y), min_y)

        self._actual_width = actual_w
        self._actual_height = actual_h
        self._actual_offset_x = actual_x
        self._actual_offset_y = actual_y

    def open(self) -> CameraOpenResult:
        ok, msg = self.available()
        if not ok:
            return CameraOpenResult(False, "Basler backend selected, but pypylon is not installed or not importable: " + msg)
        try:
            from pypylon import pylon
            self._pylon = pylon
            factory = pylon.TlFactory.GetInstance()
            devices = factory.EnumerateDevices()
            if not devices:
                return CameraOpenResult(False, "No Basler cameras found by Pylon.")

            selected = None
            if self.serial:
                for dev in devices:
                    try:
                        if dev.GetSerialNumber() == self.serial:
                            selected = dev
                            break
                    except Exception:
                        pass
                if selected is None:
                    known = []
                    for dev in devices:
                        try:
                            known.append(dev.GetSerialNumber())
                        except Exception:
                            pass
                    return CameraOpenResult(False, f"Basler serial {self.serial!r} not found. Found: {', '.join(known) or 'none'}")
            else:
                selected = devices[0]

            self.camera = pylon.InstantCamera(factory.CreateDevice(selected))
            self.camera.Open()
            try:
                self._model = selected.GetModelName()
            except Exception:
                self._model = "Basler"
            try:
                self._serial_opened = selected.GetSerialNumber()
            except Exception:
                self._serial_opened = self.serial

            self._configure_roi()

            if self.fps > 0:
                self._set_node("AcquisitionFrameRateEnable", True)
                self._set_node("AcquisitionFrameRate", float(self.fps))
            # Exposure mode: Auto enables the camera's continuous exposure loop when supported.
            # Manual turns auto exposure off and applies the requested exposure time in microseconds.
            if self.exposure_auto:
                self._set_node("ExposureAuto", "Continuous")
            else:
                self._set_node("ExposureAuto", "Off")
                if self.exposure_us > 0:
                    self._set_node("ExposureTime", float(self.exposure_us))
            if self.gain > 0:
                self._set_node("GainAuto", "Off")
                self._set_node("Gain", float(self.gain))

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._actual_width = int(self._get_node("Width", self._actual_width or self.width))
            self._actual_height = int(self._get_node("Height", self._actual_height or self.height))
            self._actual_offset_x = int(self._get_node("OffsetX", self._actual_offset_x))
            self._actual_offset_y = int(self._get_node("OffsetY", self._actual_offset_y))
            self._actual_fps = float(self._get_node("AcquisitionFrameRate", self.fps))
            try:
                self._actual_exposure_us = float(self._get_node("ExposureTime", 0.0) or 0.0)
            except Exception:
                self._actual_exposure_us = 0.0
            try:
                self._actual_exposure_auto = str(self._get_node("ExposureAuto", ""))
            except Exception:
                self._actual_exposure_auto = ""
            try:
                self._actual_pixel_format = str(self._get_node("PixelFormat", ""))
            except Exception:
                self._actual_pixel_format = ""
            try:
                self._resulting_fps = float(
                    self._get_node("ResultingAcquisitionFrameRate", self._get_node("ResultingFrameRate", 0.0)) or 0.0
                )
            except Exception:
                self._resulting_fps = 0.0
            try:
                self._throughput_limit = float(self._get_node("DeviceLinkThroughputLimit", 0.0) or 0.0)
            except Exception:
                self._throughput_limit = 0.0
            roi_msg = f" ROI X={self._actual_offset_x} Y={self._actual_offset_y} W={self._actual_width} H={self._actual_height}"
            diag = (
                f" exposure={self._actual_exposure_us:.0f}us auto={self._actual_exposure_auto or 'unknown'}"
                f" pixel={self._actual_pixel_format or 'unknown'}"
                f" resulting_fps={self._resulting_fps:.2f}"
                f" link_limit={self._throughput_limit:.0f}"
            )
            return CameraOpenResult(True, f"Basler camera opened: {self._model} SN {self._serial_opened}.{roi_msg}{diag}")
        except Exception as exc:
            self.release()
            return CameraOpenResult(False, f"Basler/Pylon camera error: {exc}")

    def _try_restart_grabbing(self) -> bool:
        """Attempt to recover after a grab stall or USB disconnect.

        A simple StartGrabbing() call is sufficient for soft stalls. For USB
        removal events ('Device has been removed') the camera object is invalid
        and a full re-open (release + open) is required instead.
        """
        # First try the cheap path: just restart the grab loop.
        try:
            if self.camera is not None and self.camera.IsOpen():
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)
                self._grab_restart_count += 1
                self.last_grab_error = f"grab restarted (restart #{self._grab_restart_count})"
                return True
        except Exception:
            pass

        # Cheap path failed — device likely disconnected. Release fully and
        # re-open (re-enumerates USB/GigE, creates a new InstantCamera).
        try:
            self.release()
        except Exception:
            pass
        try:
            result = self.open()
            if result.ok:
                self._grab_restart_count += 1
                self.last_grab_error = f"device re-opened after disconnect (restart #{self._grab_restart_count})"
                return True
            else:
                self.last_grab_error = f"device re-open failed: {result.message}"
                return False
        except Exception as exc:
            self.last_grab_error = f"device re-open exception: {exc}"
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.camera is None or self.converter is None:
            return False, None
        import time
        # If the grab loop has stopped (e.g. Basler driver event, GigE timeout),
        # restart it automatically rather than returning failed frames indefinitely.
        try:
            if not self.camera.IsGrabbing():
                self._consecutive_grab_failures += 1
                self.last_grab_error = f"IsGrabbing=False (failure #{self._consecutive_grab_failures})"
                self._try_restart_grabbing()
                return False, None
        except Exception:
            pass
        t0 = time.perf_counter()
        grab = None
        try:
            grab = self.camera.RetrieveResult(1000, self._pylon.TimeoutHandling_Return)
            t1 = time.perf_counter()
            self.last_grab_wait_ms = (t1 - t0) * 1000.0
            if grab is None or not grab.GrabSucceeded():
                try:
                    if grab is not None:
                        err_code = grab.GetErrorCode() if hasattr(grab, "GetErrorCode") else 0
                        err_desc = grab.GetErrorDescription() if hasattr(grab, "GetErrorDescription") else ""
                        self.last_grab_error = f"GrabFailed code={err_code} desc={err_desc!r}"
                        grab.Release()
                except Exception:
                    pass
                self._consecutive_grab_failures += 1
                self.last_convert_ms = 0.0
                self.last_array_ms = 0.0
                self.last_read_total_ms = (time.perf_counter() - t0) * 1000.0
                # After repeated consecutive failures, attempt to restart the grab loop.
                if self._consecutive_grab_failures >= 5:
                    self._try_restart_grabbing()
                    self._consecutive_grab_failures = 0
                return False, None
            self._consecutive_grab_failures = 0
            self.last_grab_error = ""
            t2 = time.perf_counter()
            img = self.converter.Convert(grab)
            t3 = time.perf_counter()
            arr = img.GetArray()
            t4 = time.perf_counter()
            self.last_convert_ms = (t3 - t2) * 1000.0
            self.last_array_ms = (t4 - t3) * 1000.0
            self.last_read_total_ms = (t4 - t0) * 1000.0
            try:
                grab.Release()
            except Exception:
                pass
            return True, arr
        except Exception as exc:
            try:
                if grab is not None:
                    grab.Release()
            except Exception:
                pass
            self._consecutive_grab_failures += 1
            self.last_grab_error = f"read exception: {exc}"
            self.last_read_total_ms = (time.perf_counter() - t0) * 1000.0
            if self._consecutive_grab_failures >= 5:
                self._try_restart_grabbing()
                self._consecutive_grab_failures = 0
            return False, None

    def release(self) -> None:
        try:
            if self.camera is not None and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except Exception:
            pass
        try:
            if self.camera is not None and self.camera.IsOpen():
                self.camera.Close()
        except Exception:
            pass
        self.camera = None
        self.converter = None

    def is_opened(self) -> bool:
        try:
            return bool(self.camera is not None and self.camera.IsOpen())
        except Exception:
            return False

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._actual_width or self._get_node("Width", 0))
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._actual_height or self._get_node("Height", 0))
        if prop == cv2.CAP_PROP_FPS:
            return float(self._actual_fps or self._get_node("AcquisitionFrameRate", 0))
        if prop == cv2.CAP_PROP_FOURCC:
            return 0.0
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        try:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self._actual_width = self._set_int_node("Width", int(value), self._actual_width or self.width)
                return True
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self._actual_height = self._set_int_node("Height", int(value), self._actual_height or self.height)
                return True
            if prop == cv2.CAP_PROP_FPS:
                self._set_node("AcquisitionFrameRateEnable", True)
                self._set_node("AcquisitionFrameRate", float(value))
                self._actual_fps = float(self._get_node("AcquisitionFrameRate", value))
                return True
        except Exception:
            return False
        return False

    def roi_description(self) -> str:
        if self._actual_width or self._actual_height:
            return f"ROI X={self._actual_offset_x} Y={self._actual_offset_y} W={self._actual_width} H={self._actual_height}"
        return ""

    def description(self) -> str:
        if self._model or self._serial_opened:
            return f"Basler:{self._model} SN {self._serial_opened}".strip()
        return "Basler/Pylon"


def create_camera_backend(
    backend: str,
    source_text: str = "0",
    basler_serial: str = "",
    width: int = 2592,
    height: int = 1944,
    fps: float = 30.0,
    exposure_us: float = 0.0,
    gain: float = 0.0,
    exposure_auto: bool = False,
    opencv_api: str = "auto",
    basler_roi_enabled: bool = False,
    basler_roi_offset_x: int = 0,
    basler_roi_offset_y: int = 0,
    basler_roi_width: int = 0,
    basler_roi_height: int = 0,
) -> BaseCamera:
    name = str(backend or "opencv").strip().lower()
    if name in ("basler", "pylon", "basler/pylon"):
        return BaslerPylonCamera(
            basler_serial,
            width,
            height,
            fps,
            exposure_us,
            gain,
            exposure_auto,
            basler_roi_enabled,
            basler_roi_offset_x,
            basler_roi_offset_y,
            basler_roi_width,
            basler_roi_height,
        )
    return OpenCVCamera(parse_source(source_text), width, height, fps, api=opencv_api)


def list_basler_cameras() -> list[dict[str, str]]:
    ok, _msg = BaslerPylonCamera.available()
    if not ok:
        return []
    try:
        from pypylon import pylon
        factory = pylon.TlFactory.GetInstance()
        out = []
        for dev in factory.EnumerateDevices():
            item = {}
            for key, getter in (("model", "GetModelName"), ("serial", "GetSerialNumber"), ("name", "GetFriendlyName")):
                try:
                    item[key] = str(getattr(dev, getter)())
                except Exception:
                    item[key] = ""
            out.append(item)
        return out
    except Exception:
        return []
