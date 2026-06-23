# BungVision — Project Notes

## Memory: Jetson USB camera fixes (apply the same to BungLabel Studio)

BungLabel Studio will need the **same camera fixes** that were applied to
BungVision in v0.9.93–v0.9.97. When working on BungLabel Studio, port these:

### 1. Jetson hardware MJPG decode (FPS fix)
- **Symptom:** USB camera caps at ~15fps at full resolution (e.g. 2592x1944)
  even though V4L2 reports 30fps.
- **Cause:** OpenCV's software MJPG decode (libjpeg) on the ARM cores takes
  ~30–40ms/frame. Confirmed via `cv2.getBuildInformation()` showing
  `GStreamer: NO` — the pip `opencv-python` wheel has no GStreamer support, so
  `cv2.VideoCapture(..., cv2.CAP_GSTREAMER)` can never work on these systems.
- **Fix:** A native GStreamer backend (`GstNativeCamera` in `camera_backend.py`)
  drives GStreamer directly via `python-gi` (PyGObject) + `appsink`, bypassing
  OpenCV's build flags. It tries hardware-decode pipelines in order:
  `nvv4l2decoder mjpeg=1 ! nvvidconv` → `nvjpegdec ! nvvidconv` →
  `nvjpegdec ! videoconvert` → CPU `jpegdec`. Uses the first that reaches
  PLAYING and delivers a sample. **Result: 15fps → 30fps.**
- **Deps:** `sudo apt install python3-gi gir1.2-gstreamer-1.0` (usually present
  on JetPack). Hardware decode plugins verified with:
  `gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=2592,height=1944,framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! fakesink`

### 2. Exposure for USB/GStreamer cameras
- **Symptom:** Image very dark with the GStreamer backend.
- **Cause:** Manual exposure settings were Basler-scoped (default 5000µs) and
  got applied to the USB camera as a fixed, very short exposure.
- **Fix:** Apply exposure/gain via `v4l2-ctl` on the device node (GStreamer
  `v4l2src` does not expose UVC exposure reliably). Try both new
  (`auto_exposure`, `exposure_time_absolute`) and old (`exposure_auto`,
  `exposure_absolute`) UVC control names. Default non-Basler backends to
  **auto-exposure**. exposure_us → V4L2 100µs units. Requires `v4l-utils`.

### 3. Basler USB autosuspend disconnect
- **Symptom:** Basler camera stops after ~1 minute; log shows
  `'Device has been removed from the PC.'`
- **Cause:** Jetson USB autosuspend powers down the port mid-stream.
- **System fix:**
  ```bash
  echo 'options usbcore autosuspend=-1' | sudo tee /etc/modprobe.d/usbcore.conf
  sudo update-initramfs -u && sudo reboot
  ```
- **Code fix:** `BaslerPylonCamera._try_restart_grabbing()` does a full
  `release()` + `open()` (USB re-enumeration) when the cheap `StartGrabbing()`
  path fails after a disconnect. Grab errors surfaced via `last_grab_error`.

## Workflow notes
- Develop on branch `claude/gracious-fermat-qhho2x`; open/update a PR after pushing.
- `APP_TITLE` version string is bumped on each change.
- The file/debug log (`logs/bungvision_debug.log`) filters out normal runtime
  chatter via `_should_file_log`; only errors, exceptions, and `MODEL_LOADED`
  are persisted. Add explicit `_write_debug_log(...)` for new diagnostics.
