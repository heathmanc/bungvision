# BungVision Python HMI v0.9.87 - Overlay Pixmap Cache

Baseline: v0.9.86 Pull-Based Inference Loop, based on the v0.9.74 out-of-band stop watchdog production foundation.

## Changes in v0.9.87

- Cached the detection/badge overlay as a transparent `QPixmap` inside `CameraWidget`. Qt fires `paintEvent` on every window expose, resize, and focus event — not just the 30 Hz inference timer — and the prior code re-ran all Python coordinate loops (OBB vertex scaling, font metric queries, badge placement math) on every repaint even when the `InspectionResult` had not changed. The overlay pixmap is now rebuilt only when a new result arrives or the display area changes; every other `paintEvent` is two `drawPixmap` calls with no Python iteration.
- Added `_build_overlay_pixmap(target)` to `CameraWidget`: renders detection boxes, OBB polygons, detection labels, PASS/FAIL/WAIT grade badges, and the FAIL banner into a target-sized transparent pixmap with target-relative coordinates.
- Cache keyed on `(id(result), target_width, target_height)`. Invalidated on overlay option changes (`set_overlay_options`) so toggling boxes/labels/grades takes effect immediately.
- Painting change only; no inference, grading, tracking, PLC, camera, TensorRT, model, or image-size behavior changed.

## Changes in v0.9.86

- Decoupled inference scheduling from the 30 Hz UI timer. The inference worker now pulls the newest camera frame on its own thread as soon as it finishes a prediction, instead of waiting for the UI timer to push one. This removes up to one UI-tick (~30 ms) of idle time between predictions and improves inference throughput and result freshness, especially when inference is faster than the timer.
- The UI thread now only enables/disables the worker (based on running + model-loaded state) and reads the latest result for display; it no longer hands frames to the worker each tick.
- Preserved latest-frame semantics (no frame backlog), skipped-frame accounting, the busy/stall diagnostics, and the stale-result guard. A freshly opened camera resets the worker's last-inferred sequence so its restarted counter is not mistaken for already-seen frames.
- Demo mode and operator Stop disable the worker and detach the frame source so it does not pull from a stopped or synthetic source.
- Scheduling change only; detection outputs, grading, tracking, PLC, camera, TensorRT, model, and image-size behavior are unchanged.

## Changes in v0.9.85

- Moved the operator preview downscale (full-resolution cv2.resize + BGR->RGB convert) off the Qt UI thread and into the inference worker for the production overlay-synced display path. The UI thread now only wraps the pre-scaled buffer in a QImage/QPixmap, which keeps the operator screen responsive while inference runs.
- The UI publishes its current preview content size to the inference worker each tick, so the pre-scaled buffer matches the display area; window resizes self-correct within a frame via a UI-thread fallback rescale.
- The direct-camera display path (before the first inference result, or when overlays are off) still scales on the UI thread as a fallback.
- Display-only optimization; the full-resolution frame is unchanged and still used for YOLO, tracking, PASS/FAIL, saves, and PLC logic. No grading, tracking, PLC, camera, TensorRT, model, or image-size behavior changed.

## Changes in v0.9.84

- Parse YOLO results by pulling all boxes/confidences/classes from each result in a single CPU transfer instead of one `.detach().cpu().numpy()` per detection. This removes repeated GPU->CPU syncs in the inference parse step for both OBB and detect outputs.
- Removed a redundant `QImage.copy()` in the preview pixmap build; `QPixmap.fromImage()` already copies the pixels, so the extra buffer copy was wasted work each preview update.
- Throttled PLC output submission and operator card/pill refreshes to ~10 Hz instead of the full 30 Hz UI timer rate. The PLC writer remains async with its own cadence and heartbeat, and reset pulses are still sent immediately and latched by the writer.
- Parsing/throttling change only; detection outputs, grading, tracking, PLC tag semantics, camera, TensorRT, model, and image-size behavior are unchanged.

## Changes in v0.9.83

- Warm up the model on the operator's selected device/image size during the background load ("Loading...") so the inference backend is initialized once instead of re-initializing on the first live frame. This removes the apparent "model loads twice" behavior and the first-frame stall at run start.
- Cache the operator preview pixmap so a held frame is not re-scaled/re-encoded on the Qt UI thread every timer tick during the overlay-sync hold window, improving UI responsiveness.
- Updated the title bar / footer revision string to reflect this build.
- Visual/runtime-housekeeping change only; no grading, tracking, PLC, camera, TensorRT, model, or image-size behavior was changed.

## Changes in v0.9.82

- Moved PASS/FAIL/WAIT battery badges from below the battery footprint to inside the displayed battery OBB/box bounds.
- Badges now stay visually attached to the battery they belong to when multiple batteries are visible at the same time.
- Live overlay badges use a compact two-line layout with ID/status on the first line and bung count on the second line.
- Annotated PASS/FAIL snapshot overlays use the same inside-battery badge placement strategy.
- Kept badge placement as a visual overlay-only change; no grading, tracking, PLC, camera, TensorRT, model, or image-size behavior was changed.

## Changes in v0.9.81

- Removed the visible top menu bar from the production operator screen.
- Kept Esc and F12 emergency stop keyboard shortcuts even though the menu is hidden.
- Fixed Bypass confirmation behavior so selecting No/Cancel leaves Bypass OFF and the next enable attempt asks for confirmation again.
- Bypass now remains visually/logically OFF while the confirmation dialog is open, and only turns ON after an explicit Yes.
- No inference, PLC tag, camera, TensorRT, tracking, inspection, or image-size behavior was changed.

## Changes in v0.9.80

- Replaced the image-based checkbox check mark from v0.9.79 with a Qt-drawn checkbox indicator.
  - Checked boxes now show a plain black **X** inside a bright white outlined square.
  - The indicator is painted directly with `QPainter`, so it does not depend on a PNG asset, Qt stylesheet image loading, or the platform gray checkbox theme.
  - Disabled and hover states remain high contrast.
- Removed the no-longer-needed checkbox PNG asset from the package.
- Kept the popup/dialog readability improvements from v0.9.79.
- No inference architecture changes.
- No PLC, tracking, camera, model, TensorRT, or inspection-logic changes.

## Changes in v0.9.79

- Improved popup dialog readability for warnings, confirmations, preflight checks, PLC tests, import/export messages, and model/camera notices.
  - Popups use the same dark industrial background.
  - Text is brighter and heavier.
  - Dialog buttons have clearer blue styling and larger click targets.
- v0.9.79 attempted image-based checkbox check marks; v0.9.80 replaces that approach with direct Qt painting.
- No inference architecture changes.
- No PLC, tracking, camera, model, TensorRT, or inspection-logic changes.

## Changes in v0.9.78

- Persisted the Settings dialog inspection-tuning values:
  - **Track IoU %** is saved as `track_match_iou_percent`.
  - **Locked IoU %** is saved as `committed_track_iou_percent`.
  - Both values are restored on startup and when importing a config.
- Kept the default YOLO image size at **736**.
  - This is intentional for this build.
  - If using a TensorRT `.engine`, make sure the HMI image size matches the engine export size unless the engine was exported dynamic.
- Updated production install/run notes for Jetson, TensorRT export, Basler/Pylon, PLC/pylogix, USB topology, and out-of-band Stop.
- No inference architecture changes.
- No new operator-screen diagnostic controls.

## Preserved from v0.9.81 / v0.9.80 / v0.9.79 / v0.9.78

- Normal in-process inference path.
- TensorRT OBB task/class handling.
- Basler ROI and camera settings.
- PASS/FAIL, tracking, and PLC behavior.
- Out-of-band stop watchdog from v0.9.74.
- Lean visible performance UI from v0.9.75:
  - inference FPS
  - inference ms
- Production-quiet file log behavior:
  - errors/failures/exceptions
  - confirmed `MODEL_LOADED` events only
- Removed production operator controls remain removed:
  - Preview Only
  - Pause Inference
  - software FPS cap UI
  - Jetson guard UI

## Recommended Jetson run

```bash
cd /home/enersys/bungvision_env
source bin/activate
python3 main.py
```

## Out-of-band Stop

Use this if the GUI input path stalls or the operator screen is not responding:

```bash
cd /home/enersys/bungvision_env
./request_stop.sh
```

Equivalent direct trigger:

```bash
touch /home/enersys/bungvision_env/runtime_stop.flag
```

The app writes an acknowledgement file at:

```text
/home/enersys/bungvision_env/runtime_stop_ack.txt
```

## TensorRT engine guidance

For Jetson production, export the TensorRT `.engine` on the same Jetson software stack that will run BungVision. Do not export on Windows/RTX or a different JetPack/TensorRT stack and copy the engine to the Jetson for production.

Example local export command using the current build's default image size:

```bash
yolo export model=/home/enersys/bungvision_env/best.pt \
  format=engine \
  imgsz=736 \
  half=True \
  batch=1 \
  dynamic=False \
  workspace=1 \
  device=0
```

If you intentionally export at another size, set **Settings → Runtime → Image Size** to that same value before running the engine.

## Basler / Pylon notes

Native Basler support requires the Basler Pylon runtime and `pypylon` in the active Python environment. The OpenCV backend remains available as a fallback.

Recommended USB topology check after choosing a stable port:

```bash
lsusb -t
```

For the current Jetson Orin NX / Basler USB3 setup, keep the camera on a known-good USB3 path and avoid sharing the same USB root hub with mouse/keyboard/HID devices where practical.

## PLC / pylogix notes

PLC support uses `pylogix` for Allen-Bradley / CompactLogix-style communication. Install it in the same virtual environment if PLC control is enabled:

```bash
pip install pylogix
```

PLC tag behavior is production-sensitive. Do not change tag names, Ready semantics, bypass behavior, reset behavior, heartbeat behavior, or stop/reject behavior unless the change is intentional and tested.

## Settings backup / restore

Runtime settings are stored in:

```text
config/settings.json
```

The HMI menu also includes Export Config and Import Config actions. Before changing line-side settings, export a known-good config copy.

## Validation before packaging

Before distributing a patched build, run:

```bash
python3 -m py_compile main.py camera_backend.py
unzip -t <package>.zip
```
