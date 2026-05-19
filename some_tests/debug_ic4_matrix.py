import argparse
import itertools
import statistics
import sys
import time
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mokap.core.manager import MultiCam
from mokap.utils.fileio import read_config


def _fmt_ms(value: float) -> str:
    return f"{value:.3f} ms"


def _parse_list(value: str | None, cast=float):
    if not value:
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _parse_roi_list(value: str | None):
    if not value:
        return []
    rois = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) not in (2, 4):
            raise ValueError("ROI entries must be x,y,w,h or w,h")
        rois.append([int(part) for part in parts])
    return rois


def _safe_get_feature(camera, name: str):
    try:
        return camera._get_feature_value(name)
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _dump_roll_shutter_related(camera):
    names = [
        "TriggerActivation",
        "TriggerMode",
        "AcquisitionMode",
        "AcquisitionFrameRate",
        "ExposureTime",
        "ExposureAuto",
        "FrameRate",
        "ResultingFrameRate",
        "SensorReadoutMode",
        "ShutterMode",
        "Shutter",
        "ReadoutTime",
        "LineDebouncerTime",
    ]
    print("Potential timing/shutter features:")
    for name in names:
        print(f"  {name:20s} = {_safe_get_feature(camera, name)}")


def _run_probe(manager: MultiCam, frames: int, timeout_ms: int):
    host_times = []
    device_times = []
    dropped = 0

    deadline = time.monotonic() + max(2.0, timeout_ms / 1000.0 * frames)
    last_frame_number = None

    while len(host_times) < frames and time.monotonic() < deadline:
        latest = manager._latest_frames[0]
        if latest is None:
            time.sleep(0.005)
            continue

        frame, meta = latest
        frame_number = meta.get("frame_number")
        if frame_number == last_frame_number:
            time.sleep(0.002)
            continue

        last_frame_number = frame_number
        host_times.append(time.perf_counter())
        device_times.append(meta.get("timestamp"))
        print(
            f"  frame {len(host_times) - 1:03d}: frame_number={frame_number} device_ts={meta.get('timestamp')}"
        )

    host_deltas = []
    device_deltas = []
    if len(host_times) >= 2:
        host_deltas = [(host_times[i] - host_times[i - 1]) * 1000.0 for i in range(1, len(host_times))]
        print()
        print("Host arrival intervals:")
        print(
            f"  min={_fmt_ms(min(host_deltas))} mean={_fmt_ms(statistics.fmean(host_deltas))} "
            f"max={_fmt_ms(max(host_deltas))}"
        )
        if len(host_deltas) > 1:
            print(f"  stdev={_fmt_ms(statistics.pstdev(host_deltas))}")

    valid_device_times = [ts for ts in device_times if isinstance(ts, (int, float))]
    if len(valid_device_times) >= 2:
        device_deltas = [
            (valid_device_times[i] - valid_device_times[i - 1]) / 1_000_000.0
            for i in range(1, len(valid_device_times))
        ]
        print()
        print("Camera timestamp intervals:")
        print(
            f"  min={_fmt_ms(min(device_deltas))} mean={_fmt_ms(statistics.fmean(device_deltas))} "
            f"max={_fmt_ms(max(device_deltas))}"
        )

    print()
    print(f"Captured {len(host_times)} frame(s), {dropped} timeout(s).")
    if len(host_times) >= 2:
        effective_fps = 1000.0 / statistics.fmean(
            [(host_times[i] - host_times[i - 1]) * 1000.0 for i in range(1, len(host_times))]
        )
        print(f"Effective host-side FPS: {effective_fps:.2f}")

    return len(host_times), dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep IC4 settings and measure capture timing.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--serial", default=None, help="Camera serial to target")
    parser.add_argument("--frames", type=int, default=12, help="Frames to measure per setting")
    parser.add_argument("--timeout-ms", type=int, default=2000, help="Per-frame grab timeout")
    parser.add_argument("--sweep-framerate", default=None, help="Comma-separated framerate sweep in Hz")
    parser.add_argument("--sweep-exposure-us", default=None, help="Comma-separated exposure sweep in microseconds")
    parser.add_argument(
        "--sweep-roi",
        default=None,
        help="Semicolon-separated ROI sweep, each item x,y,w,h or w,h",
    )
    parser.add_argument(
        "--hardware-trigger",
        action="store_true",
        help="Force hardware trigger on (requires an active external trigger source; this script does not start one)",
    )
    parser.add_argument("--no-hardware-trigger", action="store_true", help="Force hardware trigger off")
    args = parser.parse_args()

    # Load .env if present so RaspberryTrigger can read SSH credentials
    try:
        load_dotenv()
    except Exception:
        pass

    if args.hardware_trigger and args.no_hardware_trigger:
        parser.error("Choose only one of --hardware-trigger or --no-hardware-trigger")

    config = read_config(args.config)
    if args.hardware_trigger:
        config["hardware_trigger"] = True
    elif args.no_hardware_trigger:
        config["hardware_trigger"] = False
    else:
        # Respect the config.yaml value. Do not silently default to free-run;
        # if hardware trigger is enabled but cannot be connected, abort later.
        pass

    framerate_values = _parse_list(args.sweep_framerate, float) or [float(config.get("framerate", 60.0))]
    exposure_values = _parse_list(args.sweep_exposure_us, float) or [float(config.get("exposure", 5000.0))]
    roi_values = _parse_roi_list(args.sweep_roi) or [config.get("roi")]
    roi_values = [roi for roi in roi_values if roi]
    if not roi_values:
        roi_values = [None]

    print(f"Loaded config: {Path(args.config).resolve() if Path(args.config).exists() else args.config}")
    print(f"Base framerate: {config.get('framerate')}")
    print(f"Base exposure: {config.get('exposure')}")
    print(f"Base hardware_trigger: {config.get('hardware_trigger')}")
    print(f"Base trigger: {config.get('trigger', {})}")
    print()

    if config.get("hardware_trigger", False):
        print("[INFO] Hardware-trigger mode is enabled; the manager will start the trigger when acquisition starts.")
    else:
        print("[INFO] Running in free-run mode.")
    print()

    manager = MultiCam(config=config)
    if manager.nb_cameras == 0:
        print("No cameras found.")
        return 1

    # If we requested hardware-trigger-only, ensure the trigger is connected.
    if config.get("hardware_trigger", False):
        if not manager.hardware_triggered:
            print("[ERROR] Hardware trigger requested but no trigger connected (check .env and network). Aborting.")
            try:
                manager.disconnect_cameras()
            except Exception:
                pass
            return 2

    if args.serial:
        camera = next((cam for cam in manager.cameras if cam.unique_id == args.serial), None)
    else:
        camera = next((cam for cam in manager.cameras if cam.name or cam.unique_id), None)

    if camera is None:
        print("Could not select an IC Imaging camera instance.")
        return 1

    print(f"Probing camera: serial={camera.unique_id} name={camera.name}")

    try:
        print("Camera properties after connect:")
        print(f"  hardware_triggered = {camera.hardware_triggered}")
        print(f"  exposure           = {camera.exposure}")
        print(f"  framerate          = {camera.framerate}")
        print(f"  pixel_format       = {camera.pixel_format}")
        print(f"  roi                = {camera.roi}")
        print(f"  AcquisitionMode    = {_safe_get_feature(camera, 'AcquisitionMode')}")
        print(f"  TriggerMode        = {_safe_get_feature(camera, 'TriggerMode')}")
        print(f"  TriggerActivation  = {_safe_get_feature(camera, 'TriggerActivation')}")
        print(f"  ResultingFrameRate = {_safe_get_feature(camera, 'ResultingFrameRate')}")
        print(f"  AcquisitionFrameRate = {_safe_get_feature(camera, 'AcquisitionFrameRate')}")
        print()
        _dump_roll_shutter_related(camera)
        print()

        combos = list(itertools.product(framerate_values, exposure_values, roi_values))
        for idx, (framerate, exposure, roi) in enumerate(combos, start=1):
            print(f"=== Case {idx}/{len(combos)}: framerate={framerate:.2f} Hz exposure={exposure:.1f} us roi={roi} ===")

            try:
                manager.framerate = framerate
            except Exception as exc:
                print(f"  framerate set failed: {exc}")

            try:
                camera.exposure = exposure
            except Exception as exc:
                print(f"  exposure set failed: {exc}")

            if roi is not None:
                try:
                    camera.roi = roi
                except Exception as exc:
                    print(f"  ROI set failed: {exc}")

            print(f"  actual exposure = {camera.exposure}")
            print(f"  actual framerate = {camera.framerate}")
            print(f"  actual roi = {camera.roi}")
            manager.start_acquisition()
            time.sleep(0.5)
            _run_probe(manager, args.frames, args.timeout_ms)
            try:
                manager.stop_acquisition()
            except Exception:
                pass
            print()

        return 0

    finally:
        try:
            manager.stop_acquisition()
        except Exception:
            pass
        try:
            manager.disconnect_cameras()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())