---
name: record-scan
description: >-
  Record an RGBD turntable scan video inside the vision_comp Docker container.
  Use when the user says "record video", "record scan", or provides
  "object name" and "description" for a scan recording.
---

# Record Scan

Record an RGBD turntable scan using the `record-scan` entry point inside
the `vision_comp` container.

## Prerequisites

The vision_comp image must be built. If not already built, build it first:

```bash
cd /home/lq/lqtech/dockers/monty_isaac/docker_all && docker compose build vision_comp
```

## Workflow

### 1. Parse user input

Extract from the user's message:

| Parameter     | Required | Source                         |
|---------------|----------|--------------------------------|
| `object_name` | yes      | `object name:xxx` or `--object`|
| `description` | no       | `description:xxx`              |
| `camera`      | no       | default `orbbec`; alt `realsense` |
| `duration`    | no       | seconds; default 15 (script auto-stops, ignores Ctrl+C) |
| `no_depth`    | no       | if user says "no depth" or "rgb only" |

### 2. Run the recording command

The Orbbec (and RealSense) camera needs **USB device access** inside the
container. `docker compose run` does not support `--privileged`, so use
**`docker run`** with privileged mode and device/udev mounts.

Run the following command, replacing `<object_name>`, `<description>`, and optionally `<duration>` (default 15) with the values from the user (image name must be `monty_vision_comp:latest`). The script runs for N seconds then auto-stops and saves; **Ctrl+C is ignored**.

```bash
cd /home/lq/lqtech/dockers/monty_isaac/docker_all && docker run --rm --init --privileged \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -v "$(pwd)/data/monty:/data" \
  -e SCAN_DIR=/data/scans \
  monty_vision_comp:latest \
  record-scan \
    --object <object_name> \
    --description "<description>" \
    --no-preview \
    --scan-dir /data/scans \
    --duration <duration>
```

Use `--duration 15` if the user does not specify a duration. Omit `-it` so the command works when run by the agent (no TTY); the process exits when the duration elapses and data is written to disk.

**Privilege and device mounts:**

- `--privileged` — required so the container can open the USB camera
  (otherwise you get "No device found" or "usbEnumerator openUsbDevice failed!").
- `-v /dev:/dev` — expose host devices (including the camera).
- `-v /run/udev:/run/udev:ro` — udev data so the SDK can discover the device.

Without these, the script will fail to open the Orbbec/RealSense device.

**Behaviour:** `record-scan` runs for exactly `--duration` seconds (default 15), then stops and saves video, depth, and metadata. Ctrl+C and SIGTERM are ignored during recording so the run always completes and flushes to disk.

Key flags (record-scan):

- `--no-preview` — always pass this; the container has no GUI.
- `--scan-dir /data/scans` — writes into the `/data` volume mount
  (maps to `docker_all/data/monty/scans/` on host by default).
- `--duration N` — run for N seconds then auto-stop (default: 15). Pass explicitly when invoking (e.g. `--duration 15`).
- `--camera realsense` — only if user requests RealSense.
- `--no-depth` — only if user explicitly asks for no depth.

The `--rm` flag removes the one-off container after it exits.

When running from an agent/IDE: run the command **without** `-it`. Use a timeout long enough for warmup + duration + write (e.g. duration + 45 seconds). The process exits on its own when the duration ends.

### 3. Report results

After the recording finishes, tell the user:

- Output directory on host: `docker_all/data/monty/scans/<object_name>/` (full path: `/home/lq/lqtech/dockers/monty_isaac/docker_all/data/monty/scans/<object_name>/`)
- Files produced: `video.mp4`, `video_annotated.mp4`, `metadata.json`,
  `depth/`, `depth_color/`, `depth_float/`, `pointcloud/`
- Depth quality summary (from the command's stdout)

### Example

User says: "record video: object=yellow_block_with_sticker, description=yellow block with stickers on it"

Run:

```bash
cd /home/lq/lqtech/dockers/monty_isaac/docker_all && docker run --rm --init --privileged \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -v "$(pwd)/data/monty:/data" \
  -e SCAN_DIR=/data/scans \
  monty_vision_comp:latest \
  record-scan \
    --object yellow_block_with_sticker \
    --description "yellow block with stickers on it" \
    --no-preview \
    --scan-dir /data/scans \
    --duration 15
```

Host output at: `/home/lq/lqtech/dockers/monty_isaac/docker_all/data/monty/scans/yellow_block_with_sticker/`. Recording runs for 15 seconds then auto-stops and saves.
