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
cd docker_all && docker compose build vision_comp
```

## Workflow

### 1. Parse user input

Extract from the user's message:

| Parameter     | Required | Source                         |
|---------------|----------|--------------------------------|
| `object_name` | yes      | `object name:xxx` or `--object`|
| `description` | no       | `description:xxx`              |
| `camera`      | no       | default `orbbec`; alt `realsense` |
| `duration`    | no       | seconds; omit for manual stop  |
| `no_depth`    | no       | if user says "no depth" or "rgb only" |

### 2. Run the recording command

The Orbbec (and RealSense) camera needs **USB device access** inside the
container. `docker compose run` does not support `--privileged`, so use
**`docker run`** with privileged mode and device/udev mounts.

From the `docker_all/` directory, use this **exact** form (image name must be
`monty_vision_comp:latest`; see compose for the built image):

```bash
cd docker_all && docker run --rm -it --init --privileged \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -v "$(pwd)/data/monty:/data" \
  -e SCAN_DIR=/data/scans \
  monty_vision_comp:latest \
  record-scan \
    --object <object_name> \
    --description "<description>" \
    --no-preview \
    --scan-dir /data/scans
```

**Privilege and device mounts:**

- `--privileged` — required so the container can open the USB camera
  (otherwise you get "No device found" or "usbEnumerator openUsbDevice failed!").
- `-v /dev:/dev` — expose host devices (including the camera).
- `-v /run/udev:/run/udev:ro` — udev data so the SDK can discover the device.

Without these, the script will fail to open the Orbbec/RealSense device.

**Ctrl+C and stopping (use `-it` and `--init`):**

- **`-it`** — allocates a TTY and keeps stdin open so the terminal is attached.
  Without it, Ctrl+C may not be forwarded to the container.
- **`--init`** — runs Docker’s init (tini) as PID 1. The record-scan process runs
  as a child. Without `--init`, the Python process is PID 1; on Linux, PID 1
  often does not receive SIGINT/SIGTERM as expected, so **Ctrl+C will not stop**
  the recording. With `--init`, tini forwards signals to the child and Ctrl+C
  stops the process cleanly and writes metadata.

If the container was started in the background (e.g. `block_until_ms: 0`), the user
cannot stop it with Ctrl+C in that terminal. To stop it: run `docker ps` to get
the container ID, then `docker stop <container_id>` (sends SIGTERM; the script
handles it and exits cleanly).

Key flags (record-scan):

- `--no-preview` — always pass this; the container has no GUI.
- `--scan-dir /data/scans` — writes into the `/data` volume mount
  (maps to `docker_all/data/monty/scans/` on host by default).
- `--duration N` — optional; auto-stop after N seconds. If the user
  doesn't specify, omit it (recording stops on Ctrl+C or Enter).
- `--camera realsense` — only if user requests RealSense.
- `--no-depth` — only if user explicitly asks for no depth.

The `--rm` flag removes the one-off container after it exits.

When running from an agent/IDE: use **foreground** (no background) with `-it`
and `--init` so the user can stop with Ctrl+C. If you use `block_until_ms: 0`,
tell the user to stop with `docker stop <container_id>` or run the command
foreground so they can press Ctrl+C.

### 3. Report results

After the recording finishes, tell the user:

- Output directory on host: `docker_all/data/monty/scans/<object_name>/`
- Files produced: `video.mp4`, `video_annotated.mp4`, `metadata.json`,
  `depth/`, `depth_color/`, `depth_float/`, `pointcloud/`
- Depth quality summary (from the command's stdout)

### Example

User says: "record video with object name:yellow_cup, description:yellow ceramic cup"

Run:

```bash
cd docker_all && docker run --rm -it --init --privileged \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -v "$(pwd)/data/monty:/data" \
  -e SCAN_DIR=/data/scans \
  monty_vision_comp:latest \
  record-scan \
    --object yellow_cup \
    --description "yellow ceramic cup" \
    --no-preview \
    --scan-dir /data/scans
```

Host output at: `docker_all/data/monty/scans/yellow_cup/`. Stop with **Enter** or **Ctrl+C** (requires `-it --init`; clean exit and metadata is written).
