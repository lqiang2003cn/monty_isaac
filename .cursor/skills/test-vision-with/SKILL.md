---
name: test-vision-with
description: >-
  Runs the vision pipeline test (SAM2 + GDINO + depth) on a turntable scan.
  Use when the user says "test vision with XXX", "test vision pipeline with XXX",
  or asks to run the vision test on a scan named XXX. Resolves XXX to a folder
  under docker_all/data/monty/scans (exact match, typo correction, or closest name);
  asks if XXX does not match any scan.
---

# Test Vision With

Run the vision pipeline test script inside the vision_comp container using scan data from `docker_all/data/monty/scans/<object_name>`.

## When to use

- User says: **"test vision with XXX"**, **"test vision pipeline with XXX"**, or **"run vision test on XXX"**.
- XXX is the **object/scan name** (subfolder under `docker_all/data/monty/scans/`).

## Workflow

### 1. Resolve object name (XXX)

Scan folders live under **`docker_all/data/monty/scans/`**. List its subdirectories (e.g. with `ls docker_all/data/monty/scans/` or a glob), then:

| Case | Action |
|------|--------|
| **Exact match** | One subdir equals XXX (case-insensitive). Use it. |
| **Close match** | One subdir is clearly the intended name (e.g. `yelow_block` → `yellow_block`, `new_yellow_blok` → `new_yellow_block`). Use the closest; optionally mention "Using scan: \<name\>". |
| **Ambiguous / none** | Multiple candidates or no reasonable match. **Ask the user**: "No scan named '\<XXX\>'. Existing scans: \<list\>. Which one should I use?" or "Did you mean \<A\> or \<B\>?" |

Do **not** run the test with a guessed name if XXX is nonsense or ambiguous.

### 2. Ensure vision_comp is built

If the image might not exist:

```bash
cd docker_all && docker compose build vision_comp
```

### 3. Run the test

From the **repo root** (or `docker_all`), run the test inside the vision_comp container. The compose project uses `--profile monty` for vision_comp. Data is mounted at `/data` (host `docker_all/data/monty` → container `/data`), so scans are at `/data/scans`. The **model cache** is already mounted by the service (`${MODEL_CACHE:-../../models}` → `/models`), so the SAM2 checkpoint and other weights are reused and not re-downloaded; ensure `docker_all/models` exists (or set `MODEL_CACHE`) so the first run downloads there.

```bash
cd docker_all && docker compose --profile monty run --rm \
  -e SCAN_DIR=/data/scans \
  -e VISION_TEST_DIR=/data/vision_test \
  vision_comp \
  test-vision-pipeline --object <object_name> --scan-dir /data/scans
```

- **\<object_name\>**: the resolved folder name (e.g. `new_yellow_block`, `red_box`).
- `VISION_TEST_DIR=/data/vision_test` makes output appear on the host at **`docker_all/data/monty/vision_test/<object_name>/`** (same `/data` mount as scans).
- Optional: `--depth-source vggt` for VGGT depth; `--description "..."` to override the text prompt for GDINO.

Use the workspace path for `docker_all` (e.g. `docker_all` from repo root, or full path like `.../monty_isaac/docker_all`).

### 4. Report results

- Exit code 0 → overall pipeline **PASS**; 1 → **FAIL**.
- Output on host: **`docker_all/data/monty/vision_test/<object_name>/`** (report.json, gdino/, sam2_tracked/, combined/combined_output.mp4, etc.).
- Summarize: GDINO detection rate, SAM2 mask stats, depth result, and overall pass/fail.

## Examples

**User:** "test vision with new_yellow_block"  
→ Resolve: exact match `new_yellow_block`. Run:
`cd docker_all && docker compose --profile monty run --rm -e SCAN_DIR=/data/scans -e VISION_TEST_DIR=/data/vision_test vision_comp test-vision-pipeline --object new_yellow_block --scan-dir /data/scans`

**User:** "test vision with yelow block"  
→ Resolve: closest is `yellow_block` (or `new_yellow_block` if that’s closer). Use it and say "Using scan: yellow_block". Then run the same command with `--object yellow_block`.

**User:** "test vision with xyz"  
→ No matching scan. Reply: "No scan named 'xyz'. Existing scans: cup, gemini_test, girl_action_figure, new_yellow_block, red_box, yellow_block, yellow_cube. Which one should I use?"

## Paths reference

| What | Path |
|------|------|
| Scans on host | `docker_all/data/monty/scans/` |
| Scan dir in container | `/data/scans` |
| Test script (in container) | `test-vision-pipeline` (entry point from vision_server) |
| Test output on host | `docker_all/data/monty/vision_test/<object_name>/` (when `VISION_TEST_DIR=/data/vision_test`) |
