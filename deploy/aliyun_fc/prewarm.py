#!/usr/bin/env python3
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

from PIL import Image


def http_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> tuple[int, dict]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
        return resp.status, json.loads(data) if data else {}


def build_payload(task: str) -> dict:
    img = Image.new("RGB", (64, 64), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if task in {"segmentation", "detection"}:
        query = os.environ.get("PREWARM_QUERY", "all objects")
    else:
        query = os.environ.get("PREWARM_QUERY", "Extract the text content from this image.")

    return {
        "image": {"base64": image_b64},
        "query": query,
        "task": task,
        "max_tokens": int(os.environ.get("PREWARM_MAX_TOKENS", "32")),
        "min_image_size": int(os.environ.get("PREWARM_MIN_IMAGE_SIZE", "64")),
        "max_image_size": int(os.environ.get("PREWARM_MAX_IMAGE_SIZE", "64")),
    }


def pick_task(supported_tasks: list[str]) -> str:
    preferred = os.environ.get("PREWARM_TASK")
    if preferred and preferred in supported_tasks:
        return preferred

    for candidate in ("segmentation", "detection", "ocr_plain", "ocr_layout"):
        if candidate in supported_tasks:
            return candidate

    raise RuntimeError(f"No supported task found in health payload: {supported_tasks}")


def main() -> int:
    port = os.environ.get("PORT", "7860")
    base_url = os.environ.get("PREWARM_BASE_URL", f"http://127.0.0.1:{port}").rstrip("/")
    timeout_s = int(os.environ.get("PREWARM_TIMEOUT", "900"))
    interval_s = float(os.environ.get("PREWARM_INTERVAL", "2"))

    health_url = f"{base_url}/v1/health"
    predict_url = f"{base_url}/v1/predictions"

    start = time.time()
    health = None
    while time.time() - start < timeout_s:
        try:
            status_code, health = http_json(health_url, timeout=5.0)
            if status_code == 200 and health.get("status") == "ready":
                print(f"[prewarm] health ready: {health}")
                break
            print(f"[prewarm] waiting for ready state, payload={health}")
        except urllib.error.URLError as exc:
            print(f"[prewarm] health check failed: {exc}")
        time.sleep(interval_s)
    else:
        print(f"[prewarm] timed out waiting for {health_url}", file=sys.stderr)
        return 1

    task = pick_task(health.get("supported_tasks", []))
    payload = build_payload(task)
    print(f"[prewarm] sending warmup request for task={task}")

    try:
        status_code, resp = http_json(predict_url, payload=payload, timeout=float(os.environ.get("PREWARM_REQUEST_TIMEOUT", "120")))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[prewarm] warmup request failed: status={exc.code} body={body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"[prewarm] warmup request failed: {exc}", file=sys.stderr)
        return 1

    if status_code != 200:
        print(f"[prewarm] warmup request returned unexpected status={status_code} body={resp}", file=sys.stderr)
        return 1

    print(
        "[prewarm] success: "
        f"inference_time_ms={resp.get('inference_time_ms')} total_time_ms={resp.get('total_time_ms')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
