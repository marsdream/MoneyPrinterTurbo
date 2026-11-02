"""Agnes Video 2.5 / 2.5 Flash 异步视频生成客户端。

参照 muapi.py 的异步任务模式：
  POST /v1/videos 创建任务 → 保存 video_id
  GET  /agnesapi?video_id=...&model_name=...  轮询直到 completed/failed

Flash 限制：
  - size 固定 "720P"
  - reference 图片最多 5 张，不支持 videos
  - 当前体验期免费 ($0/秒)

API: https://www.agnes-ai.com/zh-Hans/docs/agnes-video-25
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import requests
from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect

DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_ENDPOINT = "videos"
DEFAULT_MODEL = "agnes-video-2.5-flash"
DEFAULT_SIZE = "720P"
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 12
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_RUN_TIMEOUT_SECONDS = 600.0


class AgnesError(RuntimeError):
    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        self.task_id = task_id


class AgnesAPIError(AgnesError):
    pass


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    if settings:
        return str(settings.get("agnes_api_key", "")).strip()
    return os.environ.get("AGNES_API_KEY", "").strip()


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return os.environ.get("AGNES_BASE_URL", DEFAULT_BASE_URL)


def _endpoint() -> str:
    return os.environ.get("AGNES_ENDPOINT", DEFAULT_ENDPOINT)


def _model_id() -> str:
    return os.environ.get("AGNES_MODEL", DEFAULT_MODEL)


def _size() -> str:
    return os.environ.get("AGNES_SIZE", DEFAULT_SIZE)


def _poll_interval() -> float:
    try:
        return float(os.environ.get("AGNES_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL_SECONDS)))
    except ValueError:
        return DEFAULT_POLL_INTERVAL_SECONDS


def _run_timeout() -> float:
    try:
        return float(os.environ.get("AGNES_RUN_TIMEOUT", str(DEFAULT_RUN_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_RUN_TIMEOUT_SECONDS


def _redact_secret(value: Any, secret: str) -> str:
    if not value or not secret:
        return str(value or "")
    s = str(value)
    if secret in s:
        return s.replace(secret, "***REDACTED***")
    return s


def _response_error(response: requests.Response, api_key: str) -> str:
    try:
        body = response.json()
        msg = body.get("error", {}).get("message", body.get("message", str(body)))
    except Exception:
        msg = response.text[:200]
    return f"HTTP {response.status_code}: {_redact_secret(msg, api_key)}"


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """提交一个 Agnes 文生视频任务并等待完成。"""
    base_url = _base_url()
    endpoint = _endpoint()
    model = _model_id()
    size = _size()
    api_key = get_api_key()

    if not api_key:
        raise AgnesError("Agnes API key not configured")

    prompt = str(search_term or "").strip()
    if not prompt:
        raise AgnesError("Agnes prompt must not be empty")

    try:
        aspect = VideoAspect(video_aspect)
    except (TypeError, ValueError) as exc:
        raise AgnesError("Agnes video aspect is invalid") from exc

    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AgnesError("Agnes clip duration must be a positive integer") from exc

    min_dur, max_dur = DEFAULT_MIN_DURATION_SECONDS, DEFAULT_MAX_DURATION_SECONDS
    duration = min(max(requested_duration, min_dur), max_dur)
    if duration != requested_duration:
        logger.info(
            f"Agnes clip duration clamped: requested={requested_duration}s, using={duration}s"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    aspect_ratio_map = {
        VideoAspect.landscape: "16:9",
        VideoAspect.portrait: "9:16",
        VideoAspect.square: "1:1",
    }
    aspect_ratio = aspect_ratio_map.get(aspect, "16:9")

    payload = {
        "model": model,
        "prompt": prompt,
        "mode": "text",
        "seconds": str(duration),
        "size": size,
        "aspect_ratio": aspect_ratio,
    }

    url = f"{base_url}/{endpoint}"
    logger.info(f"submitting agnes video: model={model} duration={duration}s")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise AgnesError(f"request failed: {exc}") from exc

    if resp.status_code != 200:
        raise AgnesAPIError(_response_error(resp, api_key))

    body = resp.json()
    video_id = body.get("video_id") or body.get("id") or body.get("task_id") or ""
    if not video_id:
        raise AgnesAPIError(f"no video_id in response: {body}")

    logger.info(f"agnes task created: video_id={video_id}")

    result_url = _wait_for_completion(video_id, model, headers)

    return [
        MaterialInfo(
            url=result_url,
            material_type="video",
            provider="agnes",
            duration_seconds=duration,
            source_info={"video_id": video_id, "model": model},
        )
    ]


def _wait_for_completion(video_id: str, model: str, headers: dict) -> str:
    """轮询查询 Agnes 视频任务状态，返回最终视频 URL。"""
    base_url = _base_url()
    interval = _poll_interval()
    timeout = _run_timeout()
    api_key = get_api_key()

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(
                f"{base_url}/agnesapi",
                headers=headers,
                params={"video_id": video_id, "model_name": model},
                timeout=30,
            )
        except requests.RequestException as exc:
            logger.warning(f"agnes poll error: {exc}")
            time.sleep(interval)
            continue

        if resp.status_code == 429:
            logger.warning("agnes rate limited, backing off")
            time.sleep(max(interval * 2, 10))
            continue

        if resp.status_code != 200:
            logger.warning(f"agnes poll HTTP {resp.status_code}: {_response_error(resp, api_key)}")
            time.sleep(interval)
            continue

        try:
            body = resp.json()
        except Exception:
            logger.warning(f"agnes poll invalid JSON: {resp.text[:200]}")
            time.sleep(interval)
            continue

        status = body.get("status", "")
        progress = body.get("progress", 0)
        logger.debug(f"agnes poll: status={status} progress={progress}%")

        if status == "completed":
            url = body.get("url") or ""
            if not url:
                raise AgnesAPIError(f"completed but no url in response: {body}")
            logger.info(f"agnes video ready: {url}")
            return url

        if status in ("failed", "error"):
            err_msg = body.get("error") or body.get("message", "unknown error")
            raise AgnesAPIError(f"agnes video failed: {err_msg}")

        if status not in ("pending", "processing", "queued"):
            logger.warning(f"unexpected agnes status: {status}")

        time.sleep(interval)

    raise AgnesError(f"agnes video task timed out after {timeout:.0f}s: {video_id}")
