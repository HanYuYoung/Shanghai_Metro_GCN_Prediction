import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FLOW_DIR = RAW_DIR / "passenger_flow"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = DATA_DIR / "figures"
REPORT_PATH = DATA_DIR / "report.md"

GAODE_URL = "https://map.amap.com/service/subway?_1610428198658&srhdata=3100_drw_shanghai.json"
FLOW_ZIP_URL = "https://raw.githubusercontent.com/Aplicity/metro_prediction/master/data.zip"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def ensure_dirs(extra_paths: Iterable[Path] | None = None) -> None:
    base_paths = [DATA_DIR, RAW_DIR, FLOW_DIR, PROCESSED_DIR, FIG_DIR]
    if extra_paths:
        base_paths.extend(extra_paths)
    for path in base_paths:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(text: str) -> str:
    safe = re.sub(r"[\\/:*?\"<>|]", "_", text)
    safe = re.sub(r"\s+", "", safe)
    return safe


def parse_line_station(raw: str) -> tuple[str, str]:
    if "号线" in raw:
        left, right = raw.split("号线", 1)
        return f"{left}号线", right
    if raw.startswith("浦江线"):
        return "浦江线", raw.replace("浦江线", "", 1)
    return "未知线路", raw


@dataclass
class FlowMeta:
    file: str
    line_name: str
    station_name: str


def fetch_with_retry(session: requests.Session, url: str, max_retries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt == max_retries:
                raise
    raise RuntimeError(f"Failed to fetch {url}") from last_exc

