import io
import json
import zipfile

import pandas as pd
import requests

from metro_common import (
    FLOW_DIR,
    FLOW_ZIP_URL,
    GAODE_URL,
    PROCESSED_DIR,
    RAW_DIR,
    FlowMeta,
    ensure_dirs,
    fetch_with_retry,
    parse_line_station,
    sanitize_filename,
    USER_AGENT,
)


def extract_flow_zip(content: bytes) -> list[FlowMeta]:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO(content)
    metadata: list[FlowMeta] = []
    with zipfile.ZipFile(buffer) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".csv"):
                continue
            decoded_name = info.filename.encode("cp437").decode("utf-8", errors="ignore")
            path_stem = decoded_name.split("/")[-1].rsplit(".", 1)[0]
            line_name, station_name = parse_line_station(path_stem)
            safe_line = sanitize_filename(line_name)
            safe_station = sanitize_filename(station_name or "未知站")
            out_file = f"{safe_line}_{safe_station}.csv"
            out_path = FLOW_DIR / out_file
            raw_bytes = zf.read(info)
            text = raw_bytes.decode("gbk", errors="ignore")
            out_path.write_text(text, encoding="utf-8")
            metadata.append(FlowMeta(file=out_file, line_name=line_name, station_name=station_name or "未知站"))
    return metadata


def save_metadata(metadata: list[FlowMeta]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([meta.__dict__ for meta in metadata])
    df.to_csv(PROCESSED_DIR / "passenger_flow_index.csv", index=False, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print("开始爬取高德地铁线路 JSON ...")
    network_resp = fetch_with_retry(session, GAODE_URL)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "shanghai_metro_gaode.json").write_text(
        json.dumps(network_resp.json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("完成线路 JSON 保存。")

    print("开始爬取 GitHub 客流 ZIP ...")
    flow_resp = fetch_with_retry(session, FLOW_ZIP_URL)
    (RAW_DIR / "metro_flow_data.zip").write_bytes(flow_resp.content)
    metadata = extract_flow_zip(flow_resp.content)
    save_metadata(metadata)
    print(f"完成解压客流 CSV，共 {len(metadata)} 份。")


if __name__ == "__main__":
    main()

