import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from metro_common import (
    FIG_DIR,
    FLOW_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORT_PATH,
    ensure_dirs,
)

matplotlib.use("Agg")
CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = CHINESE_FONTS
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="deep",
    rc={"font.sans-serif": CHINESE_FONTS, "axes.unicode_minus": False},
)


def parse_coord(coord_str: str) -> Tuple[float, float]:
    lon_str, lat_str = coord_str.split(",")
    return float(lon_str), float(lat_str)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def load_network_json() -> Dict:
    path = RAW_DIR / "shanghai_metro_gaode.json"
    if not path.exists():
        raise FileNotFoundError("请先运行 metro_fetch.py 获取网络数据")
    return json.loads(path.read_text(encoding="utf-8"))


def build_network_tables(raw_json: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    station_map: Dict[str, Dict] = {}
    edges: List[Dict] = []
    for line_data in raw_json.get("l", []):
        line_name = line_data.get("ln", "未知线路")
        stations = line_data.get("st", [])
        for idx, st in enumerate(stations):
            name = st.get("n", "").strip()
            english = st.get("sp", "").strip()
            lon, lat = parse_coord(st.get("sl", "0,0"))
            info = station_map.setdefault(
                name,
                {"name": name, "english": english, "lon": lon, "lat": lat, "lines": set(), "station_ids": set()},
            )
            info["lines"].add(line_name)
            info["station_ids"].add(st.get("si", ""))
            info["lon"], info["lat"] = lon, lat
            if idx < len(stations) - 1:
                next_st = stations[idx + 1]
                next_name = next_st.get("n", "").strip()
                next_lon, next_lat = parse_coord(next_st.get("sl", "0,0"))
                edges.append(
                    {
                        "line_name": line_name,
                        "station_a": name,
                        "station_b": next_name,
                        "distance_km": haversine(lon, lat, next_lon, next_lat),
                    }
                )
    station_rows = []
    for entry in station_map.values():
        station_rows.append(
            {
                "name": entry["name"],
                "english": entry["english"],
                "lon": entry["lon"],
                "lat": entry["lat"],
                "lines": sorted(entry["lines"]),
                "line_count": len(entry["lines"]),
            }
        )
    stations_df = pd.DataFrame(station_rows).sort_values("name").reset_index(drop=True)
    edges_df = pd.DataFrame(edges)
    G = nx.Graph()
    for row in stations_df.itertuples():
        G.add_node(row.name, english=row.english, lon=row.lon, lat=row.lat, lines=row.lines, line_count=row.line_count)
    for row in edges_df.itertuples():
        if G.has_edge(row.station_a, row.station_b):
            edge_data = G[row.station_a][row.station_b]
            edge_data["lines"].add(row.line_name)
            edge_data["distance_km"] = min(edge_data["distance_km"], row.distance_km)
        else:
            G.add_edge(row.station_a, row.station_b, lines={row.line_name}, distance_km=row.distance_km)
    return stations_df, edges_df, G


def load_flow_metadata() -> pd.DataFrame:
    idx_path = PROCESSED_DIR / "passenger_flow_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError("请先运行 metro_fetch.py 解压客流 CSV")
    return pd.read_csv(idx_path)


def read_flow_timeseries(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk", errors="ignore")
    rename_map = {}
    for col in df.columns:
        col_norm = col.replace(" ", "")
        if "时" in col_norm:
            rename_map[col] = "hour"
        elif "五" in col_norm or "5" in col_norm:
            rename_map[col] = "slot"
        elif "客流" in col_norm:
            rename_map[col] = "flow"
    df = df.rename(columns=rename_map)
    if not {"hour", "slot", "flow"}.issubset(df.columns):
        return None
    df = df[["hour", "slot", "flow"]].copy()
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    df["slot"] = pd.to_numeric(df["slot"], errors="coerce").fillna(0).astype(int)
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0.0)
    return df


def summarize_flow(metadata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict] = []
    for row in metadata_df.itertuples():
        csv_path = FLOW_DIR / row.file
        df = read_flow_timeseries(csv_path)
        if df is None:
            continue
        total_flow = df["flow"].sum()
        am_flow = df.loc[df["hour"].between(7, 9), "flow"].sum()
        pm_flow = df.loc[df["hour"].between(17, 19), "flow"].sum()
        peak_idx = df["flow"].idxmax()
        peak_hour = int(df.loc[peak_idx, "hour"])
        peak_slot = int(df.loc[peak_idx, "slot"])
        peak_time = f"{peak_hour:02d}:{peak_slot * 5:02d}"
        records.append(
            {
                "line": row.line_name,
                "station": row.station_name,
                "file": row.file,
                "total_flow": total_flow,
                "am_peak_flow": am_flow,
                "pm_peak_flow": pm_flow,
                "peak_time": peak_time,
                "peak_flow": float(df.loc[peak_idx, "flow"]),
            }
        )
    raw_df = pd.DataFrame(records).sort_values("total_flow", ascending=False).reset_index(drop=True)
    if raw_df.empty:
        return raw_df, raw_df
    peak_idx = raw_df.groupby("station")["peak_flow"].idxmax()
    peak_info = raw_df.loc[peak_idx, ["station", "peak_time", "peak_flow"]]
    dominant_idx = raw_df.groupby("station")["total_flow"].idxmax()
    dominant_line = raw_df.loc[dominant_idx, ["station", "line"]].rename(columns={"line": "dominant_line"})
    agg_df = (
        raw_df.groupby("station", as_index=False)
        .agg({"total_flow": "sum", "am_peak_flow": "sum", "pm_peak_flow": "sum"})
        .merge(peak_info, on="station", how="left")
        .merge(dominant_line, on="station", how="left")
    )
    return agg_df.sort_values("total_flow", ascending=False).reset_index(drop=True), raw_df


def compute_network_metrics(G: nx.Graph, stations_df: pd.DataFrame, flow_df: pd.DataFrame) -> pd.DataFrame:
    degree_dict = dict(G.degree())
    between = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)
    avg_neighbor_distance = {}
    for node in G.nodes:
        distances = [data.get("distance_km", np.nan) for _, _, data in G.edges(node, data=True)]
        avg_neighbor_distance[node] = float(np.nanmean(distances)) if distances else 0.0
    metrics = stations_df.copy()
    flow_lookup = flow_df.set_index("station")
    for col in ["total_flow", "am_peak_flow", "pm_peak_flow", "peak_time", "peak_flow"]:
        if col in flow_lookup.columns:
            metrics[col] = metrics["name"].map(flow_lookup[col])
        else:
            metrics[col] = np.nan
    metrics["degree"] = metrics["name"].map(degree_dict)
    metrics["betweenness"] = metrics["name"].map(between)
    metrics["closeness"] = metrics["name"].map(closeness)
    metrics["pagerank"] = metrics["name"].map(pagerank)
    metrics["avg_neighbor_distance"] = metrics["name"].map(avg_neighbor_distance)
    metrics["lines_joined"] = metrics["lines"].apply(lambda lst: ";".join(lst))
    return metrics


def plot_network_map(stations_df: pd.DataFrame, edges_df: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "shanghai_metro_network.png"
    plt.figure(figsize=(10, 10), facecolor="white")
    for row in edges_df.itertuples():
        coord_a = stations_df.loc[stations_df["name"] == row.station_a, ["lon", "lat"]]
        coord_b = stations_df.loc[stations_df["name"] == row.station_b, ["lon", "lat"]]
        if coord_a.empty or coord_b.empty:
            continue
        xs = [coord_a["lon"].values[0], coord_b["lon"].values[0]]
        ys = [coord_a["lat"].values[0], coord_b["lat"].values[0]]
        plt.plot(xs, ys, color="#d0d2db", linewidth=0.4, alpha=0.4)
    scatter = plt.scatter(
        stations_df["lon"],
        stations_df["lat"],
        s=30,
        c=stations_df["line_count"],
        cmap="viridis",
        alpha=0.9,
        edgecolors="#ffffff",
        linewidths=0.2,
    )
    hub = stations_df.nlargest(10, "line_count")
    plt.scatter(
        hub["lon"],
        hub["lat"],
        s=100,
        facecolors="none",
        edgecolors="#ff8f00",
        linewidths=1.2,
        label="Top10 换乘枢纽",
    )
    for _, row in hub.iterrows():
        plt.text(
            row["lon"],
            row["lat"],
            row["name"],
            fontsize=8,
            color="#333333",
            ha="left",
            va="bottom",
            clip_on=True,
        )
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label("换乘线路数量", rotation=270, labelpad=18)
    plt.legend(loc="lower left")
    plt.title("上海地铁网络拓扑 (经纬度投影)")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, facecolor="white")
    plt.close()
    return fig_path


def plot_top_flow(flow_df: pd.DataFrame, top_n: int = 15) -> Path:
    fig_path = FIG_DIR / "top_flow_stations.png"
    subset = flow_df.head(top_n)
    plt.figure(figsize=(10, 6))
    labels = subset.apply(lambda row: f"{row['station']} ({row.get('dominant_line', '未知线路') or ''})", axis=1)
    norm = plt.Normalize(subset["total_flow"].min(), subset["total_flow"].max())
    colors = plt.cm.inferno(norm(subset["total_flow"]))
    bars = plt.barh(labels, subset["total_flow"], color=colors, edgecolor="none")
    plt.gca().invert_yaxis()
    plt.xlabel("日客流量 (人次，单位:五分钟数据累计)")
    plt.title(f"客流量最高的 {top_n} 个车站")
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="日客流量")
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + max(subset["total_flow"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            va="center",
            fontsize=9,
            color="#333333",
        )
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def plot_degree_distribution(metrics_df: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "degree_distribution.png"
    plt.figure(figsize=(8, 5))
    plt.hist(
        metrics_df["degree"].dropna(),
        bins=range(1, int(metrics_df["degree"].max()) + 2),
        color="#1f77b4",
        label="节点数量分布",
    )
    plt.xlabel("节点度 (换乘能力)")
    plt.ylabel("车站数量")
    plt.title("节点度分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def plot_line_flow(flow_df: pd.DataFrame, top_n: int = 10) -> Tuple[Path, pd.DataFrame]:
    line_flow = (
        flow_df.groupby("dominant_line", as_index=False)["total_flow"].sum().sort_values("total_flow", ascending=False)
    )
    fig_path = FIG_DIR / "line_flow.png"
    plt.figure(figsize=(10, 6))
    subset = line_flow.head(top_n)
    sns.barplot(
        data=subset,
        x="total_flow",
        y="dominant_line",
        hue="dominant_line",
        dodge=False,
        palette="crest",
        legend=False,
    )
    plt.xlabel("日客流量 (人次)")
    plt.ylabel("线路")
    plt.title("线路级客流对比 (Top10)")
    for idx, row in subset.iterrows():
        plt.text(row["total_flow"], row["dominant_line"], f"{row['total_flow']:.0f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path, line_flow


def plot_peak_scatter(flow_df: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "am_pm_scatter.png"
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        x="am_peak_flow",
        y="pm_peak_flow",
        data=flow_df,
        size="total_flow",
        sizes=(10, 150),
        hue="total_flow",
        palette="viridis",
        alpha=0.7,
        legend=False,
    )
    sns.kdeplot(
        x=flow_df["am_peak_flow"],
        y=flow_df["pm_peak_flow"],
        levels=5,
        color="#ff7f0e",
        linewidths=1,
        alpha=0.4,
    )
    max_val = max(flow_df["am_peak_flow"].max(), flow_df["pm_peak_flow"].max())
    plt.plot([0, max_val], [0, max_val], "--", color="gray", linewidth=1, label="AM=PM 等值线")
    top_diff = (flow_df["pm_peak_flow"] - flow_df["am_peak_flow"]).abs().nlargest(5).index
    for idx in top_diff:
        row = flow_df.loc[idx]
        plt.text(row["am_peak_flow"], row["pm_peak_flow"], row["station"], fontsize=7)
    plt.xlabel("早高峰客流")
    plt.ylabel("晚高峰客流")
    plt.title("早晚高峰客流散点对比")
    norm = plt.Normalize(flow_df["total_flow"].min(), flow_df["total_flow"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("日客流量")
    plt.legend(handles=[plt.Line2D([0], [0], linestyle="--", color="gray", label="AM=PM 等值线")], loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def plot_flow_heatmap(stations_df: pd.DataFrame, flow_df: pd.DataFrame) -> Path:
    merged = stations_df.merge(flow_df, left_on="name", right_on="station", how="left")
    fig_path = FIG_DIR / "flow_heatmap.png"
    plt.figure(figsize=(10, 10), facecolor="white")
    scatter = plt.scatter(
        merged["lon"],
        merged["lat"],
        s=merged["total_flow"].fillna(0) / 200,
        c=merged["pm_peak_flow"].fillna(0),
        cmap="Reds",
        alpha=0.8,
    )
    plt.colorbar(label="晚高峰客流")
    plt.title("空间维度客流热力")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    size_handles = [
        plt.scatter([], [], s=s, color="#d62728", alpha=0.5)
        for s in (20, 80, 140)
    ]
    plt.legend(
        size_handles,
        ["日客流≈4万", "日客流≈12万", "日客流≈20万"],
        title="点大小 ~ 日客流",
        scatterpoints=1,
        loc="lower left",
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, facecolor="white")
    plt.close()
    return fig_path


def plot_line_overlap_heatmap(stations_df: pd.DataFrame) -> Tuple[Path, pd.DataFrame]:
    lines = sorted({line for lines in stations_df["lines"] for line in lines})
    matrix = pd.DataFrame(0, index=lines, columns=lines, dtype=int)
    for row in stations_df.itertuples():
        line_list = row.lines
        for line in line_list:
            matrix.loc[line, line] += 1
        for a, b in combinations(sorted(line_list), 2):
            matrix.loc[a, b] += 1
            matrix.loc[b, a] += 1
    fig_path = FIG_DIR / "line_overlap_heatmap.png"
    plt.figure(figsize=(11, 9), facecolor="white")
    sns.heatmap(
        matrix,
        cmap="mako",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "共用车站数量"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("线路换乘热力矩阵")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, facecolor="white")
    plt.close()
    return fig_path, matrix


def plot_centrality_vs_flow(metrics_df: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "centrality_vs_flow.png"
    df = metrics_df.dropna(subset=["total_flow", "betweenness"]).copy()
    if df.empty:
        return fig_path
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="betweenness",
        y="total_flow",
        hue="line_count",
        size="line_count",
        palette="rocket",
        sizes=(30, 200),
        alpha=0.75,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("介数中心性 (log)")
    plt.ylabel("日客流量 (log)")
    plt.title("中心性与客流的关系")
    top_points = df.nlargest(5, "total_flow")
    for _, row in top_points.iterrows():
        plt.text(row["betweenness"], row["total_flow"], row["name"], fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def plot_hourly_profile(metadata_df: pd.DataFrame) -> Tuple[Path, pd.DataFrame]:
    totals = np.zeros(24)
    counts = np.zeros(24)
    for row in metadata_df.itertuples():
        df = read_flow_timeseries(FLOW_DIR / row.file)
        if df is None:
            continue
        hourly = df.groupby("hour")["flow"].sum()
        for hour, value in hourly.items():
            if 0 <= hour < 24:
                totals[hour] += value
                counts[hour] += 1
    avg_flow = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    hourly_df = pd.DataFrame({"hour": np.arange(24), "avg_flow": avg_flow, "sample_count": counts})
    fig_path = FIG_DIR / "hourly_profile.png"
    plt.figure(figsize=(10, 5), facecolor="white")
    plt.fill_between(hourly_df["hour"], hourly_df["avg_flow"], color="#4c72b0", alpha=0.3)
    plt.plot(hourly_df["hour"], hourly_df["avg_flow"], color="#4c72b0", linewidth=2)
    plt.scatter(hourly_df["hour"], hourly_df["avg_flow"], color="#1b6ca8", s=25)
    plt.xticks(range(0, 24, 2))
    plt.xlabel("小时")
    plt.ylabel("平均客流 (人次)")
    plt.title("全天客流平均曲线 (按小时)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, facecolor="white")
    plt.close()
    return fig_path, hourly_df


def build_report(
    stations_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    figures: Dict[str, Path],
    line_flow_df: pd.DataFrame,
    am_pm_corr: float,
) -> Path:
    total_nodes = len(stations_df)
    total_edges = len(edges_df)
    avg_degree = float(metrics_df["degree"].mean())
    top_flow = flow_df.head(5)[["station", "total_flow", "peak_time", "peak_flow"]]
    top_betweenness = metrics_df.nlargest(5, "betweenness")[["name", "betweenness", "total_flow", "line_count"]]
    top_lines = line_flow_df.head(5)
    degree_stats = metrics_df["degree"].describe()

    lines = [
        "# 上海地铁网络与客流分析报告",
        "",
        "## 网络概览",
        f"- 车站数量：{total_nodes}",
        f"- 区间数量：{total_edges}",
        f"- 平均度：{avg_degree:.2f}，最大度 {degree_stats['max']:.0f} 出现在 {metrics_df.loc[metrics_df['degree'].idxmax(), 'name']}",
        "",
        "## 关键换乘枢纽（介数中心性 Top 5）",
    ]
    for _, row in top_betweenness.iterrows():
        lines.append(
            f"- {row['name']} | 介数 {row['betweenness']:.3f} | 所在线路 {row['line_count']} 条 | 已观测客流 {row['total_flow'] or 0:.0f}"
        )
    lines.append("")
    lines.append("## 客流热点车站")
    for _, row in top_flow.iterrows():
        lines.append(
            f"- {row['station']} | 总客流 {row['total_flow']:.0f} | 峰值 {row['peak_flow']:.0f} 出现在 {row['peak_time']}"
        )
    lines.append("")
    lines.append("## 线路客流对比")
    lines.append(
        "，".join(f"{row['dominant_line']}≈{row['total_flow']:.0f}" for _, row in top_lines.iterrows())
        or "暂无"
    )
    lines.append("")
    am_desc = f"{'正相关' if am_pm_corr >= 0 else '负相关'} (Pearson={am_pm_corr:.2f})"
    lines.append("## 早晚高峰特征")
    lines.append(f"- 早晚峰客流整体呈 {am_desc}，说明热门站全天承压。")
    am_top = flow_df.nlargest(5, "am_peak_flow")[["station", "am_peak_flow"]]
    pm_top = flow_df.nlargest(5, "pm_peak_flow")[["station", "pm_peak_flow"]]
    lines.append(
        "- 早高峰前五："
        + ", ".join(f"{r.station}({r.am_peak_flow:.0f})" for r in am_top.itertuples())
    )
    lines.append(
        "- 晚高峰前五："
        + ", ".join(f"{r.station}({r.pm_peak_flow:.0f})" for r in pm_top.itertuples())
    )
    lines.append("")
    lines.append("## 附图清单")
    for title, path in figures.items():
        lines.append(f"- {title}：{path}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    ensure_dirs()
    metadata_df = load_flow_metadata()
    network_json = load_network_json()

    stations_df, edges_df, G = build_network_tables(network_json)
    stations_df.to_csv(PROCESSED_DIR / "stations.csv", index=False, encoding="utf-8")
    edges_df.to_csv(PROCESSED_DIR / "edges.csv", index=False, encoding="utf-8")

    flow_summary, raw_flow = summarize_flow(metadata_df)
    flow_summary.to_csv(PROCESSED_DIR / "station_flow_summary.csv", index=False, encoding="utf-8")
    raw_flow.to_csv(PROCESSED_DIR / "station_flow_raw.csv", index=False, encoding="utf-8")

    metrics_df = compute_network_metrics(G, stations_df, flow_summary)
    metrics_df.to_csv(PROCESSED_DIR / "network_metrics.csv", index=False, encoding="utf-8")

    figures: Dict[str, Path] = {}
    figures["网络拓扑图"] = plot_network_map(stations_df, edges_df)
    figures["客流 Top15"] = plot_top_flow(flow_summary)
    figures["节点度分布"] = plot_degree_distribution(metrics_df)
    line_fig, line_flow = plot_line_flow(flow_summary)
    figures["线路客流对比"] = line_fig
    figures["早晚高峰散点"] = plot_peak_scatter(flow_summary)
    figures["空间热力"] = plot_flow_heatmap(stations_df, flow_summary)
    overlap_fig, overlap_matrix = plot_line_overlap_heatmap(stations_df)
    figures["线路换乘热力"] = overlap_fig
    overlap_matrix.to_csv(PROCESSED_DIR / "line_overlap_matrix.csv", encoding="utf-8")
    centrality_fig = plot_centrality_vs_flow(metrics_df)
    figures["中心性 vs 客流"] = centrality_fig
    hourly_fig, hourly_df = plot_hourly_profile(metadata_df)
    figures["全天客流曲线"] = hourly_fig
    hourly_df.to_csv(PROCESSED_DIR / "hourly_profile.csv", index=False, encoding="utf-8")

    am_pm_corr = flow_summary["am_peak_flow"].corr(flow_summary["pm_peak_flow"])
    report_path = build_report(stations_df, edges_df, flow_summary, metrics_df, figures, line_flow, am_pm_corr)

    print("网络与客流分析完成。")
    print(f"报告路径：{report_path}")


if __name__ == "__main__":
    main()

