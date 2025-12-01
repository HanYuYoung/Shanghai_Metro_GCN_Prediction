import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from metro_common import FIG_DIR, PROCESSED_DIR, REPORT_PATH, ensure_dirs

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, adj_norm: torch.Tensor, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("adj", adj_norm)
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adj @ self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.adj @ self.fc2(x)
        return x


def load_graph() -> Tuple[nx.Graph, pd.DataFrame]:
    stations_path = PROCESSED_DIR / "stations.csv"
    edges_path = PROCESSED_DIR / "edges.csv"
    if not stations_path.exists() or not edges_path.exists():
        raise FileNotFoundError("请先运行 metro_analysis.py 生成站点与区间表")
    stations_df = pd.read_csv(stations_path)
    edges_df = pd.read_csv(edges_path)
    G = nx.Graph()
    for row in stations_df.itertuples():
        G.add_node(row.name)
    for row in edges_df.itertuples():
        G.add_edge(row.station_a, row.station_b, distance_km=row.distance_km)
    return G, stations_df


def prepare_data() -> Tuple[nx.Graph, pd.DataFrame]:
    metrics_path = PROCESSED_DIR / "network_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("请先运行 metro_analysis.py 生成 network_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    G, _ = load_graph()
    return G, metrics_df


def run_gcn(G: nx.Graph, metrics_df: pd.DataFrame) -> Dict:
    required_cols = [
        "degree",
        "betweenness",
        "closeness",
        "pagerank",
        "line_count",
        "avg_neighbor_distance",
    ]
    features = metrics_df[required_cols].fillna(0.0)
    X = features.to_numpy(dtype=np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    nodelist = metrics_df["name"].tolist()
    adj = nx.to_numpy_array(G, nodelist=nodelist)
    adj = adj + np.eye(adj.shape[0])
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    adj_tensor = torch.from_numpy(adj_norm).float()

    labels = metrics_df["total_flow"].to_numpy(dtype=np.float32)
    mask = ~np.isnan(labels)
    labelled_indices = np.where(mask)[0]
    if len(labelled_indices) < 10:
        raise RuntimeError("带标签的站点不足，无法训练 GCN")

    scaled_labels = np.log1p(np.nan_to_num(labels, nan=0.0))

    torch.manual_seed(42)
    np.random.seed(42)

    shuffled = labelled_indices.copy()
    np.random.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    train_idx = torch.tensor(shuffled[:split], dtype=torch.long)
    test_idx = torch.tensor(shuffled[split:], dtype=torch.long)

    label_tensor = torch.from_numpy(scaled_labels).float().unsqueeze(1)
    feature_tensor = torch.from_numpy(X).float()

    model = SimpleGCN(in_dim=X.shape[1], hidden=64, out_dim=1, adj_norm=adj_tensor, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.MSELoss()

    for _ in range(300):
        model.train()
        preds = model(feature_tensor)
        loss = criterion(preds[train_idx], label_tensor[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_scaled = model(feature_tensor).squeeze().numpy()
    true_scaled = label_tensor.squeeze().numpy()

    preds = np.expm1(np.clip(preds_scaled, a_min=0.0, a_max=12.0))
    true_values = np.expm1(true_scaled)

    test_true = true_values[test_idx.numpy()]
    test_pred = preds[test_idx.numpy()]
    mae = float(np.mean(np.abs(test_true - test_pred)))
    rmse = float(np.sqrt(np.mean((test_true - test_pred) ** 2)))

    predictions = pd.DataFrame({"station": nodelist, "gcn_pred_flow": preds, "true_flow": true_values})
    predictions.to_csv(PROCESSED_DIR / "gcn_predictions.csv", index=False, encoding="utf-8")
    return {"mae": mae, "rmse": rmse, "predictions": predictions}


def plot_true_vs_pred(predictions: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "gcn_true_vs_pred.png"
    labelled = predictions[predictions["true_flow"] > 0]
    plt.figure(figsize=(6, 6))
    plt.scatter(labelled["true_flow"], labelled["gcn_pred_flow"], alpha=0.6, s=18, color="#ff7f0e", label="预测点")
    max_val = max(labelled["true_flow"].max(), labelled["gcn_pred_flow"].max())
    plt.plot([0, max_val], [0, max_val], "--", color="gray", label="理想对角线")
    plt.xlabel("真实客流")
    plt.ylabel("GCN 预测客流")
    plt.title("GCN 预测 VS 真实")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path


def update_report(metrics: Dict, cold_nodes: pd.DataFrame, fig_path: Path) -> None:
    section_lines = [
        "## GCN 预测性能",
        f"- 测试集 MAE：{metrics['mae']:.2f}",
        f"- 测试集 RMSE：{metrics['rmse']:.2f}",
        f"- 结果对比图：{fig_path}",
    ]
    if not cold_nodes.empty:
        section_lines.append(
            "- 无实测客流但预测热点："
            + ", ".join(f"{row.station}({row.gcn_pred_flow:.0f})" for row in cold_nodes.itertuples())
        )
    section_text = "\n".join(section_lines)
    if REPORT_PATH.exists():
        text = REPORT_PATH.read_text(encoding="utf-8")
        if "## GCN 预测性能" in text:
            base = text.split("## GCN 预测性能", 1)[0].rstrip()
            new_text = base + "\n\n" + section_text + "\n"
        else:
            new_text = text.rstrip() + "\n\n" + section_text + "\n"
        REPORT_PATH.write_text(new_text, encoding="utf-8")
    else:
        REPORT_PATH.write_text(section_text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    G, metrics_df = prepare_data()
    stats = run_gcn(G, metrics_df)
    fig_path = plot_true_vs_pred(stats["predictions"])
    cold_nodes = (
        stats["predictions"]
        .query("true_flow == 0")
        .nlargest(5, "gcn_pred_flow")[["station", "gcn_pred_flow"]]
    )
    update_report(stats, cold_nodes, fig_path)
    summary = {
        "mae": stats["mae"],
        "rmse": stats["rmse"],
        "top_cold_nodes": cold_nodes.to_dict(orient="records"),
    }
    (PROCESSED_DIR / "gcn_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GCN 训练完成，MAE={stats['mae']:.2f}, RMSE={stats['rmse']:.2f}")


if __name__ == "__main__":
    main()

