from __future__ import annotations

from typing import Iterable, Dict, Tuple, Optional

import torch
from sklearn.decomposition import PCA

from cluster import AutoKMeansClustering


def _orthonormal_basis_of_row_span(S: torch.Tensor) -> torch.Tensor:
    """
    S: (K, d) prototypes（每行一个 d 维向量）
    返回 Q: (d, r) 使得 columns(Q) 是 span(rows(S)) 的一组正交单位基，r <= min(K, d)
    """
    if S is None or S.numel() == 0:
        # (d, 0)
        return torch.empty((S.shape[1], 0), device=S.device, dtype=S.dtype)

    # span(rows(S)) == span(columns(S^T))
    Q, _ = torch.linalg.qr(S.T, mode="reduced")  # (d, r)
    return Q


def _project_out_span(X: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    从 X 中去除其在 span(Q) 上的投影。
    X: (M, d)
    Q: (d, r) 正交单位基（列正交）
    返回: X_perp = X - Proj_span(Q)(X)
    """
    if Q is None or Q.numel() == 0:
        return X
    return X - (X @ Q) @ Q.T


def build_residual_basis_U(
    Delta: torch.Tensor,                 # (M, d)
    selector: torch.Tensor,              # (K, d)
    k_residual: int,                     # 需要的 residual 维度
    center: bool = True
) -> torch.Tensor:
    """
    方案1的 residual 子空间构建：
      1) Q = orthonormal basis of span(selector)
      2) R = Delta - Proj_span(selector)(Delta)
      3) PCA(R) -> U_residual (d, k_residual)

    返回 U: (d, k_residual)，列正交单位。
    """
    device, dtype = Delta.device, Delta.dtype
    M, d = Delta.shape
    k_residual = int(k_residual)
    if k_residual <= 0:
        return torch.empty((d, 0), device=device, dtype=dtype)
    k_residual = min(k_residual, d)

    # 1) prototypes span 的正交基 Q
    selector = selector.to(device=device, dtype=dtype)  # (K, d)
    Q = _orthonormal_basis_of_row_span(selector)        # (d, r)

    # 2) 去投影得到残差 R
    R = _project_out_span(Delta, Q)                     # (M, d)
    if center:
        R = R - R.mean(dim=0, keepdim=True)

    # 3) PCA(R) -> U
    # sklearn PCA 需要 CPU numpy
    R_np = R.detach().float().cpu().numpy()
    pca = PCA(n_components=min(k_residual, d), svd_solver="auto")
    pca.fit(R_np)
    U = torch.from_numpy(pca.components_.T).to(device=device, dtype=dtype)  # (d, k)

    # 4) 数值保险：再去掉 U 在 Q 上的分量，并正交化
    U = _project_out_span(U.T, Q).T  # (d, k)  (把每列当成向量去投影)
    U, _ = torch.linalg.qr(U, mode="reduced")  # 列正交化

    return U


def build_selector_and_U_scheme1(
    train_ds: Iterable[Dict[str, torch.Tensor]],
    layer: int,
    k_residual: int,
    k_range: Tuple[int, int] = (3, 10),
    cluster_pca_components: int = 256,
    cluster_methods: Tuple[str, ...] = ("elbow", "silhouette", "calinski"),
    center_residual: bool = True,
) -> Tuple[None, torch.Tensor, torch.Tensor]:
    """
    方案1（ONLY）：
      - 从 train_ds 取 hc_layer{layer}, hi_layer{layer}
      - Delta = hc - hi
      - 在 Delta 上聚类得到 selector (K, d)
      - 对 Delta 去除 span(selector) 的投影后做 PCA，得到 U (d, k_residual)

    返回 (r, U, selector)，其中 r 固定为 None。
    """
    hc_list, hi_list = [], []
    for item in train_ds:
        hc_list.append(item[f"hc_layer{layer}"])
        hi_list.append(item[f"hi_layer{layer}"])

    # (M, d)
    hc = torch.stack(hc_list).squeeze(1)
    hi = torch.stack(hi_list).squeeze(1)
    device, dtype = hc.device, hc.dtype

    Delta = (hc - hi).to(device=device, dtype=dtype)  # (M, d)

    # 1) 在原始 Delta 上聚类得到 prototypes
    clusterer = AutoKMeansClustering(
        steering_vectors=Delta,          # [M, d]
        k_range=k_range,
        use_pca=True,                    # 内部先 PCA 到 cluster_pca_components 再 KMeans（更稳）
        pca_components=cluster_pca_components,
    )
    _ = clusterer.find_optimal_k(methods=list(cluster_methods))
    centers, _labels = clusterer.fit_final_clustering()
    clusterer.plot_evaluation(save_path=f"./cluster_evaluation_layer{layer}.png")
    # centers 可能是 numpy 或 torch，这里统一成 torch
    selector = torch.as_tensor(centers, device=device, dtype=dtype)  # (K, d)

    # 2) residual 子空间 U
    U = build_residual_basis_U(
        Delta=Delta,
        selector=selector,
        k_residual=k_residual,
        center=center_residual,
    )  # (d, k_residual)

    r = None
    return r, U, selector
