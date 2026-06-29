import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional,Tuple,Any
import torch.nn.functional as F
from sklearn.decomposition import PCA
from cluster import AutoKMeansClustering,ClusterBasedVectorSelector
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from scheme1 import build_residual_basis_U
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# class SteeringModel(nn.Module):
#     """
#     s(h) = s_proto(h) + U_rest @ β(h)
#     推理：h' = h + gamma * s(h)
#     - s_proto 通过 prototype attention (Q/K/V) 得到
#     - U_rest 分支由 mix(h) 得到
#     """
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         r_dim: int = 1,                      # 保留字段兼容
#         k_rest: int = 1,
#         init_selector: Optional[torch.Tensor] = None,
#         use_tanh: bool = False,
#         attn_dim: Optional[int] = None,      # attention 空间维度
#         topk: int = 2                        # top-k routing
#     ):
#         super().__init__()
#         self.use_tanh = use_tanh
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.k_rest = k_rest
#         self.topk = topk

#         # 不指定就默认等于 hidden_dim，你现在可以直接设成 3072
#         if attn_dim is None:
#             attn_dim = hidden_dim
#         self.attn_dim = attn_dim

#         # --------- Q/K/V ---------
#         self.W_q = nn.Linear(input_dim, attn_dim)   # h -> Q
#         self.W_k = nn.Linear(input_dim, attn_dim)   # selector -> K
#         self.W_v = nn.Linear(input_dim, input_dim)  # selector -> V (回到 d_in 方便加)

#         if init_selector is not None:
#             assert init_selector.dim() == 2, "init_selector 必须是 (K, d_in)"
#             assert init_selector.size(1) == input_dim, \
#                 f"init_selector.shape[1] = {init_selector.size(1)} != input_dim = {input_dim}"
#             # 原型作为 memory
#             #self.selector = nn.Parameter(init_selector.clone())  # (K, d_in)
#             self.selector = init_selector.to(self.W_q.weight.device)
#         else:
#             self.selector = None

#         # --------- U_rest 分支 ---------
#         self.scale = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, r_dim)
#         )  # 其实现在没用 r0 了，先保留
#         self.mix = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, k_rest)
#         )

#         # 这两个我们会 clamp 一下
#         #self.gamma = nn.Parameter(torch.tensor(1.0))  # 全局强度
#         self.tau   = nn.Parameter(torch.tensor(1.0))  # temperature

#     def forward(
#         self,
#         h: torch.Tensor,               # (B, d_in)
#         r0: Optional[torch.Tensor],    # 兼容 diff 模式
#         U_rest: torch.Tensor,          # (d_in, k_rest)
#         selector: Optional[torch.Tensor] = None      # 一般不用，默认用 self.selector
#     ):
#         B, d_in = h.shape
#         k_rest = U_rest.shape[1] if U_rest.ndim == 2 else 0

#         # 强制在 float32 里算 attention，防止 half 溢出
#         orig_dtype = h.dtype
#         h_f = h.to(self.W_q.weight.dtype)
#         U_rest_f = U_rest.to(self.W_q.weight.dtype)
#         selector_param = self.selector.to(self.W_k.weight.device) if self.selector is not None else selector
#         # --------- 1) prototype attention 分支 ---------
#         if selector_param is not None and selector_param.numel() > 0:
#             S = selector_param.float()          # (K, d_in)
#             K_num = S.size(0)
            
#             Q = self.W_q(h_f)                  # (B, attn_dim)
#             K_mat = self.W_k(S)                # (K, attn_dim)
#             V_mat = self.W_v(S)                # (K, d_in)

#             d_k = K_mat.size(1)
#             scores = (Q @ K_mat.T) / math.sqrt(d_k)   # (B, K)

#             # top-k routing
#             if self.topk is not None and self.topk > 0 and self.topk < K_num:
#                 k = min(self.topk, K_num)
#                 topk_scores, idx = scores.topk(k=k, dim=-1)
#                 neg_inf = torch.finfo(scores.dtype).min
#                 mask = torch.full_like(scores, neg_inf)
#                 mask.scatter_(1, idx, topk_scores)
#                 scores = mask

#             # clamp tau，避免被训练到 0 或非常小
#             tau = torch.clamp(self.tau.float(), min=0.1, max=10.0)
#             alpha = F.softmax(scores / tau, dim=-1)      # (B, K)
#             s_proto_f = alpha @ V_mat                    # (B, d_in)
#         else:
#             alpha = torch.zeros(B, 0, device=h.device, dtype=torch.float32)
#             s_proto_f = torch.zeros_like(h_f)

#         # --------- 2) U_rest 分支 ---------
#         if k_rest > 0:
#             beta = self.mix(h_f)                         # (B, k_rest)
#             U_rest_f = U_rest_f.to(h_f.device)
#             s_U_f = beta @ U_rest_f.T                    # (B, d_in)
#         else:
#             beta = torch.zeros(B, 0, device=h.device, dtype=torch.float32)
#             s_U_f = torch.zeros_like(s_proto_f)

#         # --------- 3) 合成 & dtype 转回 ---------
#         #gamma = torch.clamp(self.gamma.float(), min=-10.0, max=10.0)
#         s_f = 1 * (s_proto_f + s_U_f)                # float32

#         s = s_f.to(orig_dtype)
#         alpha = alpha.to(orig_dtype)
#         beta = beta.to(orig_dtype)

#         return alpha, beta, s

from scheme1 import build_selector_and_U_scheme1
class SteeringModel(nn.Module):
    """
    s(h) = α(h)*r0 + U_rest @ β(h)
    推理：h' = h + s(h) （无门控，所有样本都施加）
    """
    def __init__(self, input_dim: int, hidden_dim: int, r_dim: int=1, k_rest: int=1, init_selector: Optional[torch.Tensor] = None,use_tanh: bool = False):
        super().__init__()
        self.use_tanh = use_tanh
        self.query=nn.Linear(input_dim,hidden_dim)
        self.key=nn.Linear(input_dim,hidden_dim)
        #self.selector=nn.Parameter(init_selector.clone())
        self.value=nn.Linear(input_dim,hidden_dim)
        self.selector=init_selector
        self.scale = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, r_dim)
        )
        self.mix = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, k_rest)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 全局强度
        self.tau=nn.Parameter(torch.tensor(1.0))
    # def forward(self, h: torch.Tensor, r0: torch.Tensor, U_rest: torch.Tensor,selector:torch.Tensor=None, return_components: bool = False):
    #     """
    #     h:      (B,d)
    #     r0:     (d,)   单位向量
    #     U_rest: (d,k_rest) 列正交；k_rest=0 时允许传空 (d,0)
    #     """
    #     B, d = h.shape
    #     k_rest = U_rest.shape[1] if U_rest.ndim == 2 else 0
    #     h=h.to(self.query.weight.dtype)
    #     q=self.query(h)              # (B,1)
    #     d_attn=q.size(1)
    #     k_mat=self.key(self.selector.to(q.device))
    #     scores=(q@k_mat.T)/torch.sqrt(torch.tensor(d_attn,dtype=q.dtype,device=q.device))
    #     alpha=F.softmax(scores/self.tau,dim=-1)
    #     beta  = self.mix(h) if k_rest > 0 else torch.zeros(B, 0, device=h.device, dtype=h.dtype)
    #     U_rest=U_rest.to(alpha.device)
    #     if U_rest.dtype != alpha.dtype:
    #         U_rest=U_rest.to(dtype=alpha.dtype)
    #     v_mat=self.value(self.selector.to(q.device))
    #     #s_proto = alpha @ self.selector.to(alpha.device)
    #     s_proto = alpha @ v_mat
    #     # print(s_proto.shape)
    #     # exit()
    #     s_U  = beta @ U_rest.T if k_rest > 0 else torch.zeros_like(s_proto)
    #     s = self.gamma * (s_proto + s_U)      # (B,d)
    #     if return_components:
    #         return alpha, beta, s, s_proto, s_U
    #     return alpha, beta, s
    def forward(self, h: torch.Tensor, r0: torch.Tensor, U_rest: torch.Tensor, selector: torch.Tensor=None, return_components: bool = False):
    # 1. 记录原始精度（通常是 Half/Float16），用于最后统一输出
        original_dtype = h.dtype
        device = h.device

        # 2. 强制将所有涉及计算的组件转为 Float32 以保证数值稳定性
        # 同时也解决了 Linear 层和 Input 精度不匹配的问题
        h = h.to(dtype=torch.float32)
        U_rest = U_rest.to(device=device, dtype=torch.float32)
        
        # 将 nn.Module (Linear) 的计算也强制切换到 FP32
        # 注意：我们通常不直接调用 .float() 改模块，而是在 forward 里显式转换输入和权重
        def cast_linear_forward(layer, x):
            return F.linear(x, layer.weight.to(torch.float32), 
                            layer.bias.to(torch.float32) if layer.bias is not None else None)

        B, d = h.shape
        k_rest = U_rest.shape[1] if U_rest.ndim == 2 else 0

        # 3. 执行核心计算 (使用自定义的高精度转发)
        q = cast_linear_forward(self.query, h)  # (B, hidden_dim)
        
        # 确保 selector 精度对齐
        target_selector = self.selector if selector is None else selector
        target_selector = target_selector.to(device=device, dtype=torch.float32)
        
        k_mat = cast_linear_forward(self.key, target_selector)   # (N, hidden_dim)
        v_mat = cast_linear_forward(self.value, target_selector) # (N, hidden_dim)

        # 4. Attention 计算 (FP32 下进行非常稳定)
        d_attn = q.size(1)
        # scores: (B, N)
        scores = (q @ k_mat.T) / torch.sqrt(torch.tensor(d_attn, dtype=torch.float32, device=device))
        alpha = F.softmax(scores / self.tau.to(torch.float32), dim=-1)

        # 5. 其他组件计算
        # 这里 mix 和 scale 如果也是 nn.Sequential，建议也确保其内部计算是 FP32
        # 为了简化，我们可以先通过 .float() 转换这些小网络
        s_proto = alpha @ v_mat # (B, d)
        
        if k_rest > 0:
            # 同样的逻辑处理 mix 网络
            # 假设 mix 较小，可以直接临时转 float 计算
            beta = self.mix.float()(h) 
            s_U = beta @ U_rest.T
        else:
            beta = torch.zeros(B, 0, device=device, dtype=torch.float32)
            s_U = torch.zeros_like(s_proto)

        # 6. 合并生成最终的 Steering Vector
        s = self.gamma.to(torch.float32) * (s_proto + s_U)

        # 7. 关键：将输出转回原始精度 (Float16)
        # 这样才能加回到 LLM 的 Hidden States 中
        s = s.to(dtype=original_dtype)
        alpha = alpha.to(dtype=original_dtype)
        beta = beta.to(dtype=original_dtype)

        if return_components:
            return alpha, beta, s, s_proto.to(original_dtype), s_U.to(original_dtype)
        return alpha, beta, s
# class SteeringModel(nn.Module):
#     """
#     s(h) = α(h)*r0 + U_rest @ β(h)
#     推理：h' = h + s(h) （无门控，所有样本都施加）
#     """
#     def __init__(self, input_dim: int, hidden_dim: int, r_dim: int=1, k_rest: int=1, use_tanh: bool = False,init_selector: Optional[torch.Tensor] = None):
#         super().__init__()
#         self.use_tanh = use_tanh
#         self.scale = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, r_dim)
#         )
#         self.mix = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, k_rest)
#         )
#         self.gamma = nn.Parameter(torch.tensor(1.0))  # 全局强度

#     def forward(self, h: torch.Tensor, r0: torch.Tensor, U_rest: torch.Tensor,selector:torch.Tensor=None):
#         """
#         h:      (B,d)
#         r0:     (d,)   单位向量
#         U_rest: (d,k_rest) 列正交；k_rest=0 时允许传空 (d,0)
#         """
#         B, d = h.shape
#         k_rest = U_rest.shape[1] if U_rest.ndim == 2 else 0
        
#         alpha = self.scale(h)              # (B,1)
#         beta  = self.mix(h) if k_rest > 0 else torch.zeros(B, 0, device=h.device, dtype=h.dtype)
#         U_rest=U_rest.to(alpha.device)
#         if U_rest.dtype != alpha.dtype:
#             U_rest=U_rest.to(dtype=alpha.dtype)
#         if self.use_tanh:
#             alpha = torch.tanh(alpha)      # 稳定幅度（可选）
#             beta  = torch.tanh(beta)
#         if r0 is not None:
#             if r0.dtype != alpha.dtype:
#                 r0=r0.to(dtype=alpha.dtype)
#             r0=r0.to(alpha.device)
#             s_r0 = alpha * r0.unsqueeze(0)     # (B,d)
            
#         if selector is not None:
#             if selector.dtype != alpha.dtype:
#                 selector=selector.to(dtype=alpha.dtype)
#             selector=selector.to(alpha.device)
#             s_r0 = torch.matmul(alpha, selector)
#         # print(s_r0.shape)
#         # exit()
#         s_U  = beta @ U_rest.T if k_rest > 0 else torch.zeros_like(s_r0)
#         s = self.gamma * (s_r0 + s_U)      # (B,d)
#         return alpha, beta, s
    
# class SteeringModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, k):
#         super(SteeringModel, self).__init__()
        
#         # 门控头：决定是否施加 steering 向量
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # 尺度头：控制沿全局向量 r_0 的强度
#         self.scale = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#         # 混合头：在目标子空间 U 内做个性化修正
#         self.mix = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, k)  # k 是目标子空间的维度
#         )

#     def forward(self, h, r_0, U):
#         # 计算门控值
#         g = self.gate(h).squeeze(-1)  # gate output
#         g = torch.clamp(g, 0.0, 1.0)  # 确保 gate 在 [0, 1] 之间
        
#         # 计算尺度和混合头
#         alpha = self.scale(h).squeeze(-1)  # alpha 是沿 r_0 的强度
#         beta = self.mix(h)  # beta 是在 U 子空间的个性化修正
        
#         # 扩展 alpha 为与 r_0 相同的维度，逐元素乘法
#         alpha_expanded = alpha.unsqueeze(-1) * r_0  # alpha 对应每个样本与 r_0 相乘
        
#         # 计算最终的 steering 向量
#         s_raw = alpha_expanded + torch.matmul(beta, U.T)  # r_0 是全局方向，beta 在 U 中
#         return g, alpha, beta, s_raw
def normalize_vec(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def orthonormalize_columns(U: torch.Tensor) -> torch.Tensor:
    # 列正交 + 单位化
    Q, _ = torch.linalg.qr(U, mode='reduced')
    return Q

def project_out(v: torch.Tensor, u_unit: torch.Tensor) -> torch.Tensor:
    # 去掉 v 在单位向量 u_unit 上的分量
    coeff = (v @ u_unit)  # (...,)
    return v - coeff.unsqueeze(-1) * u_unit

def make_projection_mats(U_full: torch.Tensor):
    # U_full: (d,k)，列正交单位
    P = U_full @ U_full.T
    I = torch.eye(U_full.size(0), device=U_full.device, dtype=U_full.dtype)
    return P, I - P

def build_truth_subspace(
    Hc: torch.Tensor,               # (M, d) 正确回复隐藏态
    Hi: torch.Tensor,               # (M, d) 或 (M, K, d) 错误回复隐藏态
    k: int = 32,                    # 子空间维数（含 r0）
    choose_method: str = "diff",
    center_residual: bool = True,   # 是否对残差做去均值
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      r0: (d,) 单位向量，全局真相方向
      U:  (d, k) 列为正交基，第一列是 r0
    说明:
      - 若 Hi 是 (M, K, d)，会先对 K 个负例取均值。
      - 在残差上做 PCA，取前 k-1 个方向，最后与 r0 拼接并正交化。
    """
    assert Hc.ndim == 2, "Hc must be (M, d)"
    M, d = Hc.shape

    if Hi.ndim == 3:
        Hi_mean = Hi.mean(dim=1)  # (M, d)
    elif Hi.ndim == 2:
        Hi_mean = Hi
    else:
        raise ValueError("Hi must be (M, d) or (M, K, d)")

    if device is None:
        device = Hc.device
    if dtype is None:
        dtype = Hc.dtype
    Hc=Hc.to(device)
    Hi_mean=Hi_mean.to(device)
    Delta = Hc - Hi_mean                 # (M, d)
    delta_mean = Delta.mean(dim=0)       # (d,)
    print(choose_method)
    # 1) 差分与全局真相方向 r0
    if choose_method=="diff":
        
        r0 = delta_mean / (delta_mean.norm() + 1e-12)  # (d,)

        # 2) 去 r0 分量后的残差
        R = project_out(Delta, r0)           # (M, d)
        if center_residual:
            R = R - R.mean(dim=0, keepdim=True)

        # 3) 在残差上做 PCA，取 k-1 个方向（如果 k=1 则只有 r0）
        k = max(1, min(k, d))
        k_rest = max(0, k - 1)

        if k_rest > 0:
            R_np = R.detach().cpu().numpy()
            pca = PCA(n_components=k_rest, svd_solver="auto")
            pca.fit(R_np)
            U_rest = torch.from_numpy(pca.components_.T).to(device=device, dtype=dtype)  # (d, k-1)

            # 保险：确保 U_rest 与 r0 正交，且列正交单位
            # 先把 r0 分量去掉再 QR
            U_rest = U_rest - r0[:, None] * (r0 @ U_rest)
            U_rest = orthonormalize_columns(U_rest)
            U = torch.cat([r0[:, None], U_rest], dim=1)  # (d, k)
        else:
            U = r0[:, None]

        # 再做一次轻微的正交化（保持 r0 不变）
        # 用 Gram-Schmidt 对其余列做正交，r0 作为第一列固定
        if U.shape[1] > 1:
            U2 = U.clone()
            U2[:, 0] = r0
            for j in range(1, U2.shape[1]):
                v = U2[:, j]
                v = project_out(v, r0)
                # 与之前列正交
                for i in range(1, j):
                    vi = U2[:, i]
                    v = v - vi * (vi @ v)
                v = v / (v.norm() + 1e-12)
                U2[:, j] = v
            U = U2

        return r0, U  # r0:(d,), U:(d,k)
    else:
            # 可选的中心化处理
        if center_residual:
            Delta_centered = Delta - Delta.mean(dim=0, keepdim=True)
        else:
            Delta_centered = Delta
        
        # 确保k在有效范围内
        k = max(1, min(k, d))
        
        # 对Delta直接进行PCA分析
        Delta_np = Delta_centered.detach().cpu().numpy()
        pca = PCA(n_components=k, svd_solver="auto")
        pca.fit(Delta_np)
        
        # 获取主成分并转换为tensor
        U = torch.from_numpy(pca.components_.T).to(device=device, dtype=dtype)  # (d, k)
        
        # 确保U的列是正交且单位化的
        #U = orthonormalize_columns(U)
        
        return None,U  # U:(d,k)

# def get_vector_and_space(train_ds,layer,k,choose_method):
#     hc_list = []
#     hi_list = []
#     for item in train_ds:
#         # 提取 hc_layer20 并添加到 hc_list
#         hc_list.append(item[f'hc_layer{layer}'])
#         # 提取 hi_layer20 并添加到 hi_list
#         hi_list.append(item[f'hi_layer{layer}'])
#     hc_tensor = torch.stack(hc_list).squeeze(1)
#     hi_tensor = torch.stack(hi_list).squeeze(1)
#     r,U=build_truth_subspace(hc_tensor,hi_tensor,k,choose_method)
#     if choose_method=="diff":
#         r=r
#         selector=None
#     elif choose_method=="nearest" or choose_method=="top-k":
#         device=hc_tensor.device
#         Hc=hc_tensor.to(device)
#         Hi_mean=hi_tensor.to(device)
#         Delta = Hc - Hi_mean
#         clusterer = AutoKMeansClustering(
#             steering_vectors=Delta,
#             k_range=(3, 10),
#             use_pca=True,
#             pca_components=256
#         )
#         results = clusterer.find_optimal_k(methods=['elbow', 'silhouette', 'calinski'])
#         cluster_centers, labels = clusterer.fit_final_clustering()
#         # selector = ClusterBasedVectorSelector(
#         #     cluster_centers=cluster_centers,
#         #     selection_method='nearest'
#         # )
#         r=None
#         selector=cluster_centers
#     return r,U,selecto
# r
def get_vector_and_space1(
    train_ds,
    layer: int,
    k: int,
    choose_method: str="top-k",
    cluster_mode: str = "delta_pca",   # "delta_pca" 对应 1.1, "joint" 对应 1.2
):
    #论文具体实现版本
    r, U_res, selector = build_selector_and_U_scheme1(
    train_ds=train_ds,
    layer=layer,
    k_residual=k,
    k_range=(3, 10),
    cluster_pca_components=256,
    )
    return r, U_res, selector
def get_vector_and_space(
    train_ds,
    layer: int,
    k: int,
    choose_method: str="nearest",
    cluster_mode: str = "base",   # "delta_pca" 对应 1.1, "joint" 对应 1.2
    joint_lambda: float = 0.5          # 1.2 里 λ 的权重
):
    """
    根据 train_ds 中的 hc/hi 构建 (r, U, selector)

    Args:
        train_ds: 训练集，元素里至少包含 hc_layer{layer}, hi_layer{layer}
                  若 cluster_mode="joint"，还需要有 hq_layer{layer}（见下）
        layer:    指定层号
        k:        PCA 子空间维数
        choose_method:
            - "diff": 只用全局 r0 + U_rest，不做聚类
            - "nearest" / "top-k": 使用聚类 + routing
        cluster_mode:
            - "delta_pca": 1.1，在 UᵀΔ 空间上聚类
            - "joint":     1.2，在 [h_q ; λ·UᵀΔ] 联合特征上聚类
        joint_lambda:
            - 联合特征中 Δ 部分的缩放权重 λ
    """
    hc_list = []
    hi_list = []
    # 如果要用 joint 模式，需要额外取 question 的 hidden state
    hq_list = []

    for item in train_ds:
        hc_list.append(item[f'hc_layer{layer}'])   # 正确答案 hidden
        hi_list.append(item[f'hi_layer{layer}'])   # 错误答案 hidden(已均值)

        if cluster_mode == "joint":
            # 假设数据集中存了 question 的 hidden，命名为 hq_layer{layer}
            # 如果你的字段名字不一样，在这里改一下 key 即可
            if f'y_lose_layer{layer}' not in item:
                raise KeyError(
                    f"joint 模式需要在 train_ds 里提供 'hq_layer{layer}'，"
                    f"当前样本没有这个字段，请检查数据预处理。"
                )
            hq_list.append(item[f'y_lose_layer{layer}'])

    hc_tensor = torch.stack(hc_list).squeeze(1)   # (M, d)
    hi_tensor = torch.stack(hi_list).squeeze(1)   # (M, d)

    # 先构建子空间：diff 模式返回 (r0, U_full)，nearest/top-k 返回 (None, U)
    r, U = build_truth_subspace(hc_tensor, hi_tensor, k, choose_method)

    # ---------- 情况 1：diff 模式，不做聚类，直接返回 ----------
    if choose_method == "diff":
        selector = None
        return r, U, selector

    # ---------- 情况 2：nearest / top-k 模式，需要聚类 ----------
    elif choose_method in ["nearest", "top-k"]:
        device = hc_tensor.device
        Hc = hc_tensor.to(device)            # (M, d)
        Hi_mean = hi_tensor.to(device)       # (M, d)
        Delta = Hc - Hi_mean                 # (M, d)

        # U: (d, k_sub)，build_truth_subspace 在这个分支里返回的是 Delta 的 PCA basis
        # 1.1 和 1.2 都会用到 Z = Delta 在 U 子空间里的坐标
        Z = Delta @ U                        # (M, k)
        if cluster_mode == "base":
            clusterer=AutoKMeansClustering(
                steering_vectors=Delta,          # [M, d]
                k_range=(4, 10),
                use_pca=True,
                pca_components=256
            )
            _ = clusterer.find_optimal_k(methods=['elbow', 'silhouette', 'calinski'])

            cluster_centers, labels = clusterer.fit_final_clustering()
            # cluster_centers_z: (K, k)  -> 映射回原始空间
            # cluster_centers = cluster_centers_z @ U.T    # (K, d)

            r = None
            selector = cluster_centers                   # (K, d)
            return r, U, selector
        # ------------ 1.1: 在投影后的 Δ 上聚类 ------------
        elif cluster_mode == "delta_pca":
            # 在 Z 空间上聚类，这里不再让 AutoKMeansClustering 自己做 PCA
            print("delta_pca")
            clusterer = AutoKMeansClustering(
                steering_vectors=Z,          # [M, k]
                k_range=(3, 10),
                use_pca=False,               # 已经是低维特征
                pca_components=None
            )
            _ = clusterer.find_optimal_k(methods=['elbow', 'silhouette', 'calinski'])

            cluster_centers_z, labels = clusterer.fit_final_clustering()
            # cluster_centers_z: (K, k)  -> 映射回原始空间
            cluster_centers = cluster_centers_z @ U.T    # (K, d)

            r = None
            selector = cluster_centers                   # (K, d)
            return r, U, selector

        # ------------ 1.2: 在 (question, Δ) 联合特征上聚类 ------------
        elif cluster_mode == "joint":
            # 先堆叠 question hidden
            print("joint")
            Hq = torch.stack(hq_list).squeeze(1).to(device)  # (M, d)

            # 构造联合特征 F = [h_q ; λ * Z]
            # h_q: (M, d), Z: (M, k)
            F = torch.cat([Hq, joint_lambda * Z], dim=-1)    # (M, d + k)

            # 在联合特征上做 AutoKMeansClustering，内部再做一次 PCA 降到 256 维
            clusterer = AutoKMeansClustering(
                steering_vectors=F,          # [M, d + k]
                k_range=(3, 10),
                use_pca=True,
                pca_components=min(256, F.shape[1])
            )
            _ = clusterer.find_optimal_k(methods=['elbow', 'silhouette', 'calinski'])

            # 我们这里只要 label，用它在 Z 空间里算每个簇的 Δ 原型
            _, labels = clusterer.fit_final_clustering()     # labels: (M,)

            if isinstance(labels, torch.Tensor):
                labels_t = labels.to(device)
            else:
                # AutoKMeansClustering 里 labels 是 numpy array
                labels_t = torch.from_numpy(labels).to(device)

            unique_labels = labels_t.unique().tolist()

            cluster_centers_list = []
            for cid in unique_labels:
                mask = (labels_t == cid)
                if mask.sum() == 0:
                    continue
                # 在 Z (PCA 子空间) 内求该簇的均值
                z_mean = Z[mask].mean(dim=0)        # (k,)
                # 再映射回原空间 c_j = U @ z_j
                c_j = U @ z_mean                    # (d,)
                cluster_centers_list.append(c_j)

            # 堆成 (K_eff, d)；如果有空簇，K_eff 会 < 自动选出的 K，没关系
            cluster_centers = torch.stack(cluster_centers_list, dim=0)

            r = None
            selector = cluster_centers               # (K_eff, d)
            return r, U, selector

        else:
            raise ValueError(f"未知的 cluster_mode: {cluster_mode}, "
                             f"请使用 'delta_pca' 或 'joint'")

    else:
        raise ValueError(f"未知的 choose_method: {choose_method}")

