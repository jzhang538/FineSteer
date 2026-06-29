
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

class AutoKMeansClustering:
    """
    自动确定K值的K-means聚类
    """
    def __init__(
        self, 
        steering_vectors: torch.Tensor,  # [N, d]
        k_range: Tuple[int, int] = (2, 20),
        use_pca: bool = True,
        pca_components: int = 256,
        random_state: int = 42
    ):
        """
        Args:
            steering_vectors: 差值向量 [408, 3096]
            k_range: K值搜索范围 (min, max)
            use_pca: 是否降维后聚类（推荐True）
            pca_components: PCA降维目标维度
            random_state: 随机种子
        """
        self.V_original = steering_vectors.cpu().numpy()  # [N, d_original]
        self.N, self.d_original = self.V_original.shape
        self.k_range = k_range
        self.random_state = random_state
        
        # 降维（如果需要）
        if use_pca and self.d_original > pca_components:
            print(f"降维: {self.d_original} → {pca_components}")
            self.pca = PCA(n_components=pca_components, random_state=random_state)
            self.V_clustering = self.pca.fit_transform(self.V_original)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"保留方差: {explained_var:.2%}")
        else:
            self.pca = None
            self.V_clustering = self.V_original
        
        self.N, self.d_clustering = self.V_clustering.shape
        
        # 存储结果
        self.best_k = None
        self.cluster_centers_original = None  # 原始空间的centers [K, d_original]
        self.labels = None
        self.evaluation_results = {}
    
    def find_optimal_k(self, methods: List[str] = ['elbow', 'silhouette', 'calinski']) -> Dict:
        """
        使用多种方法确定最优K
        
        Args:
            methods: 评估方法列表
                - 'elbow': Elbow method (WCSS)
                - 'silhouette': Silhouette score
                - 'calinski': Calinski-Harabasz index
        
        Returns:
            各方法的评估结果
        """
        k_min, k_max = self.k_range
        k_values = range(k_min, k_max + 1)
        
        results = {
            'k_values': list(k_values),
            'wcss': [],
            'silhouette': [],
            'calinski': []
        }
        
        print(f"\n评估K值范围: {k_min} - {k_max}")
        print("=" * 60)
        
        for k in k_values:
            # 运行K-means
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(self.V_clustering)
            
            # 方法1: WCSS (Elbow)
            if 'elbow' in methods:
                wcss = kmeans.inertia_
                results['wcss'].append(wcss)
            
            # 方法2: Silhouette Score
            if 'silhouette' in methods and k > 1:
                sil_score = silhouette_score(self.V_clustering, labels, sample_size=min(5000, self.N))
                results['silhouette'].append(sil_score)
            
            # 方法3: Calinski-Harabasz Index
            if 'calinski' in methods:
                ch_score = calinski_harabasz_score(self.V_clustering, labels)
                results['calinski'].append(ch_score)
            
            print(f"K={k:2d} | WCSS={wcss:10.2f} | Silhouette={sil_score:.3f} | CH={ch_score:.2f}")
        
        self.evaluation_results = results
        
        # 自动选择最优K
        self._select_best_k(methods)
        
        return results
    
    def _select_best_k(self, methods: List[str]):
        """自动选择最优K"""
        k_values = self.evaluation_results['k_values']
        suggestions = {}
        
        # Elbow method - 找最大二阶差分
        if 'elbow' in methods and self.evaluation_results['wcss']:
            wcss = np.array(self.evaluation_results['wcss'])
            # 归一化
            wcss_norm = (wcss - wcss.min()) / (wcss.max() - wcss.min())
            
            # 计算二阶差分
            if len(wcss) > 2:
                second_diff = np.abs(np.diff(wcss_norm, n=2))
                elbow_idx = np.argmax(second_diff) + 1  # +1因为diff减少了长度
                suggestions['elbow'] = k_values[elbow_idx]
        
        # Silhouette method - 找最大值
        if 'silhouette' in methods and self.evaluation_results['silhouette']:
            sil_scores = self.evaluation_results['silhouette']
            best_idx = np.argmax(sil_scores)
            suggestions['silhouette'] = k_values[best_idx]
        
        # Calinski-Harabasz method - 找最大值
        if 'calinski' in methods and self.evaluation_results['calinski']:
            ch_scores = self.evaluation_results['calinski']
            best_idx = np.argmax(ch_scores)
            suggestions['calinski'] = k_values[best_idx]
        
        print("\n" + "=" * 60)
        print("各方法推荐的K值:")
        for method, k in suggestions.items():
            print(f"  {method:12s}: K = {k}")
        
        # 取中位数或众数
        suggested_ks = list(suggestions.values())
        self.best_k = int(np.median(suggested_ks))
        print(f"\n最终选择: K = {self.best_k}")
        print("=" * 60)
    
    def fit_final_clustering(self, k: Optional[int] = None):
        """
        用选定的K值进行最终聚类
        
        Args:
            k: 指定K值（如果为None，使用自动选择的best_k）
        """
        if k is None:
            k = self.best_k
            if k is None:
                raise ValueError("请先运行find_optimal_k()或手动指定k值")
        
        print(f"\n最终聚类: K = {k}")
        
        # 在降维空间聚类
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=20,  # 更多初始化确保稳定
            max_iter=500
        )
        self.labels = kmeans.fit_predict(self.V_clustering)
        
        # 关键：在原始空间计算cluster centers
        self.cluster_centers_original = np.zeros((k, self.d_original))
        cluster_sizes = []
        
        for i in range(k):
            mask = (self.labels == i)
            cluster_size = mask.sum()
            cluster_sizes.append(cluster_size)
            
            # 在原始3096维空间求平均
            self.cluster_centers_original[i] = self.V_original[mask].mean(axis=0)
        
        print(f"\nCluster分布:")
        for i, size in enumerate(cluster_sizes):
            print(f"  Cluster {i}: {size:3d} samples ({size/self.N*100:.1f}%)")
        
        # 转换为torch tensor
        self.cluster_centers_original = torch.tensor(
            self.cluster_centers_original, 
            dtype=torch.float32
        )
        
        return self.cluster_centers_original, self.labels
    
    def plot_evaluation(self, save_path: Optional[str] = "./"):
        """可视化K值评估结果"""
        if not self.evaluation_results:
            raise ValueError("请先运行find_optimal_k()")
        
        k_values = self.evaluation_results['k_values']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Elbow (WCSS)
        if self.evaluation_results['wcss']:
            ax = axes[0]
            wcss = self.evaluation_results['wcss']
            ax.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax.set_ylabel('WCSS', fontsize=12)
            ax.set_title('Elbow Method', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Silhouette Score
        if self.evaluation_results['silhouette']:
            ax = axes[1]
            sil = self.evaluation_results['silhouette']
            ax.plot(k_values, sil, 'go-', linewidth=2, markersize=8)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Good threshold')
            ax.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax.set_ylabel('Silhouette Score', fontsize=12)
            ax.set_title('Silhouette Method', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Calinski-Harabasz Index
        if self.evaluation_results['calinski']:
            ax = axes[2]
            ch = self.evaluation_results['calinski']
            ax.plot(k_values, ch, 'mo-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax.set_ylabel('Calinski-Harabasz Index', fontsize=12)
            ax.set_title('Calinski-Harabasz Method', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()


class ClusterBasedVectorSelector:
    """
    基于聚类的Steering Vector选择器
    """
    def __init__(
        self,
        cluster_centers: torch.Tensor,  # [K, d_model]
        selection_method: str = 'nearest'
    ):
        """
        Args:
            cluster_centers: 聚类中心 [K, 3096]
            selection_method: 选择方法
                - 'nearest': 最近邻（余弦相似度）
                - 'top_k': top-k加权组合
        """
        self.cluster_centers = cluster_centers  # [K, d_model]
        self.K, self.d_model = cluster_centers.shape
        self.selection_method = selection_method
        
        # 归一化centers（用于余弦相似度）
        self.cluster_centers_normalized = torch.nn.functional.normalize(
            cluster_centers, p=2, dim=1
        )
        
        print(f"初始化Vector Selector: {self.K} 个cluster centers, dim={self.d_model}")
    
    def select(
        self, 
        query_embedding: torch.Tensor,  # [d_model] or [batch, d_model]
        top_k: int = 1,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        为query选择steering vector
        
        Args:
            query_embedding: query的embedding [d_model] or [batch, d_model]
            top_k: 使用top-k个centers（如果method='top_k'）
            temperature: softmax温度
        
        Returns:
            selected_vector: 选择的steering vector [d_model] or [batch, d_model]
            info: 选择信息（index, similarity等）
        """
        # 处理batch维度
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # [1, d_model]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = query_embedding.shape[0]
        device = query_embedding.device
        
        # 确保centers在同一设备
        if self.cluster_centers.device != device:
            self.cluster_centers = self.cluster_centers.to(device)
            self.cluster_centers_normalized = self.cluster_centers_normalized.to(device)
        
        # 归一化query
        query_normalized = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        # 计算余弦相似度 [batch, K]
        similarities = torch.mm(query_normalized, self.cluster_centers_normalized.T)
        
        if self.selection_method == 'nearest':
            # 最近邻选择
            best_indices = similarities.argmax(dim=1)  # [batch]
            selected_vectors = self.cluster_centers[best_indices]  # [batch, d_model]
            
            info = {
                'selected_indices': best_indices.cpu().numpy(),
                'similarities': similarities.cpu().numpy(),
                'max_similarity': similarities.max(dim=1)[0].cpu().numpy()
            }
        
        elif self.selection_method == 'top_k':
            # Top-k加权组合
            topk_values, topk_indices = similarities.topk(top_k, dim=1)  # [batch, top_k]
            
            # Softmax权重
            weights = torch.softmax(topk_values / temperature, dim=1)  # [batch, top_k]
            
            # 加权组合
            selected_vectors = torch.zeros(batch_size, self.d_model, device=device)
            for i in range(batch_size):
                for j in range(top_k):
                    idx = topk_indices[i, j]
                    w = weights[i, j]
                    selected_vectors[i] += w * self.cluster_centers[idx]
            
            info = {
                'topk_indices': topk_indices.cpu().numpy(),
                'topk_similarities': topk_values.cpu().numpy(),
                'weights': weights.cpu().numpy()
            }
        
        else:
            raise ValueError(f"Unknown method: {self.selection_method}")
        
        if squeeze_output:
            selected_vectors = selected_vectors.squeeze(0)
        
        return selected_vectors, info
    
    def batch_select(
        self,
        query_embeddings: torch.Tensor,  # [batch, d_model]
        **kwargs
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """批量选择"""
        return self.select(query_embeddings, **kwargs)