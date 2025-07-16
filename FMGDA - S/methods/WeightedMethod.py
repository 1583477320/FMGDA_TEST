import torch
import numpy as np
# import quadprog

class MultiObjectiveWeightOptimizer():
    def __init__(self):
        pass

    def optimize(self, G: torch.Tensor) -> torch.Tensor:
        """
        计算最小化 ||GG @ w||^2 且满足 sum(w)=1 的 w
        Args:
            GG: 形状 (n, m) 的矩阵
        Returns:
            w: 形状 (m,) 的最优向量
        """
        # 计算 A = G^T @ G
        G = sum(G)/len(G)
        A = G.T @ G

        # 构造全 1 向量
        ones = torch.ones(G.size(1), device=G.device, dtype=G.dtype)

        # 解线性方程组 A w = ones (使用伪逆保证数值稳定性)
        try:
            # 尝试直接求解（若 A 可逆）
            v = torch.linalg.solve(A, ones)
        except RuntimeError:  # 若 A 奇异，使用伪逆
            A_pinv = torch.linalg.pinv(A)
            v = A_pinv @ ones

        # 归一化以满足 sum(w)=1
        w = v / v.sum()
        return w

# class QuadprogWeightOptimizer():
#     def __init__(self, G: torch.Tensor):
#         self.G = G
#
#     def optimize(self,) -> torch.Tensor:
#         """
#         向量化实现 - 仅满足 ∑αᵢ = 1
#         :param U: 客户端梯度列表
#         :return: 最优权重向量 alpha
#         """
#         G = self.G
#         G_matrix = sum(G)/len(G)
#         n = len(G_matrix.T)
#
#         # 计算Gram矩阵
#         K = G_matrix.T @ G_matrix  # [n, n]
#         K = K.numpy().astype(float)
#         Q = 0.5 * (K + K.T)
#         p = np.zeros(n)
#
#         # 3. 设置约束
#         G = np.ones((n, 1))  # 约束系数矩阵
#         h = np.array([1.0])   # 约束值
#
#         # 4. 求解
#         alpha = quadprog.solve_qp(Q, p, G, h, meq=1)[0]
#
#         return alpha


# 示例测试
if __name__ == "__main__":
    GG = [torch.rand((192,2)),
        torch.rand((192,2)),
        torch.rand((192,2))
    ]

    # weighted_optimizer = QuadprogWeightOptimizer(GG)
    # w_optimal = weighted_optimizer.optimize()

    weighted_optimizer_beta = MultiObjectiveWeightOptimizer()
    w_optimal = weighted_optimizer_beta.optimize(GG)
    print("GG =\n", GG)
    print("Optimal w =", w_optimal)
    print("Sum(w) =", w_optimal.sum().item())
    print("||GG @ w||^2 =", (GG @ w_optimal).pow(2).sum().item())