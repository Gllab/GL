import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from mindspore import Tensor, nn, Parameter, ops, context
from sklearn.preprocessing import normalize
from mindspore.ops import operations as P
import mindspore

context.set_context(device_target="CPU")

# SNMF模型
class C_SNMF(nn.Cell):
    # 模型初始化
    def __init__(self, num_nodes, n_components, mu, lamda):
        super(C_SNMF, self).__init__()
        self.mu = mu
        self.lamda = lamda
        self.n_components = n_components
        self.num_nodes = num_nodes
        np.random.seed(12)
        self.U = Tensor((np.abs(np.random.normal(scale=1. / n_components, size=(num_nodes, n_components)))), mindspore.float32)
        self.X = Tensor((np.abs(np.random.normal(scale=1. / n_components, size=(num_nodes, n_components)))), mindspore.float32)
        # self.U = mindspore.Tensor(shape=(n_components, num_nodes), dtype=mindspore.float32, init=Normal())
        # self.X = mindspore.Tensor(shape=(n_components, num_nodes), dtype=mindspore.float32, init=Normal())
        # self.dot = P.MatMul()
        self.transpose = ops.Transpose()
        self.perm = (1, 0)
        self.norm = P.L2Normalize(axis=0)

    # 模型更新
    def construct(self, adjacency_matrix, D):
        AX = adjacency_matrix @ self.X
        XUT = self.X @ self.transpose(self.U, self.perm)
        numerator = AX + self.mu * XUT @ self.X
        UXT = self.U @ self.transpose(self.X, self.perm)
        denominator = (1 + self.mu) * UXT @ self.X
        self.U *= numerator / denominator

        UXT = self.U @ self.transpose(self.X, self.perm)
        numerator = self.transpose(adjacency_matrix, self.perm) @ self.U + self.mu * UXT @ self.U + self.lamda * AX
        XUT = self.X @ self.transpose(self.U, self.perm)
        denominator = (1 + self.mu) * XUT @ self.U + self.lamda * D @ self.X
        self.X *= numerator / denominator

        return self.U, self.X

# 读取数据集
file_path = '../dataset/polblogs/polblogs.ungraph.txt'
ground_truth_file = '../dataset/polblogs/polblogs.community.txt'
data = np.loadtxt(file_path, delimiter='\t', dtype=int)

# 创建邻接矩阵
num_nodes = np.max(data)
adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
for edge in data:
    adjacency_matrix[edge[0] - 1, edge[1] - 1] = 1

# 创建图正则矩阵
D = np.diag(np.sum(adjacency_matrix, axis=1))

# 模型超参数设置
n_components = 2
mu = 10
lamda = 2e-4

model = C_SNMF(num_nodes, n_components, mu, lamda)

# Converting numpy arrays to MindSpore Tensors
adjacency_matrix = Tensor(adjacency_matrix, dtype=mindspore.float32)
D = Tensor(D, dtype=mindspore.float32)

# 模型训练
max_iterations = 500
tolerance = 1e-6
pre_rmse = float('inf')
count = 0

for iteration in range(max_iterations):
    U, X = model(adjacency_matrix, D)
    predicted_adjacency = np.dot(U.asnumpy(), X.asnumpy().T)
    cur_rmse = np.sqrt(np.mean((adjacency_matrix.asnumpy() - predicted_adjacency)**2))
    print("epoch %d: %.6f" % (iteration, cur_rmse))
    if pre_rmse - cur_rmse <= tolerance:
        count += 1
        if count == 5:
            break
    pre_rmse = cur_rmse

# Converting MindSpore Tensors to numpy arrays
U = U.asnumpy()
X = X.asnumpy()

# 归一化W和H以便可视化
U = normalize(U, norm='l2', axis=1)
X = normalize(X, norm='l2', axis=1)

# 绘制结果
community_assignments = np.argmax(X, axis=1)
color_map = ['b' if community_assignments[i] == 0 else 'r' for i in range(num_nodes)]

plt.scatter(X[:, 0], X[:, 1], c=color_map)
plt.title("Community Detection with NMF using Lee-Seung Rule")
plt.xlabel("Community 1")
plt.ylabel("Community 2")
plt.show()

# 输出社区分类结果
for i in range(n_components):
    nodes_in_community = np.where(community_assignments == i)[0]
    print(f"Community {i + 1}: {nodes_in_community}")

# 读取真实社区分类结果
with open(ground_truth_file, 'r') as file:
    lines = file.readlines()

ground_truth = []
for index, line in enumerate(lines):
    data = list(map(int, line.split()))
    max_value = max(data)
    if max_value >= len(ground_truth):
        ground_truth.extend([0] * (max_value - len(ground_truth)))
    for idx in data:
        ground_truth[idx-1] = index

# 计算指标
nmi = normalized_mutual_info_score(ground_truth, community_assignments)
print(f"NMI: {nmi:.4f}")
ari = adjusted_rand_score(ground_truth, community_assignments)
print(f"ARI: {ari:.4f}")
v_measure = v_measure_score(ground_truth, community_assignments)
print(f"V-measure: {v_measure:.4f}")
