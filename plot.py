import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 填入你两个 JSON 文件的路径
json_path_1 = "experiments/DPSUR, eps=3.0, 2025.0403.06.17.json"
json_path_2 = "experiments/DPSGD, eps=3.0, 2025.0403.03.24.json"

# 从 JSON 文件中读取数据
def load_experiment_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['epsilon_list'], data['loss_list'], data['args']['algorithm']

# 加载两个实验
epsilons_1, losses_1, name_1 = load_experiment_data(json_path_1)
epsilons_2, losses_2, name_2 = load_experiment_data(json_path_2)

# 每 10 个点取一个子集
epsilons_1_sub = epsilons_1[::20]
losses_1_sub = losses_1[::20]
epsilons_2_sub = epsilons_2[::20]
losses_2_sub = losses_2[::20]

# 截断 epsilon 范围到 [1.5, 3.0]
def filter_by_epsilon(epsilons, losses, lower=2.0, upper=3.0):
    return zip(*[(e, l) for e, l in zip(epsilons, losses) if lower <= e <= upper])

epsilons_1_sub, losses_1_sub = filter_by_epsilon(epsilons_1_sub, losses_1_sub)
epsilons_2_sub, losses_2_sub = filter_by_epsilon(epsilons_2_sub, losses_2_sub)

# 平滑处理
smoothed_losses_1 = gaussian_filter1d(losses_1_sub, sigma=1)
smoothed_losses_2 = gaussian_filter1d(losses_2_sub, sigma=1)

# 绘图
plt.figure(figsize=(8, 5))

plt.plot(epsilons_1_sub, smoothed_losses_1, color='red', marker='o',
         label=f"{name_1} (Smoothed)", alpha=0.7)
plt.plot(epsilons_2_sub, smoothed_losses_2, color='blue', marker='o',
         label=f"{name_2} (Smoothed)", alpha=0.7)

plt.xlabel("Epsilon")
plt.ylabel("Test Loss")
plt.title("Comparison of Test Loss vs Epsilon\n(2.0 ≤ ε ≤ 3.0, Sampled Every 20 Points + Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 可选：保存图像
plt.savefig("experiments/compare_DPSGD_vs_DPSUR_epsilon_1.5_3.0.png")
plt.show()
