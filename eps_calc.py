from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps

# DPSUR 隐私预算消耗
batch_size = 40
batch_size_valid = 10
num_examples = 2000
dp_sigma_t = 1.229 # 2.54
dp_sigma_v = 1.5
dp_delta = 1e-5
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
t = 1000

rdp_train = compute_rdp(batch_size / num_examples, dp_sigma_t, t, orders)
rdp_valid = compute_rdp(batch_size_valid / num_examples, dp_sigma_v, t, orders)
epsilon, best_alpha = compute_eps(orders, rdp_train + rdp_valid, dp_delta)
print('DPSUR epsilon:', epsilon)

# DPSGD 隐私预算消耗
batch_size = 40
num_examples = 2000
dp_sigma_t = 1.217
dp_delta = 1e-5
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
t = 1000

rdp_train = compute_rdp(batch_size / num_examples, dp_sigma_t, t, orders)
epsilon, best_alpha = compute_eps(orders, rdp_train, dp_delta)
print('DPSGD epsilon:', epsilon)


