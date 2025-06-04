import pandas as pd
import matplotlib.pyplot as plt

path_results_n = "train_n_to_convergence/results.csv"
path_results_s = "train_s_to_convergence/results.csv"

panda_n = pd.read_csv(path_results_n)
panda_s = pd.read_csv(path_results_s)

# print(panda_n.head())
# print(panda_s.head())

# print(panda_n.tail())
# print(panda_s.tail())

# print(panda_n[3:4])
# ts = panda_n[4]

# ts = ts.cumsum()
#
# plt.savefig("test.png")

print(panda_n.describe())
print(panda_s.describe())
