import pandas as pd
import matplotlib.pyplot as plt


pandas = []
for t in ["n", "s", "m", "l", "x"]:
    path_results = f"train_{t}_to_convergence_all/results.csv"
    panda = pd.read_csv(path_results)
    pandas += [panda]

for p in pandas:
    # print(p.describe())
    print(p.head())

    # print(p["epoch"])

# ts = ts.cumsum()
#
# plt.savefig("test.png")


# print(panda_s.head())
# print(panda_s.tail())
# print(panda_n[3:4])
# ts = panda_n[4]
