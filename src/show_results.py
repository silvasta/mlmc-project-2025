import polars as pl
import matplotlib.pyplot as plt

dataset_name = "african-wildlife"
data = f"{dataset_name}.yaml"
project_name = "test_yolo"

select = [
    # "epoch",
    # "time",
    # "train/box_loss",
    # "train/cls_loss",
    # "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    # "val/box_loss",
    # "val/cls_loss",
    # "val/dfl_loss",
    # "lr/pg0",
    # "lr/pg1",
    # "lr/pg2",
]
fig, axs = plt.subplots(nrows=len(select), ncols=1, figsize=(12, 16))

model_types = ["n", "s", "m", "l", "x"]
for model_type in model_types:
    experiment_name = f"train_{model_type}_to_convergence_all"
    result_csv_path = f"{project_name}/{dataset_name}/{experiment_name}/results.csv"
    df = pl.read_csv(result_csv_path)
    for i, s in enumerate(select):
        # print(i)
        # print(s)
        # print()
        selected = df.select(pl.col(s))
        extracted = pl.Series(selected).to_list()
        axs[i].plot(extracted)

for i, s in enumerate(select):
    axs[i].set_ylabel(s, fontsize=16)
plt.legend(model_types)

plt.savefig("test.png")
plt.savefig("test.svg")
