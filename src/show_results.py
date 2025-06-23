import polars as pl
import matplotlib.pyplot as plt
from polars.series import array
from PathHandler import PathHandler


ph = PathHandler


def main():
    # plot_awl()
    plot_cifar()
    # all_results()
    # all_detection()


def plot_awl():
    print()
    print("african-wildlife - create plots from csv")
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
    africa = [
        "experiments/african-wildlife/train_n_to_convergence/results.csv",
        "experiments/african-wildlife/train_n_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_s_to_convergence/results.csv",
        "experiments/african-wildlife/train_s_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_l_to_convergence/results.csv",
        "experiments/african-wildlife/train_l_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_m_to_convergence/results.csv",
        "experiments/african-wildlife/train_m_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_x_to_convergence/results.csv",
        "experiments/african-wildlife/train_x_to_convergence_all/results.csv",
    ]
    fig, axs = plt.subplots(nrows=len(select), ncols=1, figsize=(12, 16))

    for result_path in africa:
        df = pl.read_csv(result_path)
        for i, s in enumerate(select):
            selected = df.select(pl.col(s))
            extracted = pl.Series(selected).to_list()
            axs[i].plot(extracted)

    for i, s in enumerate(select):
        axs[i].set_ylabel(s, fontsize=16)

    legend = []
    for exp in africa:
        train = exp.split("/")[2]
        letter = train.split("_")[1]
        name = f"yolo_{letter}"
        legend += [name]

    plt.legend(legend)

    name = input('\nchoose plot name: f"plots/a-wl/{name}.png" \n\n')

    plt.savefig(f"plots/a-wl/{name}.png")
    plt.savefig(f"plots/a-wl/{name}.svg")


def all_results():
    all_results = [
        "experiments/VisDrone/initial_setup/results.csv",
        "experiments/VisDrone/train_n_to_convergence/result_200/results_end_200.csv",
        "experiments/VisDrone/train_n_to_convergence/results.csv",
        "experiments/african-wildlife/initial_setup/results.csv",
        "experiments/african-wildlife/train_l_to_convergence/results.csv",
        "experiments/african-wildlife/train_l_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_m_to_convergence/results.csv",
        "experiments/african-wildlife/train_m_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_n_to_convergence/results.csv",
        "experiments/african-wildlife/train_n_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_s_to_convergence/results.csv",
        "experiments/african-wildlife/train_s_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_x_to_convergence/results.csv",
        "experiments/african-wildlife/train_x_to_convergence_all/results.csv",
        "experiments/cifar/runs/classify/train10/results.csv",
        "experiments/cifar/runs/classify/train11/results.csv",
        "experiments/cifar/runs/classify/train12/results.csv",
        "experiments/cifar/runs/classify/train13/results.csv",
        "experiments/cifar/runs/classify/train14/results.csv",
        "experiments/cifar/runs/classify/train2/results.csv",
        "experiments/cifar/runs/classify/train3/results.csv",
        "experiments/cifar/runs/classify/train4/results.csv",
        "experiments/cifar/runs/classify/train9/results.csv",
        "experiments/cifar_2/runs/classify/train2/results.csv",
        "experiments/cifar_2/runs/classify/train3/results.csv",
        "experiments/cifar_2/runs/classify/train4/results.csv",
        "experiments/cifar_2/runs/classify/train5/results.csv",
        "experiments/cifar_2/runs/classify/train6/results.csv",
        "experiments/cifar_2/runs/classify/train7/results.csv",
        "experiments/coco128/initial_setup/results.csv",
        "experiments/coco128/train_n_to_convergence/results.csv",
        "experiments/coco8/initial_setup/results.csv",
        "experiments/coco8/train_n_to_convergence/results.csv",
        "experiments/coco8/train_n_to_convergence2/results.csv",
    ]
    for r in all_results:
        df = pl.read_csv(r)
        print(r.split("/")[1][:5], df.columns)


def all_detection():
    results = [
        # "experiments/VisDrone/initial_setup/results.csv",
        # "experiments/VisDrone/train_n_to_convergence/result_200/results_end_200.csv",
        # "experiments/VisDrone/train_n_to_convergence/results.csv",
        # "experiments/african-wildlife/initial_setup/results.csv",
        # "experiments/african-wildlife/train_l_to_convergence/results.csv",
        # "experiments/african-wildlife/train_l_to_convergence_all/results.csv",
        # "experiments/african-wildlife/train_m_to_convergence/results.csv",
        "experiments/african-wildlife/train_m_to_convergence_all/results.csv",
        "experiments/african-wildlife/train_n_to_convergence/results.csv",
        "experiments/african-wildlife/train_n_to_convergence_all/results.csv",
        # "experiments/african-wildlife/train_s_to_convergence/results.csv",
        "experiments/african-wildlife/train_s_to_convergence_all/results.csv",
        # "experiments/african-wildlife/train_x_to_convergence/results.csv",
        # "experiments/african-wildlife/train_x_to_convergence_all/results.csv",
        # "experiments/coco128/initial_setup/results.csv",
        # "experiments/coco128/train_n_to_convergence/results.csv",
        # "experiments/coco8/initial_setup/results.csv",
        # "experiments/coco8/train_n_to_convergence/results.csv",
        # "experiments/coco8/train_n_to_convergence2/results.csv",
    ]
    select = [
        "epoch",
        # "time",
        # "train/box_loss",
        # "train/cls_loss",
        # "train/dfl_loss",
        # "metrics/precision(B)",
        # "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        # "val/box_loss",
        # "val/cls_loss",
        # "val/dfl_loss",
        # "lr/pg0",
        # "lr/pg1",
        # "lr/pg2",
    ]
    # compare_table = pl.DataFrame(select + ["name"])
    data_collection = {}
    headers = select + ["score", "name"]
    for header in headers:
        data_collection[header] = []
    for result_path in results:
        name = result_path.split("/")[2]
        # name = result_path.split("/")[1] + "-" + result_path.split("/")[2]
        df = pl.read_csv(result_path)
        extracted = pl.Series(df)
        best_score = 0
        best_epoch = extracted[0]
        for epoch in extracted:
            score = 0.1 * epoch["metrics/mAP50(B)"] + 0.9 * epoch["metrics/mAP50-95(B)"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
        num = best_epoch["epoch"]

        for key, val in data_collection.items():
            if key in best_epoch:
                val.append(best_epoch[key])
            elif key == "name":
                val.append(name)
            elif key == "score":
                val.append(best_score)
            else:
                raise ValueError("wrong key setup")
            print(key, val)
        print(f"Best score: {round(best_score, 4)} achieved in epoch {num} - {name}")

    print(pl.DataFrame(data_collection).sort("score", descending=True))

    """
            Results detection
Best score: 0.1385 achieved in epoch 5 initial_setup
Best score: 0.1975 achieved in epoch 1 train_l_to_convergence
Best score: 0.2184 achieved in epoch 202 train_n_to_convergence
Best score: 0.2184 achieved in epoch 235 train_n_to_convergence
Best score: 0.2362 achieved in epoch 5 train_x_to_convergence
Best score: 0.5482 achieved in epoch 5 initial_setup
Best score: 0.5627 achieved in epoch 9 train_m_to_convergence
Best score: 0.6671 achieved in epoch 5 initial_setup
Best score: 0.6707 achieved in epoch 16 train_n_to_convergence
Best score: 0.6707 achieved in epoch 16 train_n_to_convergence2
Best score: 0.7509 achieved in epoch 5 initial_setup
Best score: 0.8058 achieved in epoch 83 train_n_to_convergence
Best score: 0.8094 achieved in epoch 135 train_n_to_convergence
Best score: 0.8165 achieved in epoch 105 train_n_to_convergence_all
Best score: 0.8241 achieved in epoch 198 train_x_to_convergence_all
Best score: 0.8261 achieved in epoch 161 train_m_to_convergence_all
Best score: 0.831 achieved in epoch 133 train_s_to_convergence
Best score: 0.8398 achieved in epoch 163 train_s_to_convergence_all
Best score: 0.8485 achieved in epoch 254 train_l_to_convergence_all
        """


def plot_cifar():
    print()
    print("cifar - create plots from csv")
    cifar = [
        "experiments/cifar/runs/classify/train2/results.csv",
        "experiments/cifar/runs/classify/train3/results.csv",
        "experiments/cifar/runs/classify/train4/results.csv",
        "experiments/cifar/runs/classify/train9/results.csv",
        "experiments/cifar/runs/classify/train10/results.csv",
        "experiments/cifar/runs/classify/train11/results.csv",
        "experiments/cifar/runs/classify/train12/results.csv",
        "experiments/cifar/runs/classify/train13/results.csv",
        "experiments/cifar/runs/classify/train14/results.csv",
        "experiments/cifar_2/runs/classify/train2/results.csv",
        "experiments/cifar_2/runs/classify/train3/results.csv",
        "experiments/cifar_2/runs/classify/train4/results.csv",
        "experiments/cifar_2/runs/classify/train5/results.csv",
        "experiments/cifar_2/runs/classify/train6/results.csv",
        "experiments/cifar_2/runs/classify/train7/results.csv",
    ]
    select = [
        # "epoch",
        # "time",
        "metrics/accuracy_top1",
        "metrics/accuracy_top5",
        # "train/loss",
        # "val/loss",
        # "lr/pg0",
        # "lr/pg1",
        # "lr/pg2",
    ]
    fig, axs = plt.subplots(ncols=len(select), nrows=1, figsize=(20, 5))

    legend = []
    for exp in cifar:
        name = exp.split("/")[1] + " - " + exp.split("/")[4]
        legend += [name]
    for result in cifar:
        print(result)
        df = pl.read_csv(result)
        for i, s in enumerate(select):
            selected = df.select(pl.col(s))
            extracted = pl.Series(selected).to_list()
            if len(extracted) > 230:
                extracted = extracted[0:230]
            axs[i].plot(extracted)

    for i, s in enumerate(select):
        axs[i].set_ylabel(s, fontsize=16)

    plt.legend(legend)
    # name = input("\nchoose plot name: ... \n\n")
    name = "top_1_5"

    plt.savefig(f"plots/cifar/{name}.png")
    plt.savefig(f"plots/cifar/{name}.svg")


if __name__ == "__main__":
    main()
