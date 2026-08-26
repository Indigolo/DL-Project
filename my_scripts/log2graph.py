import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def log2graph(log_file):
    epochs = []
    train_mse = []
    val_rmse = []

    with open(log_file, "r") as f:
        for line in f:
            m = re.search(
                r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*"
                r"Train MSE:\s*([\d.eE+-]+)\s+"
                r"Vali MAE:\s*([\d.eE+-]+)\s+"
                r"Vali RMSE:\s*([\d.eE+-]+)",
                line,
            )

            if m:
                epochs.append(int(m.group(1)))
                train_mse.append(float(m.group(2)))
                val_rmse.append(float(m.group(4)))

    if not epochs:
        print("No epoch results found in the log.")
        return

    print(f"Found {len(epochs)} epochs.")

    plt.figure(figsize=(20, 12))

    plt.plot(
        epochs,
        train_mse,
        marker="o",
        label="Train MSE",
    )

    plt.plot(
        epochs,
        val_rmse,
        marker="o",
        label="Validation RMSE",
    )

    plt.xlabel("Epoch", fontsize=32)
    plt.ylabel("Metric", fontsize=32)

    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))

    plt.tick_params(axis="both", labelsize=16)

    plt.title(
        "P_sLSTM Training and Validation",
        fontsize=32,
    )

    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)

    plt.show()


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        "-n",
        type=str,
        default="log.log",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    log2graph(args.filename)