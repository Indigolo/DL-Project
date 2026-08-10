import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

LOG_FILE = "submission_template/src/log.txt"
OUTPUT_FILE = ""


def read_training_log(log_file):
    epochs = []
    train_mse_values = []
    val_mae_values = []
    val_rmse_values = []

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(
                r"epoch=(\d+)\s+"
                r"train_mse=([\d.eE+-]+)\s+"
                r"val_mae=([\d.eE+-]+)\s+"
                r"val_rmse=([\d.eE+-]+)",
                line,
            )
            if match:
                epoch = int(match.group(1))
                train_mse = float(match.group(2))
                val_mae = float(match.group(3))
                val_rmse = float(match.group(4))

                epochs.append(epoch)
                train_mse_values.append(train_mse)
                val_mae_values.append(val_mae)
                val_rmse_values.append(val_rmse)

    return epochs, train_mse_values, val_mae_values, val_rmse_values


def main():
    epochs, train_mse_values, val_mae_values, val_rmse_values = read_training_log(
        LOG_FILE
    )

    # if not losses:
    # print("No training loss found in the log file.")
    # return

    plt.figure(figsize=(100, 60))
    plt.plot(epochs, train_mse_values, marker="o", label="Train MSE")
    plt.plot(epochs, val_mae_values, marker="o", label="Validation MAE")
    plt.plot(epochs, val_rmse_values, marker="o", label="Validation RMSE")

    plt.xlabel("Epoch", fontsize=32)
    plt.ylabel("", fontsize=32)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))

    plt.tick_params(axis="both", labelsize=16)

    plt.title("PatchTST Training and Validation Metrics", fontsize=32)

    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    # use when want to save the graph
    # plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()

    # print(f"Graph saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
