import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def log2graph(log_file):
    train = []
    valid = []
    test = []

    with open(log_file, "r") as f:
        for line in f:
            m = re.search(
                r"Train Loss:\s*([\d.]+)\s*Vali Loss:\s*([\d.]+)\s*Test Loss:\s*([\d.]+)",
                line,
            )
            if m:
                train.append(float(m.group(1)))
                valid.append(float(m.group(2)))
                test.append(float(m.group(3)))
                print(train)
                print(valid)
                print(test)

    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(100, 60))
    plt.plot(epochs, train, marker="o", label="Train")
    plt.plot(epochs, valid, marker="o", label="Validation")
    plt.plot(epochs, test, marker="o", label="Test")

    plt.xlabel("Epoch", fontsize=32)
    plt.ylabel("", fontsize=32)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=18)
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.show()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        "-n",
        type=str,
        default="logs/LongForecasting/P_sLSTM_Custom_336_96.log",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    log2graph(args.filename)
