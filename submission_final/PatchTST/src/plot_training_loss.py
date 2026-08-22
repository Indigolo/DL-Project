import re

import matplotlib.pyplot as plt

LOG_FILE = "submission_template/src/log.txt"
OUTPUT_FILE = ""


def read_training_log(log_file):
    epochs = []
    losses = []

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(
                r"Epoch \[(\d+(?:\.\d+)?)\] Train Loss: ([\d.eE+-]+)", line
            )

            if match:
                epoch_progress = float(match.group(1))
                loss = float(match.group(2))

                # Convert [0.04, 0.08, ...] back to epoch number
                # if the log was created with 25 epochs.
                epochs.append(epoch_progress)
                losses.append(loss)

    return epochs, losses


def main():
    epochs, losses = read_training_log(LOG_FILE)

    if not losses:
        print("No training loss found in the log file.")
        return

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, losses, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Train MSE Loss")
    plt.title("Training Loss")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # use when want to save the graph
    # plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()

    # print(f"Graph saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
