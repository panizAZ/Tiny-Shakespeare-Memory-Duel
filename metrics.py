import matplotlib.pyplot as plt
import torch


def compare_elman_jordan(elman_path, jordan_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    elman = torch.load(elman_path, map_location=device)
    jordan = torch.load(jordan_path, map_location=device)


    elman_train_loss = elman["train_losses"]
    elman_train_ppls = elman["train_ppls"]
    elman_train_accuracy = elman["train_accuracies"]
    elman_val_loss = elman["val_losses"]
    elman_val_ppls = elman["val_ppls"]
    elman_val_accuracy = elman["val_accuracies"]

    jordan_train_loss = jordan["train_losses"]
    jordan_train_ppls = jordan["train_ppls"]
    jordan_train_accuracy = jordan["train_accuracies"]
    jordan_val_loss = jordan["val_losses"]
    jordan_val_ppls = jordan["val_ppls"]
    jordan_val_accuracy = jordan["val_accuracies"]

    # Elman Val vs Train loss
    val_train_loss_plot(elman_train_loss, elman_val_loss, "Elman")
    # Jordan Val vs Train loss
    val_train_loss_plot(jordan_train_loss, jordan_val_loss, "Jordan")

    # Elman Val vs Train perplexity
    val_train_ppl_plot(elman_train_ppls, elman_val_ppls, "Elman")
    # Jordan Val vs Train perplexity
    val_train_ppl_plot(jordan_train_ppls, jordan_val_ppls, "Jordan")

    # Compare Jordan vs Elman perplexity
    compare_models(
        elman_train_ppls,
        jordan_train_ppls,
        "Train Perplexity Comparison",
        "Train Perplexity"
    )

    compare_models(
        elman_val_ppls,
        jordan_val_ppls,
        "Validation Perplexity Comparison",
        "Validation Perplexity"
    )


    compare_models(
        elman_val_accuracy,
        jordan_val_accuracy,
        "Validation Accuracy Comparison",
        "Validation Accuracy"
    )

    compare_models(
        elman_train_accuracy,
        jordan_train_accuracy,
        "Train Accuracy Comparison",
        "Train Accuracy"
    )


def val_train_loss_plot(train_loss, val_loss, rnn_model):

    plt.figure(figsize=(8,5))

    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(f"{rnn_model} Loss Curve")

    plt.legend()
    plt.grid(True)

    plt.show()


def val_train_ppl_plot(train_ppl, val_ppl, rnn_model="Elman"):
    plt.figure(figsize=(8, 5))

    plt.plot(train_ppl, label="Train PPL")
    plt.plot(val_ppl, label="Validation PPL")

    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")

    plt.title(f"{rnn_model} perplexity curve")
    plt.legend()
    plt.grid(True)

    plt.show()


def compare_models(elman_metric, jordan_metric, title, ylabel):

    plt.figure(figsize=(8,5))

    plt.plot(elman_metric, label="Elman")
    plt.plot(jordan_metric, label="Jordan")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare_elman_jordan("elman_model.pth", "jordan_model.pth")