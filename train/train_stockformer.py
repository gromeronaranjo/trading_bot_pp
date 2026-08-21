import os
import sys
import time
import math
import torch
from datetime import timedelta
from dataclasses import dataclass
from google.colab import drive

drive.mount("/content/drive")

project_directory = "/content/drive/MyDrive/personal_project"
sys.path.append(project_directory)

from models.stockformer import *


@dataclass
class StockFormerConfig:
    n_stocks: int = 23
    n_features: int = 362
    dense_size: int = 128
    n_heads: int = 1
    n_encoder_blocks: int = 2
    dropout: float = 0.2
    time_steps: int = 20
    prediction_steps: int = 2
    kernel_size: int = 2
    learning_rate: float = 0.001
    lr_decay: float = 0.1
    batch_size: int = 16
    epochs: int = 100
    classification_loss_weight: float = 2.0
    evaluation_every_steps: int = 20
    evaluation_batches: int = 4


class DataSaving:
    def __init__(self, save_loss_directory):
        self.directory = save_loss_directory
        self.regression_train_loss_list = []
        self.regression_test_loss_list = []
        self.classifier_train_loss_list = []
        self.classifier_test_loss_list = []
        self.struc2vec_train_loss_list = []
        self.combined_loss_list = []
        self.validation_accuracy_list = []
        self.test_accuracy_list = []
        self.steps = []

        os.makedirs(self.directory, exist_ok=True)

    def _to_number(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return value

    def save_data(self, step, regression_train_loss, regression_test_loss, classifier_train_loss, classifier_test_loss, struc2vec_train_loss, combined_loss, validation_accuracy, test_accuracy):
        self.steps.append(step)
        self.regression_train_loss_list.append(self._to_number(regression_train_loss))
        self.regression_test_loss_list.append(self._to_number(regression_test_loss))
        self.classifier_train_loss_list.append(self._to_number(classifier_train_loss))
        self.classifier_test_loss_list.append(self._to_number(classifier_test_loss))
        self.struc2vec_train_loss_list.append(self._to_number(struc2vec_train_loss))
        self.combined_loss_list.append(self._to_number(combined_loss))
        self.validation_accuracy_list.append(self._to_number(validation_accuracy))
        self.test_accuracy_list.append(self._to_number(test_accuracy))

    def save(self, filename="losses.pt"):
        data = {
            "steps": self.steps,
            "regression_train_loss": self.regression_train_loss_list,
            "regression_test_loss": self.regression_test_loss_list,
            "classifier_train_loss": self.classifier_train_loss_list,
            "classifier_test_loss": self.classifier_test_loss_list,
            "struc2vec_train_loss": self.struc2vec_train_loss_list,
            "combined_loss": self.combined_loss_list,
            "validation_accuracy": self.validation_accuracy_list,
            "test_accuracy": self.test_accuracy_list
        }

        torch.save(data, os.path.join(self.directory, filename))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_losses(model, x, y_regression, y_classifier, te, config):
    return_pred, direction_pred, low_return_pred, low_direction_pred, struc2vec_loss = model(x, te)

    return_pred = return_pred.squeeze(-1)
    direction_pred = direction_pred.squeeze(-1)
    low_return_pred = low_return_pred.squeeze(-1)
    low_direction_pred = low_direction_pred.squeeze(-1)

    low_y_regression, _ = wavelet_decompose(y_regression)

    regression_loss = (
        torch.nn.functional.l1_loss(return_pred, y_regression)
        + torch.nn.functional.l1_loss(low_return_pred, low_y_regression)
    )

    classifier_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            direction_pred,
            y_classifier.float()
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            low_direction_pred,
            y_classifier.float()
        )
    )

    combined_loss = (
        regression_loss
        + config.classification_loss_weight * classifier_loss
        + struc2vec_loss
    )

    return regression_loss, classifier_loss, struc2vec_loss, combined_loss, direction_pred


def evaluate(model, X, Y_regression, Y_classifier, TE, indices, config, max_batches):
    model.eval()

    regression_total = 0.0
    classifier_total = 0.0
    struc2vec_total = 0.0
    combined_total = 0.0
    correct = 0
    total = 0
    number_batches = 0

    with torch.no_grad():
        for start in range(0, len(indices), config.batch_size):
            if number_batches >= max_batches:
                break

            batch_indices = indices[start:start + config.batch_size]

            x = X[batch_indices].to(device)
            y_regression = Y_regression[batch_indices].to(device)
            y_classifier = Y_classifier[batch_indices].to(device)
            te = TE[batch_indices].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):
                regression_loss, classifier_loss, struc2vec_loss, combined_loss, direction_pred = calculate_losses(
                    model,
                    x,
                    y_regression,
                    y_classifier,
                    te,
                    config
                )

            predicted_direction = (torch.sigmoid(direction_pred) >= 0.5).long()

            correct += (predicted_direction == y_classifier).sum().item()
            total += y_classifier.numel()

            regression_total += regression_loss.item()
            classifier_total += classifier_loss.item()
            struc2vec_total += struc2vec_loss.item()
            combined_total += combined_loss.item()
            number_batches += 1

    return (
        regression_total / number_batches,
        classifier_total / number_batches,
        struc2vec_total / number_batches,
        combined_total / number_batches,
        correct / total
    )


def train():
    X = torch.load(
        os.path.join(project_directory, "data", "X.pt"),
        map_location="cpu"
    )

    Y_regression = torch.load(
        os.path.join(project_directory, "data", "Y_regression.pt"),
        map_location="cpu"
    )

    Y_classifier = torch.load(
        os.path.join(project_directory, "data", "Y_classifier.pt"),
        map_location="cpu"
    )

    TE = torch.load(
        os.path.join(project_directory, "data", "TE.pt"),
        map_location="cpu"
    )

    print("x:", X.shape)
    print("y regression:", Y_regression.shape)
    print("y classifier:", Y_classifier.shape)
    print("te:", TE.shape)
    print("device:", device)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    if len(X) != len(Y_regression) or len(X) != len(Y_classifier) or len(X) != len(TE):
        raise ValueError("x, y and te do not have the same number of samples")

    config = StockFormerConfig(
        n_stocks=X.shape[2],
        n_features=X.shape[3]
    )

    model = StockFormer(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay,
        patience=20,
        min_lr=2e-6
    )

    logs_directory = os.path.join(project_directory, "logs")
    checkpoint_directory = os.path.join(logs_directory, "checkpoints")

    data_saver = DataSaving(logs_directory)

    os.makedirs(checkpoint_directory, exist_ok=True)

    number_samples = len(X)

    train_end = int(number_samples * 0.75)
    validation_end = int(number_samples * 0.875)

    train_indices = torch.arange(0, train_end)
    validation_indices = torch.arange(train_end, validation_end)
    test_indices = torch.arange(validation_end, number_samples)

    print("train samples:", len(train_indices))
    print("validation samples:", len(validation_indices))
    print("test samples:", len(test_indices))

    steps_per_epoch = math.ceil(len(train_indices) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    print("steps per epoch:", steps_per_epoch)
    print("total steps:", total_steps)

    best_test_accuracy = 0.0
    global_step = 0
    training_start_time = time.time()

    regression_train_total = 0.0
    classifier_train_total = 0.0
    struc2vec_train_total = 0.0
    combined_train_total = 0.0
    number_train_batches = 0

    for epoch in range(config.epochs):
        model.train()

        permutation = train_indices[torch.randperm(len(train_indices))]

        for start in range(0, len(permutation), config.batch_size):
            batch_indices = permutation[start:start + config.batch_size]

            x = X[batch_indices].to(device)
            y_regression = Y_regression[batch_indices].to(device)
            y_classifier = Y_classifier[batch_indices].to(device)
            te = TE[batch_indices].to(device)

            optimizer.zero_grad()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):
                regression_loss, classifier_loss, struc2vec_loss, combined_loss, _ = calculate_losses(
                    model,
                    x,
                    y_regression,
                    y_classifier,
                    te,
                    config
                )

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            regression_train_total += regression_loss.item()
            classifier_train_total += classifier_loss.item()
            struc2vec_train_total += struc2vec_loss.item()
            combined_train_total += combined_loss.item()
            number_train_batches += 1

            global_step += 1

            elapsed_time = time.time() - training_start_time
            average_step_time = elapsed_time / global_step
            remaining_steps = total_steps - global_step
            remaining_seconds = average_step_time * remaining_steps
            eta = str(timedelta(seconds=int(remaining_seconds)))

            print(
                f"epoch {epoch + 1}/{config.epochs} "
                f"| step {global_step}/{total_steps} "
                f"| regression {regression_loss.item():.6f} "
                f"| classifier {classifier_loss.item():.6f} "
                f"| struc2vec {struc2vec_loss.item():.6f} "
                f"| total {combined_loss.item():.6f} "
                f"| eta {eta}"
            )

            if global_step % config.evaluation_every_steps == 0:
                regression_train_loss = regression_train_total / number_train_batches
                classifier_train_loss = classifier_train_total / number_train_batches
                struc2vec_train_loss = struc2vec_train_total / number_train_batches
                combined_train_loss = combined_train_total / number_train_batches

                regression_validation_loss, classifier_validation_loss, struc2vec_validation_loss, combined_validation_loss, validation_accuracy = evaluate(
                    model,
                    X,
                    Y_regression,
                    Y_classifier,
                    TE,
                    validation_indices,
                    config,
                    config.evaluation_batches
                )

                regression_test_loss, classifier_test_loss, struc2vec_test_loss, combined_test_loss, test_accuracy = evaluate(
                    model,
                    X,
                    Y_regression,
                    Y_classifier,
                    TE,
                    test_indices,
                    config,
                    config.evaluation_batches
                )

                scheduler.step(regression_validation_loss)

                data_saver.save_data(
                    step=global_step,
                    regression_train_loss=regression_train_loss,
                    regression_test_loss=regression_test_loss,
                    classifier_train_loss=classifier_train_loss,
                    classifier_test_loss=classifier_test_loss,
                    struc2vec_train_loss=struc2vec_train_loss,
                    combined_loss=combined_test_loss,
                    validation_accuracy=validation_accuracy,
                    test_accuracy=test_accuracy
                )

                data_saver.save("losses.pt")

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy

                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_directory, "best.pt")
                    )

                    print("saved new best model")

                elapsed_time = time.time() - training_start_time
                average_step_time = elapsed_time / global_step
                remaining_steps = total_steps - global_step
                remaining_seconds = average_step_time * remaining_steps
                eta = str(timedelta(seconds=int(remaining_seconds)))

                print()
                print(f"step {global_step} evaluation")
                print(f"train regression: {regression_train_loss:.6f}")
                print(f"validation regression: {regression_validation_loss:.6f}")
                print(f"test regression: {regression_test_loss:.6f}")
                print(f"train classifier: {classifier_train_loss:.6f}")
                print(f"validation classifier: {classifier_validation_loss:.6f}")
                print(f"test classifier: {classifier_test_loss:.6f}")
                print(f"validation total: {combined_validation_loss:.6f}")
                print(f"test total: {combined_test_loss:.6f}")
                print(f"validation accuracy: {validation_accuracy * 100:.2f}%")
                print(f"test accuracy: {test_accuracy * 100:.2f}%")
                print(f"best test accuracy: {best_test_accuracy * 100:.2f}%")
                print(f"lr: {optimizer.param_groups[0]['lr']:.8f}")
                print(f"eta: {eta}")
                print()

                regression_train_total = 0.0
                classifier_train_total = 0.0
                struc2vec_train_total = 0.0
                combined_train_total = 0.0
                number_train_batches = 0

                model.train()

train()