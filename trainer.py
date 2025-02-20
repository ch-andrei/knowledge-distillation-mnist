# Implemented by Andrei Chubarau 2025

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# for reproducibility
MANUAL_SEED = 1337

# multiprocessing for data loaders
NUM_WORKERS_DATALOADER = 4
USE_CUDA = True

OUTPUT_DIR = "./checkpoints"

# MNIST dataset params
MNIST_IMG_SIZE = 28  # input images are 28x28 pixels
MNIST_NUM_CLASSES = 10  # number of digits (0 through 9)

# Model and training params are inspired by
# Hinton et al. "Distilling the Knowledge in a Neural Network" 2015
# Hinton et al. "Improving neural networks by preventing co-adaptation of feature detectors" 2012
# but to speed things up, some parameters are modified to more modern standards

DROPOUT_HIDDEN = 0.5  # hidden layer dropout
DROPOUT_INPUT = 0.2  # input layer dropout

BATCH_SIZE = 100
BATCH_SIZE_TEST = 1000
NUM_TRAIN_EPOCHS = 25

OPTIMIZER = "SGD"
OPTIMIZER_LR = 0.01
OPTIMIZER_LR_EXPONENTIAL_DECAY = 0.95
SGD_MOMENTUM = 0.9

CLIP_GRAD_NORM = True


def parse_args():
    parser = argparse.ArgumentParser()

    # tag for the current run
    parser.add_argument('--output_tag', type=str, default="")

    # TODO implement no_teacher mode
    # parser.add_argument('--no_teacher', action='store_true')

    parser.add_argument('--no_student', action='store_true')

    # student network params
    parser.add_argument('--hidden_dim', type=int, default=800)

    # distillation params
    parser.add_argument('--use_kd', action='store_true')
    parser.add_argument('--temperature', type=float, default=20)
    parser.add_argument('--remove_3s', action='store_true')

    # optional path to pretrained teacher model weights
    parser.add_argument('--teacher_ckpt', type=str, default="")

    return parser.parse_args()


# simple logging function
def logger(*args, verbose=True, **kwargs):
    if verbose:
        print(*args, **kwargs)


def make_layers(dim_in, dim_out, dropout):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Network(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, use_dropout=True):
        super().__init__()
        self.fc_in = make_layers(dim_in, dim_hidden, DROPOUT_INPUT if use_dropout else 0.)
        self.fc_hidden = nn.Sequential(
            *[
                make_layers(dim_hidden, dim_hidden, DROPOUT_HIDDEN if use_dropout else 0.) for _ in range(num_hidden)
            ]
        )
        self.fc_out = make_layers(dim_hidden, dim_out, 0.)  # final layer no dropout

    def forward(self, x):
        x = x.flatten(start_dim=1)  # from BHW input, flatten HW for Linear layers
        x = self.fc_in(x)
        x = self.fc_hidden(x)
        x = self.fc_out(x)
        return x


def get_model(dim_hidden, num_hidden, device, use_dropout=False):
    return Network(
        dim_in=MNIST_IMG_SIZE * MNIST_IMG_SIZE,
        dim_out=MNIST_NUM_CLASSES,
        dim_hidden=dim_hidden,
        num_hidden=num_hidden,
        use_dropout=use_dropout,
    ).to(device)


def get_model_teacher(device):
    # teacher model is regularized with dropout
    return get_model(dim_hidden=1200, num_hidden=2, device=device, use_dropout=True)


def get_model_student(dim_hidden, device):
    # student model is not regularized with dropout
    return get_model(dim_hidden=dim_hidden, num_hidden=2, device=device, use_dropout=False)


def save_checkpoint(output_dir, output_tag, model, epoch, epoch_stats):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # save some of the params along with model weights
    params = {
        "DROPOUT_HIDDEN": DROPOUT_HIDDEN,
        "DROPOUT_VISIBLE": DROPOUT_INPUT,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS,
        "OPTIMIZER": OPTIMIZER,
        "OPTIMIZER_LR": OPTIMIZER_LR,
        "OPTIMIZER_LR_EXPONENTIAL_DECAY": OPTIMIZER_LR_EXPONENTIAL_DECAY,
        "SGD_MOMENTUM": SGD_MOMENTUM,
    }

    model_state_dict = {
        "epoch": epoch,
        "params": params,
        "epoch_stats": epoch_stats,
        "model_state_dict": model.state_dict()
    }

    output_tag = f"_{output_tag}" if output_tag else ""
    path = f"{output_dir}/ckpt{output_tag}.pth"
    torch.save(model_state_dict, path)


def load_checkpoint(path, device=None):
    with open(f"{OUTPUT_DIR}/{path}", 'rb') as file:
        return torch.load(file, device)


def load_model(model, path, device):
    checkpoint = load_checkpoint(path, device)

    # log pre-trained model stats
    try:
        # get the last stored element in stats
        # format: loss, correct_train, incorrect_train, correct_test, incorrect_test
        stats = checkpoint["epoch_stats"]
        epoch = len(stats)
        stats_final = stats[-1]
        loss, correct_train, incorrect_train, correct_test, incorrect_test = stats_final
        logger(f"Epoch {epoch}: Loss: {loss:.4f}; "
               f"train errors {incorrect_train}/{correct_train + incorrect_train}, "
               f"test errors {incorrect_test}/{correct_test + incorrect_test}.")
    except Exception:
        pass  # do nothing

    model_state_dict = checkpoint["model_state_dict"]

    try:
        # try strict load
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        logger(e)
        logger('Attempting to load model with strict=False...')
        # try relaxed load
        model.load_state_dict(model_state_dict, strict=False)


def get_optimizer_scheduler(model, regularize_weights):
    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=OPTIMIZER_LR,
            momentum=SGD_MOMENTUM,
            weight_decay=1e-4 if regularize_weights else 0
        )
    elif OPTIMIZER == "Adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=OPTIMIZER_LR,
            weight_decay=1e-2 if regularize_weights else 0
        )
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=OPTIMIZER_LR_EXPONENTIAL_DECAY,
        # verbose=True
    )

    return optimizer, scheduler


def get_dataloaders():
    transform_jitter = transforms.Compose([
        # augment by random jitter (translation)
        transforms.RandomAffine(
            degrees=0,
            translate=(2 / MNIST_IMG_SIZE, 2 / MNIST_IMG_SIZE)  # up to 2 pixels
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # only to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset_jitter = datasets.MNIST('./data', train=True, download=True, transform=transform_jitter)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader_jitter = DataLoader(train_dataset_jitter, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=NUM_WORKERS_DATALOADER)

    return train_loader_jitter, train_loader, test_loader


def loss_knowledge_distillation(logits_s, logits_t, temperature):
    # distillation loss: KL divergence with soft labels
    loss = torch.nn.functional.kl_div(
        nn.functional.log_softmax(logits_s / temperature, dim=-1),
        nn.functional.softmax(logits_t / temperature, dim=-1),
        reduction="batchmean"
    )
    loss = loss * (temperature * temperature)  # scale by T**2 to account for the effect of T on softmax magnitude
    return loss


def get_num_correct_incorrect(outputs, labels):
    _, predicted = outputs.max(1)  # get the highest probability class
    correct = (predicted == labels).sum().item()
    incorrect = labels.size(0) - correct
    return correct, incorrect


def data_apply_filter(images, labels, label_filter, remove_filter=True):
    """
    when remove_filter=True, removes all samples where label=label_filter
    when remove_filter=False, removes all samples where label!=label_filter
    :param images:
    :param labels:
    :param label_filter:
    :param remove_filter:
    :return:
    """
    indices_filtered = (labels != label_filter) if remove_filter else (labels == label_filter)
    images = images[indices_filtered]
    labels = labels[indices_filtered]
    return images, labels


def train_epoch(model, dataloader, optimizer, scheduler, device, remove_3s=False):
    model.train()

    total_loss = 0.0
    correct = incorrect = 0  # keep track of the total number of correct/incorrect predictions

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if remove_3s:
            images, labels = data_apply_filter(images, labels, label_filter=3, remove_filter=True)
            # in the unlikely case if we ended up removing all samples, go to next batch
            if labels.size(0) == 0:
                logger("Batch size is zero after removing all 3s...")
                continue

        optimizer.zero_grad()
        logits = model(images)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        if CLIP_GRAD_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct_b, incorrect_b = get_num_correct_incorrect(logits, labels)
        correct += correct_b
        incorrect += incorrect_b

    scheduler.step()  # step at the end of the epoch

    avg_loss = total_loss / len(dataloader)

    return avg_loss, correct, incorrect


def train_epoch_with_distillation(model_s, model_t, dataloader, optimizer, scheduler, device, temperature, remove_3s=False):
    model_s.train()
    model_t.eval()  # disable dropout on teacher

    total_loss = 0.0
    correct = incorrect = 0  # keep track of the total number of correct/incorrect predictions

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if remove_3s:
            images, labels = data_apply_filter(images, labels, label_filter=3, remove_filter=True)
            # in the unlikely case if we ended up removing all samples, go to next batch
            if labels.size(0) == 0:
                logger("Batch size is zero after removing all 3s...")
                continue

        optimizer.zero_grad()
        logits_s = model_s(images)

        # regular loss
        loss_hard = torch.nn.functional.cross_entropy(logits_s, labels)

        # knowledge distillation loss between student and teacher logits
        with torch.no_grad():
            logits_t = model_t(images)
        loss_soft = loss_knowledge_distillation(logits_s, logits_t, temperature)

        # combine losses
        loss = loss_hard + loss_soft

        loss.backward()
        if CLIP_GRAD_NORM:
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct_b, incorrect_b = get_num_correct_incorrect(logits_s, labels)

        correct += correct_b
        incorrect += incorrect_b

    scheduler.step()  # step at the end of the epoch

    avg_loss = total_loss / len(dataloader)

    return avg_loss, correct, incorrect


def test_model(model, dataloader, device, remove_3s=False, only_3s=False, verbose=False):
    # cant both remove and test only 3s
    if remove_3s and only_3s:
        raise ValueError()

    model.eval()
    correct = incorrect = 0  # keep track of the total number of correct/incorrect predictions
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if remove_3s or only_3s:
                images, labels = data_apply_filter(images, labels, label_filter=3, remove_filter=not only_3s)
                # in the unlikely case if we ended up removing all samples, go to next batch
                if labels.size(0) == 0:
                    logger("Batch size is zero after removing all 3s...")
                    continue

            outputs = model(images)
            correct_b, incorrect_b = get_num_correct_incorrect(outputs, labels)
            correct += correct_b
            incorrect += incorrect_b

    if verbose:
        run_tag = "All digits"
        if remove_3s:
            run_tag = "All digits except 3s"
        if only_3s:
            run_tag = "Only 3s"
        logger(f"Test set errors [{run_tag}] {incorrect}/{correct + incorrect}")

    return correct, incorrect


def train_model(model, device, train_loader, test_loader, output_dir,
                output_tag="", regularize_weights=True, remove_3s=False):
    optimizer, scheduler = get_optimizer_scheduler(model, regularize_weights)

    logger("Begin training...")
    epoch_stats = []
    for epoch in range(NUM_TRAIN_EPOCHS):
        loss, correct_train, incorrect_train = train_epoch(
            model, train_loader, optimizer, scheduler, device, remove_3s=remove_3s
        )
        correct_test, incorrect_test = test_model(
            model, test_loader, device, verbose=False
        )
        logger(f"[{output_tag}] Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS}: Loss: {loss:.4f}; "
               f"train errors {incorrect_train}/{correct_train + incorrect_train}, "
               f"test errors {incorrect_test}/{correct_test + incorrect_test}")
        epoch_stats.append(
            (loss, correct_train, incorrect_train, correct_test, incorrect_test)
        )
        if output_dir is not None:
            save_checkpoint(output_dir, output_tag, model, epoch, epoch_stats)

    return epoch_stats


def train_model_with_distillation(model_s, model_t, device, train_loader, test_loader, temperature, output_dir,
                                  output_tag="", regularize_weights=False, remove_3s=False):
    optimizer, scheduler = get_optimizer_scheduler(model_s, regularize_weights)

    if remove_3s:
        logger("Training with 3s removed from the training set.")

    logger(f"Begin training with KD (T={temperature})...")
    epoch_stats = []
    for epoch in range(NUM_TRAIN_EPOCHS):
        loss, correct_train, incorrect_train = train_epoch_with_distillation(
            model_s, model_t, train_loader, optimizer, scheduler, device, temperature, remove_3s
        )
        correct_test, incorrect_test = test_model(
            model_s, test_loader, device, verbose=False
        )
        logger(f"[{output_tag}] Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS}: Loss: {loss:.4f}; "
               f"train errors {incorrect_train}/{correct_train + incorrect_train}, "
               f"test errors {incorrect_test}/{correct_test + incorrect_test}")
        epoch_stats.append(
            (loss, correct_train, incorrect_train, correct_test, incorrect_test)
        )
        if output_dir is not None:
            save_checkpoint(output_dir, output_tag, model_s, epoch, epoch_stats)

    return epoch_stats


def seed_everything():
    torch.manual_seed(MANUAL_SEED)
    random.seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)


def main():
    options = parse_args()

    # set seed for reproducibility
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    train_loader_jitter, train_loader, test_loader = get_dataloaders()

    output_dir = f"{OUTPUT_DIR}/"
    if options.output_tag:
        output_dir += f"{options.output_tag}"  # given tag
    else:
        output_dir += f"{int(time.time())}"  # current tick number

    logger(f"Saving models to output_dir={output_dir}.")

    model_t = get_model_teacher(device)

    if options.teacher_ckpt:
        logger(f"Loading teacher network from path=[{options.teacher_ckpt}].")
        load_model(model_t, options.teacher_ckpt, device)

    else:
        logger("Training teacher network...")

        train_model(
            model_t, device, train_loader_jitter, test_loader, output_dir,
            output_tag="teacher",
            regularize_weights=True  # teacher model is trained with regularization by weight decay
        )
        test_model(model_t, test_loader, device, verbose=True)

    if options.no_student:
        exit()

    model_s = get_model_student(options.hidden_dim, device)
    remove_3s = options.remove_3s

    if options.use_kd:
        temperature = options.temperature

        logger("Training student network with knowledge distillation.")

        train_model_with_distillation(
            model_s, model_t, device, train_loader, test_loader, temperature, output_dir,
            output_tag="student",
            regularize_weights=False,  # student model is trained without regularization by weight decay
            remove_3s=remove_3s
        )

    else:
        logger("Training student network without knowledge distillation.")

        # student model is trained without regularization by weight decay
        train_model(
            model_s, device, train_loader, test_loader, output_dir,
            output_tag="student",
            regularize_weights=False,  # student model is trained without regularization by weight decay
            remove_3s=remove_3s
        )

    if options.remove_3s:
        test_model(model_s, test_loader, device, remove_3s=True, verbose=True)  # run a test on all data except 3s
        test_model(model_s, test_loader, device, only_3s=True, verbose=True)  # run a test only on 3s
    else:
        test_model(model_s, test_loader, device, verbose=True)  # run a test on all data


if __name__ == "__main__":
    main()
