# # Training and validation helpers
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from tqdm import tqdm
# from sklearn.metrics import roc_auc_score

# def train_one_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     for imgs, labels in tqdm(dataloader):
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(imgs).squeeze()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * imgs.size(0)
#         all_labels.extend(labels.detach().cpu().numpy())
#         all_preds.extend(outputs.detach().cpu().numpy())

#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_auc = roc_auc_score(all_labels, all_preds)
#     return epoch_loss, epoch_auc

# def validate(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for imgs, labels in dataloader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs).squeeze()
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * imgs.size(0)
#             all_labels.extend(labels.detach().cpu().numpy())
#             all_preds.extend(outputs.detach().cpu().numpy())

#     val_loss = running_loss / len(dataloader.dataset)
#     val_auc = roc_auc_score(all_labels, all_preds)
#     return val_loss, val_auc


# Training and validation helpers
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        outputs = outputs.view(-1)  # ensure shape (batch,)
        labels = labels.view(-1)    # ensure shape (batch,)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Handle edge case if only one class is present
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validation", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            outputs = outputs.view(-1)  # ensure shape (batch,)
            labels = labels.view(-1)    # ensure shape (batch,)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(outputs.detach().cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)

    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc
