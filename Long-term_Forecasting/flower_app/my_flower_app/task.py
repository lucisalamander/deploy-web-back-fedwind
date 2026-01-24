import os
import sys
import importlib.util
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from my_flower_app.dataloader import Dataset_Custom
import torch.nn as nn
from torch.utils.data import random_split

import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
model_path = os.path.join(project_root, "models", "GPT4TS.py")

spec = importlib.util.spec_from_file_location("GPT4TS", model_path)
GPT4TS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GPT4TS)

GPT4TS_Nonlinear = GPT4TS.GPT4TS_Nonlinear

def get_default_configs(pred_len):
    """
    Get default model configurations.

    Args:
        pred_len: Prediction length (forecast horizon). REQUIRED - must be specified. Common values: 96, 192, 336, 720
    """
    return SimpleNamespace(
        # model behavior
        is_gpt=True,
        pretrain=True,
        freeze=False,
        # data/patching
        seq_len=256,
        pred_len=pred_len,
        patch_size=4,
        stride=1,
        d_model=768,  # Must match GPT-2's hidden_size for pretrained model (GPT-2 uses 768)
        hidden_size=16,  # Intermediate MLP hidden layer size
        kernel_size=3,
        llm_layers=6,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.5,
        lora_target_modules=["c_attn", "c_fc", "c_proj"],
        dropout=0.5,  # Dropout rate for model regularization (applied in patch embedding, GPT output, pre-output layer)
    )


class Net(nn.Module):
    """
    A thin wrapper so instantiating Net() works in your Flower server/client code.
    The wrapper holds the GPT4TS model and forwards calls:
      - forward(x): expects x.shape == [B, seq_len, 1]
      returns predictions shape [B, pred_len, 1]
    """

    def __init__(self, device=None, configs=None):
        super(Net, self).__init__()
        if configs is None:
            raise ValueError("configs must be provided to Net(). Use get_default_configs(pred_len=1X) to create configs.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing Net wrapper on device: {device}")
        self.device = device
        self.configs = configs
        self.model = GPT4TS_Nonlinear(configs, device)
        self.to(device)

    def forward(self, x):
        """
        x: tensor [B, seq_len, 1]
        GPT4TS expects shape [B, L, M] where M is number of features (here 1).
        The underlying model returns [B, pred_len, M]
        """
        # ensure on device and float
        x = x.to(self.device).float()
        return self.model(x, itr=0)


def train(net, trainloader, epochs, lr, device, valloader=None):
    """
    Train loop expects `net` to be an instance of Net (wrapper above).
    trainloader yields dicts {"x": [B, seq_len, 1], "y": [B, pred_len, 1]}
    Optionally evaluates on valloader after each epoch.
    Returns:
        avg_loss: average training loss over all epochs
        history: list of dictionaries containing per-epoch metrics
    """
    net.to(device)
    # net.train() removed here as it is called at start of each epoch loop
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_count = 0
        
        # Training phase
        net.train()
        for batch in trainloader:
            if isinstance(batch, dict):
                x = batch["x"].to(device).float()
                y_true = batch["y"].to(device).float()
            else:
                x, y_true = batch[:2]
                x = x.to(device).float()
                y_true = y_true.to(device).float()

            optimizer.zero_grad()
            y_pred = net(x)

            pred_len = net.configs.pred_len
            if y_true.size(1) > pred_len:
                y_true = y_true[:, -pred_len:, :] 
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            # Record batch loss
            batch_loss = loss.item()
            total_loss += batch_loss
            epoch_loss += batch_loss
            count += 1
            epoch_count += 1

        avg_epoch_train_loss = epoch_loss / max(1, epoch_count)
        
        # Validation phase (if valloader provided)
        metrics = {"epoch": epoch + 1, "train_loss": avg_epoch_train_loss}
        
        if valloader is not None:
            # Reusing the existing test function for validation
            # Note: test() sets net.eval(), so we ensure to set net.train() back at start of loop
            val_loss, val_mae, val_rmse = test(net, valloader, device, return_predictions=False)
            metrics["val_loss"] = val_loss
            metrics["val_mae"] = val_mae
            metrics["val_rmse"] = val_rmse
            logging.info(f"Epoch {epoch+1}/{epochs}: TrainLoss={avg_epoch_train_loss:.6f}, ValLoss={val_loss:.6f}")
        else:
            logging.info(f"Epoch {epoch+1}/{epochs}: TrainLoss={avg_epoch_train_loss:.6f}")
            
        history.append(metrics)

    avg_loss = total_loss / max(1, count)
    logging.info(f"Training completed over {epochs} epochs. Average Loss: {avg_loss:.6f}")
    return avg_loss, history

def test(net, testloader, device, return_predictions=False):
    """
    Evaluate model on validation/test set.
    Returns (avg_loss, mae, rmse) or (avg_loss, mae, rmse, predictions, trues) if return_predictions=True
    """
    net.to(device)
    net.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    count = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):
                x = batch["x"].to(device).float()
                y_true = batch["y"].to(device).float()
            else:
                x, y_true = batch[:2]
                x = x.to(device).float()
                y_true = y_true.to(device).float()

            y_pred = net(x)
            pred_len = net.configs.pred_len
            if y_true.size(1) > pred_len:
                y_true = y_true[:, -pred_len:, :]
            loss = criterion(y_pred, y_true)

            if count < 5:  # print only first 5 batches to avoid spam
                logging.info(f"[TEST] y_pred: {y_pred[0].detach().cpu().numpy().flatten()}")
                logging.info(f"[TEST] y_true: {y_true[0].detach().cpu().numpy().flatten()}")

            total_loss += loss.item()
            count += 1

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_true.cpu().numpy())

    avg_loss = total_loss / max(1, count)

    preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,))
    trues = np.concatenate(all_trues, axis=0) if all_trues else np.zeros((0,))
    if preds.size:
        mae = float(np.mean(np.abs(preds - trues)))
        rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    else:
        mae = float("nan")
        rmse = float("nan")

    if return_predictions:
        return avg_loss, mae, rmse, preds, trues
    return avg_loss, mae, rmse


def load_client_train(partition_id: int, num_partitions: int, bs: int = 32, cfg: SimpleNamespace = None):
    """
    Each client loads a different partition of the *training* data (70% of full dataset).
    cfg must be provided with pred_len specified.
    """
    if cfg is None:
        raise ValueError("cfg must be provided to load_client_train(). Use get_default_configs(pred_len=1X) to create cfg.")

    root_path = os.path.join(project_root, "datasets", "custom")
    client_datasets = ["nasa_almaty.csv", "nasa_zhezkazgan.csv", "nasa_aktau.csv", "nasa_taraz.csv", "nasa_aktobe.csv"]
    data_path = client_datasets[partition_id]
    seq_len, label_len, pred_len = cfg.seq_len, 48, cfg.pred_len
    target = "WS50M"

    full_train = Dataset_Custom(
        root_path=root_path,
        flag='train',
        size=(seq_len, label_len, pred_len),
        data_path=data_path,
        target=target,
        scale=True
    )

    logging.info(f"Client {partition_id}: Loaded {len(full_train.data_x)} raw CSV rows for training.")
    logging.info(f"Client {partition_id}: Total dataset size before partitioning: {len(full_train)}")
    logging.info(f"Client {partition_id}: Partition ID: {partition_id}, Num partitions: {num_partitions}")
    trainloader = DataLoader(full_train, batch_size=bs, shuffle=True, drop_last=False)
    return trainloader


def load_client_val(partition_id: int, bs: int = 32, cfg: SimpleNamespace = None):
    """
    Each client loads their local validation data (20% of their local dataset).
    cfg must be provided with pred_len specified.
    """
    if cfg is None:
        raise ValueError("cfg must be provided to load_client_val(). Use get_default_configs(pred_len=1X) to create cfg.")

    root_path = os.path.join(project_root, "datasets", "custom")
    client_datasets = ["nasa_almaty.csv", "nasa_zhezkazgan.csv", "nasa_aktau.csv", "nasa_taraz.csv", "nasa_aktobe.csv"]
    data_path = client_datasets[partition_id]
    seq_len, label_len, pred_len = cfg.seq_len, 48, cfg.pred_len
    target = "WS50M"

    val_dataset = Dataset_Custom(
        root_path=root_path,
        flag='val',
        size=(seq_len, label_len, pred_len),
        data_path=data_path,
        target=target,
        scale=True
    )

    logging.info(f"Client {partition_id}: Loaded {len(val_dataset.data_x)} raw CSV rows for validation.")
    valloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=False)
    return valloader


def load_client_test(partition_id: int, bs: int = 32, cfg: SimpleNamespace = None):
    """
    Each client loads their local test data (10% of their local dataset).
    cfg must be provided with pred_len specified.
    """
    if cfg is None:
        raise ValueError("cfg must be provided to load_client_test(). Use get_default_configs(pred_len=1X) to create cfg.")

    root_path = os.path.join(project_root, "datasets", "custom")
    client_datasets = ["nasa_almaty.csv", "nasa_zhezkazgan.csv", "nasa_aktau.csv", "nasa_taraz.csv", "nasa_aktobe.csv"]
    data_path = client_datasets[partition_id]
    seq_len, label_len, pred_len = cfg.seq_len, 48, cfg.pred_len
    target = "WS50M"

    test_dataset = Dataset_Custom(
        root_path=root_path,
        flag='test',
        size=(seq_len, label_len, pred_len),
        data_path=data_path,
        target=target,
        scale=True
    )

    logging.info(f"Client {partition_id}: Loaded {len(test_dataset.data_x)} raw CSV rows for testing.")
    testloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=False)
    return testloader

