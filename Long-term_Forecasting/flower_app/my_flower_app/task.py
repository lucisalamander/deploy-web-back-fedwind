import os
import sys
import importlib.util
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from my_flower_app.dataloader import Dataset_Custom
import torch.nn as nn
from torch.utils.data import random_split

import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Model registry: model_name -> (module_filename, class_name, backbone_flag)
#   backbone_flag is the configs attribute that gates the LLM backbone
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "gpt4ts_nonlinear": ("GPT4TS", "GPT4TS_Nonlinear", "is_gpt"),
    "gpt4ts_linear":    ("GPT4TS", "GPT4TS_Linear",    "is_gpt"),
    "bart_nonlinear":   ("BART",   "BART_Nonlinear",   "is_bart"),
    "bart_linear":      ("BART",   "BART_Linear",      "is_bart"),
    "bert_nonlinear":   ("BERT",   "BERT_Nonlinear",   "is_bert"),
    "bert_linear":      ("BERT",   "BERT_Linear",      "is_bert"),
    "llama_nonlinear":  ("LLAMA",  "Llama_Nonlinear",  "is_llama"),
    "llama_linear":     ("LLAMA",  "Llama_Linear",     "is_llama"),
}


def _load_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    module_filename, class_name, _ = MODEL_REGISTRY[model_name]
    model_path = os.path.join(project_root, "models", f"{module_filename}.py")

    spec = importlib.util.spec_from_file_location(module_filename, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def get_default_configs(
    pred_len,
    model="gpt4ts_nonlinear",
    seq_len=336,
    patch_size=4,
    stride=1,
    d_model=768,
    hidden_size=16,
    kernel_size=3,
    llm_layers=4,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.15,
    dropout=0.15,
    use_lora=True  # Enable LoRA by default for federated learning
):
    """
    Get model configurations.

    Args:
        pred_len: Prediction length (forecast horizon). REQUIRED - must be specified.
        model: Which model to use. One of: gpt4ts_nonlinear, gpt4ts_linear,
               bart_nonlinear, bart_linear, bert_nonlinear, bert_linear,
               llama_nonlinear, llama_linear (default: gpt4ts_nonlinear)
        use_lora: Whether to enable LoRA fine-tuning (default: True).
        ... other parameters are optional and have defaults matching the baseline.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Determine which backbone flag to enable based on the selected model
    _, _, backbone_flag = MODEL_REGISTRY[model]

    return SimpleNamespace(
        model=model,
        is_gpt=(backbone_flag == "is_gpt"),
        is_bart=(backbone_flag == "is_bart"),
        is_bert=(backbone_flag == "is_bert"),
        is_llama=(backbone_flag == "is_llama"),
        pretrain=True,
        freeze=True,  # Freeze base LLM, only train adapters + LayerNorm
        use_lora=use_lora,
        # data/patching
        seq_len=seq_len,
        pred_len=pred_len,
        patch_size=patch_size,
        stride=stride,
        d_model=d_model,
        hidden_size=hidden_size,
        kernel_size=kernel_size,
        llm_layers=llm_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=["c_attn", "c_fc", "c_proj"],
        dropout=dropout,
    )


class Net(nn.Module):
    """
    A thin wrapper so instantiating Net() works in your Flower server/client code.
    The wrapper holds any model from MODEL_REGISTRY and forwards calls:
      - forward(x): expects x.shape == [B, seq_len, 1]
      returns predictions shape [B, pred_len, 1]
    """

    def __init__(self, device=None, configs=None):
        super(Net, self).__init__()
        if configs is None:
            raise ValueError("configs must be provided to Net(). Use get_default_configs(pred_len=1X) to create configs.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = getattr(configs, "model", "gpt4ts_nonlinear")
        logging.info(f"Initializing Net wrapper: model={model_name}, device={device}")
        self.device = device
        self.configs = configs

        model_class = _load_model_class(model_name)
        self.model = model_class(configs, device)
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


def train(net, trainloader, epochs, lr, device, valloader=None, weight_decay=0.01, global_weights=None, proximal_mu=None):
    """
    Train loop expects `net` to be an instance of Net (wrapper above).
    trainloader yields dicts {"x": [B, seq_len, 1], "y": [B, pred_len, 1]}
    Optionally evaluates on valloader after each epoch.

    Args:
        net: model to train
        trainloader: training data loader
        epochs: number of local training epochs
        lr: learning rate
        device: torch device
        valloader: optional validation data loader
        weight_decay: L2 regularization coefficient for AdamW optimizer (default: 0.01)
        global_weights: global model weights for FedProx proximal term (optional)
        proximal_mu: FedProx proximal term coefficient (optional, e.g., 0.01)

    Returns:
        avg_loss: average training loss over all epochs
        history: list of dictionaries containing per-epoch metrics
    """
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0
    history = []

    # Check if FedProx is enabled
    use_fedprox = (proximal_mu is not None and global_weights is not None and proximal_mu > 0)
    if use_fedprox:
        logging.info(f"FedProx enabled: proximal_mu={proximal_mu}")
        # CRITICAL FIX: Move global weights to device ONCE before training loop
        # Only move trainable parameters to save memory
        global_weights_device = {}
        for name, param in net.named_parameters():
            if param.requires_grad and name in global_weights:
                global_weights_device[name] = global_weights[name].to(device)
        logging.info(f"FedProx: Moved {len(global_weights_device)} global weight tensors to {device}")
    else:
        global_weights_device = None

    for epoch in range(epochs):
        epoch_mse_loss = 0.0  # Track MSE separately
        epoch_prox_loss = 0.0  # Track proximal term separately
        epoch_total_loss = 0.0  # Track total loss (MSE + proximal)
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

            # Base loss (MSE)
            mse_loss = criterion(y_pred, y_true)
            total_batch_loss = mse_loss

            # CRITICAL: Add FedProx proximal term: (mu/2) * ||w - w_global||^2
            # CRITICAL FIX: Use pre-moved global_weights_device (no .to() in loop)
            if use_fedprox:
                proximal_term = 0.0
                for name, param in net.named_parameters():
                    if name in global_weights_device:
                        proximal_term += torch.sum((param - global_weights_device[name]) ** 2)
                proximal_term = (proximal_mu / 2.0) * proximal_term
                total_batch_loss = mse_loss + proximal_term
                epoch_prox_loss += proximal_term.item()

            total_batch_loss.backward()
            optimizer.step()

            # CRITICAL FIX: Track MSE and total loss separately
            epoch_mse_loss += mse_loss.item()
            epoch_total_loss += total_batch_loss.item()
            total_loss += total_batch_loss.item()  # For backward compat with return value
            count += 1
            epoch_count += 1

        # CRITICAL FIX: Separate MSE from total loss for fair comparison with val/test
        avg_epoch_mse = epoch_mse_loss / max(1, epoch_count)
        avg_epoch_total = epoch_total_loss / max(1, epoch_count)

        # Validation phase (if valloader provided)
        # IMPORTANT: Use MSE as primary "train_loss" for fair comparison with val/test
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_mse,  # MSE only (comparable to val/test)
            "train_total_loss": avg_epoch_total  # MSE + proximal (actual optimized loss)
        }

        # Add proximal term info if FedProx is used
        if use_fedprox:
            avg_prox_loss = epoch_prox_loss / max(1, epoch_count)
            metrics["proximal_loss"] = avg_prox_loss

        if valloader is not None:
            # Reusing the existing test function for validation
            # Note: test() sets net.eval(), so we ensure to set net.train() back at start of loop
            val_loss, val_mae, val_rmse = test(net, valloader, device, return_predictions=False)
            metrics["val_loss"] = val_loss
            metrics["val_mae"] = val_mae
            metrics["val_rmse"] = val_rmse
            if use_fedprox:
                logging.info(f"Epoch {epoch+1}/{epochs}: TrainMSE={avg_epoch_mse:.6f}, TrainTotal={avg_epoch_total:.6f}, ProxLoss={avg_prox_loss:.6f}, ValLoss={val_loss:.6f}")
            else:
                logging.info(f"Epoch {epoch+1}/{epochs}: TrainMSE={avg_epoch_mse:.6f}, ValLoss={val_loss:.6f}")
        else:
            if use_fedprox:
                logging.info(f"Epoch {epoch+1}/{epochs}: TrainMSE={avg_epoch_mse:.6f}, TrainTotal={avg_epoch_total:.6f}, ProxLoss={avg_prox_loss:.6f}")
            else:
                logging.info(f"Epoch {epoch+1}/{epochs}: TrainMSE={avg_epoch_mse:.6f}")

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
        scale=True,
        percent=100
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


def _load_centralized(flag: str, bs: int, cfg: SimpleNamespace, shuffle: bool):
    """
    Load all 5 cities with the same per-city temporal split as federated,
    then concatenate. Each city's 70/10/20 boundary is computed independently,
    so the combined dataset is directly comparable to the federated setup.
    """
    if cfg is None:
        raise ValueError("cfg must be provided. Use get_default_configs(pred_len=...) to create cfg.")

    root_path = os.path.join(project_root, "datasets", "custom")
    client_datasets = ["nasa_almaty.csv", "nasa_zhezkazgan.csv", "nasa_aktau.csv", "nasa_taraz.csv", "nasa_aktobe.csv"]
    seq_len, label_len, pred_len = cfg.seq_len, 48, cfg.pred_len
    target = "WS50M"

    datasets = []
    for i, data_path in enumerate(client_datasets):
        ds = Dataset_Custom(
            root_path=root_path,
            flag=flag,
            size=(seq_len, label_len, pred_len),
            data_path=data_path,
            target=target,
            scale=True,
            percent=100
        )
        logging.info(f"Centralized [{flag}] city {i} ({data_path}): {len(ds.data_x)} rows, {len(ds)} samples")
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    logging.info(f"Centralized [{flag}] total: {len(combined)} samples across {len(datasets)} cities")
    return DataLoader(combined, batch_size=bs, shuffle=shuffle, drop_last=False)


def load_centralized_train(bs: int = 32, cfg: SimpleNamespace = None):
    return _load_centralized('train', bs, cfg, shuffle=True)


def load_centralized_val(bs: int = 32, cfg: SimpleNamespace = None):
    return _load_centralized('val', bs, cfg, shuffle=False)


def load_centralized_test(bs: int = 32, cfg: SimpleNamespace = None):
    return _load_centralized('test', bs, cfg, shuffle=False)

