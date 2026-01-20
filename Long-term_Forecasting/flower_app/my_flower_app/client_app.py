
"""my-flower-app: A Flower / PyTorch client that only trains locally."""
import torch
import time
import os
import pandas as pd
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from my_flower_app.task import get_default_configs, Net, load_client_train, load_client_val, load_client_test, train as train_fn, test as test_fn

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Get pred_len from train config
    pred_len = msg.content["config"].get("pred_len", 120)
    configs = get_default_configs(pred_len=pred_len)

    model = Net(configs=configs)
    state = msg.content["arrays"].to_torch_state_dict()
    if len(state) == 0:
        logging.error("Client received EMPTY weights. Skipping load_state_dict.")
    else:
        model.load_state_dict(state, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pid = context.node_config["partition-id"]
    npt = context.node_config["num-partitions"]
    bs  = context.run_config.get("batch-size", 32)
    trainloader = load_client_train(pid, npt, bs=bs, cfg=configs)

    logging.info(f"[CLIENT {pid}] Training on {len(trainloader.dataset)} samples, BatchSize={bs}")

    # --- Learning Rate Decay ---
    base_lr = msg.content["config"]["lr"]
    current_round = context.run_config.get("server_round", 1)  # add from run_config
    lr = base_lr * (0.9 ** (current_round - 1))  # decay 10% per round
    logging.info(f"[CLIENT {pid}] Using decayed LR={lr:.6f} (round {current_round})")

    train_start_time = time.time()
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        lr,
        device,
    )
    train_duration = time.time() - train_start_time

    logging.info(f"[CLIENT {pid}] Training complete. AvgTrainLoss={train_loss:.6f}, Duration={train_duration:.2f}s")

    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss": float(train_loss),
        "train_duration_sec": float(train_duration),
        "num-examples": len(trainloader.dataset),
    })
    content = RecordDict({"arrays": arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Perform local validation and test evaluation on client's datasets.
    This is called by the server to get federated evaluation metrics.
    Evaluates on both val (20%) and test (10%) splits.
    """
    # Get pred_len from run_config (server passes it)
    pred_len = context.run_config.get("pred-len", 120)
    configs = get_default_configs(pred_len=pred_len)

    model = Net(configs=configs)
    state = msg.content["arrays"].to_torch_state_dict()
    if len(state) == 0:
        logging.error("Client received EMPTY weights for evaluation. Skipping load_state_dict.")
    else:
        model.load_state_dict(state, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pid = context.node_config["partition-id"]
    bs = context.run_config.get("batch-size", 32)
    current_round = context.run_config.get("server_round", 1)

    # Get experiment directory from environment variable
    exp_dir = os.environ.get("FLOWER_EXP_DIR", ".")

    # Validation evaluation (20% of data)
    valloader = load_client_val(pid, bs=bs, cfg=configs)
    logging.info(f"[CLIENT {pid}] Evaluating on {len(valloader.dataset)} validation samples")

    val_start_time = time.time()
    val_loss, val_mae, val_rmse, val_preds, val_trues = test_fn(model, valloader, device=device, return_predictions=True)
    val_duration = time.time() - val_start_time

    logging.info(f"[CLIENT {pid}] Validation complete. ValLoss={val_loss:.6f}, MAE={val_mae:.6f}, RMSE={val_rmse:.6f}, Duration={val_duration:.2f}s")

    # Save validation predictions to CSV
    _save_predictions_to_csv(val_preds, val_trues, exp_dir, pid, current_round, "val", pred_len)

    # Test evaluation (10% of data)
    testloader = load_client_test(pid, bs=bs, cfg=configs)
    logging.info(f"[CLIENT {pid}] Evaluating on {len(testloader.dataset)} test samples")

    test_start_time = time.time()
    test_loss, test_mae, test_rmse, test_preds, test_trues = test_fn(model, testloader, device=device, return_predictions=True)
    test_duration = time.time() - test_start_time

    logging.info(f"[CLIENT {pid}] Test complete. TestLoss={test_loss:.6f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, Duration={test_duration:.2f}s")

    # Save test predictions to CSV
    _save_predictions_to_csv(test_preds, test_trues, exp_dir, pid, current_round, "test", pred_len)

    num_examples = len(valloader.dataset) + len(testloader.dataset)
    metrics = MetricRecord({
        "num-examples": num_examples,
        "val_loss": float(val_loss),
        "val_mae": float(val_mae),
        "val_rmse": float(val_rmse),
        "val_duration_sec": float(val_duration),
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_duration_sec": float(test_duration),
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)


def _save_predictions_to_csv(preds, trues, exp_dir, client_id, round_num, split_name, pred_len):
    """
    Save predictions and true values to CSV file.

    Args:
        preds: numpy array of predictions [num_samples, pred_len, 1]
        trues: numpy array of true values [num_samples, pred_len, 1]
        exp_dir: experiment directory path
        client_id: client partition ID
        round_num: current federated learning round
        split_name: 'val' or 'test'
        pred_len: prediction length
    """
    if preds.size == 0:
        logging.warning(f"[CLIENT {client_id}] No predictions to save for {split_name} split")
        return

    # Reshape predictions and trues to 2D: [num_samples, pred_len]
    preds_2d = preds.reshape(-1, pred_len)
    trues_2d = trues.reshape(-1, pred_len)

    # Create DataFrame with columns for each timestep
    data = {}
    for t in range(pred_len):
        data[f"pred_t{t}"] = preds_2d[:, t]
        data[f"true_t{t}"] = trues_2d[:, t]

    # Add metadata columns
    data["sample_idx"] = np.arange(len(preds_2d))
    data["client_id"] = client_id
    data["round"] = round_num
    data["split"] = split_name

    # Reorder columns: metadata first, then predictions and trues
    metadata_cols = ["sample_idx", "client_id", "round", "split"]
    pred_true_cols = [f"pred_t{t}" for t in range(pred_len)] + [f"true_t{t}" for t in range(pred_len)]
    df = pd.DataFrame(data)
    df = df[metadata_cols + pred_true_cols]

    # Save to CSV
    predictions_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    csv_path = os.path.join(predictions_dir, f"client{client_id}_round{round_num}_{split_name}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"[CLIENT {client_id}] Saved {len(df)} {split_name} predictions to {csv_path}")