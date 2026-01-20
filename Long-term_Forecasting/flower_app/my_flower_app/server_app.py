
"""my-flower-app: A Flower / PyTorch app."""
import copy
import torch
import pandas as pd
import time
from datetime import datetime
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import logging

logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

from my_flower_app.task import get_default_configs, Net

app = ServerApp()


class FedAvgWithMetrics(FedAvg):
    """Custom FedAvg that tracks client training and evaluation metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_train_losses = []
        self.client_train_durations = []
        self.client_val_losses = []
        self.client_val_maes = []
        self.client_val_rmses = []
        self.client_val_durations = []
        self.client_test_losses = []
        self.client_test_maes = []
        self.client_test_rmses = []
        self.client_test_durations = []

    def aggregate_fit(self, grid, config, messages):
        """Override to collect training metrics from clients."""
        self.client_train_losses = []
        self.client_train_durations = []

        logging.info(f"[DEBUG] aggregate_fit called with {len(messages)} messages")

        for msg in messages:
            logging.info(f"[DEBUG] Message content keys: {msg.content.keys()}")
            if "metrics" in msg.content:
                metrics = msg.content["metrics"]
                if hasattr(metrics, 'data'):
                    metrics_dict = metrics.data
                else:
                    metrics_dict = metrics

                logging.info(f"[DEBUG] Metrics dict: {metrics_dict}")
                if "train_loss" in metrics_dict:
                    self.client_train_losses.append(float(metrics_dict["train_loss"]))
                if "train_duration_sec" in metrics_dict:
                    self.client_train_durations.append(float(metrics_dict["train_duration_sec"]))
            else:
                logging.warning(f"[DEBUG] No 'metrics' key found in message content")

        logging.info(f"[DEBUG] Collected {len(self.client_train_losses)} train losses and {len(self.client_train_durations)} durations")

        return super().aggregate_fit(grid, config, messages)

    def aggregate_evaluate(self, grid, messages):
        """Override to collect evaluation metrics from clients."""
        self.client_val_losses = []
        self.client_val_maes = []
        self.client_val_rmses = []
        self.client_val_durations = []
        self.client_test_losses = []
        self.client_test_maes = []
        self.client_test_rmses = []
        self.client_test_durations = []

        logging.info(f"[DEBUG] aggregate_evaluate called with {len(messages)} messages")

        for msg in messages:
            if "metrics" in msg.content:
                metrics = msg.content["metrics"]
                if hasattr(metrics, 'data'):
                    metrics_dict = metrics.data
                else:
                    metrics_dict = metrics

                logging.info(f"[DEBUG] Eval Metrics dict: {metrics_dict}")

                # Validation metrics
                if "val_loss" in metrics_dict:
                    self.client_val_losses.append(float(metrics_dict["val_loss"]))
                if "val_mae" in metrics_dict:
                    self.client_val_maes.append(float(metrics_dict["val_mae"]))
                if "val_rmse" in metrics_dict:
                    self.client_val_rmses.append(float(metrics_dict["val_rmse"]))
                if "val_duration_sec" in metrics_dict:
                    self.client_val_durations.append(float(metrics_dict["val_duration_sec"]))

                # Test metrics
                if "test_loss" in metrics_dict:
                    self.client_test_losses.append(float(metrics_dict["test_loss"]))
                if "test_mae" in metrics_dict:
                    self.client_test_maes.append(float(metrics_dict["test_mae"]))
                if "test_rmse" in metrics_dict:
                    self.client_test_rmses.append(float(metrics_dict["test_rmse"]))
                if "test_duration_sec" in metrics_dict:
                    self.client_test_durations.append(float(metrics_dict["test_duration_sec"]))

        logging.info(f"[DEBUG] Collected {len(self.client_val_losses)} val losses, {len(self.client_test_losses)} test losses")
        return super().aggregate_evaluate(grid, messages)

@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train: float = context.run_config["fraction-train"]
    total_rounds:   int   = context.run_config["num-server-rounds"]
    lr:             float = context.run_config["lr"]
    bs:             int   = context.run_config.get("batch-size", 32)
    pred_len:       int   = context.run_config.get("pred-len", 120)  # Default to 120 if not specified

    # Get experiment directory from environment variable (set by run_flower_experiment.sh)
    import os
    exp_dir = os.environ.get("FLOWER_EXP_DIR", ".")

    logging.info("Loading project configuration...")
    logging.info(f"Config: rounds={total_rounds}  fraction-train={fraction_train}  lr={lr}  batch-size={bs}  pred-len={pred_len}")
    logging.info(f"Experiment directory: {exp_dir}")

    # Initialize model with configurable pred_len
    configs = get_default_configs(pred_len=pred_len)
    global_model = Net(configs=configs)
    arrays = ArrayRecord(global_model.state_dict())

    logging.info("Server initialized - no server-side datasets. All training and testing happens on clients.")


    strategy = FedAvgWithMetrics(fraction_train=fraction_train)

    best = {"round": 0, "loss": float("inf"), "arrays": None}
    logging.info("Starting federated training...")
    results = []

    training_start_time = time.time()
    for r in range(1, total_rounds + 1):
        round_start_time = time.time()
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "pred_len": pred_len, "server_round": r}),
            num_rounds=1,
        )

        logging.info(f"[DEBUG Round {r}] result type: {type(result)}")
        logging.info(f"[DEBUG Round {r}] result attributes: {dir(result)}")
        if hasattr(result, 'metrics'):
            logging.info(f"[DEBUG Round {r}] result.metrics: {result.metrics}")
        if hasattr(result, 'metrics_aggregated'):
            logging.info(f"[DEBUG Round {r}] result.metrics_aggregated: {result.metrics_aggregated}")

        new_arrays = result.arrays
        if new_arrays is None or len(new_arrays.to_torch_state_dict()) == 0:
            logging.error("WARNING: Strategy returned EMPTY weights. Keeping previous weights!")
        else:
            arrays = new_arrays

        logging.info(f"[DEBUG Round {r}] strategy.client_train_losses = {strategy.client_train_losses}")
        logging.info(f"[DEBUG Round {r}] strategy.client_train_durations = {strategy.client_train_durations}")

        # Try to get from individual client metrics first
        if strategy.client_train_losses:
            avg_train_loss = sum(strategy.client_train_losses) / len(strategy.client_train_losses)
        else:
            avg_train_loss = None
            logging.warning(f"[Round {r}] No client train losses collected from messages!")

        if strategy.client_train_durations:
            avg_client_train_duration = sum(strategy.client_train_durations) / len(strategy.client_train_durations)
            max_client_train_duration = max(strategy.client_train_durations)
        else:
            avg_client_train_duration = None
            max_client_train_duration = None
            logging.warning(f"[Round {r}] No client train durations collected from messages!")

        # Fallback: Extract from aggregated train_metrics_clientapp in result
        if hasattr(result, 'train_metrics_clientapp') and result.train_metrics_clientapp:
            # train_metrics_clientapp is a dict: {round_num: {metric_name: value}}
            # Since we call strategy.start() with num_rounds=1, the metrics are always stored with key 1
            if 1 in result.train_metrics_clientapp:
                aggregated_metrics = result.train_metrics_clientapp[1]
                logging.info(f"[Round {r}] Found aggregated metrics: {aggregated_metrics}")

                if avg_train_loss is None and 'train_loss' in aggregated_metrics:
                    avg_train_loss = float(aggregated_metrics['train_loss'])
                    logging.info(f"[Round {r}] Using aggregated train_loss: {avg_train_loss}")

                if avg_client_train_duration is None and 'train_duration_sec' in aggregated_metrics:
                    avg_client_train_duration = float(aggregated_metrics['train_duration_sec'])
                    max_client_train_duration = avg_client_train_duration
                    logging.info(f"[Round {r}] Using aggregated train_duration: {avg_client_train_duration}")

        logging.info(f"[DEBUG Round {r}] strategy.client_val_losses = {strategy.client_val_losses}")
        logging.info(f"[DEBUG Round {r}] strategy.client_val_maes = {strategy.client_val_maes}")
        logging.info(f"[DEBUG Round {r}] strategy.client_val_rmses = {strategy.client_val_rmses}")
        logging.info(f"[DEBUG Round {r}] strategy.client_test_losses = {strategy.client_test_losses}")
        logging.info(f"[DEBUG Round {r}] strategy.client_test_maes = {strategy.client_test_maes}")
        logging.info(f"[DEBUG Round {r}] strategy.client_test_rmses = {strategy.client_test_rmses}")

        if strategy.client_val_losses:
            avg_val_loss = sum(strategy.client_val_losses) / len(strategy.client_val_losses)
        else:
            avg_val_loss = None
            logging.warning(f"[Round {r}] No client val losses collected!")

        if strategy.client_val_maes:
            avg_val_mae = sum(strategy.client_val_maes) / len(strategy.client_val_maes)
        else:
            avg_val_mae = None

        if strategy.client_val_rmses:
            avg_val_rmse = sum(strategy.client_val_rmses) / len(strategy.client_val_rmses)
        else:
            avg_val_rmse = None

        if strategy.client_val_durations:
            avg_client_val_duration = sum(strategy.client_val_durations) / len(strategy.client_val_durations)
            max_client_val_duration = max(strategy.client_val_durations)
        else:
            avg_client_val_duration = None
            max_client_val_duration = None

        if strategy.client_test_losses:
            avg_test_loss = sum(strategy.client_test_losses) / len(strategy.client_test_losses)
        else:
            avg_test_loss = None
            logging.warning(f"[Round {r}] No client test losses collected!")

        if strategy.client_test_maes:
            avg_test_mae = sum(strategy.client_test_maes) / len(strategy.client_test_maes)
        else:
            avg_test_mae = None

        if strategy.client_test_rmses:
            avg_test_rmse = sum(strategy.client_test_rmses) / len(strategy.client_test_rmses)
        else:
            avg_test_rmse = None

        if strategy.client_test_durations:
            avg_client_test_duration = sum(strategy.client_test_durations) / len(strategy.client_test_durations)
            max_client_test_duration = max(strategy.client_test_durations)
        else:
            avg_client_test_duration = None
            max_client_test_duration = None

        round_duration = time.time() - round_start_time

        # Format metrics for logging
        val_loss_str = f"{avg_val_loss:.6f}" if avg_val_loss is not None else "N/A"
        val_mae_str = f"{avg_val_mae:.6f}" if avg_val_mae is not None else "N/A"
        val_rmse_str = f"{avg_val_rmse:.6f}" if avg_val_rmse is not None else "N/A"
        test_loss_str = f"{avg_test_loss:.6f}" if avg_test_loss is not None else "N/A"
        test_mae_str = f"{avg_test_mae:.6f}" if avg_test_mae is not None else "N/A"
        test_rmse_str = f"{avg_test_rmse:.6f}" if avg_test_rmse is not None else "N/A"

        logging.info(f"[Round {r}] Aggregated client val metrics: loss={val_loss_str}, MAE={val_mae_str}, RMSE={val_rmse_str}")
        logging.info(f"[Round {r}] Aggregated client test metrics: loss={test_loss_str}, MAE={test_mae_str}, RMSE={test_rmse_str}")
        logging.info(f"[Round {r}] Round time: {round_duration:.2f}s")

        results.append({
            "round": r,
            "train_loss": float(avg_train_loss) if avg_train_loss is not None else None,
            "val_loss": float(avg_val_loss) if avg_val_loss is not None else None,
            "val_mae": float(avg_val_mae) if avg_val_mae is not None else None,
            "val_rmse": float(avg_val_rmse) if avg_val_rmse is not None else None,
            "test_loss": float(avg_test_loss) if avg_test_loss is not None else None,
            "test_mae": float(avg_test_mae) if avg_test_mae is not None else None,
            "test_rmse": float(avg_test_rmse) if avg_test_rmse is not None else None,
            "best_loss": float(best["loss"]),
            "round_duration_sec": float(round_duration),
            "avg_client_train_duration_sec": float(avg_client_train_duration) if avg_client_train_duration is not None else None,
            "max_client_train_duration_sec": float(max_client_train_duration) if max_client_train_duration is not None else None,
            "avg_client_val_duration_sec": float(avg_client_val_duration) if avg_client_val_duration is not None else None,
            "max_client_val_duration_sec": float(max_client_val_duration) if max_client_val_duration is not None else None,
            "avg_client_test_duration_sec": float(avg_client_test_duration) if avg_client_test_duration is not None else None,
            "max_client_test_duration_sec": float(max_client_test_duration) if max_client_test_duration is not None else None,
        })


        if avg_val_loss is not None and avg_val_loss < best["loss"]:
            best["loss"] = float(avg_val_loss)
            best["round"] = r
            best["arrays"] = copy.deepcopy(arrays)
            logging.info(f"[ROUND {r}] Best checkpoint updated (Val Loss={avg_val_loss:.6f})")


    total_training_time = time.time() - training_start_time

    logging.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    logging.info(f"Average time per round: {total_training_time/total_rounds:.2f}s")

    logging.info("Saving final model to disk...")
    final_model_path = os.path.join(exp_dir, "final_model.pt")
    torch.save(arrays.to_torch_state_dict(), final_model_path)
    logging.info(f"Saved final model to: {final_model_path}")

    df = pd.DataFrame(results)
    training_summary_path = os.path.join(exp_dir, "training_summary.csv")
    df.to_csv(training_summary_path, index=False)
    logging.info(f"Saved training summary to: {training_summary_path}")

    timing_summary = {
        "total_training_time_sec": total_training_time,
        "total_training_time_min": total_training_time / 60,
        "num_rounds": total_rounds,
        "avg_time_per_round_sec": total_training_time / total_rounds,
        "start_timestamp": datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "end_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    timing_df = pd.DataFrame([timing_summary])
    timing_summary_path = os.path.join(exp_dir, "timing_summary.csv")
    timing_df.to_csv(timing_summary_path, index=False)
    logging.info(f"Saved timing summary to: {timing_summary_path}")
