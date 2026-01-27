from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
# from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS_Linear, GPT4TS_Nonlinear, GPT4TS_Medium_Nonlinear
from models.BERT import BERT_Linear, BERT_Nonlinear
# from models.RoBERTa import RoBERTa_Linear, RoBERTa_Nonlinear
# from models.BART import BART_Linear, BART_Nonlinear
# from models.LLAMA import Llama_Linear, Llama_Nonlinear
# from models.DLinear import DLinear

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import csv

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import json

import os, json, shutil
from datetime import datetime

try:
    from peft import PeftModel, LoraConfig, get_peft_model_state_dict
except Exception:
    PeftModel = None
    LoraConfig = None


def makedirs(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def find_peft_wrapped_module(model):
    """
    Return the first PEFT-wrapped submodule (or the model itself if already a PeftModel).
    Works when you wrap only the HF backbone (e.g., model.gpt2) inside your composite model.
    """
    if PeftModel is None:
        return None
    if isinstance(model, PeftModel):
        return model
    for _name, m in model.named_modules():
        if isinstance(m, PeftModel):
            return m
    return None


def save_lora_artifacts(model, args, run_dir, metrics=None):
    """
    Save:
    - adapter-only (adapter_config.json + adapter_model.bin) via PEFT
    - merged base model (optional; for inference without PEFT)
    - LoRA metadata (json)
    """
    makedirs(run_dir)
    lora_meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "dataset_name": args.dataset_name,
        "model": args.model,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "llm_layers": args.llm_layers,
        # LoRA hyperparams if present on args
        "lora_r": getattr(args, "lora_r", getattr(args, "lora_rank", None)),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", None),
        "lora_target_modules": getattr(args, "lora_target_modules", None),
        # Trainable stats
        "params_total_M": metrics.get("params_total_M") if metrics else None,
        "params_trainable_M": metrics.get("params_trainable_M") if metrics else None,
        "params_trainable_pct": metrics.get("params_trainable_pct") if metrics else None,
    }

    # 1) Save adapter-only (best for fine-tune portability)
    adapter_dir = makedirs(os.path.join(run_dir, "lora_adapter"))
    peft_mod = find_peft_wrapped_module(model)
    if peft_mod is not None:
        # Prefer the official save API (writes adapter_config.json + adapter_model.bin)
        try:
            peft_mod.save_pretrained(adapter_dir)
        except Exception:
            # Fallback: raw adapter state dict
            torch.save(get_peft_model_state_dict(peft_mod), os.path.join(adapter_dir, "adapter_model.bin"))
    else:
        # If we cannot find a PEFT module, warn into metadata
        lora_meta["warning"] = "No PeftModel found; adapter not saved."

    # 2) (Optional) Save merged base model (so you can run without PEFT at inference)
    merged_dir = makedirs(os.path.join(run_dir, "merged_base_model"))
    if peft_mod is not None:
        try:
            merged = peft_mod.merge_and_unload()  # returns a HF model with LoRA merged
            try:
                merged.save_pretrained(merged_dir)  # HF format
            except Exception:
                # Fallback to plain PyTorch checkpoint
                torch.save(merged.state_dict(), os.path.join(merged_dir, "pytorch_model.bin"))
        except Exception as e:
            lora_meta["merge_warning"] = f"merge_and_unload failed: {str(e)}"
    else:
        # As a fallback, still save the overall model state dict
        torch.save(model.state_dict(), os.path.join(merged_dir, "model_state_dict.bin"))

    # 3) Save LoRA metadata
    with open(os.path.join(run_dir, "lora_metadata.json"), "w") as f:
        json.dump(lora_meta, f, indent=2)

    # 4) (Nice-to-have) copy your metrics CSV/JSON into the same folder
    try:
        if os.path.exists(comprehensive_results_path):
            shutil.copy2(comprehensive_results_path,
                         os.path.join(run_dir, os.path.basename(comprehensive_results_path)))
        if os.path.exists(experiment_log_path):
            shutil.copy2(experiment_log_path, os.path.join(run_dir, os.path.basename(experiment_log_path)))
    except Exception:
        pass

    print(f"[LoRA] Saved adapter to: {adapter_dir}")
    print(f"[LoRA] Saved merged model to: {merged_dir}")
    print(f"[LoRA] Saved metadata to: {os.path.join(run_dir, 'lora_metadata.json')}")


warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='LLM')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--data', type=str, default='custom')

parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freq', type=str, default='10min')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--dataset_name', type=str, default='vietnam')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--target', type=str, default='speed')

parser.add_argument('--is_gpt', type=int, default=0)
parser.add_argument('--is_bert', type=int, default=0)
parser.add_argument('--is_roberta', type=int, default=1)
parser.add_argument('--is_bart', type=int, default=0)
parser.add_argument('--is_llama', type=int, default=0)

parser.add_argument('--is_nonlinear', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--llm_layers', type=int, default=6)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--train_epochs', type=int, default=10)

parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--learning_rate', type=float, default=0.0001)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)

parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--n_heads', type=int, default=16)

parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--kernel_size', type=int, default=3)

parser.add_argument('--cos', type=int, default=0)

parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--num_workers', type=int, default=10)

parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--loss_func', type=str, default='mse')

parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)

parser.add_argument('--experiment_type', type=str, default='experiment_results')

# LoRA Configuration
parser.add_argument('--use_lora', type=int, default=0, help='Enable LoRA (1) or disable (0)')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (lower = fewer parameters)')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA scaling parameter')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')

args = parser.parse_args()
args.checkpoints = os.path.join(args.checkpoints, args.experiment_type)

# Create results directory
results_base_dir = './experiment_results'
if not os.path.exists(results_base_dir):
    os.makedirs(results_base_dir)

# Create comprehensive results file
timestamp = time.strftime("%Y%m%d_%H%M%S")
comprehensive_results_path = os.path.join(results_base_dir,
                                          f'{args.dataset_name}_{args.model}_{args.experiment_type}_{args.seq_len}.csv')
experiment_log_path = os.path.join(results_base_dir, f'experiment_{args.model_id}_{timestamp}.json')


# Function to write comprehensive results
def write_comprehensive_results(args, metrics, iteration_results):
    """Write all parameters and results to a comprehensive CSV file"""

    # Check if file exists to write header
    write_header = not os.path.exists(comprehensive_results_path)

    with open(comprehensive_results_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if write_header:
            header = [
                # Experiment identification
                'timestamp', 'model_id', 'model_type',
                # Key variable parameters
                'pred_len', 'learning_rate', 'decay_fac', 'dropout',
                'n_heads', 'patch_size', 'stride', 'label_len',
                # Other parameters
                'hidden_size', 'kernel_size', 'seq_len', 'batch_size', 'train_epochs', 'd_model', 'd_ff',
                'llm_layers', 'e_layers', 'enc_in', 'c_out', 'freq',
                'loss_func', 'cos', 'tmax', 'percent', 'patience', 'params_total_M', 'params_trainable_M',
                'params_trainable_pct',
                # Metrics (mean and std)
                'epoch_mean', 'epoch_std', 'mse_mean', 'mse_std', 'mae_mean', 'mae_std',
                'rmse_mean', 'rmse_std', 'mape_mean', 'mape_std',
                'mspe_mean', 'mspe_std', 'smape_mean', 'smape_std',
                'nd_mean', 'nd_std',
                # Training info
                'iterations', 'early_stopped', 'final_epoch',
                # Individual iteration results
                'mse_all', 'mae_all', 'rmse_all', 'mape_all',
                'mspe_all', 'smape_all', 'nd_all'
            ]
            writer.writerow(header)

        # Prepare row data
        row = [
            timestamp,
            args.model_id,
            args.model,
            # Key variable parameters
            args.pred_len,
            args.learning_rate,
            args.decay_fac,
            args.dropout,
            args.n_heads,
            args.patch_size,
            args.stride,
            args.label_len,
            # Other parameters
            args.hidden_size,
            args.kernel_size,
            args.seq_len,
            args.batch_size,
            args.train_epochs,
            args.d_model,
            args.d_ff,
            args.llm_layers,
            args.e_layers,
            args.enc_in,
            args.c_out,
            args.freq,
            args.loss_func,
            args.cos,
            args.tmax,
            args.percent,
            args.patience,
            # Metrics
            metrics['params_total_M'],
            metrics['params_trainable_M'],
            metrics['params_trainable_pct'],
            metrics['epoch_mean'],
            metrics['epoch_std'],
            metrics['mse_mean'],
            metrics['mse_std'],
            metrics['mae_mean'],
            metrics['mae_std'],
            metrics['rmse_mean'],
            metrics['rmse_std'],
            metrics['mape_mean'],
            metrics['mape_std'],
            metrics['mspe_mean'],
            metrics['mspe_std'],
            metrics['smape_mean'],
            metrics['smape_std'],
            metrics['nd_mean'],
            metrics['nd_std'],
            # Training info
            args.itr,
            iteration_results.get('early_stopped', False),
            iteration_results.get('final_epoch', args.train_epochs),
            # Individual results as strings
            ','.join(map(str, iteration_results['mses'])),
            ','.join(map(str, iteration_results['maes'])),
            ','.join(map(str, iteration_results['rmses'])),
            ','.join(map(str, iteration_results['mapes'])),
            ','.join(map(str, iteration_results['mspes'])),
            ','.join(map(str, iteration_results['smapes'])),
            ','.join(map(str, iteration_results['nds']))
        ]

        writer.writerow(row)


def save_experiment_log(args, metrics, iteration_results, training_history):
    """Save detailed experiment information in JSON format"""

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    experiment_data = {
        'timestamp': timestamp,
        'model_id': args.model_id,
        'parameters': convert_to_native(vars(args)),
        'metrics_summary': convert_to_native(metrics),
        'iteration_results': convert_to_native({
            'mses': iteration_results['mses'],
            'maes': iteration_results['maes'],
            'rmses': iteration_results['rmses'],
            'mapes': iteration_results['mapes'],
            'mspes': iteration_results['mspes'],
            'smapes': iteration_results['smapes'],
            'nds': iteration_results['nds']
        }),
        'training_history': convert_to_native(training_history)
    }

    with open(experiment_log_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)


# Main experiment loop
mses, maes, rmses, mapes, mspe_list, smapes, nds = [], [], [], [], [], [], []
training_history = []
epoch_durations = []
early_stopped = False
final_epoch = args.train_epochs
params_total_M = params_trainable_M = params_trainable_pct = None

for ii in range(args.itr):
    print(f"\n{'=' * 50}")
    print(f"Starting iteration {ii + 1}/{args.itr}")
    print(f"{'=' * 50}\n")

    setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(
        args.model, args.dataset_name, args.model_id, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.llm_layers,
        args.d_ff, args.embed, ii
    )
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = len(train_loader)

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    elif args.model == 'BERT_Linear':
        model = BERT_Linear(args, device)
        model.to(device)
    elif args.model == 'BERT_Nonlinear':
        model = BERT_Nonlinear(args, device)
        model.to(device)
    elif args.model == 'GPT4TS_Linear':
        model = GPT4TS_Linear(args, device)
        model.to(device)
    elif args.model == 'GPT4TS_Nonlinear':
        model = GPT4TS_Nonlinear(args, device)
        model.to(device)
    elif args.model == 'GPT4TS_Medium_Nonlinear':
        model = GPT4TS_Medium_Nonlinear(args, device)
        model.to(device)
    elif args.model == 'RoBERTa_Linear':
        model = RoBERTa_Linear(args, device)
        model.to(device)
    elif args.model == 'RoBERTa_Nonlinear':
        model = RoBERTa_Nonlinear(args, device)
        model.to(device)
    elif args.model == 'BART_Linear':
        model = BART_Linear(args, device)
        model.to(device)
    elif args.model == 'BART_Nonlinear':
        model = BART_Nonlinear(args, device)
        model.to(device)
    elif args.model == 'Llama_Linear':
        model = Llama_Linear(args, device)
        model.to(device)
    elif args.model == 'Llama_Nonlinear':
        model = Llama_Nonlinear(args, device)
        model.to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = model.parameters()

    if params_total_M is None:
        params_total_M = total_params / 1e6
        params_trainable_M = trainable_params / 1e6
        params_trainable_pct = 100.0 * trainable_params / total_params
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()

            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))


        criterion = SMAPE()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    iteration_training_history = []
    epochs_all = []

    for epoch in range(args.train_epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.time()
        iter_count = 0
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, ii)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_duration = time.time() - epoch_start
        epochs_all.append(epoch_duration)

        # Record training history
        epoch_history = {
            'iteration': ii,
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'vali_loss': float(vali_loss),
            'learning_rate': model_optim.param_groups[0]['lr'],
            'epoch_duration': epoch_duration
        }
        iteration_training_history.append(epoch_history)

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            early_stopped = True
            final_epoch = epoch + 1
            break

    training_history.extend(iteration_training_history)
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_model_path = path + '/' + f'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    lora_run_dir = os.path.join(results_base_dir, "lora_runs",
                                f"{args.dataset_name}_{args.model}_{args.model_id}_sl{args.seq_len}_pl{args.pred_len}_ll{args.llm_layers}",
                                f"iter_{ii + 1}")
    # save_lora_artifacts(model, args, lora_run_dir, metrics)
    print("------------------------------------")

    # Test the model
    mse, mae, rmse, mape, mspe, smape, nd = test(model, test_data, test_loader, args, device, ii)

    mses.append(mse)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    mspe_list.append(mspe)
    smapes.append(smape)
    nds.append(nd)
    epoch_durations.append(float(np.mean(epochs_all)))

    print(f"Iteration {ii + 1} Results:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}, MSPE: {mspe:.4f}, SMAPE: {smape:.4f}, ND: {nd:.4f}")

# Convert to numpy arrays
mses = np.array(mses)
maes = np.array(maes)
rmses = np.array(rmses)
mapes = np.array(mapes)
mspe_list = np.array(mspe_list)
smapes = np.array(smapes)
nds = np.array(nds)
epoch_time = np.array(epoch_durations)

# Compute mean & std for each metric
metrics = {
    'mse_mean': mses.mean(),
    'mse_std': mses.std(),
    'mae_mean': maes.mean(),
    'mae_std': maes.std(),
    'rmse_mean': rmses.mean(),
    'rmse_std': rmses.std(),
    'mape_mean': mapes.mean(),
    'mape_std': mapes.std(),
    'mspe_mean': mspe_list.mean(),
    'mspe_std': mspe_list.std(),
    'smape_mean': smapes.mean(),
    'smape_std': smapes.std(),
    'nd_mean': nds.mean(),
    'nd_std': nds.std(),
    'epoch_mean': epoch_time.mean(),
    'epoch_std': epoch_time.std(),
    'params_total_M': params_total_M,
    'params_trainable_M': params_trainable_M,
    'params_trainable_pct': params_trainable_pct,
}

# Prepare iteration results
iteration_results = {
    'mses': mses.tolist(),
    'maes': maes.tolist(),
    'rmses': rmses.tolist(),
    'mapes': mapes.tolist(),
    'mspes': mspe_list.tolist(),
    'smapes': smapes.tolist(),
    'nds': nds.tolist(),
    'early_stopped': early_stopped,
    'final_epoch': final_epoch
}
print("JELLO")
# Write comprehensive results
write_comprehensive_results(args, metrics, iteration_results)
final_lora_dir = os.path.join(results_base_dir, "lora_runs",
                              f"{args.dataset_name}_{args.model}_{args.model_id}_sl{args.seq_len}_pl{args.pred_len}_ll{args.llm_layers}",
                              "final")
# Reload best model from last iteration again (or keep current)
save_lora_artifacts(model, args, final_lora_dir, metrics)
# Save detailed experiment log
save_experiment_log(args, metrics, iteration_results, training_history)

# Also write to the original results file for backward compatibility
results_path = os.path.join(args.checkpoints, f'results_TRAVIHN_{args.seq_len}.csv')
write_header = not os.path.exists(results_path)
with open(results_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            'pred_len',
            'MSE',
            'MAE',
            'RMSE',
            'MAPE',
            'MSPE',
            'SMAPE',
            'ND',
            'Epoch_Time'
        ])
    writer.writerow([
        args.pred_len,
        metrics['mse_mean'],
        metrics['mae_mean'],
        metrics['rmse_mean'],
        metrics['mape_mean'],
        metrics['mspe_mean'],
        metrics['smape_mean'],
        metrics['nd_mean'],
        metrics['epoch_mean'],
    ])

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
print(f"MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
print(f"RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
print(f"MAPE: {metrics['mape_mean']:.4f} ± {metrics['mape_std']:.4f}")
print(f"MSPE: {metrics['mspe_mean']:.4f} ± {metrics['mspe_std']:.4f}")
print(f"SMAPE: {metrics['smape_mean']:.4f} ± {metrics['smape_std']:.4f}")
print(f"ND: {metrics['nd_mean']:.4f} ± {metrics['nd_std']:.4f}")

print(f"\nResults saved to:")
print(f"  - Comprehensive CSV: {comprehensive_results_path}")
print(f"  - Detailed JSON log: {experiment_log_path}")
print(f"  - Original format: {results_path}")