import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# LOAD DATA
# ============================================================
rows = []
with open('./master_experiment_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        cleaned = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in r.items()}
        rows.append(cleaned)

print(f"Loaded {len(rows)} experiments")

# Split by dataset
nasa_rows = [r for r in rows if r.get('dataset_name', '').strip() in ('', 'None')]
vnmet_rows = [r for r in rows if r.get('dataset_name', '').strip() == 'VNMET']
print(f"NASA: {len(nasa_rows)} experiments, VNMET: {len(vnmet_rows)} experiments")

# ============================================================
# HELPER: extract best experiment per model for a given horizon
# ============================================================
def get_best_per_model(data_rows, horizon, all_models):
    best = {}
    for r in data_rows:
        if r['pred_len'] != str(horizon):
            continue
        m = r['model'].replace('_nonlinear', '').upper()
        if r['best_test_mae'] == '' or r['client_val_mae_min'] == '':
            continue
        mae = float(r['best_test_mae'])
        if m not in best or mae < best[m]['mae']:
            best[m] = {
                'mae': mae,
                'algo': r['fl_algorithm'],
                'client_min': float(r['client_val_mae_min']),
                'client_max': float(r['client_val_mae_max']),
                'client_mean': float(r['client_val_mae_mean']),
                'client_std': float(r['client_val_mae_std']),
                'best_city': r['client_val_mae_best_city'],
                'worst_city': r['client_val_mae_worst_city'],
                'fairness': float(r['fairness_ratio']),
            }
    models = [m for m in all_models if m in best]
    return best, models


# ============================================================
# HELPER: plot one row of 3 horizon panels
# ============================================================
def plot_fairness_row(fig, axes, dataset_rows, all_models, model_label_map, dataset_name):
    horizons = [1, 72, 432]
    horizon_titles = ['h=1 (1-hour forecast)', 'h=72 (3-day forecast)', 'h=432 (18-day forecast)']
    width = 0.25
    colors = {'min': '#57bb8a', 'mean': '#4f8fc6', 'max': '#d95c54'}

    for col, (h, title) in enumerate(zip(horizons, horizon_titles)):
        ax = axes[col]
        best, models = get_best_per_model(dataset_rows, h, all_models)

        if not models:
            ax.text(0.5, 0.5, f'No data for h={h}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13, color='gray')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            continue

        labels = [model_label_map[m] for m in models]
        x = np.arange(len(models))

        min_vals = [best[m]['client_min'] for m in models]
        mean_vals = [best[m]['client_mean'] for m in models]
        max_vals = [best[m]['client_max'] for m in models]
        fairness = [best[m]['fairness'] for m in models]
        best_city = best[models[0]]['best_city']
        worst_city = best[models[0]]['worst_city']

        ax.bar(x - width, min_vals, width, label=f'Best Client ({best_city})',
               color=colors['min'], edgecolor='white', linewidth=1, zorder=3)
        ax.bar(x, mean_vals, width, label='Mean (all clients)',
               color=colors['mean'], edgecolor='white', linewidth=1, zorder=3)
        ax.bar(x + width, max_vals, width, label=f'Worst Client ({worst_city})',
               color=colors['max'], edgecolor='white', linewidth=1, zorder=3)

        # Fairness ratio annotations
        for i, f in enumerate(fairness):
            ax.text(i, max_vals[i] + (max(max_vals) - min(min_vals)) * 0.04,
                    f'×{f:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#2C3E50')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax.tick_params(axis='y', labelsize=11)

        # Smart y-axis limits
        data_min = min(min_vals)
        data_max = max(max_vals)
        margin = (data_max - data_min) * 0.35
        if data_min > 0.5:
            # For h=72, h=432 — zoom in
            ax.set_ylim(data_min - margin * 0.5, data_max + margin)
        else:
            # For h=1 — start from 0
            ax.set_ylim(0, data_max + margin)

        if col == 0:
            ax.set_ylabel('Validation MAE per Client', fontsize=13, fontweight='bold')

        # Print summary
        for m in models:
            d = best[m]
            print(f"  {dataset_name} h={h} {m}: min={d['client_min']:.4f} ({d['best_city']})  "
                  f"mean={d['client_mean']:.4f}  max={d['client_max']:.4f} ({d['worst_city']})  "
                  f"fairness={d['fairness']:.4f}  algo={d['algo']}")


# ============================================================
# MODEL CONFIG
# ============================================================
all_models = ['GPT4TS', 'BERT', 'BART', 'LLAMA']
model_label_map = {
    'GPT4TS': 'GPT4TS',
    'BERT': 'BERT',
    'BART': 'BART',
    'LLAMA': 'LLaMA'
}

# ============================================================
# FIGURE 1: NASA
# ============================================================
print("\n=== NASA Dataset ===")
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 7))
plot_fairness_row(fig1, axes1, nasa_rows, all_models, model_label_map, 'NASA')
fig1.suptitle('Per-Client Fairness — NASA Wind Dataset (5 Kazakhstan Cities)',
              fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig1.savefig('./fig2a_fairness_nasa.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("Saved: fig2a_fairness_nasa.png")

# ============================================================
# FIGURE 2: VNMET
# ============================================================
print("\n=== VNMET Dataset ===")
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
plot_fairness_row(fig2, axes2, vnmet_rows, all_models, model_label_map, 'VNMET')
fig2.suptitle('Per-Client Fairness — VNMET Wind Dataset (5 Stations)',
              fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('./fig2b_fairness_vnmet.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("Saved: fig2b_fairness_vnmet.png")

# ============================================================
# LOAD DATA
# ============================================================
rows = []
with open('./master_experiment_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        cleaned = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in r.items()}
        rows.append(cleaned)

# Filter to NASA-only (experiments without dataset_name or with empty dataset_name)
rows = [r for r in rows if r.get('dataset_name', '').strip() in ('', 'None')]

print(f"Loaded {len(rows)} experiments (NASA only)")

# ============================================================
# FIGURE 1: Convergence Rate — EXTRACTED FROM DATA
# ============================================================

strategies_order = ['fedavg', 'fedprox', 'fedln', 'fedper']
strategy_labels = ['FedAvg', 'FedProx', 'FedLN', 'FedPer']

convergence_stats = {}
for algo in strategies_order:
    exps = [r for r in rows if r['fl_algorithm'] == algo]
    dec = sum(1 for r in exps if r['loss_trend_last3'] == 'decreasing')
    inc = sum(1 for r in exps if r['loss_trend_last3'] == 'increasing')
    total = len(exps)
    pct = dec / total * 100 if total > 0 else 0
    convergence_stats[algo] = {'decreasing': dec, 'total': total, 'pct': pct}
    print(f"  {algo:<10}: {dec}/{total} decreasing ({pct:.1f}%)")

decreasing = [convergence_stats[a]['decreasing'] for a in strategies_order]
totals = [convergence_stats[a]['total'] for a in strategies_order]
pcts = [convergence_stats[a]['pct'] for a in strategies_order]

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#d95c54', '#57bb8a', '#f3c78b', '#eda17a']
bars = ax.bar(strategy_labels, pcts, color=colors, width=0.6, edgecolor='white', linewidth=1.5, zorder=3)

for bar, d, t, p in zip(bars, decreasing, totals, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{p:.1f}%', ha='center', va='bottom', fontsize=18, fontweight='bold',
            color=bar.get_facecolor())
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{d}/{t}', ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')

ax.set_ylabel('Experiments with Decreasing Val Loss (%)', fontsize=13, fontweight='bold')
ax.set_title('Convergence Success Rate by FL Strategy\n',
             fontsize=15, fontweight='bold', pad=10)

max_pct = max(pcts)
ax.set_ylim(0, max_pct + 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(axis='both', labelsize=13)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

# Find the best strategy dynamically
best_algo_idx = np.argmax(pcts)
worst_algo_idx = np.argmin(pcts)
ratio = pcts[best_algo_idx] / pcts[worst_algo_idx] if pcts[worst_algo_idx] > 0 else float('inf')

ax.annotate(f'{strategy_labels[best_algo_idx]} achieves {ratio:.0f}× higher\nconvergence rate than {strategy_labels[worst_algo_idx]}',
            xy=(best_algo_idx, pcts[best_algo_idx] + 2), xytext=(best_algo_idx + 1.3, pcts[best_algo_idx] + 12),
            fontsize=11, style='italic', color='#2C3E50',
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5),
            ha='center')

plt.tight_layout()
plt.savefig('./fig1_convergence_rate.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 1 saved")