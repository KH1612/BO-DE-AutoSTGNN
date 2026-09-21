import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from search_space import get_search_space
from trainer import (
    set_seed,
    get_model,
    build_metrics,
)
from dataloader import get_dataloaders
from utils import to_native
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE

SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'convergence_plots')


def sample_n_configs(model_name, n=5, seed=SEED):
    cs = get_search_space(model_name)
    cs.seed(seed)
    configs = cs.sample_configuration(n)
    if n == 1:
        configs = [configs]
    return [dict(c) for c in configs]


def train_one_config(
    dataset_name,
    model_name,
    config,
    max_epochs,
    window=12,
    horizon=12,
    base_root='./data',
    log_dir=None,
    run_name=None,
    seed=SEED,
):
    set_seed(seed)
    torch.set_float32_matmul_precision('medium')

    cfg = dict(config)
    lr = float(cfg.pop('lr'))
    batch_size = int(cfg.pop('batch_size'))
    model_kwargs = cfg

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        base_root=base_root,
    )

    sample_batch = next(iter(train_loader))
    n_nodes = sample_batch.input.x.shape[2]
    input_size = sample_batch.input.x.shape[-1]
    output_size = sample_batch.target.y.shape[-1]


    exog_size = sample_batch.input.u.shape[-1] if 'u' in sample_batch.input else 0
    model_kwargs = dict(model_kwargs) if model_kwargs else {}
    model_kwargs.setdefault('exog_size', exog_size)

    model, used_kwargs = get_model(
        model_name=model_name,
        n_nodes=n_nodes,
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        model_kwargs=model_kwargs,
    )
    print(f"  [{run_name}] Training {model_name} with: {used_kwargs}")

    loss_fn = MaskedMAE()
    metrics = build_metrics()

    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': lr},
        loss_fn=loss_fn,
        metrics=metrics,
    )

    callbacks = [EarlyStopping(monitor='val_mae', patience=30, mode='min')]

    logger = CSVLogger(save_dir=log_dir, name=run_name)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = trainer.test(predictor, dataloaders=test_loader)

    test_mae = test_results[0].get('test_mae')
    print(f"  [{run_name}] Finished. test_mae={test_mae:.4f}")

    return {
        'lr': lr,
        'batch_size': batch_size,
        'model_kwargs': model_kwargs,
        'test_mae': test_mae,
        'metrics_csv': os.path.join(logger.log_dir, 'metrics.csv'),
    }


def run_convergence_analysis(
    model_name='graphwavenet',
    dataset_name='metrla',
    n_configs=5,
    max_epochs=15,
    window=12,
    horizon=12,
    base_root='./data',
    output_root=DEFAULT_OUTPUT_ROOT,
    seed=SEED,
):

    model_out_dir = os.path.join(output_root, model_name)
    log_dir = os.path.join(model_out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    configs = sample_n_configs(model_name, n=n_configs, seed=seed)

    configs_path = os.path.join(model_out_dir, f'{model_name}_configs.json')
    with open(configs_path, 'w') as f:
        json.dump(to_native(configs), f, indent=2)
    print(f"Sampled {n_configs} configs, saved to {configs_path}")

    results = []
    for i, config in enumerate(configs):
        run_name = f'config{i}'
        print(f"\n=== {model_name} — {run_name} ({i + 1}/{n_configs}) ===")
        print(f"Config: {config}")
        result = train_one_config(
            dataset_name=dataset_name,
            model_name=model_name,
            config=config,
            max_epochs=max_epochs,
            window=window,
            horizon=horizon,
            base_root=base_root,
            log_dir=log_dir,
            run_name=run_name,
            seed=seed,
        )
        result['config_id'] = i
        result['run_name'] = run_name
        results.append(result)

    results_path = os.path.join(model_out_dir, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nAll configs finished. Results saved to {results_path}")

    plot_convergence(model_name, results, output_dir=model_out_dir, metric='val_mae')
    plot_convergence(model_name, results, output_dir=model_out_dir, metric='train_mae')

    return results


def plot_convergence(model_name, results, output_dir, metric='val_mae'):
    """Plots the given metric vs epoch for every config, saves a PNG."""
    plt.figure(figsize=(10, 6))

    for r in results:
        csv_path = r['metrics_csv']
        if not os.path.exists(csv_path):
            print(f"  WARNING: missing log file for {r['run_name']}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            print(f"  WARNING: '{metric}' not found in {csv_path}")
            continue

        df = df[df[metric].notna()]
        if df.empty:
            continue

        label = f"config {r['config_id']} (test_mae={r['test_mae']:.3f})"
        plt.plot(df['epoch'], df[metric], marker='o', markersize=3, label=label)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{model_name}: {metric} convergence across {len(results)} configs (seed={SEED})')
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(output_dir, f'{model_name}_{metric}_convergence.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convergence analysis across N sampled configs")
    parser.add_argument("--model", type=str, required=True,
                         choices=["graphwavenet","stgcn", "agcrn"])
    parser.add_argument("--dataset", type=str, default="metrla",
                         choices=["metrla", "pemsbay", "electricity"])
    parser.add_argument("--n_configs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--base_root", type=str, default="./data")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    run_convergence_analysis(
        model_name=args.model,
        dataset_name=args.dataset,
        n_configs=args.n_configs,
        max_epochs=args.epochs,
        window=args.window,
        horizon=args.horizon,
        base_root=args.base_root,
        output_root=args.output_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()