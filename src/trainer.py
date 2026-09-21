import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from tsl.nn.models import GraphWaveNetModel, STCNModel, AGCRNModel
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataloader import get_dataloaders


_torch_load = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


MODEL_MAP = {
    'graphwavenet': GraphWaveNetModel,
    'stgcn': STCNModel,
    'agcrn': AGCRNModel
}

DEFAULT_MODEL_KWARGS = {
    'graphwavenet': {
        'exog_size': 0,
        'hidden_size': 32,
        'ff_size': 256,
        'n_layers': 8,
        'temporal_kernel_size': 2,
        'spatial_kernel_size': 2,
        'learned_adjacency': True,
        'emb_size': 10,
        'dilation': 2,
        'dilation_mod': 2,
        'norm': 'batch',
        'dropout': 0.3,
    },
    'stgcn': {
        'exog_size': 0,
        'hidden_size': 64,
        'ff_size': 128,
        'n_layers': 1,
        'temporal_kernel_size': 3,
        'spatial_kernel_size': 2,
        'dropout': 0.3,
    },
    'agcrn': {
        'hidden_size': 64,
        'emb_size': 10,
        'n_layers': 1,
    },
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def get_model(model_name, n_nodes, input_size, output_size, horizon, model_kwargs=None):
    model_name = model_name.lower()
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_MAP.keys())}"
        )

    model_cls = MODEL_MAP[model_name]

    kwargs = dict(DEFAULT_MODEL_KWARGS[model_name])
    if model_kwargs:
        kwargs.update(model_kwargs)

    if model_name == 'graphwavenet' and kwargs.get('learned_adjacency', True):
        kwargs['n_nodes'] = n_nodes

    if model_name == 'agcrn':
        kwargs['n_nodes'] = n_nodes

    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        **kwargs
    )

    return model, kwargs


class MaskedRMSE(MaskedMSE):
    def compute(self):
        return torch.sqrt(super().compute())


def build_metrics(horizon_steps=(3, 6, 12)):

    metrics = {
        'mae': MaskedMAE(),
        'mape': MaskedMAPE(),
        'rmse': MaskedRMSE(),
    }

    for step in horizon_steps:
        label = str(step)
        at = step - 1
        metrics[f'mae_at_{label}'] = MaskedMAE(at=at)
        metrics[f'mape_at_{label}'] = MaskedMAPE(at=at)
        metrics[f'rmse_at_{label}'] = MaskedRMSE(at=at)

    return metrics


def train(
    dataset_name='metrla',
    model_name='stcn',
    window=None,
    horizon=12,
    batch_size=64,
    lr=1e-3,
    max_epochs=50,
    base_root='./data',
    model_kwargs=None,
    checkpoint_dir='./checkpoints',
    save_best=True,
    seed=42,
    horizon_steps=(3, 6, 12),
    patience = 30
):
    set_seed(seed)
    torch.set_float32_matmul_precision('medium')

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
        model_kwargs=model_kwargs
    )
    print(f"Training {model_name} on {dataset_name} with: {used_kwargs}")

    loss_fn = MaskedMAE()
    metrics = build_metrics(horizon_steps=horizon_steps)

    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': lr},
        loss_fn=loss_fn,
        metrics=metrics
    )

    callbacks = [EarlyStopping(monitor='val_mae', patience=patience, mode='min')]

    checkpoint_callback = None
    if save_best:
        run_dir = os.path.join(checkpoint_dir, dataset_name, model_name)
        os.makedirs(run_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir,
            filename='best-{epoch:02d}-{val_mae:.4f}',
            monitor='val_mae',
            mode='min',
            save_top_k=1,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
    )

    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None
    test_results = trainer.test(predictor, dataloaders=test_loader,ckpt_path=best_model_path)

    
    best_val_mae = (
        float(checkpoint_callback.best_model_score)
        if checkpoint_callback and checkpoint_callback.best_model_score is not None
        else None
    )
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")

    return predictor, trainer, test_results, best_model_path, best_val_mae