import os
import numpy as np
from tsl.datasets import MetrLA, PemsBay, ElectricityBenchmark
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.ops.connectivity import adj_to_edge_index


DATASET_MAP = {
    'metrla': MetrLA,
    'pemsbay': PemsBay,
    'electricity': ElectricityBenchmark,
}


DATASET_KWARGS = {}

CONN_DEFAULTS = {
    'electricity': {'k': 6, 'include_self': False, 'layout': 'edge_index'},
}

DEFAULT_WINDOW = {
    'electricity': 12,
}


def _knn_connectivity(dataset, root, k=6, include_self=False, **kwargs):
    cache_path = os.path.join(root, 'correlation_adj.npy')

    if os.path.exists(cache_path):
        adj = np.load(cache_path)
    else:
        values = dataset.dataframe().values
        adj = np.abs(np.corrcoef(values, rowvar=False))
        adj = np.nan_to_num(adj)
        np.save(cache_path, adj)

    adj = adj.copy()  
    n = adj.shape[0]

    if not include_self:
        np.fill_diagonal(adj, -np.inf)  

    if k >= n - 1:
        raise ValueError(f"k={k} must be smaller than n_nodes-1={n - 1}")


    topk_idx = np.argpartition(-adj, kth=k, axis=1)[:, :k]

    knn_mask = np.zeros_like(adj, dtype=bool)
    rows = np.repeat(np.arange(n), k)
    cols = topk_idx.ravel()
    knn_mask[rows, cols] = True
    knn_mask = knn_mask | knn_mask.T

    if not include_self:
        np.fill_diagonal(knn_mask, False)

    out_adj = np.where(knn_mask, np.abs(adj), 0.0)
    if include_self:
        np.fill_diagonal(out_adj, 1.0)

    return adj_to_edge_index(out_adj)


def get_dataloaders(
    dataset_name='metrla',
    window=None,
    horizon=12,
    batch_size=64,
    val_len=0.1,
    test_len=0.2,
    conn_threshold=0.1,
    base_root='./data',
    workers=None,
):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_MAP.keys())}"
        )

    if window is None:
        window = DEFAULT_WINDOW.get(dataset_name, 12)

    root = os.path.join(base_root, dataset_name)
    os.makedirs(root, exist_ok=True)

    dataset_cls = DATASET_MAP[dataset_name]
    extra_kwargs = DATASET_KWARGS.get(dataset_name, {})
    dataset = dataset_cls(root=root, **extra_kwargs)

    conn_kwargs = CONN_DEFAULTS.get(dataset_name, {
        'threshold': conn_threshold,
        'include_self': False,
        'normalize_axis': 1,
        'layout': 'edge_index',
    })

    if dataset.similarity_options is None: # Triggers for electricity
        connectivity = _knn_connectivity(dataset, root, **conn_kwargs)
    else:
        connectivity = dataset.get_connectivity(**conn_kwargs)

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        mask=dataset.mask,
        horizon=horizon,
        window=window,
        stride=1
    )
    # Electricity nodes normalised by own histrical value, between [0,1] tested but gave poor results
    scalers = {'target': StandardScaler(axis=(0, 1) if dataset_name != 'electricity' else (0,))}
    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)

    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=workers,
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    return train_loader, val_loader, test_loader