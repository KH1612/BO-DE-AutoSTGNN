
import sys
from pathlib import Path


def show_graph(
    dataset_name,
    src_dir='.',
    base_root='./data',
    batch_size=64,
    window=12,
    horizon=12,
    out_path='graph.png',
    layout='spring',
    node_size=20,
):
    sys.path.insert(0, str(Path(src_dir).resolve()))

    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from dataloader import get_dataloaders

    train_loader, _, _ = get_dataloaders(
        dataset_name=dataset_name, window=window, horizon=horizon,
        batch_size=batch_size, base_root=base_root,
    )
    sample_batch = next(iter(train_loader))
    n_nodes = sample_batch.input.x.shape[2]
    edge_index = sample_batch.edge_index.cpu().numpy()


    degree = np.zeros(n_nodes, dtype=int)
    for src, dst in edge_index.T:
        degree[src] += 1

    n_edges = edge_index.shape[1]
    isolated = np.where(degree == 0)[0]
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else float('nan')

    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {n_nodes}   Edges (directed): {n_edges}   Density: {density:.4f}")
    print(f"Degree: min={degree.min()}  max={degree.max()}  "
          f"mean={degree.mean():.2f}  median={int(np.median(degree))}")
    print(f"Isolated nodes (degree=0): {len(isolated)} / {n_nodes}"
          f"{f'  -> indices: {isolated.tolist()}' if 0 < len(isolated) <= 20 else ''}")


    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout '{layout}'")

    node_colors = degree
    fig, ax = plt.subplots(figsize=(10, 10))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors,
        cmap='viridis', ax=ax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False, ax=ax)
    if isolated.size:
        nx.draw_networkx_nodes(
            G, pos, nodelist=isolated.tolist(), node_size=node_size * 1.5,
            node_color='red', ax=ax,
        )
    plt.colorbar(nodes, ax=ax, label='degree')
    ax.set_title(f"{dataset_name}: connectivity graph "
                f"({n_nodes} nodes, {n_edges} edges, "
                f"{len(isolated)} isolated in red)")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'degree': degree,
        'isolated': isolated,
    }

if __name__ == '__main__':
    show_graph('electricity')