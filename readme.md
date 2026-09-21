# AutoSTGNN BO-DE code
[Github](https://github.com/KH1612/AutoSTGNN)

This code is for benchmarking BO-DE against a Random Search baseline.

Models: Graph Wavenet, STCN, AGCRN

Datasets: Metr-LA, PEMS-BAY, ElectricityBenchmark

Models and datasets are sourced from [tsl (Torch Spatiotemporal)](https://torch-spatiotemporal.readthedocs.io/en/latest/)

All code is designed to run on a Google Colab notebook. 
Quick start and usage instructions shown in [AutoSTGNN_example.ipynb](AutoSTGNN_example.ipynb)


## Results

- [Training results for BO-DE vs RS](search_results)
- [All solver comparison](search_results/crosssolver/)
- [Final training results](search_results/final_runs/)
- [Plots](plots)

## File description

All source code is stored in [src](src)

### Setup
- **dataloader.py** - Builds train/val/test dataloaders per dataset (MetrLA, PemsBay, Electricity). Uses tsl's built-in sensor-similarity graph for MetrLA/PemsBay, and a correlation-based k-NN graph for Electricity (which has no built-in connectivity).

- **trainer.py** - Core `train()` function used by solver scripts. builds the model, wraps it in a tsl `Predictor`, runs a PyTorch Lightning `Trainer`, and returns the trained predictor, trainer, test results, best checkpoint path, and best validation MAE.

- **search_space.py** - Defines each model's hyperparameter search space (`ConfigSpace`). Used identically by `random_search.py` and `bo_de.py`, so both methods are compared over exactly the same space.

- **utils.py** - Contains shared helpers

### Solver code
- **standard_stgnn.py** - Provides a no search baseline for each model-dataset combination

- **convergence.py** - Creates convergence plots for model-dataset combinations

- **random_search.py** - Baseline hyperparameter search: samples configs uniformly at random from `search_space.py` and trains each one, logging results to JSON/Excel. Supports resuming of crashed runs.

- **bo_de.py** - The BO-DE solver: Bayesian Optimization with a Gaussian Process surrogate and Expected Improvement acquisition, maximized via Differential Evolution ([Algorithm 2](https://www.nature.com/articles/s41598-023-32027-3) of the reference paper). Supports resuming of crashed runs. 

- **train_final.py** - Takes a completed search's results `.xlsx`, extracts the best-found config, and retrains it for a longer, fixed number of epochs to produce a final reportable result.

### Plotting
- **plot_trials.py** - Plots best-val-MAE-found-so-far vs. trial number, one panel per model, comparing RS vs BO-DE, from the `.xlsx` result files in a results directory.

- **trial_analysis.py** - Builds a summary table (mean/median/std/best val_mae, trials-to-best, per-trial timing) comparing RS vs BO-DE per (model, dataset), from the same result files.

- **solver_compare.py** - Anytime-performance plot: best value found so far vs. cumulative wall-clock training time, comparing arbitrary solvers (BO-DE, RS, and external baselines like DEHB/BOHB) on the same model-dataset combination.

- **graph.py** - Diagnostic tool: loads a dataset's connectivity graph and visualizes it 

