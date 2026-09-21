import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
from datetime import datetime

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm

import ConfigSpace.hyperparameters as CSH

from search_space import get_search_space
from trainer import train
from utils import results_to_excel, to_native


def iter_rng(seed, phase, idx):

    return np.random.default_rng([int(seed), int(phase), int(idx)])


'''
All hyperparams map to [0,1]^D (D is dimensions) so its compactible with DE. GP is also fit with these vectors.

config_to_vector encodes, vector_to_config decodes. 

quantize prevents GP from seeing 2 different configs as same

'''
def cs_to_bounds(cs):
    bounds = []
    for hp in list(cs.values()):
        if isinstance(hp, CSH.UniformFloatHyperparameter):
            bounds.append({'name': hp.name, 'type': 'float',
                           'lower': hp.lower, 'upper': hp.upper, 'log': hp.log})
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            if hp.log:
                raise NotImplementedError(
                    f"'{hp.name}': log-scaled integers use uniform index bins here, "
                    "which would misrepresent their spacing. Encode as a Categorical "
                    "with explicit levels instead."
                )
            bounds.append({'name': hp.name, 'type': 'int',
                           'lower': int(hp.lower), 'upper': int(hp.upper)})
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            bounds.append({'name': hp.name, 'type': 'cat', 'choices': list(hp.choices)})
        else:
            raise TypeError(f"Unsupported hyperparameter type for '{hp.name}': {type(hp)}")
    return bounds


def vector_to_config(vector, bounds):
    config = {}
    for i, b in enumerate(bounds):
        v = float(np.clip(vector[i], 0.0, 1.0))
        if b['type'] == 'cat':
            n = len(b['choices'])
            config[b['name']] = b['choices'][min(int(v * n), n - 1)]
        elif b['type'] == 'int':
            n = b['upper'] - b['lower'] + 1
            config[b['name']] = int(b['lower'] + min(int(v * n), n - 1))
        else:
            lower, upper = b['lower'], b['upper']
            if b.get('log'):
                config[b['name']] = float(
                    np.exp(np.log(lower) + v * (np.log(upper) - np.log(lower))))
            else:
                config[b['name']] = float(lower + v * (upper - lower))
    return config


def config_to_vector(config, bounds):
    vector = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        if b['name'] not in config:
            raise KeyError(
                f"'{b['name']}' missing from config {sorted(config)}. If the search "
                "space changed since the run being resumed, the old results are not "
                "reusable - start a fresh run."
            )
        val = config[b['name']]
        if b['type'] == 'cat':
            if val not in b['choices']:
                raise ValueError(
                    f"'{b['name']}'={val!r} is not among the current choices "
                    f"{b['choices']}. The search space changed since this run."
                )
            vector[i] = (b['choices'].index(val) + 0.5) / len(b['choices'])
        elif b['type'] == 'int':
            n = b['upper'] - b['lower'] + 1
            vector[i] = (int(val) - b['lower'] + 0.5) / n
        else:
            lower, upper = b['lower'], b['upper']
            v = float(val)
            if b.get('log'):
                vector[i] = (np.log(v) - np.log(lower)) / (np.log(upper) - np.log(lower))
            else:
                vector[i] = (v - lower) / (upper - lower)
        vector[i] = float(np.clip(vector[i], 0.0, 1.0))
    return vector


def quantize(vector, bounds):
    return config_to_vector(vector_to_config(vector, bounds), bounds)


def initial_design(n_init, bounds, seed):
    D = len(bounds)
    rng = np.random.default_rng(seed)
    return rng.random((n_init, D))



def acquisition(X_new, gp, y_best, xi=0.0): # EI function
    X_new = np.atleast_2d(X_new)
    mu, sigma = gp.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    z = (y_best - mu - xi) / sigma
    return (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def ei_terms(x, gp, y_best, xi=0.0): # Save EI terms for diagnositic
    mu, sigma = gp.predict(np.atleast_2d(x), return_std=True)
    mu, sigma = float(mu[0]), max(float(sigma[0]), 1e-12)
    z = (y_best - mu - xi) / sigma
    exploit = (y_best - mu - xi) * norm.cdf(z)
    explore = sigma * norm.pdf(z)
    return exploit, explore, mu, sigma



def de_maximise_ei(gp, y_best, bounds, rng, n_pop=10, k=20, f=0.8, p_c=0.9,
                   xi=0.0, tol=1e-6, patience=20, verbose=True): # Differential Evolution loop
    D = len(bounds)
    pop = rng.random((n_pop, D))
    ei = acquisition(pop, gp, y_best, xi=xi)
    best_ei, stale, converged_at = ei.max(), 0, k

    for gen in range(k):
        # three distinct donors per individual, none equal to the target
        r = rng.random((n_pop, n_pop))
        r[np.arange(n_pop), np.arange(n_pop)] = np.inf
        idx = np.argsort(r, axis=1)[:, :3]

        donors = pop[idx[:, 0]] + f * (pop[idx[:, 1]] - pop[idx[:, 2]])

        oob = (donors < 0.0) | (donors > 1.0)
        if oob.any():
            donors[oob] = rng.random((n_pop, D))[oob]

        mask = rng.random((n_pop, D)) <= p_c
        mask[np.arange(n_pop), rng.integers(0, D, n_pop)] = True  # forced delta
        trials = np.where(mask, donors, pop)

        ei_trials = acquisition(trials, gp, y_best, xi=xi)  # one batched call

        improved = ei_trials >= ei
        pop[improved] = trials[improved]
        ei[improved] = ei_trials[improved]

        if ei.max() > best_ei + tol:
            best_ei, stale = ei.max(), 0
        else:
            stale += 1
            if stale >= patience:
                converged_at = gen
                break

    best_idx = int(np.argmax(ei))
    if verbose:
        print(f"  DE: converged at gen {converged_at}/{k}, "
              f"max EI={ei[best_idx]:.6g}", flush=True)
    return pop[best_idx], float(ei[best_idx]), converged_at


# Gaussian process as defined in the paper
def build_gp(seed):
    k = ConstantKernel(1.0) * RBF(length_scale=1.0)
    return GaussianProcessRegressor(
        kernel=k, n_restarts_optimizer=10, normalize_y=True, random_state=seed)


def fit_gp(gp, X_obs, y_obs):

    X = np.asarray(X_obs)
    y = np.asarray(y_obs, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return False, None
    gp.fit(X[ok], y[ok])
    return True, float(y[ok].min())


# For resuming
def record_to_config(record): # Rebuilt huperparam dict from saved trials
    cfg = dict(record.get('model_kwargs', {}))
    cfg['lr'] = record['lr']
    cfg['batch_size'] = record['batch_size']
    return cfg


def load_state(resume_from, bounds): # Reload completed trials from json
    with open(resume_from, 'r') as fp:
        results_log = json.load(fp)

    X_obs, y_obs = [], []
    n_init_done = n_iters_done = 0
    best_mae, best_config, best_model_path = float('inf'), None, None

    for rec in results_log:
        cfg = record_to_config(rec)
        X_obs.append(config_to_vector(cfg, bounds))

        objective = rec.get('val_mae')
        objective = float('inf') if objective is None else float(objective)
        y_obs.append(objective)

        if rec.get('phase') == 'init':
            n_init_done += 1
        else:
            n_iters_done += 1

        if objective < best_mae:
            best_mae = objective
            best_config = cfg
            best_model_path = rec.get('best_model_path')

    return (X_obs, y_obs, results_log, n_init_done, n_iters_done,
            best_mae, best_config, best_model_path)


# Objective function
def train_one_run(base_args, config, dataset_name, seed=None, patience = 30):
    config = dict(config)
    lr = float(config.pop('lr'))
    batch_size = int(config.pop('batch_size'))
    model_kwargs = config

    if seed is not None:
        try:
            import pytorch_lightning as pl
            pl.seed_everything(seed, workers=True)
        except Exception:
            try:
                import lightning.pytorch as pl
                pl.seed_everything(seed, workers=True)
            except Exception:
                pass

    predictor, pl_trainer, test_results, best_model_path, best_val_mae = train(
        dataset_name=dataset_name,
        model_name=base_args.model,
        window=base_args.window,
        horizon=base_args.horizon,
        batch_size=batch_size,
        lr=lr,
        max_epochs=base_args.epochs,
        base_root=base_args.base_root,
        model_kwargs=model_kwargs,
        patience = patience
    )

    test_results_dict = test_results[0]
    metrics = {k: v for k, v in test_results_dict.items() if k.startswith('test_')}
    metrics['best_model_path'] = best_model_path
    metrics['best_val_mae'] = best_val_mae

    if best_val_mae is not None:
        objective = float(best_val_mae)
    elif metrics.get('test_mae') is not None:
        objective = float(metrics['test_mae'])
    else:
        objective = float('inf')

    return objective, metrics


def record_trial(results_log, out_path, search_start, phase, iteration,
                 config, objective, metrics, trial_start, error=None, diag=None):
    cfg = dict(config)
    lr = cfg.pop('lr')
    batch_size = cfg.pop('batch_size')
    trial_end = time.time()

    record = {
        'trial': len(results_log),
        'phase': phase,
        'iteration': iteration,
        'lr': lr,
        'batch_size': batch_size,
        'model_kwargs': cfg,
        'val_mae': objective if np.isfinite(objective) else None,
        'trial_duration_sec': trial_end - trial_start,
        'elapsed_since_start_sec': trial_end - search_start,
    }
    if diag:
        record.update(diag)

    if error is not None:
        record['error'] = error
    else:
        record['best_model_path'] = metrics.get('best_model_path') if metrics else None
        if metrics:
            for key, value in metrics.items():
                if key != 'best_model_path':
                    record[key] = value

    results_log.append(record)

    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as fp:
        json.dump(to_native(results_log), fp, indent=2)
    os.replace(tmp_path, out_path)

    try:
        results_to_excel(out_path)
    except Exception as e:
        print(f"Excel export failed: {e}", flush=True)

    return record


# Main BODE 
def bo_de(base_args, dataset_name, T, n_init, n_pop, k, f, p_c, results_dir,
          seed=42, xi=0.0, train_seed=0, resume_from=None, patience = 5):
    os.makedirs(results_dir, exist_ok=True)

    cs = get_search_space(base_args.model)
    bounds = cs_to_bounds(cs)
    D = len(bounds)

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from not found: {resume_from}")
        (X_obs, y_obs, results_log, n_init_done, n_iters_done,
         best_mae, best_config, best_model_path) = load_state(resume_from, bounds)
        out_path = resume_from
        print(f"=== Resuming from {os.path.basename(resume_from)} ===", flush=True)
        print(f"    {len(results_log)} trials reloaded "
              f"({n_init_done} init, {n_iters_done} BO-DE)", flush=True)
        print(f"    best val_mae so far: {best_mae:.4f}", flush=True)
        if n_init_done < n_init:
            print(f"    -> {n_init - n_init_done} init trials remaining, "
                  f"then {T} BO-DE iterations", flush=True)
        else:
            print(f"    -> {max(0, T - n_iters_done)} BO-DE iterations remaining",
                  flush=True)
        if n_init_done > n_init:
            print(f"    NOTE: log has {n_init_done} init trials but n_init={n_init}; "
                  "all are kept and fed to the GP", flush=True)
    else:
        X_obs, y_obs, results_log = [], [], []
        n_init_done = n_iters_done = 0
        best_mae, best_config, best_model_path = float('inf'), None, None
        timestamp = datetime.now().strftime('%Y%m%d')
        out_path = os.path.join(
            results_dir,
            f"{base_args.model}_{dataset_name}_BODE__{timestamp}.json")

    search_start = time.time()

    def update_best(objective, config, metrics):
        nonlocal best_mae, best_config, best_model_path
        if metrics and np.isfinite(objective) and objective < best_mae:
            best_mae, best_config = objective, config
            best_model_path = metrics.get('best_model_path')
            print(f"*** New best val_mae: {best_mae:.4f} ***", flush=True)
            if best_model_path:
                print(f"    checkpoint: {best_model_path}", flush=True)

    if n_init_done < n_init:
        design = initial_design(n_init, bounds, seed)
        print(f"=== Initialising: {n_init - n_init_done} random observations "
              f"(D={D}) ===", flush=True)
        for i in range(n_init_done, n_init):
            vec = quantize(design[i], bounds)
            config = vector_to_config(vec, bounds)
            print(f"\n--- Init {i + 1}/{n_init} ---", flush=True)
            print("Config:", config, flush=True)

            trial_start, error = time.time(), None
            try:
                objective, metrics = train_one_run(
                    base_args, config, dataset_name, seed=train_seed, patience=patience)
            except Exception as e:
                print(f"Init {i + 1} failed: {e}", flush=True)
                objective, metrics, error = float('inf'), None, str(e)

            X_obs.append(vec)
            y_obs.append(objective)
            record_trial(results_log, out_path, search_start, "init", None,
                         config, objective, metrics, trial_start, error=error)
            update_best(objective, config, metrics)
            print(f"Init {i + 1} objective (val_mae): {objective:.4f}", flush=True)
    else:
        print(f"=== Initial design already complete "
              f"({n_init_done}/{n_init}) ===", flush=True)

    remaining = T - n_iters_done
    if remaining <= 0:
        print(f"\n=== BO-DE already complete ({n_iters_done}/{T}) ===", flush=True)
    else:
        print(f"\n=== BO-DE: {remaining} iterations (kernel=rbf, acq=ei, "
              f"n_pop={n_pop}, k={k}, f={f}, p_c={p_c}) ===", flush=True)

    gp = build_gp(seed)

    for t in range(n_iters_done + 1, T + 1):
        print(f"\n--- BO-DE Iteration {t}/{T} ---", flush=True)

        fitted, y_best = fit_gp(gp, X_obs, y_obs)
        diag = {}

        if not fitted:
            print("  <2 finite observations, sampling at random", flush=True)
            vec = quantize(iter_rng(seed, 2, t).random(D), bounds)
        else:
            raw_vec, ei_val, conv_gen = de_maximise_ei(
                gp, y_best, bounds, iter_rng(seed, 1, t),
                n_pop=n_pop, k=k, f=f, p_c=p_c, xi=xi)
            vec = quantize(raw_vec, bounds)

            exploit, explore, mu, sigma = ei_terms(vec, gp, y_best, xi=xi)
            y_arr = np.asarray(y_obs, dtype=float)
            X_fin = np.asarray(X_obs)[np.isfinite(y_arr)]
            diag = {
                'de_converged_gen': int(conv_gen),
                'ei_at_selection': float(ei_val),
                'ei_exploit_term': float(exploit),
                'ei_explore_term': float(explore),
                'ei_explore_exploit_ratio': (
                    float(explore / exploit) if abs(exploit) > 1e-12 else None),
                'gp_mu_at_selection': float(mu),
                'gp_sigma_at_selection': float(sigma),
                'nn_distance': float(np.min(np.linalg.norm(X_fin - vec, axis=1))),
                'gp_kernel': str(gp.kernel_),
            }
            ratio = diag['ei_explore_exploit_ratio']
            print(f"  exploit={exploit:.4g}  explore={explore:.4g}  "
                  f"ratio={ratio if ratio is None else round(ratio, 2)}  "
                  f"nn_dist={diag['nn_distance']:.3f}", flush=True)
            print(f"  kernel: {gp.kernel_}", flush=True)

        config = vector_to_config(vec, bounds)
        print("Next config (from DE):", config, flush=True)

        trial_start, error = time.time(), None
        try:
            objective, metrics = train_one_run(
                base_args, config, dataset_name, seed=train_seed, patience = patience)
        except Exception as e:
            print(f"Iteration {t} failed: {e}", flush=True)
            objective, metrics, error = float('inf'), None, str(e)

        X_obs.append(vec)
        y_obs.append(objective)
        record_trial(results_log, out_path, search_start, "bode", t,
                     config, objective, metrics, trial_start,
                     error=error, diag=diag)
        update_best(objective, config, metrics)
        print(f"Iteration {t} objective (val_mae): {objective:.4f}", flush=True)

    return best_config, best_mae, best_model_path, results_log, out_path


def run_bode(model_name='graphwavenet', dataset_name='metrla',
             T=20, n_init=5, n_pop=10, k=20, f=0.8, p_c=0.9,
             window=12, horizon=12, max_epochs=10, base_root='./data',
             results_dir='./search_results', seed=42, xi=0.0, train_seed=0,
             resume_from=None, patience = 30):
    base_args = argparse.Namespace(
        model=model_name, window=window, horizon=horizon,
        epochs=max_epochs, base_root=base_root,
    )

    print("Model:", model_name, flush=True)
    print("Dataset:", dataset_name, flush=True)
    print(f"BO-DE: T={T}, n_init={n_init}, total budget={T + n_init}, "
          f"seed={seed}", flush=True)

    best_config, best_mae, best_model_path, results_log, out_path = bo_de(
        base_args, dataset_name, T=T, n_init=n_init, n_pop=n_pop, k=k,
        f=f, p_c=p_c, results_dir=results_dir, seed=seed, xi=xi,
        train_seed=train_seed, resume_from=resume_from, patience=patience
    )

    print("\n========== BO-DE Complete ==========", flush=True)
    print(f"Best Val MAE: {best_mae:.4f}", flush=True)
    print("Best Config:", best_config, flush=True)
    print("Best Checkpoint:", best_model_path, flush=True)
    print("Results saved to:", out_path, flush=True)
    print("Excel saved to:", out_path.replace('.json', '.xlsx'), flush=True)
    return results_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["metrla", "pemsbay", "electricity"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["graphwavenet", "stgcn", "agcrn"])

    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--n_pop", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--f", type=float, default=0.8)
    parser.add_argument("--p_c", type=float, default=0.9)

    parser.add_argument("--xi", type=float, default=0.0,
                        help="EI exploration offset; 0 matches the paper")

    parser.add_argument("--seed", type=int, default=42,
                        help="search seed: vary this across repeats")
    parser.add_argument("--train_seed", type=int, default=0,
                        help="training seed, held fixed across trials (CRN)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="path to a run's JSON; continues it and appends "
                             "to the same file")

    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./search_results")

    args = parser.parse_args()

    run_bode(
        model_name=args.model, dataset_name=args.dataset,
        T=args.T, n_init=args.n_init, n_pop=args.n_pop, k=args.k,
        f=args.f, p_c=args.p_c, xi=args.xi,
        seed=args.seed, train_seed=args.train_seed,
        resume_from=args.resume_from,
        window=args.window, horizon=args.horizon, max_epochs=args.epochs,
        base_root=args.base_root, results_dir=args.results_dir, patience=5
    )


if __name__ == "__main__":
    main()