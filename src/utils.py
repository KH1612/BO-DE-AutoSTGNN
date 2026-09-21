import json
import numpy as np
import pandas as pd


def to_native(value):
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def results_to_excel(json_path, excel_path=None):
    """Convert a random_search results JSON file into a flat Excel spreadsheet.
    Generically flattens any nested dict fields (model_kwargs, horizon_mae, etc.)
    into their own columns, and includes every top-level scalar field present.
    """
    with open(json_path, 'r') as f:
        results_log = json.load(f)

    rows = []
    for r in results_log:
        row = {}
        for key, value in r.items():
            if isinstance(value, dict):
                # flatten nested dicts (model_kwargs, horizon_mae, test_results, etc.)
                prefix = 'kw_' if key == 'model_kwargs' else ''
                for sub_key, sub_value in value.items():
                    col_name = f'{prefix}{sub_key}' if prefix else sub_key
                    row[col_name] = sub_value
            else:
                row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    if excel_path is None:
        excel_path = json_path.replace('.json', '.xlsx')

    df.to_excel(excel_path, index=False)
    print(f"Saved: {excel_path}")
    return df