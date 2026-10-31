from __future__ import annotations

import numpy as np


def _correlation_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denominator = np.sqrt(
        np.sum(a_centered**2, axis=1) * np.sum(b_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _correlation_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _correlation_rows(a.T, b.T)



def protein_pearson_mean(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")
    return float(_correlation_columns(prediction, target).mean())


def _correlation_matrix(array: np.ndarray, row_variables: bool) -> np.ndarray:
    values = array if row_variables else array.T
    result = np.corrcoef(values)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_matrix_distance(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.sum(a * b))
    denominator = float(np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro"))
    return 1.0 - numerator / max(denominator, 1e-12)


def rbf_mmd(
    prediction: np.ndarray,
    target: np.ndarray,
    max_samples: int = 2000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    if prediction.shape[0] > max_samples:
        prediction = prediction[rng.choice(prediction.shape[0], max_samples, replace=False)]
    if target.shape[0] > max_samples:
        target = target[rng.choice(target.shape[0], max_samples, replace=False)]
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = (
            np.sum(a * a, axis=1, keepdims=True)
            + np.sum(b * b, axis=1, keepdims=True).T
            - 2.0 * (a @ b.T)
        )
        squared = np.maximum(squared, 0.0) / a.shape[1]
        return np.exp(-squared)

    xx = kernel(x, x)
    yy = kernel(y, y)
    xy = kernel(x, y)
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def evaluate(
    prediction: np.ndarray,
    target: np.ndarray,
    cell_matrix_limit: int = 5000,
    seed: int = 42,
) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    pcc_protein = _correlation_columns(prediction, target)
    pcc_cell = _correlation_rows(prediction, target)
    protein_corr_prediction = _correlation_matrix(prediction, row_variables=False)
    protein_corr_target = _correlation_matrix(target, row_variables=False)

    rng = np.random.default_rng(seed)
    if prediction.shape[0] > cell_matrix_limit:
        selected = rng.choice(prediction.shape[0], cell_matrix_limit, replace=False)
        prediction_cells = prediction[selected]
        target_cells = target[selected]
    else:
        prediction_cells = prediction
        target_cells = target
    cell_corr_prediction = _correlation_matrix(prediction_cells, row_variables=True)
    cell_corr_target = _correlation_matrix(target_cells, row_variables=True)

    return {
        "pcc_protein": pcc_protein.astype(np.float32),
        "pcc_cell": pcc_cell.astype(np.float32),
        "pcc_protein_mean": float(pcc_protein.mean()),
        "pcc_cell_mean": float(pcc_cell.mean()),
        "cmd_protein": correlation_matrix_distance(
            protein_corr_prediction, protein_corr_target
        ),
        "cmd_cell": correlation_matrix_distance(
            cell_corr_prediction, cell_corr_target
        ),
        "mmd": rbf_mmd(prediction, target, seed=seed),
        "mse": float(np.mean((prediction - target) ** 2)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }
