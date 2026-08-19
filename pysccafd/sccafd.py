"""Native Python port of the official SCCAF-D deconvolution chain.

Implements ``SCCAF_D()`` from rnacentre/SCCAF-D @d60fabd with the packaged
compatibility semantics (configurable Harmony batch key, within-label top-100
cell ranking, proportions-only fast path):

* the scanpy/Harmony stage uses public APIs and the small SCCAF self-projection
  routine is reproduced locally, avoiding SCCAF's obsolete Scanpy pin while
  preserving its stochastic split and logistic-regression behaviour;
* TMM scaling, the MAST-style signature builder and the SCCAF-D DWLS variant
  are native ports (see ``tmm.py``, ``mast_de.py``, ``dwls.py``);
* assembly semantics (row order = first-appearance cell types then
  ``gtools::mixedsort``, round-to-3-digits long format, column normalisation)
  follow the official Deconvolution/adapter code.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pydwls import solve_dampened_wls
from .mast_de import build_signature_matrix_mast
from .tmm import scaling_tmm

try:  # pragma: no cover - import cost only paid by the sccaf_d path
    import anndata as ad
    import scanpy as sc
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "sccaf_d requires scanpy, harmonypy, anndata and scikit-learn in the runtime "
        "python environment"
    ) from exc


def _sample_group(frame: pd.DataFrame, *, n: int, frac: float) -> pd.DataFrame:
    """SCCAF ``msample`` without depending on the legacy SCCAF package."""
    if len(frame) <= np.floor(n / frac):
        return frame.sample(frac=0.9 if len(frame) < 10 else frac)
    return frame.sample(n=n)


def _sccaf_assessment(X, labels: pd.Series, *, n: int = 500):
    """The SCCAF 0.0.10 logistic-regression self-projection path."""
    label_frame = pd.DataFrame({"class": labels.to_numpy()}, index=np.arange(len(labels)))
    train_indices: list[int] = []
    for _, group in label_frame.groupby("class", sort=True):
        train_indices.extend(_sample_group(group, n=n, frac=0.5).index.tolist())
    train_indices_array = np.asarray(train_indices, dtype=int)
    test_indices = np.flatnonzero(~np.isin(np.arange(len(labels)), train_indices_array))
    X_train, X_test = X[train_indices_array, :], X[test_indices, :]
    y_train, y_test = labels.iloc[train_indices_array], labels.iloc[test_indices]

    classifier_args = {
        "random_state": 1,
        "penalty": "l1",
        "C": 0.5,
        "solver": "liblinear",
    }
    try:
        classifier = LogisticRegression(multi_class="ovr", **classifier_args)
    except TypeError:  # scikit-learn versions where multi_class was removed
        classifier = LogisticRegression(**classifier_args)
    cv_mean = float(cross_val_score(
        classifier, X_train, np.asarray(y_train), cv=5, scoring="accuracy"
    ).mean())
    classifier.fit(X_train, y_train)
    accuracy = float(classifier.score(X_test, y_test))
    return classifier.predict_proba(X_test), classifier.predict(X_test), y_test, classifier, cv_mean, accuracy


def scanpy_workflow(adata, *, label_key: str, batch_key: str, span: float = 0.3):
    """Patched official scanpy_workflow.py, executed in-process.

    Capability checks keep this path compatible with supported Scanpy releases
    without locking the shared Omicos kernel to one version.
    """
    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=2000,
        batch_key=batch_key,
        subset=False,
        span=span,
    )
    try:
        sc.tl.pca(adata, svd_solver="arpack", mask_var="highly_variable")
    except TypeError:  # Scanpy <1.10
        sc.tl.pca(adata, svd_solver="arpack", use_highly_variable=True)
    assessment_basis = "X_pca"
    if adata.obs[batch_key].nunique() > 1:
        sc.external.pp.harmony_integrate(
            adata, key=batch_key, basis="X_pca",
            adjusted_basis="X_pca_harmony", max_iter_harmony=50,
        )
        assessment_basis = "X_pca_harmony"
    y_prob, y_pred, y_test, clf, cvsm, acc = _sccaf_assessment(
        adata.obsm[assessment_basis], adata.obs[label_key], n=500
    )
    return y_prob, y_pred, y_test, clf


def selection_cells(y_prob, y_pred, y_test, clf) -> list[str]:
    """Patched official selection_cells: within-label top-100 by probability."""
    prob = pd.DataFrame(np.asarray(y_prob), columns=clf.classes_)
    pred = pd.DataFrame(np.asarray(y_pred), columns=["predict"])
    sccaf = pd.concat([pred, prob], axis=1)
    sccaf.insert(0, "cellType", np.asarray(y_test))
    if hasattr(y_test, "index"):
        sccaf.index = y_test.index
    else:
        sccaf.index = np.arange(len(y_test))
    sccaf.index = sccaf.index.astype(str)
    sccaf = sccaf[sccaf["cellType"].astype(str) == sccaf["predict"].astype(str)]
    selected: list[str] = []
    for class_name in [c for c in sccaf.columns if c not in ("cellType", "predict")]:
        class_cells = sccaf[sccaf["cellType"] == class_name]
        class_cells = class_cells.sort_values(class_name, ascending=False, kind="stable")
        selected.extend(class_cells.index[:100].astype(str).tolist())
    return selected


def _r_intersect_order(a, b) -> list[str]:
    set_b = set(b)
    seen: set[str] = set()
    out: list[str] = []
    for item in a:
        if item in set_b and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _mixedsort_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def run_sccaf_d(reference_h5ad: str, bulk: pd.DataFrame, reference_counts=None,
                reference_genes=None, reference_labels=None, reference_batches=None,
                label_key: str = "cellType", batch_key: str = "sampleID",
                layer: str | None = None, span: float = 0.3,
                stage_times: dict | None = None) -> pd.DataFrame:
    """Full SCCAF-D proportions chain; returns the 4 x N cell-type x sample frame.

    Either reads the reference H5AD itself (default) or uses the supplied
    in-memory counts (genes x cells) for testing against stored artifacts.
    """
    import time as _time

    times = stage_times if stage_times is not None else {}

    t0 = _time.perf_counter()
    adata = ad.read_h5ad(reference_h5ad)
    missing = [key for key in (label_key, batch_key) if key not in adata.obs]
    if missing:
        raise ValueError(f"reference metadata keys are missing: {missing}")
    if layer and layer not in adata.layers:
        raise ValueError(f"reference layer is missing: {layer}")
    source_matrix = adata.layers[layer] if layer else adata.X
    values = source_matrix.data if hasattr(source_matrix, "toarray") else np.asarray(source_matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("reference counts must be finite and non-negative")
    reference_cells = [str(c) for c in adata.obs_names]
    assessment = ad.AnnData(
        X=source_matrix.copy(),
        obs=adata.obs[[label_key, batch_key]].copy(),
        var=adata.var.copy(),
    )
    times["load_reference_s"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    y_prob, y_pred, y_test, clf = scanpy_workflow(
        assessment, label_key=label_key, batch_key=batch_key, span=span
    )
    times["scanpy_assessment_s"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    selected = selection_cells(y_prob, y_pred, y_test, clf)
    if not selected:
        raise ValueError("SCCAF assessment selected no cells")
    cell_pos = {c: i for i, c in enumerate(reference_cells)}
    missing = [c for c in selected if c not in cell_pos]
    if missing:
        raise ValueError(f"SCCAF selection returned unknown cells: {missing[:5]}")
    sel_idx = np.asarray([cell_pos[c] for c in selected])
    if reference_counts is None:
        selected_matrix = source_matrix[sel_idx, :]
        selected_matrix = selected_matrix.toarray() if hasattr(selected_matrix, "toarray") else np.asarray(selected_matrix)
        counts_sel = np.asarray(selected_matrix, dtype=float).T
        reference_genes = [str(g) for g in adata.var_names]
        labels_sel = adata.obs[label_key].astype(str).to_numpy()[sel_idx]
    else:
        counts_sel = reference_counts[:, sel_idx]
        labels_sel = np.asarray([reference_labels[i] for i in sel_idx])
    times["selection_s"] = _time.perf_counter() - t0

    # Gene universe: intersect(rownames(X1), rownames(X2)) keeps the bulk order.
    t0 = _time.perf_counter()
    to_keep = _r_intersect_order(list(bulk.index), reference_genes)
    bulk_pos = {g: i for i, g in enumerate(bulk.index)}
    T = bulk.loc[to_keep]
    ref_pos = {g: i for i, g in enumerate(reference_genes)}
    keep_idx = np.asarray([ref_pos[g] for g in to_keep])
    C = pd.DataFrame(counts_sel[keep_idx, :], index=to_keep,
                     columns=[f"c{i}" for i in range(counts_sel.shape[1])])
    C.columns = labels_sel.astype(str)                          # read_data semantics

    T_scaled = scaling_tmm(T)
    C_scaled = scaling_tmm(C)
    times["tmm_s"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    signature = build_signature_matrix_mast(C_scaled, list(C_scaled.columns))
    times["mast_signature_s"] = _time.perf_counter() - t0

    # Deconvolution: keep = intersect(rownames(C), rownames(T)) -> C row order
    t0 = _time.perf_counter()
    keep2 = _r_intersect_order(list(C_scaled.index), list(T_scaled.index))
    T_final = T_scaled.loc[keep2]

    type_order = list(dict.fromkeys(C_scaled.columns.tolist()))   # unique(id)
    results = {}
    for sample in T_final.columns:
        b = T_final[sample]
        genes_sig = _r_intersect_order(list(signature.index), list(b.index))
        sig = signature.loc[genes_sig].to_numpy(dtype=float)
        y = b.loc[genes_sig].to_numpy(dtype=float)
        sol = solve_dampened_wls(sig, y, variant="sccafd")
        results[sample] = {t: float(v) for t, v in zip(signature.columns, sol)}

    frame = pd.DataFrame(results, index=type_order)[list(T_final.columns)]
    frame = frame.clip(lower=0.0)
    frame = frame / frame.sum(axis=0)
    frame = frame.sort_index(key=lambda idx: pd.Index([_mixedsort_key(x) for x in idx]))
    # official Deconvolution rounds long-format values to 3 digits
    frame = frame.round(3)
    frame = frame / frame.sum(axis=0)
    times["dwls_assembly_s"] = _time.perf_counter() - t0
    return frame
