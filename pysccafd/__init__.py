"""pysccafd: native Python port of the SCCAF-D deconvolution chain.

Ports SCCAF_D (rnacentre/SCCAF-D @d60fabd, MIT) with the MAST-style hurdle
signature stage (derived from RGLab/MAST, GPL-2+) and edgeR's TMM
normalisation (GPL-2+); the scanpy/harmony/SCCAF assessment stage reuses the
pinned upstream Python packages unchanged. The DWLS solver comes from the
companion package pydwls.

License: GPL-2.0-or-later; see NOTICE.md for per-file attributions.
"""

from .mast_de import build_signature_matrix_mast
from .mast_fast import hurdle_lr_pvalues_batch
from .tmm import calc_norm_factors_tmm, scaling_tmm

__version__ = "0.1.0"


def __getattr__(name):
    if name == "run_sccaf_d":
        from .sccafd import run_sccaf_d
        return run_sccaf_d
    if name == "scanpy_workflow":
        from .sccafd import scanpy_workflow
        return scanpy_workflow
    if name == "selection_cells":
        from .sccafd import selection_cells
        return selection_cells
    raise AttributeError(name)


__all__ = [
    "build_signature_matrix_mast",
    "hurdle_lr_pvalues_batch",
    "calc_norm_factors_tmm",
    "scaling_tmm",
    "run_sccaf_d",
    "__version__",
]
