# pysccafd

Native Python port of the **SCCAF-D** deconvolution chain:
TMM normalisation -> Harmony -> SCCAF self-projection assessment ->
MAST-style hurdle signature -> DWLS.

Ported from [rnacentre/SCCAF-D](https://github.com/rnacentre/SCCAF-D) @
`d60fabd5a2ac8fe95cb712ce730d01391210cfcd` (MIT) with the audited
compatibility semantics (configurable Harmony batch key, within-label top-100
cell selection, proportions-only fast path). The hurdle signature stage
follows [RGLab/MAST](https://github.com/RGLab/MAST) 1.28.0 (bayesglm IRLS
with prior-augmented rows, empirical-Bayes inverse-gamma variance prior,
hurdle likelihood-ratio tests) and TMM follows edgeR 4.0.16; both are
GPL-2+ derivations, hence this package's license. The scanpy / harmony /
SCCAF assessment stage reuses the pinned upstream Python packages unchanged
(the same versions the R chain drove through reticulate). The DWLS solver
comes from [pydwls](https://github.com/LCGaoZzz/pydwls).

## Install

```bash
pip install git+https://github.com/LCGaoZzz/pydwls.git
pip install git+https://github.com/LCGaoZzz/pysccafd.git
# needs the pinned stack: scanpy 1.9.x, harmonypy 0.0.9, SCCAF 0.0.10, anndata
```

## Usage

```python
import pandas as pd
from pysccafd import run_sccaf_d

bulk = pd.read_csv("bulk.tsv", sep="\t", index_col=0)   # genes x samples, counts
proportions = run_sccaf_d(
    "reference.h5ad",    # raw counts with cellType + sampleID in .obs
    bulk,
    batch_key="sampleID",
    span=0.3,
)
# proportions: cell types x samples, column-normalised
```

## Evidence

On the 14-sample / 4-cell-type PBMC gold standard:

- TMM scaling matches edgeR to **5e-11** (both the bulk and reference legs),
  including the quantile/lib.size reference-column rule;
- MAST-style signature: 212 genes (the R run's count), 207 common, common-gene
  values within **4.5e-12**; the vectorised batch solver agrees with the
  reference-loop implementation on 99.9% of threshold calls;
- end-to-end accuracy vs mixture truth: MAE 0.1121 / RMSE 0.1484 / r 0.7427 /
  max 0.3225 against the R chain's 0.1106 / 0.1491 / 0.7500 / 0.3306 —
  RMSE and max error improve, MAE is +1.4% relative;
- runtime: ~51 s vs 197.8 s for the R chain (3.9x).

## Known deviations (documented, not hidden)

1. The hurdle p-values are a faithful re-implementation of MAST's compiled
   IRLS, not bit-identical; the visible consequence is ~5 of 212 signature
   genes flipping on the padj=0.01 razor edge (hundreds of genes sit within
   +/-0.0002 of the threshold). This drives the small end-to-end deltas above.
2. The upstream SCCAF assessment train/test split is unseeded in the upstream
   package; this port preserves that behaviour exactly (do not assume
   bit-reproducibility across environments for the cell selection).

## License

GPL-2.0-or-later (combined work). Per-file attributions in `NOTICE.md`.
