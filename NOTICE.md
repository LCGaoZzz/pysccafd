# Third-party notices

This package is licensed GPL-2.0-or-later because it combines code derived
from GPL and MIT upstreams:

- `pysccafd/sccafd.py` is a Python port of `SCCAF_D` and its companion files
  from [rnacentre/SCCAF-D](https://github.com/rnacentre/SCCAF-D)
  (commit `d60fabd5a2ac8fe95cb712ce730d01391210cfcd`, MIT, (c) 2024 RNA
  centre), with the audited compatibility semantics (configurable Harmony
  batch key, within-label top-100 cell selection, proportions-only fast path).
  The MIT-licensed portions remain available under MIT.
- `pysccafd/mast_de.py` and `pysccafd/mast_fast.py` re-implement the MAST-style
  hurdle differential-expression stage (bayesglm IRLS with prior-augmented
  rows, empirical-Bayes inverse-gamma variance prior, hurdle likelihood-ratio
  tests) following
  [RGLab/MAST](https://github.com/RGLab/MAST) 1.28.0 (GPL-2+, (c) Andrew
  McDavid, Greg Finak, Masanao Yajima) and its vendored arm::bayesglm.
- `pysccafd/tmm.py` ports `calcNormFactors(method="TMM")` and `cpm` from
  edgeR 4.0.16 ((c) Yunshun Chen, Aaron Lun, Davis McCarthy, Gordon Smyth et
  al., GPL-2+).
- The Harmony/PCA/SCCAF-assessment stage reuses the pinned upstream Python
  packages (SCCAF 0.0.10, MIT; harmonypy 0.0.9; scanpy) unchanged — they are
  runtime dependencies, not vendored.
- The DWLS solver (`solve_dampened_wls`) comes from
  [pydwls](https://github.com/LCGaoZzz/pydwls) (GPL-2.0-or-later, derived from
  sistia01/DWLS).
