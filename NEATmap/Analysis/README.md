# Analysis utilities

The analysis modules can be imported without loading the neural-network training
environment. The Python analysis dependencies are NumPy, pandas, SciPy and
tifffile. Run imports from the `NEATmap` directory, for example:

```python
from Analysis import Statistics

Statistics.ttest_level6(root, experimental_samples, control_samples, region_num=205)
```

## CSV and statistical conventions

- Group comparisons align rows by `Region` or `Region_name`, retaining the first
  experimental sample's order. Every sample must have the same unique region
  labels; missing and duplicate labels produce an error. Group sample counts may
  differ. Measurements retain floating-point precision.
- The existing independent, equal-variance t-test is retained. BH/FDR and
  Bonferroni corrections operate on finite p-values, with missing tests retained
  as NaN. `ttest_multi_layer(remove_nan=True)` conservatively substitutes 1 for
  missing tests before correction, retaining that option's original behavior.
- Significance classes use `<0.001`, `<0.01`, `<0.05`, and `>=0.05`.
  Exact threshold values retain their rows; non-finite results are marked `NA`.
  `DNN_result_ranksum` retains its separate historical marker convention.
- `log2change` keeps the original raw `pValue` column and adds a `qValue` column
  containing BH-adjusted values. Zero means produce NaN or signed infinity in
  the log ratio; no pseudocount is added.
- Hemisphere summaries group rows by acronym. Left/right exports retain the
  existing convention that the first of each region's two rows is left.
  Missing or negative cell counts are rejected rather than silently discarded.
- `cortex_stats` and `level2_count` retain their existing atlas row mappings.
  Input files must use the corresponding atlas order. The legacy SSp-bfd
  layer-six slice overlaps layer five; that scientific mapping is retained for
  review rather than changed as part of this optimization.

## Images

Atlas exporters process one input slice at a time instead of holding repeated
full atlas volumes. Region masks and point masks retain their uint16 0/255
representation. Missing slices in `concat_image` are zero-filled in the output
without creating or overwriting source files. Input slice shapes are validated.

Density maps normally retain uint16 TIFF output. If a count exceeds 65535, the
writer emits a warning and uses uint32 to preserve counts; downstream readers
must support that dtype. Internal uint32 count overflow raises an error.
Existing voxel scales, density constants, intensity thresholds and histogram
ranges are retained.

## R plots

The scripts require ggplot2. They accept input CSV and output PDF paths without
changing the working directory:

```text
Rscript NEATmap/Analysis/R_code/vocano_plot.R input.csv volcano.pdf
Rscript NEATmap/Analysis/R_code/heatmap_bupple.R correlations.csv correlations.pdf
```

The volcano plot uses `qValue` when available and otherwise labels the raw
`pValue` correctly. Correlation bubbles use the supplied significance values
and do not recompute a correction. Both scripts can also be sourced to call
their plotting functions directly.

## Validation

From the repository root with the analysis dependencies installed:

```text
python -m unittest discover -s NEATmap/Analysis/tests -p "test_*.py" -v
```

Tests use synthetic CSVs and 2-D/3-D TIFFs. They cover region reordering, unequal
group sizes, decimal measurements, significance boundaries, missing data,
image boundaries, density overflow, and numerical agreement with reference
calculations. Python 3.6 syntax is retained; the regression suite was executed
with Python 3.12 and current NumPy/pandas/SciPy/tifffile. The original pinned
environment and real whole-brain datasets require separate validation. The R
scripts have been statically checked but have not been executed in an R runtime.
