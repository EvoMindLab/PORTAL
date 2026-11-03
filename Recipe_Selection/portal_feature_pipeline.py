"""
PORTAL Feature Preparation Pipeline
====================================
Author: Danial Yazdani / Claude Sonnet 4.6 — June 2026

Prepares cleaned, standardized ELA-feature and algorithm-performance matrices from raw
PORTAL data files for use in downstream curation (near-duplicate detection,
coverage-driven subset selection).

The ELA inputs are collected with the adapted pflacco implementation:
https://github.com/Reiyan/pflacco (MIT; see pflacco/LICENSE).
R. P. Prager and H. Trautmann (2024), Pflacco: Feature-Based Landscape Analysis
of Continuous and Constrained Optimization Problems in Python, Evolutionary
Computation 32(3), 211-216. https://doi.org/10.1162/evco_a_00341
The preparation and performance-aggregation workflow is PORTAL-owned code.

Pipeline stages
---------------
  Performance-1  Load raw Performance CSV
  Performance-2  Log-transform absolute errors: log(e + epsilon)
  Performance-3  Aggregate: per-(recipe, algorithm, checkpoint) median across runs
  Performance-4  Pivot to (n_recipes x 35) matrix [7 algorithms x 5 checkpoints]
  Performance-5  Robust standardization (median/MAD per coordinate)

  ELA-S1 Load raw ELA CSV
  ELA-S2 Drop costs_runtime columns (computational timing, not landscape features)
  ELA-S3 Drop near-constant features: IQR < iqr_floor OR MAD == 0
         (MAD==0 catches binary-valued features where >50% of recipes share the
          median value; these cannot be meaningfully standardized and contribute
          only a binary signal that distorts L2 distances.)
  ELA-S4 Impute sporadic missing values (column median)
  ELA-S4b Winsorize at [winsor_low, winsor_high] quantiles to cap sentinel values
          (pflacco can return sentinel values such as cond=1e6; winsorization
           prevents these from dominating pairwise distances after standardization.)
  ELA-S5 Greedy global Pearson correlation pruning (|r| >= corr_threshold)
         Retain the feature with larger IQR in each correlated pair
  ELA-S6 Robust standardization (median/MAD per feature)

  OUT    Save cleaned matrices, metadata, and feature lists to output directory

Usage
-----
  # Supply raw tables explicitly (run from the repository root):
  python Recipe_Selection/portal_feature_pipeline.py \\
      --performance_file path/to/performance_raw.csv --ela_file path/to/ela.csv

  # Specify custom paths:
  python Recipe_Selection/portal_feature_pipeline.py \\
      --performance_file path/to/performance_raw.csv \\
      --ela_file path/to/ela.csv \\
      --out_dir  path/to/output

  # Adjust thresholds:
  python Recipe_Selection/portal_feature_pipeline.py \\
      --performance_file path/to/performance_raw.csv --ela_file path/to/ela.csv \\
      --corr_threshold 0.90 --iqr_floor 1e-4

Outputs (written to out_dir)
------------------------------
  performance_cleaned.csv           Standardized Performance matrix, one row per recipe
  ela_cleaned.csv           Standardized ELA matrix, one row per recipe (retained features only)
  ela_retained_features.csv Feature-level metadata (group, IQR, drop reasons)
  ela_dropped_features.csv  Dropped features with reason and (if applicable) correlated partner
  pipeline_metadata.json    All pipeline parameters and summary statistics
  pipeline_report.txt       Human-readable log of every pipeline decision
"""

# Read-only compatibility for historical characterization artifacts.
try:
    from Recipe_Selection._terminology import read_characterization_csv
except ModuleNotFoundError as _terminology_error:
    if _terminology_error.name != 'Recipe_Selection':
        raise
    import sys as _source_sys
    from pathlib import Path as _SourcePath
    _source_sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[1]))
    from Recipe_Selection._terminology import read_characterization_csv

import os
import json
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    performance_file       = None,   # resolved relative to script dir if None
    ela_file       = None,
    out_dir        = None,
    log_eps        = 1e-10,  # floor for log(error + epsilon)
    iqr_floor      = 1e-6,   # features with IQR below this are near-constant
    corr_threshold = 0.95,   # Pearson |r| threshold for correlation pruning
    mad_floor      = 1e-8,   # floor for MAD in robust standardization
    winsor_low     = 0.01,   # lower winsorization quantile (P1)
    winsor_high    = 0.99,   # upper winsorization quantile (P99)
    algorithms     = ['CMA_ES', 'DE', 'LSHADE', 'PSO', 'CLPSO', 'PatternSearch', 'Powell'],
    checkpoints    = [30000, 50000, 100000, 200000, 300000],
)

_ELA_META_COLS = {
    'family', 'family_id', 'recipe_index', 'recipe_id',
    'sample', 'instance_type', 'd', 'json_path', 'ela_failed_groups',
}

_GROUP_PREFIXES = [
    'ela_conv', 'ela_disp', 'ela_distr', 'ela_fitness_distance',
    'ela_gradient', 'ela_ic', 'ela_level', 'ela_meta', 'ela_nbc', 'ela_pca',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feat_group(col: str) -> str:
    for g in _GROUP_PREFIXES:
        if col.startswith(g):
            return g
    return 'other'


def _robust_standardize(df: pd.DataFrame, mad_floor: float) -> pd.DataFrame:
    """Standardize columns by (x - median) / max(MAD, mad_floor)."""
    meds = df.median()
    mads = df.apply(lambda s: np.median(np.abs(s - s.median())))
    scales = mads.clip(lower=mad_floor)
    return (df - meds) / scales


def _resolve_path(path, default_relative, base_dir):
    if path is not None:
        return path
    return os.path.join(base_dir, default_relative)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PortalFeaturePipeline:
    """
    End-to-end PORTAL feature preparation pipeline.

    Parameters
    ----------
    performance_file, ela_file : str
        Paths to raw Performance and ELA CSV files.
    out_dir : str
        Directory where all outputs are written.
    log_eps : float
        Floor added before log-transforming absolute Performance errors.
    iqr_floor : float
        Features with global IQR below this are treated as near-constant.
    corr_threshold : float
        Pearson |r| threshold for greedy correlation pruning.
    mad_floor : float
        Minimum MAD used in robust standardization to avoid division by zero.
    algorithms : list[str]
        Ordered list of algorithm names expected in the Performance table.
    checkpoints : list[int]
        Ordered list of FE checkpoints expected in the Performance table.
    """

    def __init__(
        self,
        performance_file,
        ela_file,
        out_dir,
        log_eps        = _DEFAULTS['log_eps'],
        iqr_floor      = _DEFAULTS['iqr_floor'],
        corr_threshold = _DEFAULTS['corr_threshold'],
        mad_floor      = _DEFAULTS['mad_floor'],
        winsor_low     = _DEFAULTS['winsor_low'],
        winsor_high    = _DEFAULTS['winsor_high'],
        algorithms     = None,
        checkpoints    = None,
    ):
        self.performance_file       = performance_file
        self.ela_file       = ela_file
        self.out_dir        = out_dir
        self.log_eps        = log_eps
        self.iqr_floor      = iqr_floor
        self.corr_threshold = corr_threshold
        self.mad_floor      = mad_floor
        self.winsor_low     = winsor_low
        self.winsor_high    = winsor_high
        self.algorithms     = algorithms or _DEFAULTS['algorithms']
        self.checkpoints    = checkpoints or _DEFAULTS['checkpoints']

        os.makedirs(out_dir, exist_ok=True)

        self._log_lines = []
        self._drop_records = []   # {feature, reason, partner, r}
        self._retain_records = [] # {feature, group, global_iqr}

        # Outputs populated by run()
        self.performance_meta      = None   # DataFrame: recipe_id, family, family_id
        self.performance_raw_wide  = None   # (n, 35) unscaled log-error medians
        self.performance_scaled    = None   # (n, 35) standardized
        self.performance_cols      = None   # list of 35 column names

        self.ela_meta      = None   # DataFrame: recipe_id, family, family_id
        self.ela_raw_ret   = None   # (n, d_ELA) unscaled retained features
        self.ela_scaled    = None   # (n, d_ELA) standardized
        self.ela_feat_cols = None   # list of retained feature names

        self.summary = {}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg=''):
        print(msg)
        self._log_lines.append(str(msg))

    def _section(self, title):
        bar = '=' * 70
        self._log(''); self._log(bar)
        self._log(f'  {title}')
        self._log(bar)

    def _sub(self, title):
        self._log(f'\n--- {title} ---')

    # ------------------------------------------------------------------
    # Performance stages
    # ------------------------------------------------------------------

    def _run_performance(self):
        self._section('Performance pipeline')

        # Performance-1  Load
        self._log(f'  Performance-1  Loading {self.performance_file}')
        performance_raw = read_characterization_csv(self.performance_file)
        self._log(f'    rows          : {len(performance_raw):,}')
        self._log(f'    unique recipes: {performance_raw["recipe_id"].nunique()}')
        self._log(f'    algorithms    : {sorted(performance_raw["algorithm"].unique())}')
        self._log(f'    checkpoints   : {sorted(performance_raw["checkpoint_fe"].unique())}')
        self._log(f'    runs per cell : {performance_raw.groupby(["recipe_id","algorithm","checkpoint_fe"])["run_id"].count().unique().tolist()}')

        # Performance-2  Log-transform
        self._log(f'\n  Performance-2  Log-transform: log(absolute_error + {self.log_eps})')
        performance_raw['log_error'] = np.log(performance_raw['absolute_error'] + self.log_eps)

        # Performance-3  Per-(recipe, algorithm, checkpoint) median across runs
        self._log('\n  Performance-3  Aggregate: median log-error per (recipe, algorithm, checkpoint)')
        performance_med = (
            performance_raw
            .groupby(['recipe_id', 'family', 'family_id',
                      'algorithm', 'algorithm_id', 'checkpoint_fe'])
            ['log_error']
            .median()
            .reset_index()
        )

        # Performance-4  Pivot to wide form
        self._log('\n  Performance-4  Pivot to wide (n_recipes x 35) matrix')
        performance_med['coord'] = performance_med['algorithm'] + '@' + performance_med['checkpoint_fe'].astype(str)
        performance_wide = performance_med.pivot_table(
            index=['recipe_id', 'family', 'family_id'],
            columns='coord',
            values='log_error',
        ).reset_index()

        ordered_cols = [f'{a}@{c}' for a in self.algorithms for c in self.checkpoints]
        present_cols = [c for c in ordered_cols if c in performance_wide.columns]
        missing_cols = [c for c in ordered_cols if c not in performance_wide.columns]
        if missing_cols:
            self._log(f'  WARNING: Missing Performance coordinates: {missing_cols}')

        self.performance_meta     = performance_wide[['recipe_id', 'family', 'family_id']].copy().reset_index(drop=True)
        self.performance_raw_wide = performance_wide[present_cols].copy().reset_index(drop=True)
        self.performance_cols     = present_cols

        n_miss = self.performance_raw_wide.isnull().sum().sum()
        self._log(f'    shape         : {self.performance_raw_wide.shape}')
        self._log(f'    missing values: {n_miss}')

        # Performance-5  Robust standardization
        self._log('\n  Performance-5  Robust standardization (median / MAD)')
        self.performance_scaled = _robust_standardize(self.performance_raw_wide, self.mad_floor)
        self._log(f'    done. Output shape: {self.performance_scaled.shape}')

        self.summary['performance_n_recipes']    = len(self.performance_meta)
        self.summary['performance_n_coords']     = len(present_cols)
        self.summary['performance_missing_vals'] = int(n_miss)

    # ------------------------------------------------------------------
    # ELA stages
    # ------------------------------------------------------------------

    def _run_ela(self):
        self._section('ELA pipeline')

        # ELA-S1  Load
        self._log(f'  ELA-S1  Loading {self.ela_file}')
        ela_raw  = read_characterization_csv(self.ela_file)
        all_feat = [c for c in ela_raw.columns if c not in _ELA_META_COLS]
        self.ela_meta = ela_raw[['recipe_id', 'family', 'family_id']].copy().reset_index(drop=True)
        ela_F = ela_raw[all_feat].copy().astype(float)
        self._log(f'    recipes       : {len(ela_raw)}')
        self._log(f'    raw features  : {len(all_feat)}')

        # ELA-S2  Drop costs_runtime
        self._sub('ELA-S2  Drop costs_runtime columns')
        runtime_cols = [c for c in all_feat if c.endswith('costs_runtime')]
        self._log(f'  Dropped {len(runtime_cols)} costs_runtime features:')
        for c in runtime_cols:
            self._log(f'    {c}')
            self._drop_records.append({
                'feature': c, 'group': _feat_group(c),
                'reason': 'costs_runtime', 'partner': '', 'r': '',
            })
        feats_s2 = [c for c in all_feat if c not in runtime_cols]
        ela_s2   = ela_F[feats_s2].copy()

        # ELA-S3  Drop near-constant (IQR < floor OR MAD == 0)
        self._sub('ELA-S3  Drop near-constant features')
        iqr_s2 = ela_s2.quantile(0.75) - ela_s2.quantile(0.25)
        mad_s2 = ela_s2.apply(lambda s: np.median(np.abs(s - s.median())))

        nc_low_iqr  = [c for c in feats_s2 if iqr_s2[c] < self.iqr_floor]
        nc_zero_mad = [c for c in feats_s2
                       if c not in nc_low_iqr and mad_s2[c] == 0.0]
        nc_cols = nc_low_iqr + nc_zero_mad

        self._log(f'  Dropped {len(nc_low_iqr)} low-IQR features (IQR < {self.iqr_floor}):')
        for c in nc_low_iqr:
            self._log(f'    {c}  (IQR={iqr_s2[c]:.2e})')
            self._drop_records.append({
                'feature': c, 'group': _feat_group(c),
                'reason': 'near_constant_low_iqr', 'partner': '', 'r': '',
            })
        self._log(f'  Dropped {len(nc_zero_mad)} zero-MAD features (>50% recipes at single value):')
        for c in nc_zero_mad:
            mode_val = ela_s2[c].mode().iloc[0]
            mode_frac = (ela_s2[c] == mode_val).mean()
            n_unique = ela_s2[c].nunique()
            self._log(f'    {c}  (IQR={iqr_s2[c]:.2e}, MAD=0, '
                      f'mode_frac={mode_frac:.2%}, n_unique={n_unique})')
            self._drop_records.append({
                'feature': c, 'group': _feat_group(c),
                'reason': 'near_constant_zero_mad', 'partner': '', 'r': '',
            })

        feats_s3 = [c for c in feats_s2 if c not in nc_cols]
        ela_s3   = ela_s2[feats_s3].copy()

        # ELA-S4  Impute sporadic missing
        self._sub('ELA-S4  Impute sporadic missing values')
        n_miss_before = ela_s3.isnull().sum().sum()
        for col in feats_s3:
            if ela_s3[col].isnull().any():
                med = ela_s3[col].median()
                ela_s3[col] = ela_s3[col].fillna(med)
                self._log(f'    Imputed: {col} (median={med:.4f})')
        n_miss_after = ela_s3.isnull().sum().sum()
        self._log(f'  Missing before imputation: {n_miss_before}  after: {n_miss_after}')

        # ELA-S4b  Winsorize at [P{winsor_low*100}, P{winsor_high*100}]
        self._sub(f'ELA-S4b  Winsorize at [P{self.winsor_low*100:.0f}, P{self.winsor_high*100:.0f}]')
        self._log('  Clips sentinel/extreme values before standardization.')
        n_clipped_total = 0
        winsor_log = []
        for col in feats_s3:
            lo = ela_s3[col].quantile(self.winsor_low)
            hi = ela_s3[col].quantile(self.winsor_high)
            n_lo = (ela_s3[col] < lo).sum()
            n_hi = (ela_s3[col] > hi).sum()
            if n_lo > 0 or n_hi > 0:
                ela_s3[col] = ela_s3[col].clip(lower=lo, upper=hi)
                winsor_log.append((col, lo, hi, n_lo, n_hi))
                n_clipped_total += n_lo + n_hi
        if winsor_log:
            self._log(f'  Features with clipped values ({len(winsor_log)}):')
            for col, lo, hi, n_lo, n_hi in winsor_log:
                self._log(f'    {col}: [{lo:.4g}, {hi:.4g}]  '
                          f'clipped_low={n_lo}, clipped_high={n_hi}')
        else:
            self._log('  No values clipped (no outliers beyond winsorization bounds).')
        self._log(f'  Total cell values clipped: {n_clipped_total}')

        # ELA-S5  Greedy global correlation pruning
        self._sub(f'ELA-S5  Greedy global Pearson correlation pruning (|r| >= {self.corr_threshold})')
        iqr_s3   = ela_s3.quantile(0.75) - ela_s3.quantile(0.25)
        corr_mat = ela_s3.corr(method='pearson')

        upper = corr_mat.where(
            np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        )
        pairs = [
            (fa, fb, float(upper.loc[fa, fb]))
            for fa in upper.index for fb in upper.columns
            if pd.notna(upper.loc[fa, fb]) and abs(upper.loc[fa, fb]) >= self.corr_threshold
        ]
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
        self._log(f'  High-correlation pairs found: {len(pairs_sorted)}')

        to_drop = set()
        for fa, fb, r in pairs_sorted:
            if fa in to_drop or fb in to_drop:
                continue
            # Drop the feature with smaller IQR; tie-break by name
            if iqr_s3[fa] < iqr_s3[fb] or (iqr_s3[fa] == iqr_s3[fb] and fa > fb):
                dropped, kept = fa, fb
            else:
                dropped, kept = fb, fa
            to_drop.add(dropped)
            self._drop_records.append({
                'feature': dropped, 'group': _feat_group(dropped),
                'reason': 'correlation_pruning',
                'partner': kept, 'r': f'{r:.4f}',
            })

        feats_retained = [c for c in feats_s3 if c not in to_drop]
        ela_retained   = ela_s3[feats_retained].copy()
        self._log(f'  Features dropped by pruning : {len(to_drop)}')
        self._log(f'  Features retained           : {len(feats_retained)}')

        # Verify: no residual pairs at threshold
        corr_ret = ela_retained.corr(method='pearson')
        up_ret   = corr_ret.where(np.triu(np.ones(corr_ret.shape), k=1).astype(bool))
        residual = [
            (fa, fb, float(up_ret.loc[fa, fb]))
            for fa in up_ret.index for fb in up_ret.columns
            if pd.notna(up_ret.loc[fa, fb]) and abs(up_ret.loc[fa, fb]) >= self.corr_threshold
        ]
        self._log(f'  Residual pairs >= {self.corr_threshold} in retained set: {len(residual)}')
        if residual:
            for fa, fb, r in residual:
                self._log(f'    WARN: {fa}  vs  {fb}  r={r:.4f}')

        # ELA-S6  Robust standardization
        self._sub('ELA-S6  Robust standardization (median / MAD)')
        self.ela_feat_cols = feats_retained
        self.ela_raw_ret   = ela_retained.reset_index(drop=True)
        self.ela_scaled    = _robust_standardize(ela_retained, self.mad_floor).reset_index(drop=True)
        self._log(f'    done. Output shape: {self.ela_scaled.shape}')

        # Build retain records
        iqr_ret = ela_retained.quantile(0.75) - ela_retained.quantile(0.25)
        for c in feats_retained:
            self._retain_records.append({
                'feature': c, 'group': _feat_group(c),
                'global_iqr': float(iqr_ret[c]),
            })

        self.summary['ela_raw_features']       = len(all_feat)
        self.summary['ela_drop_runtime']       = len(runtime_cols)
        self.summary['ela_drop_low_iqr']       = len(nc_low_iqr)
        self.summary['ela_drop_zero_mad']      = len(nc_zero_mad)
        self.summary['ela_drop_constant']      = len(nc_cols)
        self.summary['ela_winsor_features']    = len(winsor_log)
        self.summary['ela_winsor_cells']       = int(n_clipped_total)
        self.summary['ela_drop_corr']          = len(to_drop)
        self.summary['ela_retained']           = len(feats_retained)
        self.summary['ela_residual_pairs']     = len(residual)
        self.summary['ela_corr_threshold']     = self.corr_threshold
        self.summary['ela_iqr_floor']          = self.iqr_floor
        self.summary['ela_winsor_low']         = self.winsor_low
        self.summary['ela_winsor_high']        = self.winsor_high

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def _save_outputs(self):
        self._section('Saving outputs')

        # Performance cleaned
        performance_out = pd.concat([self.performance_meta, self.performance_scaled], axis=1)
        performance_path = os.path.join(self.out_dir, 'performance_cleaned.csv')
        performance_out.to_csv(performance_path, index=False)
        self._log(f'  [saved] performance_cleaned.csv  ({performance_out.shape})')

        # Performance raw (unscaled, for reference)
        performance_raw_out = pd.concat([self.performance_meta, self.performance_raw_wide], axis=1)
        performance_raw_path = os.path.join(self.out_dir, 'performance_logmedian_unscaled.csv')
        performance_raw_out.to_csv(performance_raw_path, index=False)
        self._log('  [saved] performance_logmedian_unscaled.csv')

        # ELA cleaned
        ela_out = pd.concat([self.ela_meta, self.ela_scaled], axis=1)
        ela_path = os.path.join(self.out_dir, 'ela_cleaned.csv')
        ela_out.to_csv(ela_path, index=False)
        self._log(f'  [saved] ela_cleaned.csv  ({ela_out.shape})')

        # ELA raw retained (unscaled)
        ela_raw_out = pd.concat([self.ela_meta, self.ela_raw_ret], axis=1)
        ela_raw_path = os.path.join(self.out_dir, 'ela_retained_unscaled.csv')
        ela_raw_out.to_csv(ela_raw_path, index=False)
        self._log('  [saved] ela_retained_unscaled.csv')

        # Retained feature list
        ret_df = pd.DataFrame(self._retain_records)
        ret_df.to_csv(os.path.join(self.out_dir, 'ela_retained_features.csv'), index=False)
        self._log(f'  [saved] ela_retained_features.csv  ({len(ret_df)} features)')

        # Dropped feature list
        drop_df = pd.DataFrame(self._drop_records)
        drop_df.to_csv(os.path.join(self.out_dir, 'ela_dropped_features.csv'), index=False)
        self._log(f'  [saved] ela_dropped_features.csv  ({len(drop_df)} features)')

        # Pipeline metadata JSON
        meta = {
            'performance_file'       : self.performance_file,
            'ela_file'       : self.ela_file,
            'out_dir'        : self.out_dir,
            'log_eps'        : self.log_eps,
            'iqr_floor'      : self.iqr_floor,
            'corr_threshold' : self.corr_threshold,
            'mad_floor'      : self.mad_floor,
            'algorithms'     : self.algorithms,
            'checkpoints'    : self.checkpoints,
            'summary'        : self.summary,
            'ela_retained_features': self.ela_feat_cols,
            'performance_coords'     : self.performance_cols,
        }
        meta_path = os.path.join(self.out_dir, 'pipeline_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        self._log('  [saved] pipeline_metadata.json')

        # Text report
        report_path = os.path.join(self.out_dir, 'pipeline_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self._log_lines))
        self._log('  [saved] pipeline_report.txt')

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full pipeline and save all outputs."""
        self._section('PORTAL Feature Preparation Pipeline')
        self._log(f'  Performance file : {self.performance_file}')
        self._log(f'  ELA file : {self.ela_file}')
        self._log(f'  Out dir  : {self.out_dir}')
        self._log(f'  log_eps  = {self.log_eps}')
        self._log(f'  iqr_floor = {self.iqr_floor}')
        self._log(f'  corr_threshold = {self.corr_threshold}')
        self._log(f'  mad_floor = {self.mad_floor}')

        self._run_performance()
        self._run_ela()
        self._save_outputs()
        self._print_summary()

        return self

    def _print_summary(self):
        self._section('PIPELINE SUMMARY')
        s = self.summary
        self._log(f"""
  Performance
  ---
  Recipes          : {s['performance_n_recipes']}
  Coordinates      : {s['performance_n_coords']}  (algorithms x checkpoints)
  Missing values   : {s['performance_missing_vals']}
  Transform        : log(e + {self.log_eps})
  Standardization  : median / MAD (floor = {self.mad_floor})

  ELA
  ---
  Raw features     : {s['ela_raw_features']}
  Dropped runtime  : {s['ela_drop_runtime']}
  Dropped low-IQR  : {s['ela_drop_low_iqr']}  (IQR < {s['ela_iqr_floor']})
  Dropped zero-MAD : {s['ela_drop_zero_mad']}  (>50% values at median, MAD=0)
  Winsorized feats : {s['ela_winsor_features']}  cells clipped: {s['ela_winsor_cells']}  (P{s['ela_winsor_low']*100:.0f}/P{s['ela_winsor_high']*100:.0f})
  Dropped corr     : {s['ela_drop_corr']}  (|r| >= {s['ela_corr_threshold']})
  Retained         : {s['ela_retained']}
  Residual pairs   : {s['ela_residual_pairs']}  (should be 0)
  Standardization  : median / MAD (floor = {self.mad_floor})
""")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description='PORTAL Feature Preparation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--performance_file', '--performance-file', '--isa_file', dest='performance_file', required=True,
                   help='Path to raw algorithm-performance CSV; supplied automatically by run_suite_pipeline.py.')
    p.add_argument('--ela_file',       required=True,
                   help='Path to raw ELA CSV; supplied automatically by run_suite_pipeline.py.')
    p.add_argument('--out_dir',        default=None,
                   help='Output directory. Defaults to pipeline_output/ at the repository root.')
    p.add_argument('--log_eps',        type=float, default=_DEFAULTS['log_eps'])
    p.add_argument('--iqr_floor',      type=float, default=_DEFAULTS['iqr_floor'])
    p.add_argument('--corr_threshold', type=float, default=_DEFAULTS['corr_threshold'])
    p.add_argument('--mad_floor',      type=float, default=_DEFAULTS['mad_floor'])
    p.add_argument('--winsor_low',     type=float, default=_DEFAULTS['winsor_low'])
    p.add_argument('--winsor_high',    type=float, default=_DEFAULTS['winsor_high'])
    p.add_argument(
        '--algorithms',
        default=','.join(_DEFAULTS['algorithms']),
        help='Comma-separated Performance algorithm names expected in the raw table.',
    )
    p.add_argument(
        '--checkpoints',
        default=','.join(str(value) for value in _DEFAULTS['checkpoints']),
        help='Comma-separated FE checkpoints expected in the raw Performance table.',
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)  # project root (one level up from Recipe_Selection/)
    performance_file = args.performance_file
    ela_file = args.ela_file
    for label, path in (('--performance_file', performance_file), ('--ela_file', ela_file)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'{label} input does not exist: {path}. '
                'Pass your own raw measurement CSVs explicitly, or use '
                'run_suite_pipeline.py to generate them. Historical research '
                'measurements are not bundled with the public release.'
            )
    out_dir  = args.out_dir  or os.path.join(root_dir, 'pipeline_output')
    algorithms = [value.strip() for value in args.algorithms.split(',') if value.strip()]
    checkpoints = [int(value.strip()) for value in args.checkpoints.split(',') if value.strip()]
    if not algorithms or len(algorithms) != len(set(algorithms)):
        raise ValueError('algorithms must be a non-empty comma-separated list without duplicates.')
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        raise ValueError('checkpoints must be unique and strictly increasing.')

    pipeline = PortalFeaturePipeline(
        performance_file       = performance_file,
        ela_file       = ela_file,
        out_dir        = out_dir,
        log_eps        = args.log_eps,
        iqr_floor      = args.iqr_floor,
        corr_threshold = args.corr_threshold,
        mad_floor      = args.mad_floor,
        winsor_low     = args.winsor_low,
        winsor_high    = args.winsor_high,
        algorithms     = algorithms,
        checkpoints    = checkpoints,
    )
    pipeline.run()
    return pipeline


if __name__ == '__main__':
    main()
