"""Part II supplement, Section I: derived reporting, never objective rescaling."""
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
from portal_studio.experiment_runner.convergence import read_aligned_moments, standard_error

METRIC_CHOICES = [('Raw error', 'final_error'),
                  ('Normalised error', 'normalised_error'),
                  ('Reference-scale error (%)', 'percentage_error')]


def reference_scale(instance):
    """Return (S_ref, reason); unavailable calibration never changes raw runs."""
    try:
        count = int(instance['K'])
        p = instance['P']
        beta = np.asarray(p['beta'], dtype=float).reshape(count)
        delta = np.asarray(p['Delta'], dtype=float).reshape(count)
        enabled = np.asarray(p['neutralize_enable'], dtype=bool).reshape(count)
        if not np.all(enabled):
            return None, 'Neutralisation was disabled for one or more components in this run.'
        if (not np.all(np.isfinite(beta))
                or not np.all(np.isfinite(delta)) or np.any(delta <= 0)):
            return None, 'Requires enabled neutralisation and positive reference rises for every component.'
        blocks = instance.get('_schema_blocks') or [dict(K=count, weight=1.)]
        total, start = 0., 0
        for block in blocks:
            end = start + int(block['K'])
            weight = float(block.get('weight', 1.))
            if end <= start or end > count or not np.isfinite(weight) or weight <= 0:
                raise ValueError('Invalid block scale.')
            # Subtract the common level first to avoid cancellation after shifts.
            total += weight * float(np.max((beta[start:end] - np.min(beta[start:end]))
                                          + delta[start:end]))
            start = end
        if start != count or not np.isfinite(total) or total <= 0:
            raise ValueError('Invalid reference scale.')
        return total, None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, 'The evaluated instance has no verified reference scale.'


def factor(metric):
    if metric not in dict((key, label) for label, key in METRIC_CHOICES):
        raise ValueError(f'Unknown error metric: {metric}')
    return 100. if metric == 'percentage_error' else 1.


def read_normalised(bundle, uid, metric='normalised_error', *, trend=False):
    """Read normalised results using recorded per-run reference scales only."""
    if metric == 'final_error':
        raise ValueError('Use the raw-result loader for raw errors.')
    multiplier = factor(metric)
    path = (Path(bundle) / 'summary.sqlite').resolve()
    if not path.is_file():
        raise ValueError('Result database is missing.')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as cx:
        columns = {r[1] for r in cx.execute('PRAGMA table_info(runs)')}
        if 'reference_scale' not in columns:
            raise ValueError('Normalised error is unavailable for these older results; rerun to record the evaluated scale.')
        rows = cx.execute('SELECT run_uid, final_error, reference_scale FROM runs '
                          'WHERE runset_uid=? ORDER BY run_index', (uid,)).fetchall()
        if not rows or any(r[1] is None or r[2] is None or not np.isfinite(r[1])
                           or r[1] < 0 or not np.isfinite(r[2]) or r[2] <= 0 for r in rows):
            if 'normalisation_reason' in columns:
                reason = cx.execute(
                    'SELECT normalisation_reason FROM runs WHERE runset_uid=? '
                    'AND reference_scale IS NULL AND normalisation_reason IS NOT NULL LIMIT 1',
                    (uid,),
                ).fetchone()
                if reason and reason[0]:
                    raise ValueError(f'Normalised error is unavailable: {reason[0]}')
            raise ValueError('Normalised error is unavailable: every run needs a finite error and verified positive reference scale.')
        if not trend:
            values = np.asarray([(r[1] / r[2]) * multiplier for r in rows])
            if not np.all(np.isfinite(values)):
                raise ValueError('Normalised errors exceed the supported numerical range.')
            return values
        tables = {r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'normalised_traces' not in tables:
            raise ValueError('Normalised convergence traces are unavailable; rerun this experiment.')
        result = read_aligned_moments(cx, uid, normalised=True)
    if result is None or not len(result[0]):
        raise ValueError('Normalised convergence traces are missing for some runs.')
    fe, mean, m2, count = result
    mean = mean * multiplier
    se = standard_error(m2, count) * multiplier
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(se)):
        raise ValueError('Normalised summaries exceed the supported numerical range.')
    return fe, mean, se
