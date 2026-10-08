"""Cross-validated discrimination of 90-day mortality with and without mean sigma (Table S6).

Supplementary Methods S3. Patients are assigned at random to five folds; within each training fold missing values are
replaced by the median and flagged, the mean and 10th percentile of mu, the mean session SBP and the 10th percentile of
the session nadir enter as three-knot restricted cubic splines, all variables are standardised, and ridge logistic
regressions with a fixed penalty (C = 1) are fitted. Out-of-fold predictions are pooled; 95% CIs are from 300 paired
patient-level bootstrap resamples. Basic model: age, sex, log(1 + sessions), calendar year, observed hypotension history
and fraction of forecasts with mu below 90 mmHg. Extended model: the basic model plus the mean, SD and 10th percentile of
mu, the mean and between-session SD of session-mean SBP, the mean within-session SD and the 10th percentile of the
session nadir. Each is compared with the same model plus mean sigma.

Usage: python -m analysis.discrimination --frame <landmark_frame.parquet> --forecasts <forecasts.parquet> --source <CY|TT> --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .landmarks import P, window_sigma

SEED = 20260914
REPS = 300
PT = 'prior90_point_nadir90_burden'
OBS = 'prior90_observed_nadir90_burden'
PROB = 'mean_probability'
SIG = 'sigma_mean'
COV = ['age_at_index_years', 'sex_female', 'log1p_prior90_sessions', 'index_year']
BP = ['mu_mean', 'mu_sd', 'mu_p10', 'observed_mean', 'observed_between_sd', 'observed_within_sd', 'observed_nadir_p10']
SPLINES = {'mu_mean', 'mu_p10', 'observed_mean', 'observed_nadir_p10', PROB}
BASE = COV + [OBS, PT]
MODELS = {'B0': BASE, 'B0_S': BASE + [SIG], 'B1': BASE + BP, 'B1_S': BASE + BP + [SIG]}
PAIRS = [('B1', 'B1_S'), ('B0', 'B0_S')]
METRICS = ['auc', 'auprc', 'brier', 'log_loss', 'mean_predicted', 'event_rate']


def boot_patient_weights(patients, seed=SEED, reps=REPS):
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=reps).astype(float)


def interval_frame(records, draws, keys_, metrics):
    a = pd.DataFrame(records)
    b = pd.DataFrame(draws)
    for metric in metrics:
        q = b.groupby(keys_)[metric].quantile([.025, .975]).unstack()
        q.columns = [metric + '_low', metric + '_high']
        a = a.merge(q.reset_index(), on=keys_, how='left', validate='one_to_one')
    return a


class ClinicalDesign:
    """Median imputation with missing indicators, three-knot restricted cubic splines, standardisation; fitted on the
    training fold only."""

    def fit(self, frame, columns):
        self.columns = columns
        self.fill = {}
        self.knots = {}
        self.missing = []
        for col in columns:
            x = frame[col].to_numpy(float)
            self.fill[col] = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.
            if (~np.isfinite(x)).any():
                self.missing.append(col)
            x = np.where(np.isfinite(x), x, self.fill[col])
            if col == PROB:
                x = logit(np.clip(x, 1e-6, 1 - 1e-6))
            if col in SPLINES:
                knots = np.quantile(x, [.1, .5, .9])
                if np.all(np.diff(knots) > 1e-8):
                    self.knots[col] = knots
        raw = self.raw(frame)
        self.center = raw.mean(axis=0)
        self.sd = raw.std(axis=0, ddof=0)
        self.keep = self.sd > 1e-10
        return self

    def raw(self, frame):
        arrays, names = [], []
        for col in self.columns:
            x = frame[col].to_numpy(float)
            x = np.where(np.isfinite(x), x, self.fill[col])
            if col == PROB:
                x = logit(np.clip(x, 1e-6, 1 - 1e-6))
            arrays.append(x)
            names.append(col)
            if col in self.knots:
                a, b, c = self.knots[col]
                term = (np.maximum(x - a, 0) ** 3 - np.maximum(x - b, 0) ** 3 * (c - a) / (c - b) + np.maximum(x - c, 0) ** 3 * (b - a) / (c - b)) / (c - a) ** 2
                arrays.append(term)
                names.append(col + '_rcs')
            if col in self.missing:
                arrays.append((~np.isfinite(frame[col].to_numpy(float))).astype(float))
                names.append(col + '_missing')
        self.names = names
        return np.column_stack(arrays)

    def transform(self, frame):
        return (self.raw(frame)[:, self.keep] - self.center[self.keep]) / self.sd[self.keep]


def calibration_diagnostic(y, p, w=None):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    x = logit(p)
    if w is None:
        w = np.ones(len(y))
    w = w / w.sum()

    def obj(beta, offset):
        z = x + beta[0] if offset else beta[0] + beta[1] * x
        prob = expit(z)
        loss = np.sum(w * (np.logaddexp(0, z) - y * z))
        res = w * (prob - y)
        grad = np.array([res.sum()]) if offset else np.array([res.sum(), np.dot(res, x)])
        return loss, grad

    citl = minimize(lambda b: obj(b, True), [0.], jac=True, method='BFGS', options={'gtol': 1e-8, 'maxiter': 300})
    cal = minimize(lambda b: obj(b, False), [0., 1.], jac=True, method='BFGS', options={'gtol': 1e-8, 'maxiter': 300})
    return dict(calibration_in_large=float(citl.x[0]), calibration_intercept=float(cal.x[0]), calibration_slope=float(cal.x[1]))


def weighted_score(y, p, w):
    """Identical-score groups make weighted AUC and average precision exact under ties."""
    order = np.argsort(-p, kind='stable')
    ys, ps, ws = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    tp = np.cumsum(ws * ys)[ends]
    fp = np.cumsum(ws * (1 - ys))[ends]
    assert tp[-1] > 0 and fp[-1] > 0
    auc = float(np.trapezoid(np.r_[0, tp] / tp[-1], np.r_[0, fp] / fp[-1]))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    ap = float(np.dot(np.diff(np.r_[0, tp]) / tp[-1], precision))
    pp = np.clip(p, 1e-12, 1 - 1e-12)
    n = w.sum()
    return {'auc': auc, 'auprc': ap, 'brier': float(np.dot(w, (y - p) ** 2) / n),
            'log_loss': float(np.dot(w, -y * np.log(pp) - (1 - y) * np.log1p(-pp)) / n),
            'mean_predicted': float(np.dot(w, p) / n), 'event_rate': float(np.dot(w, y) / n)}


def evaluate(lm, source, reps=REPS, seed=SEED):
    """Out-of-fold predictions of the four models and paired patient-bootstrap scores."""
    y = lm.outcome.to_numpy(int)
    pred = {name: np.full(len(lm), np.nan) for name in MODELS}
    fitlog = []
    for fold in range(5):
        frame = lm.copy()
        test = frame.fold.eq(fold).to_numpy()
        train = ~test
        assert not set(frame.loc[test, P]) & set(frame.loc[train, P]), 'patient_overlap'
        for name, cols in MODELS.items():
            design = ClinicalDesign().fit(frame.loc[train], cols)
            xt, xv = design.transform(frame.loc[train]), design.transform(frame.loc[test])
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=3000, tol=1e-9).fit(xt, y[train])
            pred[name][test] = model.predict_proba(xv)[:, 1]
            fitlog.append(dict(source=source, fold=fold, model=name, n_train=int(train.sum()), n_test=int(test.sum()),
                               parameters=xt.shape[1] + 1, iterations=int(model.n_iter_[0]), converged=True))
        print(f'{source}: fold {fold + 1}/5 fitted', flush=True)
    patients = np.sort(lm[P].unique())
    codes = pd.Categorical(lm[P], categories=patients).codes
    weights = np.vstack([np.ones(len(patients)), boot_patient_weights(patients, seed, reps)])
    performance, pdraws, differences, ddraws, scores_by_model = [], [], [], [], {}
    for name, p in pred.items():
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), 'invalid_prediction'
        scores = [weighted_score(y, p, wp[codes]) for wp in weights]
        scores_by_model[name] = scores
        performance.append(dict(source=source, model=name, n_windows=len(lm), n_patients=len(patients), n_events=int(y.sum()),
                                **scores[0], **calibration_diagnostic(y, p)))
        for rep in range(reps):
            pdraws.append(dict(source=source, model=name, replicate=rep, **scores[rep + 1]))
    for ref, cand in PAIRS:
        dd = [{m: b[m] - a[m] for m in a} for a, b in zip(scores_by_model[ref], scores_by_model[cand])]
        differences.append(dict(source=source, reference=ref, candidate=cand, **dd[0]))
        for rep in range(reps):
            ddraws.append(dict(source=source, reference=ref, candidate=cand, replicate=rep, **dd[rep + 1]))
    return performance, pdraws, differences, ddraws, fitlog


def assign_folds(lm, seed=SEED):
    tokens = np.sort(lm[P].unique())
    foldmap = {}
    for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(tokens)):
        foldmap.update({p: fold for p in tokens[test]})
    return lm[P].map(foldmap).astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame', required=True, help='landmark frame of one system (parquet) from analysis.landmarks')
    p.add_argument('--forecasts', required=True, help='forecasts of the same system (parquet)')
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reps', type=int, default=REPS)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm = pd.read_parquet(args.frame).reset_index(drop=True)
    lm['landmark_date'] = pd.to_datetime(lm.landmark_date)
    rows = pd.read_parquet(args.forecasts, columns=[P, 'dialysis_date', 'mu_sbp', 'sigma_sbp'])
    rows['dialysis_date'] = pd.to_datetime(rows.dialysis_date)
    sig, mu, n = window_sigma(lm, rows)
    assert np.array_equal(n, lm.prior90_n_rows.to_numpy(int)) and np.allclose(mu, lm.mu_mean90.to_numpy(float))
    lm = lm.rename(columns={'mu_mean90': 'mu_mean'})
    lm[SIG] = sig
    lm['outcome'] = lm.death90.astype(int)
    lm['fold'] = assign_folds(lm)
    perf, pdraws, diff, ddraws, fitlog = evaluate(lm, args.source, args.reps)
    interval_frame(perf, pdraws, ['source', 'model'], METRICS).to_csv(out / 'sigma_performance.csv', index=False)
    interval_frame(diff, ddraws, ['source', 'reference', 'candidate'], METRICS).to_csv(out / 'sigma_differences.csv', index=False)
    pd.DataFrame(fitlog).to_csv(out / 'sigma_fit_audit.csv', index=False)
    print('done', out)


if __name__ == '__main__':
    main()
