"""Train the five-member NAMLSS ensemble of one forecasting model (Model A or Model B).

Input: the hourly feature table (see docs/data_dictionary.md) and a model spec (spec_model_A.json or spec_model_B.json).
Steps, as in Supplementary Methods S1: patients of the fitting sources are split at random into a fitting partition and a
development partition (about 20% of patients); the preprocessor (median imputation, missing indicators, standardisation)
is fitted on the fitting partition; each member is trained on a patient-level bootstrap sample of the fitting partition
with a short mean-only warm-up, then the Gaussian negative log-likelihood of next-hour SBP and DBP (AdamW); training
stops when the development negative log-likelihood has not improved for 100 epochs and the member is kept at its best
development value. Targets are modelled as changes from the current-hour mean SBP and DBP.

Usage: python -m model.train_members --spec model/spec_model_A.json --table <feature_table.parquet> --out <dir> [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
import torch

from .config import ModelConfig
from .losses import gaussian_nll_with_clamped_sigma, mse_mu_loss
from .model import GaussianAdditiveNAMLSS
from .preprocessing import ConstantSupportPreprocessor, MedianMissingIndicatorPreprocessor
from .training import bootstrap_indices_by_group, ridge_initial_residual_scale, set_global_seed

IDS = ['site_group', 'source_system', 'hd_session_id', 'patient_token_v1', 'hour_index']
P = 'patient_token_v1'
TARGETS = ['target_sbp_next_hour_mean_mmhg', 'target_dbp_next_hour_mean_mmhg']
ELIGIBLE = 'forecast_eligible'
OFFSET_COLUMNS = ['sbp_mean_all_mmhg', 'dbp_mean_all_mmhg']


def dump(data, path):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf8')


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def keys(frame):
    frame = frame.copy()
    for col in IDS[:-1]:
        if col in frame:
            frame[col] = frame[col].astype(str)
    return frame


def split_pool(pool, seed, development_fraction=0.2):
    """Patient-level split: about 20% of the patients of each source form the development partition."""
    patient_keys = pool.site_group.astype(str) + '__' + pool[P].astype(str)
    unique = np.unique(patient_keys.to_numpy())
    np.random.default_rng(seed).shuffle(unique)
    dev = patient_keys.isin(set(unique[:max(1, int(round(len(unique) * development_fraction)))]))
    return pool.loc[~dev], pool.loc[dev], patient_keys.loc[~dev].to_numpy()


def offsets(frame, fills):
    """The mean is predicted as a change from the current-hour mean SBP and DBP (median-filled when missing)."""
    value = frame[OFFSET_COLUMNS].to_numpy(dtype=np.float32)
    return np.where(np.isfinite(value), value, np.asarray(fills, np.float32))


def load_data(spec, table):
    features = spec['features']
    cols = list(dict.fromkeys(IDS + features + TARGETS + [ELIGIBLE]))
    cache = keys(pd.read_parquet(table, columns=cols))
    pool = cache.loc[cache.site_group.isin(spec['fitting_sources'])].reset_index(drop=True)
    assert len(pool), 'no rows from the fitting sources'
    fit, dev, groups = split_pool(pool, spec['seed'], spec.get('development_fraction', 0.2))
    keep = fit[ELIGIBLE].fillna(False).to_numpy(bool)
    groups = groups[keep]
    fit = fit.loc[keep].reset_index(drop=True)
    dev = dev.loc[dev[ELIGIBLE].fillna(False)].reset_index(drop=True)
    pre = MedianMissingIndicatorPreprocessor(features, add_missing_indicator=True, scale_numeric=True)
    xf = pre.fit_transform(fit).to_numpy(np.float32)
    xd = pre.transform(dev).to_numpy(np.float32)
    fill = np.nanmedian(fit[OFFSET_COLUMNS].to_numpy(float), axis=0)
    yf = fit[TARGETS].to_numpy(np.float32) - offsets(fit, fill)
    yd = dev[TARGETS].to_numpy(np.float32) - offsets(dev, fill)
    dg = dev.site_group.astype(str) + '__' + dev[P].astype(str)
    assert not (set(groups) & set(dg)), 'source-patient split leakage'
    assert not (set(fit[P]) & set(dev[P])), 'a patient token appears in both partitions'
    for a in (xf, xd, yf, yd):
        assert np.isfinite(a).all()
    dcode, uniques = pd.factorize(dg, sort=True)
    split = dict(fit_rows=len(fit), dev_rows=len(dev), fit_patients=len(np.unique(groups)), dev_patients=len(uniques))
    # support guard: indicators constant in the fitting partition stay frozen at that state (no encoded value changes)
    missing = fit[features].apply(pd.to_numeric, errors='coerce').isna()
    frozen = {f'{f}_is_missing': int(missing[f].iloc[0]) for f in features if missing[f].nunique() == 1}
    guard = copy.deepcopy(pre)
    guard.__class__ = ConstantSupportPreprocessor
    guard.frozen_missing_indicators_ = frozen
    assert np.array_equal(guard.transform(fit).to_numpy(np.float32), xf) and np.array_equal(guard.transform(dev).to_numpy(np.float32), xd)
    extra = dict(feature_columns=features, preprocessor_output_columns=list(pre.transform(fit.iloc[:1]).columns),
                 offset_fill=fill.tolist(), offset_columns=OFFSET_COLUMNS, frozen_missing_indicators=frozen)
    return dict(label=spec['model'], pre=guard, xf=xf, xd=xd, yf=yf, yd=yd, groups=groups, dev_codes=dcode,
                dev_sources=dev.site_group.to_numpy(), split=split, extra=extra)


@torch.inference_mode()
def prediction(model, x, cfg):
    model.eval()
    mu, sigma = [], []
    for start in range(0, len(x), cfg.batch_size):
        p = model(x[start:start + cfg.batch_size], sigma_floor=(cfg.sigma_floor_sbp, cfg.sigma_floor_dbp), sigma_cap=cfg.sigma_cap)
        mu.append(p['mu'].cpu().numpy())
        sigma.append(p['sigma'].cpu().numpy())
    return np.concatenate(mu), np.concatenate(sigma)


def score(y, mu, sigma, cfg, codes=None, sources=None):
    y, mu, sigma = (np.asarray(a, dtype=np.float64) for a in (y, mu, sigma))
    e = y - mu
    z = e / sigma
    nll = np.log(sigma + cfg.epsilon) + .5 * z * z + .5 * math.log(2 * math.pi)
    joint = np.mean(nll * np.array([cfg.loss_weight_sbp, cfg.loss_weight_dbp]), axis=1)
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    out = {'joint_nll': float(joint.mean())}
    for j, bp in enumerate(('sbp', 'dbp')):
        out.update({f'{bp}_nll': float(nll[:, j].mean()), f'{bp}_mae': float(np.abs(e[:, j]).mean()),
                    f'{bp}_rmse': float(np.sqrt(np.mean(e[:, j] ** 2))), f'{bp}_crps': float(crps[:, j].mean()),
                    f'{bp}_coverage95': float(np.mean(np.abs(z[:, j]) <= 1.96))})
    if codes is not None:
        out['patient_equal_joint_nll'] = float(np.mean(np.bincount(codes, weights=joint) / np.bincount(codes)))
    if sources is not None:
        for s in np.unique(sources):
            out[f'source_{s}_joint_nll'] = float(joint[sources == s].mean())
    return out


def train_member(data, spec, rule, member, cap, device, out):
    label = data['label']
    ck = out / 'checkpoints'
    ck.mkdir(parents=True, exist_ok=True)
    final = ck / f'model_{label}_member_{member}.joblib'
    resume_path = ck / f'member_{member}_resume.pt'
    history_path = out / f'member_{member}_learning_curve.csv'
    if final.exists():
        item = joblib.load(final)
        if item['info']['stop_reason'] == 'development_patience':
            print(json.dumps(dict(event='already_complete', model=label, member=member, **item['info']['selection'])), flush=True)
            return item
    started = time.perf_counter()
    cfg = ModelConfig(**spec['training_config'])
    cfg.n_epochs = cap
    cfg.patience = rule['patience_epochs']
    seed = spec['seed'] + member
    indices = bootstrap_indices_by_group(data['groups'], seed=seed)
    x, y = data['xf'][indices], data['yf'][indices]
    set_global_seed(seed)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(y, device=device)
    xd = torch.as_tensor(data['xd'], device=device)
    sample_idx = np.linspace(0, len(data['xf']) - 1, min(20000, len(data['xf'])), dtype=int)
    xs = torch.as_tensor(data['xf'][sample_idx], device=device)
    ys = data['yf'][sample_idx]
    model = GaussianAdditiveNAMLSS(xt.shape[1], cfg.num_bases).to(device)
    generator = torch.Generator(device='cpu').manual_seed(seed + 11)
    histories = []
    best_nll, anchor = np.inf, np.inf
    best_epoch, last_significant = 0, 0
    best_state = None
    elapsed_before = 0.
    if resume_path.exists():
        r = torch.load(resume_path, map_location=device, weights_only=False)
        assert r['rule'] == rule and r['seed'] == seed
        model.load_state_dict(r['model'])
        model.set_sigma_trainable(True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        opt.load_state_dict(r['optimizer'])
        generator.set_state(r['generator'].cpu())
        torch.set_rng_state(r['torch_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([a.cpu() for a in r['cuda_rng']])
        histories, best_nll, anchor = r['history'], r['best_nll'], r['anchor']
        best_epoch, last_significant = r['best_epoch'], r['last_significant']
        best_state = {k: v.cpu() for k, v in r['best_state'].items()}
        elapsed_before = r['elapsed_seconds']
        first = r['epoch'] + 1
    else:
        mean, rs = ridge_initial_residual_scale(x, y)
        model.set_mu_intercepts(*map(float, mean))
        model.reset_log_sigma_intercepts(*map(float, rs))
        model.set_sigma_trainable(False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for epoch in range(1, cfg.warmup_epochs + 1):                       # mean-only warm-up
            perm = torch.randperm(len(xt), generator=generator, device='cpu').to(device)
            losses = []
            for start in range(0, len(xt), cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                p = model(xt.index_select(0, idx))
                loss = mse_mu_loss(yt.index_select(0, idx), p['mu'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            histories.append(dict(epoch=epoch, stage='warmup_mu_mse', train_online_loss=float(np.mean(losses))))
        with torch.no_grad():                                               # sigma intercepts from the warm-up residuals
            p = model(xt)['mu'].cpu().numpy()
        warm = np.nanstd(y - p, axis=0)
        warm = np.where(np.isfinite(warm) & (warm > 1e-6), warm, rs)
        model.reset_log_sigma_intercepts(*map(float, warm))
        model.set_sigma_trainable(True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        first = cfg.warmup_epochs + 1
    stop_reason = 'epoch_cap'
    epoch = first - 1                                                   # defined even when a resumed member is already at the cap
    for epoch in range(first, cap + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=generator, device='cpu').to(device)
        losses = []
        for start in range(0, len(xt), cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            p = model(xt.index_select(0, idx))
            loss = gaussian_nll_with_clamped_sigma(yt.index_select(0, idx), p['mu'], p['log_sigma'],
                                                   sigma_floor=(cfg.sigma_floor_sbp, cfg.sigma_floor_dbp), sigma_cap=cfg.sigma_cap,
                                                   epsilon=cfg.epsilon, loss_weights=(cfg.loss_weight_sbp, cfg.loss_weight_dbp))
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss {label}/{member}/{epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = dict(epoch=epoch, stage='nll', train_online_loss=float(np.mean(losses)))
        if epoch % rule['evaluate_every'] == 0 or epoch == 12 or epoch == cap:
            mu, sd = prediction(model, xd, cfg)
            scores = score(data['yd'], mu, sd, cfg, data['dev_codes'], data['dev_sources'])
            assert np.isfinite(list(scores.values())).all()
            row.update({f'dev_{k}': v for k, v in scores.items()})
            sm, ss = prediction(model, xs, cfg)
            row['fit_sample_joint_nll'] = score(ys, sm, ss, cfg)['joint_nll']
            val = scores['joint_nll']
            if val < best_nll:
                best_nll, best_epoch, best_state = val, epoch, cpu_state(model)
            if val < anchor - rule['min_delta']:
                anchor, last_significant = val, epoch
            row.update(best_epoch=best_epoch, best_dev_joint_nll=best_nll, last_significant_epoch=last_significant)
            if epoch - last_significant >= rule['patience_epochs']:
                stop_reason = 'development_patience'
        histories.append(row)
        should_stop = stop_reason == 'development_patience'
        if epoch % 25 == 0 or epoch == 12 or epoch == cap or should_stop:
            pd.DataFrame(histories).to_csv(history_path, index=False)
            elapsed = elapsed_before + time.perf_counter() - started
            torch.save(dict(rule=rule, seed=seed, epoch=epoch, model=cpu_state(model), optimizer=opt.state_dict(),
                            generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [], history=histories,
                            best_nll=best_nll, best_epoch=best_epoch, best_state=best_state, anchor=anchor,
                            last_significant=last_significant, elapsed_seconds=elapsed), resume_path)
            print(json.dumps(dict(event='progress', model=label, member=member, epoch=epoch, best_epoch=best_epoch,
                                  best_dev_joint_nll=best_nll, current_dev_joint_nll=row.get('dev_joint_nll'),
                                  last_significant_epoch=last_significant, elapsed_seconds=round(elapsed, 1))), flush=True)
        if should_stop:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    selection = dict(best_epoch=best_epoch, stop_epoch=epoch, best_development_joint_nll=best_nll, last_significant_epoch=last_significant)
    info = dict(selection=selection, stop_reason=stop_reason, rule=rule, history=histories, config=vars(cfg))
    item = dict(model=model.cpu(), info=info, seed=seed, elapsed_seconds=elapsed_before + time.perf_counter() - started)
    joblib.dump(item, final)
    dump({k: v for k, v in info.items() if k != 'history'}, out / f'member_{member}_selection.json')
    print(json.dumps(dict(event='member_complete', model=label, member=member, stop_reason=stop_reason, **selection)), flush=True)
    del xt, yt, xd, xs, opt, model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True, help='spec_model_A.json or spec_model_B.json')
    p.add_argument('--table', required=True, help='hourly feature table (parquet), see docs/data_dictionary.md')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--members', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--max-epochs', type=int, default=1500)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    spec = json.loads(Path(args.spec).read_text(encoding='utf8'))
    assert len(spec['features']) == 13
    rule = spec['early_stopping']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(spec, args.table)
    ck = out / 'checkpoints'
    ck.mkdir(parents=True, exist_ok=True)
    label = spec['model']
    dump(dict(model=label, rule=rule, split=d['split'], features=spec['features'], training_config=spec['training_config']), ck / f'model_{label}_training_spec.json')
    joblib.dump(d['pre'], ck / f'model_{label}_preprocessor.joblib')
    print(json.dumps(dict(event='split', model=label, **d['split'])), flush=True)
    for member in args.members:
        assert member in range(spec['training_config']['ensemble_members'])
        train_member(d, spec, rule, member, args.max_epochs, args.device, out)
    files = [ck / f'model_{label}_member_{m}.joblib' for m in range(spec['training_config']['ensemble_members'])]
    if all(f.exists() for f in files):
        items = [joblib.load(f) for f in files]
        complete = all(i['info']['stop_reason'] == 'development_patience' for i in items)
        cfg = ModelConfig(**spec['training_config'])
        cfg.n_epochs = args.max_epochs
        cfg.patience = rule['patience_epochs']
        selections = [dict(member=m, seed=i['seed'], stop_reason=i['info']['stop_reason'], **i['info']['selection']) for m, i in enumerate(items)]
        extra = {**d['extra'], 'epoch_selection': dict(rule=rule, members=selections, complete=complete)}
        joblib.dump(dict(models=[i['model'] for i in items], config=cfg, extra=extra), ck / f'model_{label}_bundle.joblib')
        dump(dict(model=label, complete=complete, members=selections), out / f'model_{label}_completion.json')


if __name__ == '__main__':
    main()
