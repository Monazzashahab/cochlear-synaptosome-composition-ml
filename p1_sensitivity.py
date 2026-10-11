#!/usr/bin/env python3
"""
Sensitivity analyses (run after p1_analysis.py)

1. Sample influence: remove each Control or Cisplatin sample and recompute the
   Cisplatin vs Control signal (four-group moderated t model).
2. All 35 partitions of the eight Control and Cisplatin samples.
3. Exploratory reduced analysis without the most influential Control and
   Cisplatin sample (identified in step 1, so not an independent test).
4. Inclusion of the QC-failed channel C1.
"""
import os, io, json, pickle, itertools, contextlib
import numpy as np
import pandas as pd
from scipy import stats

import p1_analysis as P
from limma_stats import moderated_t, bh

OUT = P.OUT_DIR
with contextlib.redirect_stdout(io.StringIO()):
    prot, Lall, L, ml_idx, info = P.load_and_qc()
R = pickle.load(open(os.path.join(OUT, 'results.pkl'), 'rb'))
de = R['de']
G = P.G_K; S = np.array(P.SAMPLES_K)


def bva(Lm, groups):
    r, _ = moderated_t(Lm, groups, [('B', 'A')], levels=P.GROUPS)
    p = r[('B', 'A')]['p']; q = bh(p)
    return {'p001': int(np.nansum(p < 0.01)), 'frac_p005': float(np.nanmean(p < 0.05)),
            'q020': int(np.nansum(q < 0.20)), 'q010': int(np.nansum(q < 0.10)),
            'pi0': float(min(1, np.nanmean(p > 0.5) / 0.5))}, r[('B', 'A')]


base, _ = bva(L, G)
print('Full data:', base)

# 1. influence
infl = []
for j in np.where(np.isin(G, ['A', 'B']))[0]:
    keep = np.arange(len(S)) != j
    r, _ = bva(L[:, keep], G[keep])
    infl.append({'sample_removed': S[j], 'group': P.GROUP_NAMES[G[j]], **r,
                 'delta_frac_p005': r['frac_p005'] - base['frac_p005']})
infl = pd.DataFrame(infl)
infl.to_csv(os.path.join(OUT, 'TableS12_sample_influence.csv'), index=False)
print(infl.round(3).to_string(index=False))

# 2. partitions
ab = np.where(np.isin(G, ['A', 'B']))[0]
rows, seen = [], set()
for comb in itertools.combinations(range(8), 4):
    key = frozenset(comb) if 0 in comb else frozenset(set(range(8)) - set(comb))
    if key in seen:
        continue
    seen.add(key)
    gp = G.copy(); gp[ab[list(key)]] = 'B'; gp[ab[[i for i in range(8) if i not in key]]] = 'A'
    r, _ = bva(L, gp)
    rows.append({'set_B': '/'.join(S[ab[sorted(key)]]), 'set_A': '/'.join(S[ab[[i for i in range(8) if i not in key]]]),
                 'n_A_in_set_B': int(np.sum(G[ab[list(key)]] == 'A')), **r})
parts = pd.DataFrame(rows)
parts['is_true'] = parts['n_A_in_set_B'].isin([0, 4])
parts = parts.sort_values('p001', ascending=False).reset_index(drop=True)
parts['rank'] = np.arange(1, len(parts) + 1)
parts.to_csv(os.path.join(OUT, 'TableS13_all_partitions.csv'), index=False)
print('\nTop partitions:\n', parts.head(5)[['set_B', 'set_A', 'p001', 'q020', 'frac_p005', 'is_true']].to_string(index=False))
true_rank = int(parts.loc[parts.is_true, 'rank'].iloc[0])
print('True labelling rank (by proteins at p<0.01):', true_rank)

# 3. reduced: drop the most influential Control and Cisplatin sample
excl = [infl[infl.group == 'Control'].sort_values('delta_frac_p005').iloc[-1]['sample_removed'],
        infl[infl.group == 'Cisplatin'].sort_values('delta_frac_p005').iloc[-1]['sample_removed']]
keep = ~np.isin(S, excl)
red, red_r = bva(L[:, keep], G[keep])
print(f'\nWithout {excl}:', red)
# overlap of FDR<0.20 sets
sig_full = set(np.where(de['q_BvsA'].values < 0.20)[0])
q_red = bh(red_r['p']); sig_red = set(np.where(q_red < 0.20)[0])
same = sum(1 for i in sig_full & sig_red if np.sign(red_r['logFC'][i]) == np.sign(de['log2FC_BvsA'].values[i]))
# nested classification on the 6 remaining samples
Xr = L[ml_idx][:, keep & np.isin(G, ['A', 'B'])]
yr = (G[keep & np.isin(G, ['A', 'B'])] == 'B').astype(int)
clf_red = {}
for name, mk in P.classifiers().items():
    s, pr, _ = P.nested_loo(Xr, yr, mk)
    m = P.metrics(yr, pr, s)
    pa = []
    for comb in itertools.combinations(range(6), 3):
        yp = np.array([1 if i in comb else 0 for i in range(6)])
        sp, prp, _ = P.nested_loo(Xr, yp, mk)
        pa.append(P.metrics(yp, prp, sp)['accuracy'])
    m['perm_p_accuracy'] = float(np.mean(np.array(pa) >= m['accuracy'] - 1e-12))
    clf_red[name] = m
    print(f'  {name:20s} acc {m["accuracy"]:.3f} AUROC {m["auroc"]:.3f} perm p {m["perm_p_accuracy"]:.2f}')

# 4. include C1
G16 = P.GROUP_OF
r16, _ = moderated_t(Lall, G16, [('C', 'B'), ('C', 'A'), ('B', 'A')], levels=P.GROUPS)
c1 = {k: {'q005': int(np.nansum(bh(v['p']) < 0.05)), 'q020': int(np.nansum(bh(v['p']) < 0.20))}
      for k, v in {'CvsB': r16[('C', 'B')], 'CvsA': r16[('C', 'A')], 'BvsA': r16[('B', 'A')]}.items()}
# C1 on the cisplatin axis (top-50 selected on all 8 Control/Cisplatin samples)
X16 = Lall[ml_idx]
ab16 = np.isin(G16, ['A', 'B']); y16 = (G16[ab16] == 'B').astype(int)
ok = ~np.isnan(X16).any(1)
idx = P.select_top(X16[ok][:, ab16], y16, P.K_PRIMARY)
Xo = X16[ok][idx]
mA, mB = Xo[:, ab16][:, y16 == 0].mean(1), Xo[:, ab16][:, y16 == 1].mean(1)
d = mB - mA
c1_axis = float(np.sum(d * (Xo[:, 8] - mA)) / np.sum(d ** 2))
c1_r = float(np.mean([np.corrcoef(Xo[:, 8] - Xo.mean(1), Xo[:, j] - Xo.mean(1))[0, 1] for j in (9, 10, 11)]))
print(f'\nWith C1 (n=4 MnTBAP+Cisplatin): {c1}; C1 cisplatin-axis score {c1_axis:.2f}')

sens = {'base': base, 'influence': infl.to_dict('records'), 'true_rank': true_rank,
        'top_partition': parts.iloc[0].to_dict(), 'n_partitions': int(len(parts)),
        'reduced': {'excluded': excl, **red, 'overlap_q020': len(sig_full & sig_red), 'overlap_same_dir': same,
                    'classification': clf_red},
        'C1': {'counts': c1, 'axis': c1_axis, 'mean_r_with_C2_C4': c1_r,
               'fraction_of_median_total': info['C1_fraction_of_median_total']}}
json.dump(sens, open(os.path.join(OUT, 'sensitivity_summary.json'), 'w'), indent=2,
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
pickle.dump({'infl': infl, 'parts': parts, 'sens': sens},
            open(os.path.join(OUT, 'sensitivity.pkl'), 'wb'))

from relabel_outputs import relabel
relabel(OUT)
