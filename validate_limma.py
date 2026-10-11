#!/usr/bin/env python3
"""Validate limma_stats against inmoose.limma (Python port of limma).
Checks: (1) group-means model on all retained proteins (with missing values),
(2) the same on complete-case proteins, (3) group model + PC1 covariate.
All five contrasts are compared."""
import io, contextlib, pickle, os
import numpy as np, pandas as pd
from inmoose.limma import lmFit, eBayes, contrasts_fit
import p1_analysis as P
from limma_stats import moderated_t, moderated_t_design

with contextlib.redirect_stdout(io.StringIO()):
    prot, Lall, L, ml_idx, info = P.load_and_qc(verbose=False)
groups = P.GROUP_OF[P.KEEP]
CONS = {'BvsA': [-1, 1, 0, 0], 'CvsA': [-1, 0, 1, 0], 'CvsB': [0, -1, 1, 0], 'DvsA': [-1, 0, 0, 1], 'CvsD': [0, 0, 1, -1]}
PAIRS = {'BvsA': ('B', 'A'), 'CvsA': ('C', 'A'), 'CvsB': ('C', 'B'), 'DvsA': ('D', 'A'), 'CvsD': ('C', 'D')}
design = pd.get_dummies(pd.Series(groups)).astype(float)


def ref_fit(Y, D, cons):
    fit = lmFit(Y, D)
    cols = list(fit.coefficients.columns)
    C = pd.DataFrame(np.array([cons[k] for k in cons]).T, index=cols, columns=list(cons)).astype(float)
    return eBayes(contrasts_fit(fit, C))


report = []
for label, Y in [('all retained proteins (with missing values)', L), ('complete-case proteins', L[ml_idx])]:
    ref = ref_fit(Y, design, CONS)
    mine, prior = moderated_t(Y, groups, list(PAIRS.values()), levels=P.GROUPS)
    dt = max(np.nanmax(np.abs(mine[PAIRS[k]]['t'] - np.asarray(ref.t)[:, j])) for j, k in enumerate(CONS))
    dp = max(np.nanmax(np.abs(mine[PAIRS[k]]['p'] - np.asarray(ref.p_value)[:, j])) for j, k in enumerate(CONS))
    report.append(f'{label}: n = {len(Y)}; prior d0 {prior["d0"]:.6f} vs {float(np.ravel(ref.df_prior)[0]):.6f}, '
                  f's0^2 {prior["s02"]:.6f} vs {float(np.ravel(ref.s2_prior)[0]):.6f}; max |dt| {dt:.1e}, max |dp| {dp:.1e}')

res = pickle.load(open(os.path.join(P.OUT_DIR, 'results.pkl'), 'rb'))
pc1 = res['compo']['pc1']
Y = L[ml_idx]
Xg = np.column_stack([(groups == g).astype(float) for g in P.GROUPS])
D2 = pd.DataFrame(np.column_stack([Xg, pc1]), columns=['A', 'B', 'C', 'D', 'PC1'])
cons2 = {k: v + [0] for k, v in CONS.items()}
ref = ref_fit(Y, D2, cons2)
mine, prior = moderated_t_design(Y, D2.values, cons2)
dt = max(np.nanmax(np.abs(mine[k]['t'] - np.asarray(ref.t)[:, j])) for j, k in enumerate(cons2))
dp = max(np.nanmax(np.abs(mine[k]['p'] - np.asarray(ref.p_value)[:, j])) for j, k in enumerate(cons2))
report.append(f'group + PC1 covariate model: n = {len(Y)}; prior d0 {prior["d0"]:.6f} vs '
              f'{float(np.ravel(ref.df_prior)[0]):.6f}; max |dt| {dt:.1e}, max |dp| {dp:.1e}')
txt = '\n'.join(report)
print(txt)
open(os.path.join(P.OUT_DIR, 'limma_validation.txt'), 'w').write(txt + '\n')

# load_and_qc rewrites Table S1 with raw channel codes; restore the sample names used in the paper
from relabel_outputs import relabel
relabel(P.OUT_DIR)
