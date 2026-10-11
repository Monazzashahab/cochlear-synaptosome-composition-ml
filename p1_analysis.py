#!/usr/bin/env python3
"""
Analysis pipeline: nested machine learning and composition reanalysis of a
cochlear synaptosome TMT-16plex dataset (cisplatin and MnTBAP), starting from
the raw reporter ion abundances

Cochlear synaptosomes, TMT-16plex, four groups x four biological replicates:
  A = Control, B = Cisplatin, C = MnTBAP+Cisplatin, D = MnTBAP

Input: Proteome Discoverer protein export, sheet 'Proteins', columns
'Abundances (Grouped): 126 ... 134N' (raw reporter abundances). No derived
column of the sheet (normalized abundances, group statistics, fold changes,
p values) is used.

Pre-specified decisions
  * Channel QC: a channel whose total reporter abundance is < 25% of the
    median channel total is excluded (C1, channel 130C, is excluded).
  * Proteins: Protein FDR confidence 'High'; quantified in >= 3 retained
    replicates of every group.
  * Normalization: log2, then per-channel median centering on proteins
    quantified in every retained channel (grand median added back).
  * Differential abundance: empirical Bayes moderated t (limma eBayes) on a
    four-group means model, Benjamini-Hochberg FDR. DE: FDR < 0.05.
    Exploratory cisplatin discovery signature: Cisplatin vs Control FDR < 0.20.
  * Machine learning: nested leave-one-out; in every fold, proteins are ranked
    by a two-group moderated t on the training samples only and the top 50 are
    used (primary). Exact permutation test over all 70 labelings.
  * MnTBAP+Cisplatin and MnTBAP samples are never used to train the binary
    models or to select the top 50/150 proteins (held-out projection onto the
    cisplatin axis).
"""
import os, re, json, pickle, itertools, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA

from limma_stats import moderated_t, bh

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ config
BASE = os.path.dirname(os.path.abspath(__file__))
# Proteome Discoverer export (sheet 'Proteins'). The data are not distributed with this code.
# Place the file at data/2363_MC-Loading_2.xlsx or set the environment variable P1_DATA_FILE.
DATA_FILE = os.environ.get('P1_DATA_FILE', os.path.join(BASE, 'data', '2363_MC-Loading_2.xlsx'))
OUT_DIR = os.path.join(BASE, 'output'); os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
LOADING_QC_FRACTION = 0.25
FDR_DE = 0.05
FDR_SIGNATURE = 0.20
K_PRIMARY = 50
N_TREES = 300

GROUPS = ['A', 'B', 'C', 'D']
GROUP_NAMES = {'A': 'Control', 'B': 'Cisplatin', 'C': 'MnTBAP+Cisplatin', 'D': 'MnTBAP'}
RAW_COLS = list(range(15, 31))              # Abundances (Grouped): 126 ... 134N
CHANNELS = ['126', '127N', '127C', '128N', '128C', '129N', '129C', '130N',
            '130C', '131N', '131C', '132N', '132C', '133N', '133C', '134N']
SAMPLES = [f'{g}{i}' for g in GROUPS for i in range(1, 5)]
GROUP_OF = np.array([s[0] for s in SAMPLES])
CONTRASTS = {'BvsA': ('B', 'A'), 'CvsA': ('C', 'A'), 'CvsB': ('C', 'B'),
             'DvsA': ('D', 'A'), 'CvsD': ('C', 'D')}
KEEP = None          # set in load_and_qc
SAMPLES_K = None
G_K = None


def log(msg=''):
    print(msg, flush=True)


# ------------------------------------------------------------------ 1. load, QC, normalize
def load_and_qc(verbose=True):
    global KEEP, SAMPLES_K, G_K
    say = log if verbose else (lambda *a: None)
    say('=' * 66); say('1. RAW DATA, CHANNEL QC, NORMALIZATION'); say('=' * 66)
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f'Input data not found: {DATA_FILE}\nSee data/README.md for how to obtain the Proteome Discoverer export.')
    raw_df = pd.read_excel(DATA_FILE, sheet_name='Proteins', header=None)
    hdr = [str(x) for x in raw_df.iloc[0, RAW_COLS]]
    assert all(h.startswith('Abundances (Grouped)') for h in hdr), hdr
    d = raw_df.iloc[5:]
    # the workbook holds spreadsheet working rows (a channel-sum row, loading-factor rows and
    # blank rows) between and after the protein rows; keep only rows with a protein accession
    d = d[d[1].notna()].reset_index(drop=True)

    def gene(s):
        m = re.search(r'GN=(\S+)', str(s)); return m.group(1) if m else None

    def pname(s):
        m = re.match(r'^(.+?)\s+OS=', str(s)); return m.group(1) if m else str(s)[:60]

    prot = pd.DataFrame({
        'Accession': d[1].astype(str).values, 'Description': d[2].values,
        'FDR_confidence': d[0].values,
        'Coverage_pct': pd.to_numeric(d[5], errors='coerce').values,
        'Peptides': pd.to_numeric(d[6], errors='coerce').values,
        'Unique_peptides': pd.to_numeric(d[9], errors='coerce').values,
        'PSMs': pd.to_numeric(d[8], errors='coerce').values,
    })
    prot['Gene'] = [gene(s) or a for s, a in zip(prot['Description'], prot['Accession'])]
    prot['Protein'] = prot['Description'].apply(pname)
    raw = np.column_stack([pd.to_numeric(d[c], errors='coerce').values for c in RAW_COLS]).astype(float)
    raw[raw <= 0] = np.nan
    n_sheet = len(prot)

    # ---- channel QC on all protein groups
    tot = np.nansum(raw, 0)
    hi = (prot['FDR_confidence'] == 'High').values
    qc = pd.DataFrame({'sample': SAMPLES, 'channel': CHANNELS,
                       'group': [GROUP_NAMES[g] for g in GROUP_OF],
                       'total_abundance': tot,
                       'fraction_of_median_total': tot / np.median(tot),
                       'median_log2_high_conf': np.nanmedian(np.log2(raw[hi]), 0),
                       'missing_high_conf': np.isnan(raw[hi]).sum(0)})
    qc['excluded'] = qc['fraction_of_median_total'] < LOADING_QC_FRACTION
    KEEP = ~qc['excluded'].values
    SAMPLES_K = [s for s, k in zip(SAMPLES, KEEP) if k]
    G_K = GROUP_OF[KEEP]
    qc.to_csv(os.path.join(OUT_DIR, 'TableS1_channel_QC.csv'), index=False)
    say(qc.round(3).to_string(index=False))
    say(f'Excluded channels: {", ".join(qc.loc[qc.excluded, "sample"] + " (" + qc.loc[qc.excluded, "channel"] + ")")}')

    # ---- proteins: High confidence, >=3 valid retained replicates per group
    prot = prot[hi].reset_index(drop=True)
    Lraw = np.log2(raw[hi])
    valid = np.all([np.sum(~np.isnan(Lraw[:, KEEP][:, G_K == g]), 1) >= 3 for g in GROUPS], axis=0)
    prot = prot[valid].reset_index(drop=True)
    Lraw = Lraw[valid]

    # ---- median normalization (retained channels; C1 normalized the same way for sensitivity)
    comp_k = ~np.isnan(Lraw[:, KEEP]).any(1)
    med = np.median(Lraw[comp_k][:, KEEP], 0)
    grand = np.median(med)
    L = Lraw[:, KEEP] - med + grand
    comp_all = ~np.isnan(Lraw).any(1)
    med_all = np.median(Lraw[comp_all], 0)
    Lall = Lraw - med_all + grand
    ml_idx = np.where(~np.isnan(L).any(1))[0]

    info = {'n_sheet': n_sheet, 'n_high_conf': int(hi.sum()), 'n_quant': len(prot),
            'n_complete_retained': int(len(ml_idx)),
            'n_missing_values': int(np.isnan(L).sum()), 'n_values': int(L.size),
            'excluded': qc.loc[qc.excluded, 'sample'].tolist(),
            'C1_fraction_of_median_total': float(qc.loc[qc['sample'] == 'C1', 'fraction_of_median_total'].iloc[0]),
            'qc': qc.to_dict('records')}
    say(f'Protein groups: {n_sheet}; High confidence: {hi.sum()}; quantified in >=3 retained '
        f'replicates of every group: {len(prot)}; complete across retained channels: {len(ml_idx)}')
    say(f'Missing values among retained channels: {info["n_missing_values"]} of {info["n_values"]} '
        f'({100 * info["n_missing_values"] / info["n_values"]:.2f}%), no imputation')
    return prot, Lall, L, ml_idx, info


# ------------------------------------------------------------------ 2. differential abundance
def differential(prot, L, ml_idx):
    log('\n' + '=' * 66); log('2. DIFFERENTIAL ABUNDANCE (moderated t, BH FDR)'); log('=' * 66)
    res, prior = moderated_t(L, G_K, list(CONTRASTS.values()), levels=GROUPS)
    log(f'Empirical Bayes prior: d0 = {prior["d0"]:.2f}, s0^2 = {prior["s02"]:.4f}')
    de = prot[['Accession', 'Gene', 'Protein', 'Peptides', 'Unique_peptides']].copy()
    for g in GROUPS:
        de[f'mean_log2_{g}'] = np.nanmean(L[:, G_K == g], 1)
    counts = {}
    for name, c in CONTRASTS.items():
        r = res[c]; q = bh(r['p'])
        de[f'log2FC_{name}'] = r['logFC']; de[f'FC_{name}'] = 2 ** r['logFC']
        de[f't_{name}'] = r['t']; de[f'p_{name}'] = r['p']; de[f'q_{name}'] = q
        de[f'sig_{name}'] = np.where(q < FDR_DE, np.where(r['logFC'] > 0, 'Up', 'Down'), 'NS')
        counts[name] = {
            'q005_up': int(np.sum((q < 0.05) & (r['logFC'] > 0))),
            'q005_down': int(np.sum((q < 0.05) & (r['logFC'] < 0))),
            'q010': int(np.sum(q < 0.10)), 'q020': int(np.sum(q < 0.20)),
            'q020_up': int(np.sum((q < 0.20) & (r['logFC'] > 0))),
            'q020_down': int(np.sum((q < 0.20) & (r['logFC'] < 0))),
            'p005': int(np.sum(r['p'] < 0.05)), 'p001': int(np.sum(r['p'] < 0.01)),
            'frac_p005': float(np.nanmean(r['p'] < 0.05)),
            'pi0': float(min(1, np.nanmean(r['p'] > 0.5) / 0.5)), 'min_q': float(np.nanmin(q))}
        log(f'{GROUP_NAMES[c[0]]:>17s} vs {GROUP_NAMES[c[1]]:<10s}: FDR<0.05 {counts[name]["q005_up"]}↑ '
            f'{counts[name]["q005_down"]}↓ | FDR<0.10 {counts[name]["q010"]} | FDR<0.20 {counts[name]["q020"]} | '
            f'p<0.05 {counts[name]["p005"]} ({100 * counts[name]["frac_p005"]:.1f}%) | pi0 {counts[name]["pi0"]:.2f}')
    de['signature'] = de['q_BvsA'] < FDR_SIGNATURE
    de['in_ML_matrix'] = False
    de.loc[ml_idx, 'in_ML_matrix'] = True

    # ---- global signal and empirical null for Cisplatin vs Control
    p = de['p_BvsA'].values
    ab = np.where(np.isin(G_K, ['A', 'B']))[0]
    nulls, seen = [], set()
    for comb in itertools.combinations(range(8), 4):
        key = frozenset(comb) if 0 in comb else frozenset(set(range(8)) - set(comb))
        if key in seen:
            continue
        seen.add(key)
        nA = int(np.sum(G_K[ab[list(key)]] == 'A'))
        if nA in (0, 4):
            continue
        gp = G_K.copy()
        gp[ab[list(key)]] = 'B'
        gp[ab[[i for i in range(8) if i not in key]]] = 'A'
        rr, _ = moderated_t(L, gp, [('B', 'A')], levels=GROUPS)
        pp = rr[('B', 'A')]['p']; qq = bh(pp)
        nulls.append({'balanced': nA == 2, 'p001': int(np.nansum(pp < 0.01)),
                      'frac_p005': float(np.nanmean(pp < 0.05)), 'q020': int(np.nansum(qq < 0.20)),
                      'min_q': float(np.nanmin(qq))})
    nulls = pd.DataFrame(nulls)
    bal = nulls[nulls.balanced]
    obs = counts['BvsA']
    glob = {'p_hist': np.histogram(p[~np.isnan(p)], bins=20, range=(0, 1))[0].tolist(),
            'null': nulls.to_dict('list'),
            'null_bal_p001_median': float(bal.p001.median()), 'null_bal_p001_max': int(bal.p001.max()),
            'null_bal_q020_median': float(bal.q020.median()), 'null_bal_q020_max': int(bal.q020.max()),
            'null_bal_frac_p005_median': float(bal.frac_p005.median()),
            'null_bal_frac_p005_max': float(bal.frac_p005.max()),
            'n_null_balanced': int(len(bal)), 'n_null': int(len(nulls)),
            'perm_rank_p001': int(1 + np.sum(nulls.p001 >= obs['p001'])),
            'empirical_FDR_signature': float(bal.q020.median() / max(obs['q020'], 1))}
    log(f'Null (18 balanced relabelings): proteins at p<0.01 median {bal.p001.median():.0f}, max '
        f'{bal.p001.max()} (observed {obs["p001"]}); FDR<0.20 median {bal.q020.median():.0f}, max '
        f'{bal.q020.max()} (observed {obs["q020"]}); fraction p<0.05 max {bal.frac_p005.max():.3f} '
        f'(observed {obs["frac_p005"]:.3f})')
    counts['global'] = glob
    counts['prior'] = prior

    de.to_csv(os.path.join(OUT_DIR, 'TableS2_differential_abundance_all.csv'), index=False)
    de[de.signature].sort_values('p_BvsA').to_csv(
        os.path.join(OUT_DIR, 'TableS3_cisplatin_signature_FDR020.csv'), index=False)
    de[de.sig_CvsB != 'NS'].sort_values('p_CvsB').to_csv(
        os.path.join(OUT_DIR, 'TableS4_DE_MnTBAPcis_vs_cisplatin_FDR005.csv'), index=False)
    return de, counts


# ------------------------------------------------------------------ 3. classifiers
class SignatureScore(ClassifierMixin, BaseEstimator):
    """Equal-weight signature score (diagonal LDA with sign weights)."""

    def fit(self, X, y):
        X0, X1 = X[y == 0], X[y == 1]
        m0, m1 = X0.mean(0), X1.mean(0)
        n0, n1 = len(X0), len(X1)
        sp = np.sqrt(((n0 - 1) * X0.var(0, ddof=1) + (n1 - 1) * X1.var(0, ddof=1)) / (n0 + n1 - 2))
        self.mid_ = (m0 + m1) / 2
        self.w_ = np.sign(m1 - m0) / np.where(sp > 0, sp, 1)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return ((X - self.mid_) * self.w_).mean(1)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def classifiers(include_signature=True):
    d = {'Signature score': lambda: SignatureScore()} if include_signature else {}
    return d | {
        'Logistic regression': lambda: LogisticRegression(penalty='l2', C=0.1, max_iter=5000,
                                                          random_state=SEED),
        'Linear SVM': lambda: SVC(kernel='linear', C=1.0, random_state=SEED),
        'Random forest': lambda: RandomForestClassifier(n_estimators=N_TREES, max_features='sqrt',
                                                        random_state=SEED, n_jobs=2),
        'Gradient boosting': lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                                learning_rate=0.1, random_state=SEED),
    }


def score_of(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return model.decision_function(X)


def rank_two_group(X, y):
    """Two-group moderated t on training samples; returns p (NaN -> 1) and logFC."""
    r, _ = moderated_t(X, np.where(y == 1, 'B', 'A'), [('B', 'A')], levels=['A', 'B'])
    p = np.where(np.isnan(r[('B', 'A')]['p']), 1.0, r[('B', 'A')]['p'])
    return p, r[('B', 'A')]['logFC']


def select_top(X, y, k):
    p, _ = rank_two_group(X, y)
    o = np.argsort(p)
    return o if k is None else o[:k]


def nested_loo(X, y, make_clf, k=K_PRIMARY, fixed_idx=None):
    """X: proteins x samples. Selection repeated inside each fold unless fixed_idx."""
    n = len(y); score = np.zeros(n); pred = np.zeros(n, int); sel = []
    for i in range(n):
        tr = np.arange(n) != i
        idx = fixed_idx if fixed_idx is not None else select_top(X[:, tr], y[tr], k)
        m = make_pipeline(StandardScaler(), make_clf()).fit(X[idx][:, tr].T, y[tr])
        Xte = X[idx][:, [i]].T
        score[i] = score_of(m, Xte)[0]; pred[i] = m.predict(Xte)[0]; sel.append(idx)
    return score, pred, sel


def metrics(y, pred, score):
    out = {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
           'f1': f1_score(y, pred, zero_division=0)}
    try:
        out['auroc'] = roc_auc_score(y, score)
    except ValueError:
        out['auroc'] = np.nan
    return out


def binary_ml(L, ml_idx, de):
    log('\n' + '=' * 66); log('3. CONTROL vs CISPLATIN CLASSIFICATION (nested LOO, n = 8)'); log('=' * 66)
    ab = np.isin(G_K, ['A', 'B'])
    X = L[ml_idx][:, ab]
    y = (G_K[ab] == 'B').astype(int)
    names_ab = [s for s, g in zip(SAMPLES_K, G_K) if g in 'AB']
    clfs = classifiers()
    leaky_idx = select_top(X, y, K_PRIMARY)
    res, leaky = {}, {}
    for name, mk in clfs.items():
        s, pr, sel = nested_loo(X, y, mk)
        res[name] = {**metrics(y, pr, s), 'score': s, 'pred': pr}
        s2, pr2, _ = nested_loo(X, y, mk, fixed_idx=leaky_idx)
        leaky[name] = {**metrics(y, pr2, s2), 'score': s2, 'pred': pr2}
        log(f'{name:20s} nested acc {res[name]["accuracy"]:.3f} AUROC {res[name]["auroc"]:.3f} | '
            f'selection outside CV acc {leaky[name]["accuracy"]:.3f} AUROC {leaky[name]["auroc"]:.3f}')
    # fold-to-fold overlap of the selected top-50 sets
    _, _, sel = nested_loo(X, y, clfs['Signature score'])
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(sel, 2)]
    fold_overlap = {'jaccard_median': float(np.median(jac)), 'jaccard_min': float(np.min(jac)),
                    'n_in_all_folds': int(len(set.intersection(*[set(s) for s in sel])))}
    log(f'Top-50 sets across folds: median Jaccard {fold_overlap["jaccard_median"]:.2f}, '
        f'{fold_overlap["n_in_all_folds"]} proteins selected in all 8 folds')

    per_sample = pd.DataFrame({'sample': names_ab, 'true': np.where(y == 1, 'Cisplatin', 'Control')})
    for name, r in res.items():
        per_sample[f'{name} score'] = r['score']; per_sample[f'{name} correct'] = r['pred'] == y
    per_sample['n_classifiers_correct'] = per_sample[[c for c in per_sample if c.endswith('correct')]].sum(1)
    per_sample.to_csv(os.path.join(OUT_DIR, 'TableS6_per_sample_nested_predictions.csv'), index=False)
    log(per_sample[['sample', 'true', 'n_classifiers_correct']].to_string(index=False))

    log('\nExact permutation test (70 labelings)...'); t0 = time.time()
    perms = [np.array([1 if j in c else 0 for j in range(8)]) for c in itertools.combinations(range(8), 4)]
    perm = {n: {'accuracy': [], 'auroc': []} for n in clfs}
    for yp in perms:
        for name, mk in clfs.items():
            s, pr, _ = nested_loo(X, yp, mk)
            m = metrics(yp, pr, s)
            perm[name]['accuracy'].append(m['accuracy']); perm[name]['auroc'].append(m['auroc'])
    for name in clfs:
        a = np.array(perm[name]['accuracy']); u = np.array(perm[name]['auroc'])
        res[name]['perm_p_accuracy'] = float(np.mean(a >= res[name]['accuracy'] - 1e-12))
        res[name]['perm_p_auroc'] = float(np.mean(u >= res[name]['auroc'] - 1e-12))
        perm[name] = {'accuracy': a, 'auroc': u}
        log(f'{name:20s} permutation p: accuracy {res[name]["perm_p_accuracy"]:.3f}, '
            f'AUROC {res[name]["perm_p_auroc"]:.3f}')
    log(f'  ({time.time() - t0:.0f} s)')

    log('\nSignature-size curve...')
    ks = [5, 10, 20, 50, 100, 200, 500, None]
    rows = []
    for name in ['Signature score', 'Logistic regression', 'Linear SVM', 'Random forest']:
        for k in ks:
            s, pr, _ = nested_loo(X, y, clfs[name], k=k)
            rows.append({'classifier': name, 'k': 'all' if k is None else str(k), **metrics(y, pr, s)})
    size_curve = pd.DataFrame(rows)
    log(size_curve.pivot(index='k', columns='classifier', values='auroc').round(3).to_string())

    pd.DataFrame([{'Classifier': n, 'Accuracy': r['accuracy'], 'Balanced accuracy': r['balanced_accuracy'],
                   'F1': r['f1'], 'AUROC': r['auroc'], 'Permutation p (accuracy)': r['perm_p_accuracy'],
                   'Permutation p (AUROC)': r['perm_p_auroc'],
                   'Accuracy, selection outside CV': leaky[n]['accuracy'],
                   'AUROC, selection outside CV': leaky[n]['auroc']} for n, r in res.items()]).to_csv(
        os.path.join(OUT_DIR, 'Table1_binary_classification.csv'), index=False)
    size_curve.to_csv(os.path.join(OUT_DIR, 'TableS5_signature_size_curve.csv'), index=False)
    return {'res': res, 'leaky': leaky, 'perm': perm, 'size_curve': size_curve, 'y': y,
            'names': names_ab, 'fold_overlap': fold_overlap, 'per_sample': per_sample}


# ------------------------------------------------------------------ 4. consensus ranking
def hedges_g(L0, L1):
    n0, n1 = L0.shape[1], L1.shape[1]
    sp = np.sqrt(((n0 - 1) * L0.var(1, ddof=1) + (n1 - 1) * L1.var(1, ddof=1)) / (n0 + n1 - 2))
    return (L1.mean(1) - L0.mean(1)) / sp * (1 - 3 / (4 * (n0 + n1) - 9))


def biomarker_ranking(L, ml_idx, de):
    log('\n' + '=' * 66); log('4. CONSENSUS RANKING OF THE CISPLATIN SIGNATURE'); log('=' * 66)
    ab = np.isin(G_K, ['A', 'B'])
    y = (G_K[ab] == 'B').astype(int)
    cand = np.array([i for i in ml_idx if de['signature'].values[i]])
    log(f'Signature proteins (FDR < {FDR_SIGNATURE}): {int(de.signature.sum())}; complete across '
        f'retained channels: {len(cand)}')
    X = L[cand][:, ab]
    # stability: moderated t on all complete proteins (proteome-wide prior), then keep signature rows
    Xall = L[ml_idx][:, ab]
    pos = np.searchsorted(ml_idx, cand)
    assert np.all(np.asarray(ml_idx)[pos] == cand)
    a_idx, b_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    sign_full = np.sign(de['log2FC_BvsA'].values[cand])
    freq = np.zeros(len(cand)); n_sub = 0
    for ca in itertools.combinations(a_idx, 3):
        for cb in itertools.combinations(b_idx, 3):
            cols = np.array(ca + cb)
            p, lfc = rank_two_group(Xall[:, cols], y[cols])
            p, lfc = p[pos], lfc[pos]
            freq += (p < 0.05) & (np.sign(lfc) == sign_full); n_sub += 1
    freq /= n_sub
    Xs = StandardScaler().fit_transform(X.T)
    lr = LogisticRegression(penalty='l2', C=0.1, max_iter=5000, random_state=SEED).fit(Xs, y)
    rf = RandomForestClassifier(n_estimators=5000, max_features='sqrt', random_state=SEED, n_jobs=2).fit(Xs, y)
    g = hedges_g(X[:, y == 0], X[:, y == 1])
    rk = pd.DataFrame({'idx': cand, 'Gene': de['Gene'].values[cand], 'Accession': de['Accession'].values[cand],
                       'Protein': de['Protein'].values[cand], 'log2FC_BvsA': de['log2FC_BvsA'].values[cand],
                       'FC_BvsA': de['FC_BvsA'].values[cand], 'p_BvsA': de['p_BvsA'].values[cand],
                       'q_BvsA': de['q_BvsA'].values[cand], 'stability': freq,
                       'LR_coef': lr.coef_[0], 'abs_LR_coef': np.abs(lr.coef_[0]),
                       'RF_importance': rf.feature_importances_, 'abs_hedges_g': np.abs(g)})
    comps = ['stability', 'abs_LR_coef', 'RF_importance', 'abs_hedges_g']
    for c in comps:
        rk[f'pct_{c}'] = rk[c].rank(pct=True)
    rk['consensus_score'] = rk[[f'pct_{c}' for c in comps]].mean(1)
    rk = rk.sort_values('consensus_score', ascending=False).reset_index(drop=True)
    rk['consensus_rank'] = np.arange(1, len(rk) + 1)
    rk['direction'] = np.where(rk['log2FC_BvsA'] > 0, 'Up', 'Down')
    for c in ['log2FC_CvsA', 'log2FC_CvsB', 'q_CvsB', 'log2FC_DvsA', 'q_DvsA']:
        rk[c] = de[c].values[rk['idx']]
    rk['recovery_fraction'] = 1 - rk['log2FC_CvsA'] / rk['log2FC_BvsA']
    rk.drop(columns='idx').to_csv(os.path.join(OUT_DIR, 'TableS7_consensus_ranking.csv'), index=False)
    rk.head(20)[['consensus_rank', 'Gene', 'Accession', 'Protein', 'FC_BvsA', 'p_BvsA', 'q_BvsA',
                 'stability', 'consensus_score', 'recovery_fraction']].to_csv(
        os.path.join(OUT_DIR, 'Table2_top20_consensus.csv'), index=False)
    log(f'Stability >= 0.75: {(rk.stability >= 0.75).sum()}; = 1: {(rk.stability >= 0.999).sum()}')
    log(rk.head(20)[['Gene', 'FC_BvsA', 'p_BvsA', 'q_BvsA', 'stability', 'consensus_score']].round(4).to_string())
    return rk


# ------------------------------------------------------------------ 5. MnTBAP validation
def mntbap_validation(L, ml_idx, de, rk):
    log('\n' + '=' * 66); log('5. HELD-OUT VALIDATION WITH MnTBAP GROUPS'); log('=' * 66)
    X = L[ml_idx]
    ab = np.isin(G_K, ['A', 'B'])
    y = (G_K[ab] == 'B').astype(int)
    pos = {gi: j for j, gi in enumerate(ml_idx)}
    sig_rows = np.array([pos[i] for i in rk['idx']])          # signature proteins in X

    rows = []
    for name, mk in classifiers().items():
        for label, idx in [('top50', select_top(X[:, ab], y, K_PRIMARY)), ('signature', sig_rows)]:
            m = make_pipeline(StandardScaler(), mk()).fit(X[idx][:, ab].T, y)
            sc = score_of(m, X[idx].T); pr = m.predict(X[idx].T)
            for j, s in enumerate(SAMPLES_K):
                if not ab[j]:
                    rows.append({'classifier': name, 'features': label, 'sample': s,
                                 'group': GROUP_NAMES[s[0]], 'score': sc[j], 'cisplatin_like': int(pr[j])})
    calls = pd.DataFrame(rows)
    calls.to_csv(os.path.join(OUT_DIR, 'Table3_heldout_classifier_calls.csv'), index=False)
    summ = calls.groupby(['features', 'classifier', 'group'])['cisplatin_like'].sum().unstack()
    log(summ.to_string())

    def axis_scores(k):
        out = np.zeros(len(SAMPLES_K)); abpos = np.where(ab)[0]
        for j in range(len(SAMPLES_K)):
            if ab[j]:
                tr = abpos[abpos != j]
            else:
                tr = abpos
            ytr = (G_K[tr] == 'B').astype(int)
            idx = select_top(X[:, tr], ytr, k)
            mA = X[idx][:, tr[ytr == 0]].mean(1); mB = X[idx][:, tr[ytr == 1]].mean(1)
            dlt = mB - mA
            out[j] = np.sum(dlt * (X[idx, j] - mA)) / np.sum(dlt ** 2)
        return out

    ax = {'top50': axis_scores(K_PRIMARY), 'top150': axis_scores(150)}
    axis_df = pd.DataFrame({'sample': SAMPLES_K, 'group': [GROUP_NAMES[g] for g in G_K],
                            'evaluation': np.where(ab, 'leave-one-out', 'held-out'),
                            'axis_top50': ax['top50'], 'axis_top150': ax['top150']})
    axis_df.to_csv(os.path.join(OUT_DIR, 'TableS8_cisplatin_axis_scores.csv'), index=False)
    axis_stats = {}
    for col in ['axis_top50', 'axis_top150']:
        gm = {g: axis_df.loc[G_K == g, col].values for g in GROUPS}
        axis_stats[col] = {'mean': {g: float(v.mean()) for g, v in gm.items()},
                           'sd': {g: float(v.std(ddof=1)) for g, v in gm.items()},
                           'p_C_vs_B': float(stats.ttest_ind(gm['C'], gm['B']).pvalue),
                           'p_C_vs_A': float(stats.ttest_ind(gm['C'], gm['A']).pvalue),
                           'p_D_vs_A': float(stats.ttest_ind(gm['D'], gm['A']).pvalue),
                           'p_D_vs_B': float(stats.ttest_ind(gm['D'], gm['B']).pvalue),
                           'fraction_back_toward_control': float((gm['B'].mean() - gm['C'].mean()) /
                                                                 (gm['B'].mean() - gm['A'].mean()))}
        log(f'{col}: ' + ', '.join(f'{GROUP_NAMES[g]} {gm[g].mean():.2f}±{gm[g].std(ddof=1):.2f}'
                                   for g in GROUPS) +
            f' | C vs B p={axis_stats[col]["p_C_vs_B"]:.3f}, C vs A p={axis_stats[col]["p_C_vs_A"]:.3f}, '
            f'back toward control {100 * axis_stats[col]["fraction_back_toward_control"]:.0f}%')

    # protein-level, descriptive, on the signature
    sub = de.iloc[rk['idx'].values].copy()
    lba, lca = sub['log2FC_BvsA'].values, sub['log2FC_CvsA'].values
    slope = float(np.sum(lba * lca) / np.sum(lba ** 2))
    r_p, p_p = stats.pearsonr(lba, lca)
    rec = 1 - lca / lba
    cat = np.select([rec >= 0.75, rec >= 0.25, rec >= 0], ['Complete (>=75%)', 'Partial (25-75%)',
                                                           'Minimal (0-25%)'], 'Exacerbated (<0%)')
    sub['recovery_fraction'] = rec; sub['recovery_category'] = cat
    sub['sig_reversed_CvsB'] = (sub['q_CvsB'] < FDR_DE) & (np.sign(sub['log2FC_CvsB']) == -np.sign(lba))
    sub.to_csv(os.path.join(OUT_DIR, 'TableS9_signature_recovery.csv'), index=False)
    prot_level = {'n': int(len(sub)), 'slope': slope, 'pearson_r': float(r_p), 'pearson_p': float(p_p),
                  'reference_heldout_cisplatin': float(axis_df.loc[G_K == 'B', 'axis_top150'].mean()),
                  'reference_heldout_control': float(axis_df.loc[G_K == 'A', 'axis_top150'].mean()),
                  'median_recovery': float(np.median(rec)),
                  'categories': {k: int(v) for k, v in pd.Series(cat).value_counts().items()},
                  'n_sig_reversed_CvsB': int(sub['sig_reversed_CvsB'].sum())}
    log(f'Signature protein level: slope {slope:.2f}, r {r_p:.2f}, held-out cisplatin reference '
        f'{prot_level["reference_heldout_cisplatin"]:.2f}; categories {prot_level["categories"]}; '
        f'significantly reversed (FDR<0.05) {prot_level["n_sig_reversed_CvsB"]}')
    return {'calls': calls, 'axis_df': axis_df, 'axis_stats': axis_stats, 'prot_level': prot_level,
            'recovery_table': sub}


# ------------------------------------------------------------------ 6. MnTBAP+Cisplatin vs Cisplatin
def combination_effect(de, counts):
    log('\n' + '=' * 66); log('6. MnTBAP+CISPLATIN vs CISPLATIN (strongest contrast)'); log('=' * 66)
    s = de[de.sig_CvsB != 'NS'].copy()
    up = s[s.sig_CvsB == 'Up'].sort_values('p_CvsB'); dn = s[s.sig_CvsB == 'Down'].sort_values('p_CvsB')
    # how many of these also differ from Control, and is MnTBAP alone in the same direction?
    both = s[(s.q_CvsA < FDR_DE)]
    same_dir_D = np.mean(np.sign(s.log2FC_DvsA) == np.sign(s.log2FC_CvsB))
    # where does the C vs B difference come from: reversal of cisplatin, or new change?
    rev = s[np.sign(s.log2FC_BvsA) == -np.sign(s.log2FC_CvsB)]
    out = {'n': int(len(s)), 'up': int(len(up)), 'down': int(len(dn)),
           'also_FDR005_vs_control': int(len(both)),
           'frac_same_direction_as_MnTBAP_alone': float(same_dir_D),
           'n_opposite_to_cisplatin_change': int(len(rev)),
           'top_up': up.head(15)['Gene'].tolist(), 'top_down': dn.head(15)['Gene'].tolist(),
           'corr_CvsB_with_DvsA': float(stats.pearsonr(de['log2FC_CvsB'].values, de['log2FC_DvsA'].values)[0]),
           'corr_CvsA_with_DvsA': float(stats.pearsonr(de['log2FC_CvsA'].values, de['log2FC_DvsA'].values)[0]),
           'corr_BvsA_with_DvsA': float(stats.pearsonr(de['log2FC_BvsA'].values, de['log2FC_DvsA'].values)[0])}
    log(f'{out["n"]} proteins ({out["up"]} up, {out["down"]} down) at FDR<0.05; also FDR<0.05 vs control '
        f'{out["also_FDR005_vs_control"]}; same direction as MnTBAP alone {100 * same_dir_D:.0f}%; '
        f'opposite to the cisplatin change {out["n_opposite_to_cisplatin_change"]}')
    log(f'Correlation of log2FC: (MnTBAP+Cis vs Cis) with (MnTBAP vs Ctrl) r = {out["corr_CvsB_with_DvsA"]:.2f}; '
        f'(MnTBAP+Cis vs Ctrl) with (MnTBAP vs Ctrl) r = {out["corr_CvsA_with_DvsA"]:.2f}; '
        f'(Cis vs Ctrl) with (MnTBAP vs Ctrl) r = {out["corr_BvsA_with_DvsA"]:.2f}')
    log(f'Top increased: {", ".join(out["top_up"])}')
    log(f'Top decreased: {", ".join(out["top_down"])}')
    return out


# ------------------------------------------------------------------ 7. four-group classification
def multiclass_ml(L, ml_idx, n_feat=300):
    log('\n' + '=' * 66); log(f'7. FOUR-GROUP CLASSIFICATION (nested LOO, n = {len(SAMPLES_K)})'); log('=' * 66)
    X = L[ml_idx]; y = np.array([GROUPS.index(g) for g in G_K]); n = len(y)
    out = {}
    for name, mk in classifiers(include_signature=False).items():
        pred = np.zeros(n, int)
        for i in range(n):
            tr = np.arange(n) != i
            F, _ = f_classif(X[:, tr].T, y[tr])
            idx = np.argsort(np.nan_to_num(F, nan=-1))[::-1][:n_feat]
            m = make_pipeline(StandardScaler(), mk()).fit(X[idx][:, tr].T, y[tr])
            pred[i] = m.predict(X[idx][:, [i]].T)[0]
        cm = confusion_matrix(y, pred, labels=range(4))
        out[name] = {'accuracy': accuracy_score(y, pred), 'pred': pred, 'cm': cm,
                     'recall': {GROUPS[i]: float(cm[i, i] / cm[i].sum()) for i in range(4)}}
        log(f'{name:20s} accuracy {out[name]["accuracy"]:.3f}; recall ' +
            ', '.join(f'{GROUP_NAMES[g]} {v:.2f}' for g, v in out[name]['recall'].items()))
    pd.DataFrame([{'Classifier': n_, 'Accuracy': r['accuracy'],
                   **{f'Recall {GROUP_NAMES[g]}': v for g, v in r['recall'].items()}}
                  for n_, r in out.items()]).to_csv(os.path.join(OUT_DIR, 'TableS10_four_group.csv'), index=False)
    return out


# ------------------------------------------------------------------ 8. unsupervised
def unsupervised(L, Lall, ml_idx, rk):
    log('\n' + '=' * 66); log('8. UNSUPERVISED STRUCTURE'); log('=' * 66)
    X = L[ml_idx]
    pca = PCA(svd_solver='full').fit(StandardScaler().fit_transform(X.T))
    pcs = pca.transform(StandardScaler().fit_transform(X.T))
    # QC PCA including the excluded channel
    comp16 = ~np.isnan(Lall).any(1)
    Xa = Lall[comp16]
    pca16 = PCA(n_components=3, svd_solver='full').fit(StandardScaler().fit_transform(Xa.T))
    pcs16 = pca16.transform(StandardScaler().fit_transform(Xa.T))
    # discovery PCA on signature, fit on A/B, project C/D
    ab = np.isin(G_K, ['A', 'B'])
    S = L[rk['idx'].values]
    sc = StandardScaler().fit(S[:, ab].T)
    pab = PCA(n_components=2, svd_solver='full').fit(sc.transform(S[:, ab].T))
    proj = pab.transform(sc.transform(S.T))
    if proj[G_K == 'B', 0].mean() < proj[G_K == 'A', 0].mean():
        proj[:, 0] *= -1
    Z = X - X.mean(1, keepdims=True)
    log(f'PCA (all {len(ml_idx)} complete proteins): PC1 {100 * pca.explained_variance_ratio_[0]:.1f}%, '
        f'PC2 {100 * pca.explained_variance_ratio_[1]:.1f}%; discovery PCA PC1 {100 * pab.explained_variance_ratio_[0]:.1f}%')
    return {'pcs': pcs, 'var': pca.explained_variance_ratio_, 'pcs16': pcs16,
            'var16': pca16.explained_variance_ratio_, 'n16': int(comp16.sum()),
            'proj': proj, 'var_ab': pab.explained_variance_ratio_, 'sample_corr': np.corrcoef(Z.T)}


# ------------------------------------------------------------------ 9. co-expression
def coexpression(L, de, rk):
    log('\n' + '=' * 66); log('9. CO-EXPRESSION OF THE CISPLATIN SIGNATURE (15 samples)'); log('=' * 66)
    idx = rk['idx'].values
    Z = L[idx]; Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True)
    C = np.corrcoef(Z)
    D = np.clip(1 - C, 0, 2); np.fill_diagonal(D, 0)
    link = linkage(squareform(D, checks=False), 'average')
    best = None
    for k in range(2, 9):
        lab = fcluster(link, k, 'maxclust')
        if np.min(np.bincount(lab)[1:]) < 5:
            continue
        s = silhouette_score(D, lab, metric='precomputed')
        if best is None or s > best[1]:
            best = (k, s, lab)
    k, sil, lab = best
    order = pd.Series(lab).value_counts().index.tolist()
    lab = np.array([{o: i + 1 for i, o in enumerate(order)}[x] for x in lab])
    A_adj = np.abs(C) ** 6; np.fill_diagonal(A_adj, 0)
    eig, rows, mstats = {}, [], []
    for m in range(1, k + 1):
        mi = np.where(lab == m)[0]
        u, s, vt = np.linalg.svd(Z[mi], full_matrices=False)
        e = vt[0] * np.sign(np.corrcoef(vt[0], Z[mi].mean(0))[0, 1]); eig[m] = e
        gm = {g: e[G_K == g] for g in GROUPS}
        kin = A_adj[np.ix_(mi, mi)].sum(1)
        mstats.append({'module': f'M{m}', 'n_proteins': len(mi),
                       'n_up_cisplatin': int(np.sum(rk['log2FC_BvsA'].values[mi] > 0)),
                       'n_down_cisplatin': int(np.sum(rk['log2FC_BvsA'].values[mi] < 0)),
                       'var_explained': float(s[0] ** 2 / np.sum(s ** 2)),
                       'p_B_vs_A': stats.ttest_ind(gm['B'], gm['A']).pvalue,
                       'p_C_vs_B': stats.ttest_ind(gm['C'], gm['B']).pvalue,
                       'p_C_vs_A': stats.ttest_ind(gm['C'], gm['A']).pvalue,
                       'p_D_vs_A': stats.ttest_ind(gm['D'], gm['A']).pvalue,
                       'recovery': float((gm['B'].mean() - gm['C'].mean()) / (gm['B'].mean() - gm['A'].mean())),
                       **{f'eig_mean_{g}': float(v.mean()) for g, v in gm.items()}})
        for j, kk in zip(mi, kin):
            rows.append({'module': f'M{m}', 'Gene': rk['Gene'].values[j], 'Accession': rk['Accession'].values[j],
                         'Protein': rk['Protein'].values[j], 'log2FC_BvsA': rk['log2FC_BvsA'].values[j],
                         'kME': float(np.corrcoef(Z[j], e)[0, 1]), 'k_within': float(kk)})
    mod_df = pd.DataFrame(rows).sort_values(['module', 'k_within'], ascending=[True, False])
    mstats = pd.DataFrame(mstats)
    mod_df.to_csv(os.path.join(OUT_DIR, 'TableS11_modules.csv'), index=False)
    mstats.to_csv(os.path.join(OUT_DIR, 'Table4_module_summary.csv'), index=False)
    log(f'{k} modules, silhouette {sil:.2f}')
    log(mstats[['module', 'n_proteins', 'n_up_cisplatin', 'n_down_cisplatin', 'p_C_vs_B', 'p_D_vs_A',
                'recovery']].round(4).to_string())
    for m in range(1, k + 1):
        log(f'  M{m} hubs: {", ".join(mod_df[mod_df.module == f"M{m}"].head(8).Gene)}')
    return {'labels': lab, 'corr': C, 'linkage': link, 'eig': eig, 'mod_df': mod_df, 'mod_stats': mstats,
            'k': k, 'silhouette': float(sil)}


# ------------------------------------------------------------------ 10. tissue composition
PANELS = {
    'Synaptic/neuronal': ['Syp', 'Syn1', 'Syn2', 'Snap25', 'Stx1a', 'Stx1b', 'Stxbp1', 'Vamp2', 'Syt1', 'Syt2',
                          'Rab3a', 'Cplx1', 'Cplx2', 'Dlg4', 'Gria1', 'Gria2', 'Gria3', 'Gria4', 'Grin1', 'Camk2a',
                          'Nefl', 'Nefm', 'Nefh', 'Tubb3', 'Map2', 'Gap43', 'Bsn', 'Pclo', 'Rims1', 'Ctbp2', 'Otof',
                          'Slc17a8', 'Atp1a3', 'Nsf', 'Napa', 'Sv2a', 'Syngr1'],
    'Bone/ECM': ['Col1a1', 'Col1a2', 'Ibsp', 'Spp1', 'Bglap', 'Bglap2', 'Sparc', 'Dcn', 'Bgn', 'Omd', 'Ogn', 'Aspn',
                 'Postn', 'Col2a1', 'Col11a1', 'Cilp', 'Crtap', 'Tnfrsf11b', 'Mepe', 'Dmp1', 'Phex', 'Alpl',
                 'Ifitm5', 'Col5a1', 'Col12a1'],
    'Blood/plasma': ['Hba', 'Hba-a1', 'Hbb-b1', 'Hbb-bs', 'Hbb-b2', 'Alb', 'Apoa1', 'Apoa2', 'Apoh', 'Trf', 'Hpx',
                     'Serpina1a', 'Serpina1b', 'Serpina1c', 'Serpina3k', 'Fga', 'Fgb', 'Fgg', 'C3', 'Mug1', 'Ighm',
                     'Igkc', 'Plg', 'Ahsg', 'Kng1', 'Hrg', 'Itih4', 'Afm', 'Klkb1', 'Car2', 'Bpgm', 'Slc4a1',
                     'Spta1', 'Sptb', 'Gc', 'Apob', 'A2m', 'F2', 'Cfh', 'Hp', 'Prdx2', 'Alad'],
    'Marrow/immune': ['Ptprc', 'Lcp1', 'Coro1a', 'Ptpn6', 'Plek', 'Itgb2', 'Lyz2', 'Lyz1', 'S100a8', 'S100a9', 'Mpo',
                      'Ltf', 'Camp', 'Ngp', 'Chil3', 'Cd74', 'H2-Aa', 'H2-Ab1', 'Arhgdib', 'Lsp1', 'Was', 'Hcls1',
                      'Elane', 'Prtn3', 'Ctsg', 'Lcn2', 'Cd177', 'Itga2b', 'Gp1bb', 'Pf4', 'Ppbp', 'Fermt3', 'Myh9',
                      'Lcp2', 'Rac2', 'Dock2', 'Samhd1', 'Stk4'],
    'Myelin/glia': ['Mbp', 'Plp1', 'Mpz', 'Pmp22', 'Mag', 'Mog', 'Prx', 'Ncmap', 'Cnp', 'Pllp', 'Gfap', 'S100b'],
    'Nuclear': ['Lmna', 'Lmnb1', 'Lmnb2', 'Tmpo', 'Lbr', 'Nup93', 'Nup98', 'Nup155', 'Npm1', 'Ncl', 'Hnrnpa1',
                'Hnrnpa2b1', 'Hnrnpc', 'Hnrnpk', 'Hnrnpu', 'Sfpq', 'Nono', 'Top1', 'Top2b', 'Parp1', 'Pcna',
                'Smarca5', 'Hmgb1', 'Hmgb2', 'Matr3', 'Safb'],   # plus all quantified histones (added below)
}
HISTONE_RE = re.compile(r'^(H1-\d|H1f|H2a[cwxz]|H2az|H2b[cw]|H3c|H3-\d|H3f3|H4c|H4f|Hist|Macroh2a)')


def composition(L, ml_idx, de, uns, bin_out, net=None, rk=None):
    from limma_stats import moderated_t_design
    log('\n' + '=' * 66); log('10. TISSUE COMPOSITION OF THE PREPARATIONS'); log('=' * 66)
    genes = de['Gene'].values
    X = L[ml_idx]
    pc1 = uns['pcs'][:, 0].copy()
    scores, members = {}, {}
    for name, gl in PANELS.items():
        rows = [j for j, i in enumerate(ml_idx) if genes[i] in gl or
                (name == 'Nuclear' and HISTONE_RE.match(str(genes[i])) is not None)]
        Z = X[rows]; Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True)
        scores[name] = np.median(Z, 0); members[name] = [genes[ml_idx[j]] for j in rows]
    # orient PC1 so that positive = synapse-enriched
    if np.corrcoef(pc1, scores['Synaptic/neuronal'])[0, 1] < 0:
        pc1 = -pc1
    panel_stats = []
    for name, s in scores.items():
        gm = {g: s[G_K == g] for g in GROUPS}
        panel_stats.append({'panel': name, 'n_markers': len(members[name]),
                            **{f'mean_{g}': float(v.mean()) for g, v in gm.items()},
                            'p_B_vs_A': stats.ttest_ind(gm['B'], gm['A']).pvalue,
                            'p_C_vs_B': stats.ttest_ind(gm['C'], gm['B']).pvalue,
                            'p_C_vs_A': stats.ttest_ind(gm['C'], gm['A']).pvalue,
                            'p_D_vs_A': stats.ttest_ind(gm['D'], gm['A']).pvalue,
                            'r_with_PC1': float(np.corrcoef(s, pc1)[0, 1])})
    panel_stats = pd.DataFrame(panel_stats)
    log(panel_stats.round(4).to_string(index=False))
    sc_df = pd.DataFrame(scores, index=SAMPLES_K); sc_df['PC1_synaptic_axis'] = pc1
    sc_df.to_csv(os.path.join(OUT_DIR, 'TableS14_composition_scores.csv'), index_label='sample')
    panel_stats.to_csv(os.path.join(OUT_DIR, 'Table5_composition_panels.csv'), index=False)
    pc1_group_p = float(stats.f_oneway(*[pc1[G_K == g] for g in GROUPS]).pvalue)

    # PC1 loadings
    Xs = StandardScaler().fit_transform(X.T)
    pca = PCA(n_components=2, svd_solver='full').fit(Xs)
    load = pca.components_[0] * np.sign(np.corrcoef(pca.transform(Xs)[:, 0], pc1)[0, 1])
    o = np.argsort(load)
    loadings = {'synaptic_side': [genes[ml_idx[j]] for j in o[::-1][:25]],
                'contamination_side': [genes[ml_idx[j]] for j in o[:25]]}
    log(f'PC1 synaptic side: {", ".join(loadings["synaptic_side"][:15])}')
    log(f'PC1 other side:    {", ".join(loadings["contamination_side"][:15])}')

    # adjusted differential abundance (complete-case proteins)
    Xg = np.column_stack([(G_K == g).astype(float) for g in GROUPS])
    cons = {'BvsA': [-1, 1, 0, 0], 'CvsA': [-1, 0, 1, 0], 'CvsB': [0, -1, 1, 0], 'DvsA': [-1, 0, 0, 1],
            'CvsD': [0, 0, 1, -1]}
    un, _ = moderated_t_design(X, Xg, cons)
    adj, prior_adj = moderated_t_design(X, np.column_stack([Xg, pc1]), {k: v + [0] for k, v in cons.items()})
    adj_rows, adj_tab = [], de.iloc[ml_idx][['Accession', 'Gene', 'Protein']].copy()
    for k in cons:
        for lab, r in [('unadjusted', un[k]), ('PC1-adjusted', adj[k])]:
            q = bh(r['p'])
            adj_rows.append({'contrast': k, 'model': lab, 'FDR005': int(np.sum(q < 0.05)),
                             'FDR020': int(np.sum(q < 0.20)), 'p001': int(np.sum(r['p'] < 0.01)),
                             'frac_p005': float(np.mean(r['p'] < 0.05)),
                             'pi0': float(min(1, np.mean(r['p'] > 0.5) / 0.5))})
        adj_tab[f'log2FC_{k}_adj'] = adj[k]['logFC']; adj_tab[f'p_{k}_adj'] = adj[k]['p']
        adj_tab[f'q_{k}_adj'] = bh(adj[k]['p'])
    adj_df = pd.DataFrame(adj_rows)
    # covariate sensitivity: how strongly PC1 tracks group, and alternative covariates for Cisplatin vs Control
    gmean = np.array([pc1[G_K == g].mean() for g in GROUPS])[[GROUPS.index(g) for g in G_K]]
    pc1_group_r2 = float(1 - np.sum((pc1 - gmean) ** 2) / np.sum((pc1 - pc1.mean()) ** 2))
    pc1_resid = pc1 - gmean
    cov_rows = []
    for lab, cov in [('unadjusted', None), ('PC1 (as in main analysis)', pc1),
                     ('PC1 residualized on group (group-orthogonal)', pc1_resid),
                     ('Synaptic marker score', scores['Synaptic/neuronal'])]:
        if cov is None:
            r = un['BvsA']
        else:
            r = moderated_t_design(X, np.column_stack([Xg, cov]), {'BvsA': [-1, 1, 0, 0, 0]})[0]['BvsA']
        q = bh(r['p'])
        cov_rows.append({'covariate': lab, 'FDR005': int(np.sum(q < 0.05)), 'FDR020': int(np.sum(q < 0.20)),
                         'p001': int(np.sum(r['p'] < 0.01)), 'frac_p005': float(np.mean(r['p'] < 0.05)),
                         'pi0': float(min(1, np.mean(r['p'] > 0.5) / 0.5))})
    cov_df = pd.DataFrame(cov_rows); cov_df.insert(0, 'contrast', 'Cisplatin vs Control')
    cov_df['PC1_group_R2'] = pc1_group_r2
    cov_df.to_csv(os.path.join(OUT_DIR, 'TableS16_covariate_sensitivity.csv'), index=False)
    log(f'PC1 variance explained by group: R2 = {pc1_group_r2:.3f}')
    log(cov_df.round(3).to_string(index=False))
    adj_df.to_csv(os.path.join(OUT_DIR, 'Table6_adjusted_DE_summary.csv'), index=False)
    adj_tab.to_csv(os.path.join(OUT_DIR, 'TableS15_PC1_adjusted_DE.csv'), index=False)
    log(adj_df.round(3).to_string(index=False))
    robust = {}
    for k in ['DvsA', 'CvsB', 'CvsA', 'BvsA']:
        q = adj_tab[f'q_{k}_adj'].values; lf = adj_tab[f'log2FC_{k}_adj'].values; o2 = np.argsort(adj_tab[f'p_{k}_adj'].values)
        robust[k] = {'up': [adj_tab['Gene'].values[i] for i in o2 if q[i] < 0.05 and lf[i] > 0][:20],
                     'down': [adj_tab['Gene'].values[i] for i in o2 if q[i] < 0.05 and lf[i] < 0][:20]}

    # correlation of each signature protein with the composition axis (added to Tables 2 and S7)
    if rk is not None:
        rk['r_PC1'] = [float(np.corrcoef(L[i], pc1)[0, 1]) for i in rk['idx']]
        rk.drop(columns='idx').to_csv(os.path.join(OUT_DIR, 'TableS7_consensus_ranking.csv'), index=False)
        rk.head(20)[['consensus_rank', 'Gene', 'Accession', 'Protein', 'FC_BvsA', 'p_BvsA', 'q_BvsA', 'stability',
                     'consensus_score', 'r_PC1', 'recovery_fraction']].to_csv(
            os.path.join(OUT_DIR, 'Table2_top20_consensus.csv'), index=False)

    # increased-signature module (M1) after adjustment for PC1 and nuclear content
    m1_check = None
    if net is not None:
        m1 = set(net['mod_df'].loc[net['mod_df'].module == 'M1', 'Accession'])
        rows_m1 = [j for j, i in enumerate(ml_idx) if de['Accession'].values[i] in m1]
        m1_check = {}
        for lab, cov in [('PC1', [pc1]), ('Nuclear', [scores['Nuclear']]), ('PC1+Nuclear', [pc1, scores['Nuclear']])]:
            a, _ = moderated_t_design(X, np.column_stack([Xg] + cov), {'BvsA': [-1, 1, 0, 0] + [0] * len(cov)})
            lf = a['BvsA']['logFC'][rows_m1]; pp = a['BvsA']['p'][rows_m1]
            m1_check[lab] = {'n': len(rows_m1), 'mean_log2FC': float(lf.mean()), 'frac_up': float(np.mean(lf > 0)),
                             'n_p005': int(np.sum(pp < 0.05)), 'proteome_frac_p005': float(np.mean(a['BvsA']['p'] < 0.05)),
                             'proteome_FDR020': int(np.sum(bh(a['BvsA']['p']) < 0.20))}
        log(f'M1 after adjustment: {m1_check}')

    # classifiers versus composition
    ab = np.isin(G_K, ['A', 'B']); y = bin_out['y']
    ps = bin_out['per_sample']
    clf_r = {c.replace(' score', ''): float(np.corrcoef(ps[c], pc1[ab])[0, 1]) for c in ps.columns if c.endswith(' score')}
    pred = []
    for i in range(8):
        tr = np.arange(8) != i
        m0, m1 = pc1[ab][tr][y[tr] == 0].mean(), pc1[ab][tr][y[tr] == 1].mean()
        pred.append(int((pc1[ab][i] - (m0 + m1) / 2) * np.sign(m1 - m0) > 0))
    pc1_only_acc = float(np.mean(np.array(pred) == y))
    Xc = X - X.mean(1, keepdims=True)
    Xr = (X - np.outer(Xc @ pc1 / np.sum(pc1 ** 2), pc1))[:, ab]
    perms = [np.array([1 if j in c else 0 for j in range(8)]) for c in itertools.combinations(range(8), 4)]
    removed = {}
    for name in ['Signature score', 'Logistic regression', 'Linear SVM']:
        mk = classifiers()[name]
        s, pr, _ = nested_loo(Xr, y, mk); m = metrics(y, pr, s)
        acc, auc = [], []
        for yp in perms:
            s2, pr2, _ = nested_loo(Xr, yp, mk); mm = metrics(yp, pr2, s2)
            acc.append(mm['accuracy']); auc.append(mm['auroc'])
        m['perm_p_accuracy'] = float(np.mean(np.array(acc) >= m['accuracy'] - 1e-12))
        m['perm_p_auroc'] = float(np.mean(np.array(auc) >= m['auroc'] - 1e-12))
        removed[name] = m
        log(f'PC1 removed, {name:20s} acc {m["accuracy"]:.3f} (p {m["perm_p_accuracy"]:.3f}) AUROC '
            f'{m["auroc"]:.3f} (p {m["perm_p_auroc"]:.3f})')
    log(f'Classifier scores vs PC1: ' + ', '.join(f'{k} r={v:.2f}' for k, v in clf_r.items()) +
        f'; PC1 alone LOO accuracy {pc1_only_acc:.3f}; PC1 group ANOVA p {pc1_group_p:.3f}')
    return {'scores': scores, 'members': members, 'panel_stats': panel_stats, 'pc1': pc1,
            'pc1_group_p': pc1_group_p, 'loadings': loadings, 'adj_summary': adj_df, 'robust': robust,
            'clf_r_pc1': clf_r, 'pc1_only_acc': pc1_only_acc, 'pc1_removed_ml': removed,
            'prior_adj': prior_adj, 'm1_check': m1_check}


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    prot, Lall, L, ml_idx, info = load_and_qc()
    de, counts = differential(prot, L, ml_idx)
    cache = os.path.join(OUT_DIR, 'bin_cache.pkl')
    if os.path.exists(cache) and os.environ.get('P1_RECOMPUTE') != '1':
        log('\n(3. classification loaded from cache; set P1_RECOMPUTE=1 to recompute)')
        bin_out = pickle.load(open(cache, 'rb'))
    else:
        bin_out = binary_ml(L, ml_idx, de)
        pickle.dump(bin_out, open(cache, 'wb'))
    rk = biomarker_ranking(L, ml_idx, de)
    val = mntbap_validation(L, ml_idx, de, rk)
    comb = combination_effect(de, counts)
    multi = multiclass_ml(L, ml_idx)
    uns = unsupervised(L, Lall, ml_idx, rk)
    net = coexpression(L, de, rk)
    compo = composition(L, ml_idx, de, uns, bin_out, net, rk)

    results = dict(prot=prot, L=L, Lall=Lall, ml_idx=ml_idx, info=info, de=de, counts=counts, bin=bin_out,
                   rk=rk, val=val, comb=comb, multi=multi, uns=uns, net=net, compo=compo, SAMPLES=SAMPLES,
                   SAMPLES_K=SAMPLES_K, G_K=G_K, KEEP=KEEP, GROUP_OF=GROUP_OF)
    pickle.dump(results, open(os.path.join(OUT_DIR, 'results.pkl'), 'wb'))

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, np.ndarray): return None
        if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
        return o

    summary = {
        'load': {k: v for k, v in info.items()}, 'de_counts': counts,
        'binary_nested': {n: {k: v for k, v in r.items() if k not in ('score', 'pred')}
                          for n, r in bin_out['res'].items()},
        'binary_selection_outside_cv': {n: {k: v for k, v in r.items() if k not in ('score', 'pred')}
                                        for n, r in bin_out['leaky'].items()},
        'fold_overlap': bin_out['fold_overlap'],
        'per_sample_correct': dict(zip(bin_out['per_sample']['sample'],
                                       bin_out['per_sample']['n_classifiers_correct'])),
        'signature_n': int(de.signature.sum()), 'signature_n_complete': int(len(rk)),
        'stability_ge_075': int((rk.stability >= 0.75).sum()), 'stability_eq_1': int((rk.stability >= 0.999).sum()),
        'top20': rk.head(20)[['Gene', 'FC_BvsA', 'p_BvsA', 'q_BvsA', 'stability', 'consensus_score',
                              'recovery_fraction']].to_dict('records'),
        'heldout_calls': val['calls'].groupby(['features', 'classifier', 'group'])['cisplatin_like'].sum()
                                     .unstack().reset_index().to_dict('records'),
        'axis_stats': val['axis_stats'], 'protein_level': val['prot_level'], 'combination': comb,
        'four_group': {n: {'accuracy': r['accuracy'], 'recall': r['recall']} for n, r in multi.items()},
        'pca_var': uns['var'][:5].tolist(), 'pca16_var': uns['var16'].tolist(), 'pca_ab_var': uns['var_ab'].tolist(),
        'modules': net['mod_stats'].to_dict('records'), 'n_modules': net['k'], 'silhouette': net['silhouette'],
        'composition': {'panels': compo['panel_stats'].to_dict('records'), 'pc1_group_p': compo['pc1_group_p'],
                        'loadings': compo['loadings'], 'adjusted': compo['adj_summary'].to_dict('records'),
                        'robust_after_adjustment': compo['robust'], 'clf_r_pc1': compo['clf_r_pc1'],
                        'pc1_only_acc': compo['pc1_only_acc'], 'pc1_removed_ml': compo['pc1_removed_ml'],
                        'prior_adj': compo['prior_adj'], 'm1_check': compo['m1_check']},
    }
    json.dump(clean(summary), open(os.path.join(OUT_DIR, 'analysis_summary.json'), 'w'), indent=2, default=str)
    from relabel_outputs import relabel
    relabel(OUT_DIR)
    log(f'\nDone in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
