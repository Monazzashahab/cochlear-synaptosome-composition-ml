#!/usr/bin/env python3
"""Figures 1 to 6 and S1 to S5 (reads output/results.pkl and output/sensitivity.pkl)."""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_curve
from adjustText import adjust_text

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'output'); FIG = os.path.join(BASE, 'figures'); os.makedirs(FIG, exist_ok=True)
RES = pickle.load(open(os.path.join(OUT, 'results.pkl'), 'rb'))
SENS = pickle.load(open(os.path.join(OUT, 'sensitivity.pkl'), 'rb'))

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.8, 'xtick.labelsize': 7.8, 'ytick.labelsize': 7.8,
    'legend.fontsize': 7.2, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
GROUPS = ['A', 'B', 'C', 'D']
GN = {'A': 'Control', 'B': 'Cisplatin', 'C': 'MnTBAP+Cisplatin', 'D': 'MnTBAP'}
GC = {'A': '#2166AC', 'B': '#B2182B', 'C': '#1B9E77', 'D': '#7570B3'}
UP, DOWN, NS = '#C0392B', '#2E6DB4', '#C8C8C8'
CLF_C = {'Signature score': '#000000', 'Logistic regression': '#E69F00', 'Linear SVM': '#56B4E9',
         'Random forest': '#009E73', 'Gradient boosting': '#CC79A7'}
PANEL_C = {'Synaptic/neuronal': '#6A3D9A', 'Bone/ECM': '#B15928', 'Blood/plasma': '#E31A1C',
           'Marrow/immune': '#FF7F00', 'Myelin/glia': '#33A02C'}
de = RES['de']; L = RES['L']; ml = RES['ml_idx']; S = RES['SAMPLES_K']; G = RES['G_K']
# Sample nomenclature: group name + replicate number (short forms defined by KEY in every figure)
SHORT = {'A': 'Ctrl', 'B': 'Cis', 'C': 'M+C', 'D': 'Mn'}


def lab(s):
    return f'{SHORT[s[0]]} {s[1]}'


SL = [lab(s) for s in S]
LAB2G = {lab(s): s[0] for s in RES['SAMPLES']}
KEY = 'Ctrl = Control,  Cis = Cisplatin,  M+C = MnTBAP+Cisplatin,  Mn = MnTBAP  (number = replicate)'
info = RES['info']; counts = RES['counts']; compo = RES['compo']


def P(s):
    """Mouse protein symbol convention: uppercase, roman."""
    return str(s).upper()


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')


def save(fig, name, key=True):
    if key:
        fig.text(0.5, -0.01, KEY, ha='center', va='top', fontsize=7, style='italic', color='#333333')
    fig.savefig(os.path.join(FIG, name + '.png')); fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig); print('saved', name)


def group_handles(groups=GROUPS, marker='o'):
    return [Line2D([0], [0], marker=marker, ls='', color=GC[g], label=GN[g], ms=5.5) for g in groups]


def strip(ax, values, groups=G, rng_seed=1, label_samples=True, labels=SL, short=False):
    rng = np.random.default_rng(rng_seed)
    for i, g in enumerate(GROUPS):
        m = groups == g; v = values[m]; xs = i + rng.uniform(-0.12, 0.12, m.sum())
        ax.scatter(xs, v, c=GC[g], s=22, zorder=3)
        if label_samples:
            for xx, yy, nm in zip(xs, v, np.array(labels)[m]):
                ax.annotate(nm, (xx, yy), fontsize=5, xytext=(3, 0), textcoords='offset points', color='#666666',
                            va='center')
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color='black', lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Ctrl', 'Cis', 'M+C', 'Mn'] if short else ['Control', 'Cisplatin', 'MnTBAP\n+Cisplatin', 'MnTBAP'], fontsize=7)


# =============================================================== Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis('off')

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=col, ec='#333333', lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=8.2, fontweight='bold')
        ax.text(x + w / 2, y + h - 6.3, body, ha='center', va='top', fontsize=6.9, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#333333'))

    box(1, 46, 30, 18, 'Raw reporter abundances', 'TMT-16plex, cochlear synaptosomes\nControl | Cisplatin\nMnTBAP+Cisplatin | MnTBAP\n4 biological replicates per group', '#EAEAEA')
    box(35, 46, 30, 18, 'Channel QC and filtering',
        f'MnTBAP+Cisplatin 1 at {100 * info["C1_fraction_of_median_total"]:.0f}% loading:\nexcluded. High-confidence proteins\n'
        f'≥3 replicates in every group: {info["n_quant"]:,}\nlog2, median normalization', '#EAEAEA')
    box(69, 46, 30, 18, 'Differential abundance', 'Empirical Bayes moderated t\n(validated against limma)\nBenjamini-Hochberg FDR\nlabel-permutation null', '#FBE3E3')
    box(1, 22, 30, 19, 'Classification', 'Control vs Cisplatin (n = 8)\n5 classifiers, nested LOO\n(top 50 proteins re-selected\nin every fold), exact\npermutation test (70 labelings)', '#DCE9F5')
    box(35, 22, 30, 19, 'Tissue composition', 'Unsupervised PC1 and six\nmarker panels: synaptic, bone,\nblood, marrow, myelin, nuclear\ncomposition-adjusted models', '#F5EBDC')
    box(69, 22, 30, 19, 'Held-out projection', 'MnTBAP+Cisplatin (n = 3)\nMnTBAP (n = 4)\nnot used for training\nor top-50 selection', '#DDF1E8')
    box(18, 1, 30, 16, 'Consensus biomarkers', 'stability, LR, RF, Hedges g\non the FDR < 0.20 signature', '#E8EEF7')
    box(52, 1, 30, 16, 'Sensitivity analyses', 'sample influence, 35 partitions,\nreduced design,\nMnTBAP+Cisplatin 1 restored', '#EFE8F7')
    arrow(31.5, 55, 34.5, 55); arrow(65.5, 55, 68.5, 55)
    arrow(84, 45.5, 84, 41.5); arrow(16, 45.5, 16, 41.5); arrow(50, 45.5, 50, 41.5)
    arrow(31.5, 31.5, 34.5, 31.5); arrow(65.5, 31.5, 68.5, 31.5)
    arrow(40, 21.5, 34, 17.5); arrow(60, 21.5, 66, 17.5)
    save(fig, 'Figure1_workflow', key=False)


# =============================================================== Figure 2: QC and global signal
def fig2():
    glob = counts['global']; qc = pd.DataFrame(info['qc'])
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = gridspec.GridSpec(2, 3, hspace=0.62, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    col = ['#999999' if e else GC[s[0]] for s, e in zip(qc['sample'], qc['excluded'])]
    ax.bar(range(16), qc['total_abundance'] / 1e5, color=col)
    ax.axhline(qc['total_abundance'].median() * 0.25 / 1e5, color='black', ls='--', lw=0.7)
    ax.set_xticks(range(16)); ax.set_xticklabels([lab(x) for x in qc['sample']], rotation=90, fontsize=6.3)
    ax.set_ylabel('Total reporter abundance (×10$^5$)')
    ax.text(8, qc['total_abundance'].iloc[8] / 1e5 + 0.3, 'M+C 1 excluded', rotation=90, ha='center', va='bottom', fontsize=6.3)
    ax.set_title('Dashed line: QC threshold\n(25% of median channel)', fontsize=6.5)
    panel(ax, 'A', -0.32)

    ax = fig.add_subplot(gs[0, 1])
    x = de['log2FC_BvsA']; yv = -np.log10(de['p_BvsA'])
    sig = de['q_BvsA'] < 0.20
    c = np.where(sig & (x > 0), UP, np.where(sig & (x < 0), DOWN, NS))
    ax.scatter(x, yv, c=c, s=4, lw=0, alpha=0.8, rasterized=True)
    top = de[sig].nsmallest(10, 'p_BvsA')
    texts = [ax.text(r['log2FC_BvsA'], -np.log10(r['p_BvsA']), P(r['Gene']), fontsize=5.6) for _, r in top.iterrows()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', lw=0.3, color='#666666'))
    ax.set_xlabel('log$_2$ FC, Cisplatin / Control'); ax.set_ylabel('$-$log$_{10}$ p (moderated t)')
    ax.set_ylim(0, yv.max() * 1.12)
    ax.text(0.02, 0.02, f'FDR < 0.20: {counts["BvsA"]["q020_up"]}↑ {counts["BvsA"]["q020_down"]}↓\n'
                        f'FDR < 0.05: {counts["BvsA"]["q005_up"] + counts["BvsA"]["q005_down"]}',
            transform=ax.transAxes, va='bottom', fontsize=6.3)
    panel(ax, 'B', -0.3)

    ax = fig.add_subplot(gs[0, 2])
    h = np.array(glob['p_hist']); n = h.sum()
    ax.bar(np.linspace(0, 0.95, 20), h, width=0.05, align='edge', color='#7F8C8D', ec='white', lw=0.4)
    ax.axhline(n / 20, color='black', ls=':', lw=0.8, label='No protein changed')
    ax.axhline(counts['BvsA']['pi0'] * n / 20, color=UP, ls='--', lw=0.9, label=f'π0 = {counts["BvsA"]["pi0"]:.2f}')
    ax.legend(frameon=False, fontsize=6, loc='upper right')
    ax.set_xlabel('p value, Cisplatin vs Control'); ax.set_ylabel('Proteins')
    panel(ax, 'C', -0.3)

    ax = fig.add_subplot(gs[1, 0])
    nd = pd.DataFrame(glob['null'])
    rng = np.random.default_rng(1)
    for xpos, m, colr in [(0, nd.balanced, '#555555'), (1, ~nd.balanced, '#AAAAAA')]:
        v = nd.loc[m, 'p001'].values
        ax.scatter(xpos + rng.uniform(-0.15, 0.15, len(v)), v, s=12, c=colr, lw=0)
    ax.scatter([2], [counts['BvsA']['p001']], s=36, c=GC['B'], marker='D')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Balanced\nnull', 'Unbalanced', 'True\nlabels'], fontsize=6.5)
    ax.set_ylabel('Proteins at p < 0.01'); ax.set_xlim(-0.6, 2.6)
    panel(ax, 'D', -0.32)

    ax = fig.add_subplot(gs[1, 1:])
    pcs = RES['uns']['pcs']; v = RES['uns']['var']
    for g in GROUPS:
        m = G == g
        ax.scatter(pcs[m, 0], pcs[m, 1], c=GC[g], s=30, ec='white', lw=0.5, zorder=3)
        for i in np.where(m)[0]:
            ax.annotate(SL[i], (pcs[i, 0], pcs[i, 1]), fontsize=5.6, xytext=(3, 2), textcoords='offset points', color='#555555')
    ax.set_xlabel(f'PC1 ({100 * v[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({100 * v[1]:.1f}%)')
    ax.legend(handles=group_handles(), frameon=False, fontsize=6.3, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f'{len(ml):,} proteins, 15 retained samples', fontsize=7.5)
    panel(ax, 'E', -0.12)
    save(fig, 'Figure2_QC_global_signal')


# =============================================================== Figure 3: classification
def fig3():
    b = RES['bin']; res = b['res']; leaky = b['leaky']; y = b['y']
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = gridspec.GridSpec(2, 3, hspace=0.75, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    for name, r in res.items():
        fpr, tpr, _ = roc_curve(y, r['score'])
        ax.plot(fpr, tpr, color=CLF_C[name], lw=1.3, label=f'{name} ({r["auroc"]:.2f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=0.6)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.5, -0.25),
              title='Nested LOO (AUROC)', title_fontsize=6)
    panel(ax, 'A', -0.25)

    ax = fig.add_subplot(gs[0, 1:])
    names = list(res); x = np.arange(len(names)); w = 0.2
    ax.bar(x - 1.5 * w, [leaky[n]['accuracy'] for n in names], w, color='#BBBBBB', label='Accuracy, selection outside CV')
    ax.bar(x - 0.5 * w, [res[n]['accuracy'] for n in names], w, color='#333333', label='Accuracy, nested')
    ax.bar(x + 0.5 * w, [leaky[n]['auroc'] for n in names], w, color='#F2C6C6', label='AUROC, selection outside CV')
    ax.bar(x + 1.5 * w, [res[n]['auroc'] for n in names], w, color=UP, label='AUROC, nested')
    ax.axhline(0.5, color='grey', ls=':', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    ax.set_ylim(0, 1.25); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel('Score')
    ax.legend(frameon=False, ncol=2, fontsize=6.3, loc='upper center')
    panel(ax, 'B', -0.1)

    ax = fig.add_subplot(gs[1, 0])
    best = max(res, key=lambda n: res[n]['auroc'])
    pa = b['perm'][best]['auroc']
    ax.hist(pa, bins=np.linspace(0, 1, 17), color='#AAAAAA', ec='white')
    ax.axvline(res[best]['auroc'], color=UP, lw=1.5)
    ax.text(0.03, 0.97, f'{best}\nobserved {res[best]["auroc"]:.2f}\nexact p = {res[best]["perm_p_auroc"]:.3f}',
            transform=ax.transAxes, va='top', fontsize=6.3)
    ax.set_xlabel('Nested AUROC under\n70 label permutations'); ax.set_ylabel('Labelings')
    panel(ax, 'C', -0.25)

    ax = fig.add_subplot(gs[1, 1])
    sc = b['size_curve']; order = ['5', '10', '20', '50', '100', '200', '500', 'all']
    for name in sc['classifier'].unique():
        s = sc[sc.classifier == name].set_index('k').loc[order]
        ax.plot(range(len(order)), s['auroc'], 'o-', ms=3, lw=1, color=CLF_C[name], label=name)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=6.5, rotation=45)
    ax.axhline(0.5, color='grey', ls=':', lw=0.7); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Proteins per fold (top k)'); ax.set_ylabel('Nested AUROC')
    ax.legend(frameon=False, fontsize=5.5, loc='lower right')
    panel(ax, 'D', -0.25)

    ax = fig.add_subplot(gs[1, 2])
    s = res['Signature score']['score']
    ax.bar(range(8), s, color=[GC['B'] if v else GC['A'] for v in y])
    ax.axhline(0, color='black', lw=0.7)
    wrong = [i for i, (v, t) in enumerate(zip(res['Signature score']['pred'], y)) if v != t]
    ax.scatter(wrong, [s[i] + np.sign(s[i]) * 0.15 for i in wrong], marker='x', s=22, c='black', lw=1.2)
    ax.set_xticks(range(8)); ax.set_xticklabels([lab(x) for x in b['names']], fontsize=6.5, rotation=90)
    ax.set_ylabel('Held-out signature score\n(>0 = cisplatin-like)')
    ax.legend(handles=[Patch(color=GC['A'], label='Control'), Patch(color=GC['B'], label='Cisplatin'),
                       Line2D([0], [0], marker='x', ls='', color='black', label='Misclassified')],
              frameon=False, fontsize=6, loc='upper left')
    ax.set_ylim(min(s) - 0.4, max(s) + 0.9)
    panel(ax, 'E', -0.3)
    save(fig, 'Figure3_classification')


# =============================================================== Figure 4: composition
def fig4():
    sc = compo['scores']; pc1 = compo['pc1']; ps = compo['panel_stats']
    fig = plt.figure(figsize=(7.4, 9.0))
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.85, 1, 1], hspace=0.78, wspace=0.62)

    ax = fig.add_subplot(gs[0, :])
    names = list(sc)
    M = np.array([sc[n] for n in names])
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list('d', [DOWN, 'white', UP]), vmin=-1.6, vmax=1.6, aspect='auto')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f'{n} ({int(ps.loc[ps.panel == n, "n_markers"].iloc[0])})' for n in names], fontsize=6.8)
    ax.set_xticks(range(15)); ax.set_xticklabels(SL, fontsize=6.3, rotation=90)
    for xx in (3.5, 7.5, 10.5):
        ax.axvline(xx, color='black', lw=0.9)
    for xc, g in zip([1.5, 5.5, 9, 12.5], GROUPS):
        ax.text(xc, -0.8, GN[g], ha='center', fontsize=6.8, color=GC[g], fontweight='bold')
    for i, n in enumerate(names):
        r = ps.loc[ps.panel == n, 'r_with_PC1'].iloc[0]
        ax.text(14.7, i, f'  r(PC1) = {r:+.2f}', va='center', fontsize=6.2, color='#444444')
    ax.spines[:].set_visible(True)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.16); cb.set_label('Panel score (median z)', fontsize=6.3)
    cb.ax.tick_params(labelsize=6)
    panel(ax, 'A', -0.2, 1.1)

    ax = fig.add_subplot(gs[1, 0])
    strip(ax, pc1, short=True)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_ylabel('PC1 score (+ = synapse-enriched)', fontsize=7.5)
    ax.set_title(f'PC1, {100 * RES["uns"]["var"][0]:.1f}% of variance', fontsize=7.2)
    panel(ax, 'B', -0.42)

    ax = fig.add_subplot(gs[1, 1])
    ab = np.isin(G, ['A', 'B'])
    s = RES['bin']['res']['Signature score']['score']
    for g in 'AB':
        m = G[ab] == g
        ax.scatter(pc1[ab][m], s[m], c=GC[g], s=26, zorder=3)
    for xx, yy, nm in zip(pc1[ab], s, np.array(SL)[ab]):
        ax.annotate(nm, (xx, yy), fontsize=5.5, xytext=(3, 2), textcoords='offset points', color='#555555')
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel('PC1 score'); ax.set_ylabel('Held-out signature score')
    ax.set_title(f'r = {compo["clf_r_pc1"]["Signature"]:.2f}', fontsize=7.2)
    panel(ax, 'C', -0.42)

    adj = compo['adj_summary']
    cons = ['BvsA', 'CvsB', 'DvsA', 'CvsA', 'CvsD']
    lab = ['Cis vs Ctrl', 'M+C vs Cis', 'Mn vs Ctrl', 'M+C vs Ctrl', 'M+C vs Mn']
    for slot, (metric, ylab, lett) in [((1, 2), ('frac_p005', '% proteins p < 0.05', 'D')),
                                       ((2, 0), ('pi0', 'π0 (proportion unchanged)', 'E'))]:
        ax = fig.add_subplot(gs[slot])
        x = np.arange(len(cons)); w = 0.38
        u = [adj[(adj.contrast == c) & (adj.model == 'unadjusted')][metric].iloc[0] for c in cons]
        a = [adj[(adj.contrast == c) & (adj.model == 'PC1-adjusted')][metric].iloc[0] for c in cons]
        f = 100 if metric == 'frac_p005' else 1
        ax.bar(x - w / 2, np.array(u) * f, w, color='#888888', label='Group only')
        ax.bar(x + w / 2, np.array(a) * f, w, color='#B15928', label='+ PC1 covariate')
        if metric == 'frac_p005':
            ax.axhline(5, color='black', ls=':', lw=0.8); ax.set_ylim(0, 34)
        else:
            ax.axhline(1, color='black', ls=':', lw=0.8); ax.set_ylim(0, 1.25)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.3, rotation=60, ha='right')
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.legend(frameon=False, fontsize=6, loc='upper left', ncol=1)
        panel(ax, lett, -0.42)

    ax = fig.add_subplot(gs[2, 1:])
    ld = compo['loadings']
    left = [P(g) for g in ld['synaptic_side'][:12]]; right = [P(g) for g in ld['contamination_side'][:12]]
    ax.axis('off')
    ax.text(0.0, 1.0, 'PC1 loadings, synapse-enriched side', fontsize=7.2, fontweight='bold', color=PANEL_C['Synaptic/neuronal'], va='top')
    ax.text(0.52, 1.0, 'PC1 loadings, opposite side', fontsize=7.2, fontweight='bold', color=PANEL_C['Bone/ECM'], va='top')
    for i in range(12):
        ax.text(0.0 + (i // 6) * 0.25, 0.86 - (i % 6) * 0.14, f'{i + 1}. {left[i]}', fontsize=6.8, va='top')
        ax.text(0.52 + (i // 6) * 0.25, 0.86 - (i % 6) * 0.14, f'{i + 1}. {right[i]}', fontsize=6.8, va='top')
    panel(ax, 'F', -0.06, 1.0)
    save(fig, 'Figure4_tissue_composition')


# =============================================================== Figure 5: MnTBAP held-out and combination contrast
def fig5():
    v = RES['val']
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.55, 1.15], hspace=0.65, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    proj = RES['uns']['proj']; va = RES['uns']['var_ab']
    for g in GROUPS:
        m = G == g
        ax.scatter(proj[m, 0], proj[m, 1], c=GC[g], s=30, ec='white', lw=0.5, marker='o' if g in 'AB' else '^', zorder=3)
        for i in np.where(m)[0]:
            ax.annotate(SL[i], (proj[i, 0], proj[i, 1]), fontsize=5.5, xytext=(3, 2), textcoords='offset points', color='#555555')
    ax.set_xlabel(f'Discovery PC1 ({100 * va[0]:.0f}%)'); ax.set_ylabel(f'Discovery PC2 ({100 * va[1]:.0f}%)')
    h = [Line2D([0], [0], marker='o' if g in 'AB' else '^', ls='', color=GC[g],
                label=GN[g] + (' (fit)' if g in 'AB' else ' (projected)'), ms=5) for g in GROUPS]
    ax.legend(handles=h, frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2,
              handletextpad=0.1, columnspacing=0.6)
    panel(ax, 'A', -0.22)

    ax = fig.add_subplot(gs[0, 1])
    ad = v['axis_df']; st = v['axis_stats']['axis_top50']
    strip(ax, ad['axis_top50'].values)
    ax.axhline(0, color=GC['A'], ls=':', lw=0.8); ax.axhline(1, color=GC['B'], ls=':', lw=0.8)
    ax.set_xticklabels(['Control\n(LOO)', 'Cisplatin\n(LOO)', 'MnTBAP\n+Cisplatin\n(held out)', 'MnTBAP\n(held out)'], fontsize=6.3)
    ax.set_ylabel('Cisplatin-axis score\n(0 = Control, 1 = Cisplatin centroid)', fontsize=7.5)
    ax.text(0.02, 1.02, f'MnTBAP+Cis vs Cis p = {st["p_C_vs_B"]:.2f}; vs Control p = {st["p_C_vs_A"]:.2f}',
            transform=ax.transAxes, fontsize=6.2)
    panel(ax, 'B', -0.3)

    ax = fig.add_subplot(gs[1, :])
    calls = v['calls'][v['calls'].features == 'top50']
    piv = calls.pivot(index='classifier', columns='sample', values='cisplatin_like').loc[list(RES['bin']['res'])]
    ps_ = RES['bin']['per_sample']
    ab_rows = np.array([RES['bin']['res'][n]['pred'] for n in piv.index])
    full = np.column_stack([ab_rows, piv.values])
    cols = [lab(x) for x in list(ps_['sample']) + list(piv.columns)]
    ax.imshow(full, cmap=LinearSegmentedColormap.from_list('b', ['#F4F4F4', GC['B']]), aspect='auto', vmin=0, vmax=1)
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            if full[i, j]:
                ax.text(j, i, 'C', ha='center', va='center', fontsize=5.5, color='white')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6.5)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=6.5)
    for xx in (3.5, 7.5, 10.5):
        ax.axvline(xx, color='black', lw=1)
    for xc, g, t in [(1.5, 'A', 'nested LOO'), (5.5, 'B', 'nested LOO'), (9, 'C', 'held out'), (12.5, 'D', 'held out')]:
        ax.text(xc, -0.85, f'{GN[g]}, {t}', ha='center', fontsize=6.3, color=GC[g], fontweight='bold')
    ax.spines[:].set_visible(True)
    panel(ax, 'C', -0.16, 1.12)

    ax = fig.add_subplot(gs[2, 0])
    x = de['log2FC_CvsB']; yv = -np.log10(de['p_CvsB'])
    ax.scatter(x, yv, s=3, c='#D5D5D5', lw=0, rasterized=True)
    genes = de['Gene'].values
    for name in ['Bone/ECM', 'Blood/plasma', 'Synaptic/neuronal', 'Marrow/immune']:
        m = np.isin(genes, compo['members'][name])
        ax.scatter(x[m], yv[m], s=9, c=PANEL_C[name], lw=0, label=name, zorder=3)
    qthr = de.loc[de.q_CvsB < 0.05, 'p_CvsB'].max()
    ax.axhline(-np.log10(qthr), color='grey', ls='--', lw=0.6)
    ax.set_xlabel('log$_2$ FC, MnTBAP+Cisplatin / Cisplatin'); ax.set_ylabel('$-$log$_{10}$ p')
    ax.legend(frameon=False, fontsize=5.8, loc='upper left', markerscale=1.5)
    ax.text(0.98, 0.02, f'FDR < 0.05: {counts["CvsB"]["q005_up"]}↑ {counts["CvsB"]["q005_down"]}↓',
            transform=ax.transAxes, ha='right', fontsize=6.3)
    panel(ax, 'D', -0.22)

    ax = fig.add_subplot(gs[2, 1])
    adj = compo['adj_summary']
    cons = ['CvsB', 'DvsA', 'BvsA']
    labs = ['MnTBAP+Cis\nvs Cis', 'MnTBAP\nvs Ctrl', 'Cis\nvs Ctrl']
    xx = np.arange(3); w = 0.38
    u = [adj[(adj.contrast == c) & (adj.model == 'unadjusted')]['FDR005'].iloc[0] for c in cons]
    a = [adj[(adj.contrast == c) & (adj.model == 'PC1-adjusted')]['FDR005'].iloc[0] for c in cons]
    b1 = ax.bar(xx - w / 2, u, w, color='#888888', label='Group only')
    b2 = ax.bar(xx + w / 2, a, w, color='#B15928', label='+ PC1 covariate')
    for bars in (b1, b2):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 3, int(bb.get_height()), ha='center', fontsize=6.3)
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=6.8)
    ax.set_ylabel('Proteins at FDR < 0.05\n(complete-case set)')
    ax.legend(frameon=False, fontsize=6.3, loc='upper right')
    panel(ax, 'E', -0.3)
    save(fig, 'Figure5_MnTBAP')


# =============================================================== Figure 6: consensus biomarkers
def fig6():
    rk = RES['rk']; pc1 = compo['pc1']
    rpc = np.array([np.corrcoef(L[i], pc1)[0, 1] for i in rk['idx']])
    rk = rk.assign(r_pc1=rpc)
    fig = plt.figure(figsize=(7.4, 7.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.38, wspace=0.6)
    ax = fig.add_subplot(gs[0, 0])
    t = rk.head(20).iloc[::-1]
    ax.barh(range(20), t['consensus_score'], color=[UP if d == 'Up' else DOWN for d in t['direction']], height=0.7)
    ax.set_yticks(range(20)); ax.set_yticklabels([P(g) for g in t['Gene']], fontsize=6.8)
    lo = np.floor(t['consensus_score'].min() * 20) / 20
    ax.set_xlim(lo, 1.0); ax.set_xlabel('Consensus score')
    for i, rr in enumerate(t['r_pc1']):
        ax.text(1.005, i, f'{rr:+.2f}', va='center', fontsize=5.6, color='#B15928' if abs(rr) >= 0.7 else '#555555',
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.005, 20.2, 'r(PC1)', fontsize=5.8, transform=ax.get_yaxis_transform())
    ax.legend(handles=[Patch(color=UP, label='Increased'), Patch(color=DOWN, label='Decreased')], frameon=False,
              fontsize=6.3, loc='lower right', title='In cisplatin', title_fontsize=6.3)
    panel(ax, 'A', -0.32)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(rk['r_pc1'], rk['log2FC_BvsA'], s=8, c=[UP if d == 'Up' else DOWN for d in rk['direction']], lw=0, alpha=0.8)
    ax.axvline(0, color='grey', lw=0.5); ax.axhline(0, color='grey', lw=0.5)
    for vline in (-0.7, 0.7):
        ax.axvline(vline, color='#B15928', ls=':', lw=0.7)
    n_track = int(np.sum(np.abs(rpc) >= 0.7))
    ax.set_xlabel('Correlation with PC1 (synaptic axis)'); ax.set_ylabel('log$_2$ FC, Cisplatin / Control')
    ax.set_title(f'{n_track} of {len(rk)} signature proteins with |r| ≥ 0.7', fontsize=7.2)
    panel(ax, 'B', -0.3)

    ax = fig.add_subplot(gs[1, :])
    idx = rk.head(25)['idx'].values
    M = L[idx]; Z = (M - M.mean(1, keepdims=True)) / M.std(1, ddof=1, keepdims=True)
    im = ax.imshow(Z, aspect='auto', cmap=LinearSegmentedColormap.from_list('d', [DOWN, 'white', UP]), vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(15)); ax.set_xticklabels(SL, fontsize=6.5, rotation=90)
    ax.set_yticks(range(25)); ax.set_yticklabels([P(g) for g in rk.head(25)['Gene']], fontsize=6.3)
    for xx in (3.5, 7.5, 10.5):
        ax.axvline(xx, color='black', lw=0.9)
    for xc, g in zip([1.5, 5.5, 9, 12.5], GROUPS):
        ax.text(xc, -1.2, GN[g], ha='center', fontsize=7.2, fontweight='bold', color=GC[g])
    ax.spines[:].set_visible(True)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('Row z-score (log$_2$)', fontsize=7)
    panel(ax, 'C', -0.08, 1.04)
    save(fig, 'Figure6_consensus_biomarkers')


# =============================================================== Supplementary
def figS1():
    fig = plt.figure(figsize=(7.4, 3.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.1, 1.25], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    p16 = RES['uns']['pcs16']; v16 = RES['uns']['var16']; S16 = RES['SAMPLES']; G16 = RES['GROUP_OF']
    for g in GROUPS:
        m = G16 == g
        ax.scatter(p16[m, 0], p16[m, 2], c=GC[g], s=24, zorder=3)
    for i, s in enumerate(S16):
        ax.annotate(lab(s), (p16[i, 0], p16[i, 2]), fontsize=5.5, xytext=(3, 2), textcoords='offset points',
                    color='black' if s == 'C1' else '#666666', fontweight='bold' if s == 'C1' else 'normal')
    ax.set_xlabel(f'PC1 ({100 * v16[0]:.1f}%)'); ax.set_ylabel(f'PC3 ({100 * v16[2]:.1f}%)')
    ax.set_title(f'All 16 channels ({RES["uns"]["n16"]:,} proteins)', fontsize=7.2)
    panel(ax, 'A', -0.3)
    ax = fig.add_subplot(gs[1])
    X = L[ml]; Xc = X - X.mean(1, keepdims=True)
    Zl = linkage(pdist(Xc.T, 'correlation'), 'average')
    dendrogram(Zl, labels=SL, ax=ax, leaf_rotation=90, leaf_font_size=6.3, color_threshold=0, above_threshold_color='#555555')
    for t in ax.get_xticklabels():
        t.set_color(GC[LAB2G[t.get_text()]])
    ax.set_ylabel('1 − Pearson r')
    panel(ax, 'B', -0.3)
    ax = fig.add_subplot(gs[2])
    C = RES['uns']['sample_corr']
    im = ax.imshow(C, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(15)); ax.set_xticklabels(SL, fontsize=5.5, rotation=90)
    ax.set_yticks(range(15)); ax.set_yticklabels(SL, fontsize=5.5)
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_color(GC[LAB2G[t.get_text()]])
    for xx in (3.5, 7.5, 10.5):
        ax.axvline(xx, color='black', lw=0.6); ax.axhline(xx, color='black', lw=0.6)
    plt.colorbar(im, ax=ax, fraction=0.045, label='Pearson r')
    panel(ax, 'C', -0.25)
    save(fig, 'FigureS1_sample_structure')


def figS2():
    infl = SENS['infl']; parts = SENS['parts']; s = SENS['sens']
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = gridspec.GridSpec(1, 3, wspace=0.75, width_ratios=[1, 1.3, 1])
    ax = fig.add_subplot(gs[0])
    ax.bar(range(8), 100 * infl['delta_frac_p005'], color=[GC[x[0]] for x in infl['sample_removed']])
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(range(8)); ax.set_xticklabels([lab(x) for x in infl['sample_removed']], fontsize=6.5, rotation=90)
    ax.set_ylabel('Change in % proteins p < 0.05\nwhen sample removed')
    panel(ax, 'A', -0.35)
    ax = fig.add_subplot(gs[1])
    ax.bar(range(len(parts)), parts['p001'], color=['#B2182B' if t else '#888888' for t in parts['is_true']])
    ax.set_xlabel('All 35 partitions (ranked)'); ax.set_ylabel('Proteins at p < 0.01'); ax.set_xticks([])
    tl = parts[parts.is_true].iloc[0]
    ax.annotate(f'True labels (rank {int(tl["rank"])})', (tl['rank'] - 1, tl['p001']), xytext=(8, tl['p001'] + 60),
                fontsize=6.3, arrowprops=dict(arrowstyle='->', lw=0.6))
    t0 = parts.iloc[0]
    ax.annotate('/'.join(lab(x) for x in t0['set_B'].split('/')) + '\nvs ' + '/'.join(lab(x) for x in t0['set_A'].split('/')), (0, t0['p001']), xytext=(5, t0['p001'] - 20), fontsize=6,
                arrowprops=dict(arrowstyle='->', lw=0.6))
    panel(ax, 'B', -0.2)
    ax = fig.add_subplot(gs[2])
    sc = compo['scores']['Synaptic/neuronal']
    ab = np.isin(G, ['A', 'B'])
    for g in 'AB':
        m = (G == g)
        ax.scatter(np.where(m)[0][:0], [], c=GC[g])
    xs = np.arange(8)
    ax.bar(xs, sc[ab], color=[GC[g] for g in G[ab]])
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(np.array(SL)[ab], fontsize=6.5, rotation=90)
    ax.set_ylabel('Synaptic/neuronal panel score')
    panel(ax, 'C', -0.4)
    save(fig, 'FigureS2_sensitivity')


def figS3():
    rk = RES['rk'].head(8)
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9))
    for ax, (_, r) in zip(axes.flat, rk.iterrows()):
        strip(ax, L[r['idx']], label_samples=False)
        ax.set_xticklabels(['Ctrl', 'Cis', 'M+C', 'Mn'], fontsize=6.5)
        ax.set_title(f'{P(r["Gene"])}  (FC {r["FC_BvsA"]:.2f}, FDR {r["q_BvsA"]:.2f})', fontsize=6.6)
        ax.tick_params(labelsize=6)
    for a in axes[:, 0]:
        a.set_ylabel('log$_2$ abundance', fontsize=7)
    plt.tight_layout()
    save(fig, 'FigureS3_top_biomarkers')


def figS4():
    mu = RES['multi']
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2))
    lab = ['Ctrl', 'Cis', 'M+C', 'Mn']
    for ax, (name, r) in zip(axes, mu.items()):
        cm = r['cm']
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=4)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7.5, color='white' if cm[i, j] >= 3 else 'black')
        ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=6.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(lab if ax is axes[0] else [], fontsize=6.5)
        ax.set_title(f'{name}\naccuracy {r["accuracy"]:.2f}', fontsize=7); ax.set_xlabel('Predicted', fontsize=7)
        ax.spines[:].set_visible(True)
    axes[0].set_ylabel('True', fontsize=7)
    save(fig, 'FigureS4_four_group')


def figS5():
    net = RES['net']; k = net['k']; ms = net['mod_stats']
    mcol = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3'][:k]
    fig = plt.figure(figsize=(7.4, 3.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    order = leaves_list(net['linkage']); C = net['corr']; lab = net['labels']; n = len(order)
    im = ax.imshow(C[np.ix_(order, order)], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for i, j in enumerate(order):
        ax.add_patch(plt.Rectangle((-0.06 * n, i - 0.5), 0.04 * n, 1, color=mcol[lab[j] - 1], clip_on=False, lw=0))
    ax.set_xlim(-0.065 * n, n - 0.5)
    ax.set_title(f'{n} signature proteins (Pearson r, 15 samples)', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.045)
    ax.legend(handles=[Patch(color=mcol[m], label=f'M{m + 1}') for m in range(k)], frameon=False, fontsize=6.3,
              loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=k)
    ax.spines[:].set_visible(False)
    panel(ax, 'A', -0.12)
    ax = fig.add_subplot(gs[1])
    for m in range(k):
        e = net['eig'][m + 1]
        ax.errorbar(np.arange(4) + (m - (k - 1) / 2) * 0.07, [e[G == g].mean() for g in GROUPS],
                    yerr=[e[G == g].std(ddof=1) / np.sqrt((G == g).sum()) for g in GROUPS], color=mcol[m],
                    marker='o', ms=3.5, lw=1.3, capsize=2,
                    label=f'M{m + 1}: {int(ms.iloc[m]["n_proteins"])} proteins, '
                          f'{"increased" if ms.iloc[m]["n_up_cisplatin"] else "decreased"} in cisplatin')
    ax.set_xticks(range(4)); ax.set_xticklabels(['Control', 'Cisplatin', 'MnTBAP\n+Cisplatin', 'MnTBAP'], fontsize=7)
    ax.axhline(0, color='grey', lw=0.5); ax.set_ylabel('Module eigengene (mean ± SEM)')
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    panel(ax, 'B', -0.2)
    save(fig, 'FigureS5_coexpression', key=False)


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, figS1, figS2, figS3, figS4, figS5):
        f()
