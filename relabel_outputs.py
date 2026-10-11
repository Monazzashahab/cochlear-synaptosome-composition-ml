#!/usr/bin/env python3
"""Rewrite sample IDs in output tables to the group nomenclature used in the manuscript:
A1 -> Control 1, B1 -> Cisplatin 1, C1 -> MnTBAP+Cisplatin 1, D1 -> MnTBAP 1.
Idempotent. Run after p1_analysis.py and p1_sensitivity.py (both call it)."""
import os, re, glob, json

NAMES = {'A': 'Control', 'B': 'Cisplatin', 'C': 'MnTBAP+Cisplatin', 'D': 'MnTBAP'}
TOKEN = re.compile(r'(?<![A-Za-z0-9_+])([ABCD])([1-4])(?![A-Za-z0-9_])')


def name(m):
    return f'{NAMES[m.group(1)]} {m.group(2)}'


def relabel(out_dir):
    import pandas as pd
    for f in glob.glob(os.path.join(out_dir, '*.csv')):
        df = pd.read_csv(f)
        changed = False
        for c in df.columns:
            if df[c].dtype == object:
                vals = df[c].dropna().astype(str)
                if len(vals) and vals.str.fullmatch(r'([ABCD][1-4])(/[ABCD][1-4])*').all():
                    df[c] = df[c].astype(str).str.replace(TOKEN, name, regex=True)
                    changed = True
        if changed:
            df.to_csv(f, index=False)
    for f in glob.glob(os.path.join(out_dir, '*.json')):
        txt = open(f).read()
        new = TOKEN.sub(name, txt)
        if new != txt:
            open(f, 'w').write(new)


if __name__ == '__main__':
    relabel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
