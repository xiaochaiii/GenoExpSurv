"""Hallmark gene-set validation and preprocessing utilities."""
from __future__ import annotations
import argparse
from pathlib import Path


def validate_hallmark(path):
    items={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        f=line.split('\t')
        if len(f)<3:raise ValueError('Each GMT line requires name, description, gene(s)')
        name=f[0]
        if name in items:raise ValueError('Duplicate gene set: '+name)
        if not name.startswith('HALLMARK_'):
            raise ValueError('Non-Hallmark set name: '+name)
        genes=set(g.strip() for g in f[2:] if g.strip())
        if not genes:raise ValueError('Empty gene set: '+name)
        items[name]=genes
    if len(items)!=50:raise ValueError(f'Expected 50 Hallmark gene sets, found {len(items)}')
    return {'pathways':len(items),'unique_genes':len(set().union(*items.values())),
            'min_genes_per_pathway':min(map(len,items.values())),
            'max_genes_per_pathway':max(map(len,items.values()))}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmt',required=True)
    print(validate_hallmark(p.parse_args().gmt))
