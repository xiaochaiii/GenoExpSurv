"""Sequencing-panel gene coverage and mask construction."""
from __future__ import annotations
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

OFFICIAL_PANEL_FILES = {
    panel: f'https://media.githubusercontent.com/media/cBioPortal/datahub/master/'
           f'reference_data/gene_panels/data_gene_panel_{panel.lower()}.txt'
    for panel in ('IMPACT341', 'IMPACT410', 'IMPACT468')
}


def parse_official_panel(text, expected_id):
    """Read official cBioPortal stable_id/gene_list lines; reject Git LFS pointers."""
    if 'git-lfs.github.com/spec' in text:
        raise ValueError('Downloaded a Git-LFS pointer, not a gene list. Use media.githubusercontent.com/media/ URL.')
    fields = {}
    for line in text.splitlines():
        if ': ' in line:
            k, v = line.split(': ', 1)
            fields[k.strip()] = v.strip()
    if fields.get('stable_id') != expected_id:
        raise ValueError(f'Incorrect official panel ID; expected {expected_id}, got {fields.get("stable_id")}')
    genes = [g.strip() for g in fields.get('gene_list','').split('\t') if g.strip()]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError(f'Empty/duplicated gene list for {expected_id}')
    if not all(g.replace('-', '').replace('.', '').isalnum() for g in genes):
        raise ValueError(f'Unexpected gene symbol in {expected_id}')
    return genes


def download_official_panels(csv_path, *, timeout=40):
    """Fetch 3 gene lists; write a single panel,gene CSV and provenance JSON.

    Explicit network action requiring user to opt into --download-official;
    never downloads patient data or writes model scores.
    """
    csv_path=Path(csv_path)
    all_rows=[]; provenance=[]
    for panel, url in OFFICIAL_PANEL_FILES.items():
        with urllib.request.urlopen(urllib.request.Request(url,headers={
                'User-Agent':'GenoExpSurv-panel-reference/1.0'}), timeout=timeout) as response:
            content=response.read()
        if len(content) > 300_000:
            raise ValueError(f'Reference panel {panel} unexpectedly large')
        genes=parse_official_panel(content.decode('utf-8'),panel)
        all_rows.extend({'panel':panel,'gene':g} for g in genes)
        provenance.append({'panel':panel,'url':url,'gene_count':len(genes),
                           'sha256':hashlib.sha256(content).hexdigest()})
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows,columns=['panel','gene']).to_csv(csv_path,index=False)
    payload={'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
             'source_repository':'https://github.com/cBioPortal/datahub/tree/master/reference_data/gene_panels',
             'notes':'Panel lists may differ from their numeric labels; verify panel identities and gene symbols against the analysis protocol.',
             'files':provenance}
    csv_path.with_name(csv_path.stem+'_provenance.json').write_text(
        json.dumps(payload,indent=2),encoding='utf-8')
    return provenance


def build_panel_mask(coverage):
    if isinstance(coverage,(str,Path)):
        coverage=pd.read_csv(coverage,dtype=str)
    if not {'panel','gene'}.issubset(coverage.columns):
        raise ValueError('Coverage file must contain panel,gene columns')
    rows=coverage[['panel','gene']].dropna().drop_duplicates()
    if rows.empty:
        raise ValueError('Empty panel coverage')
    panels=sorted(rows.panel.unique());genes=sorted(rows.gene.unique())
    i={p:k for k,p in enumerate(panels)};j={g:k for k,g in enumerate(genes)}
    mask=np.zeros((len(panels),len(genes)),dtype=np.uint8)
    for panel,gene in rows.itertuples(index=False,name=None):
        mask[i[panel],j[gene]]=1
    return panels,genes,mask


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--coverage',help='Existing verified coverage CSV with panel,gene')
    ap.add_argument('--download-official',action='store_true',
                    help='Download the cBioPortal DataHub 341/410/468 definitions (requires Internet)')
    ap.add_argument('--out-csv',default='data/raw/panel_genes.csv',
                    help='Local output for official target gene mapping when downloading')
    ap.add_argument('--out',help='Optional generated numpy coverage matrix .npz; omit to inspect only')
    a=ap.parse_args()
    if a.download_official and a.coverage:
        ap.error('Choose either --download-official or --coverage, not both')
    if not a.download_official and not a.coverage:
        ap.error('Provide --coverage PATH or --download-official')
    if a.download_official:
        provenance=download_official_panels(a.out_csv)
        print('Downloaded verified panel files into',a.out_csv)
        for entry in provenance:
            print(f'  {entry["panel"]}: {entry["gene_count"]} genes ({entry["sha256"][:12]})')
        cover=a.out_csv
    else:
        cover=a.coverage
    panels,genes,mask=build_panel_mask(cover)
    print(f'Panel coverage: {len(panels)} panels; {len(genes)} assayed genes in union')
    for panel,k in zip(panels,mask.sum(axis=1)):
        print(f'  {panel}: {int(k)} covered genes')
    if a.out:
        p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,panel=np.array(panels),gene=np.array(genes),covered=mask)


if __name__=='__main__':
    main()
