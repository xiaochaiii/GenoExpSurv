"""Feature matrix preparation and split-specific transformations."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import numpy as np
# Allow running from data/preprocessing/ regardless of installation.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from util import Cohort, FeatureBuilder
from GenoExpSurv_main import make_split, read_manifest


def build(args):
    cohort=Cohort.load(args.data)
    split=read_manifest(args.manifest,cohort,args.split_id) if args.manifest else make_split(cohort,args.seed)
    fb=FeatureBuilder().fit(cohort,split['train'])
    summary={}
    for part,ids in split.items():
        a=fb.transform(cohort,ids)
        summary[part]={'patients':len(a.ids),'clinical_dim':int(a.clinical.shape[1]),
                       'genomic_dim':int(a.genomic.shape[1]),
                       'anchor_dim':int(a.anchor.shape[1]),
                       'semantic_aspects':int(a.aspects.shape[1])}
        if args.out:
            target=Path(args.out);target.mkdir(parents=True,exist_ok=True)
            t,e=cohort.labels(ids)
            np.savez_compressed(target/f'{part}.npz',patient_id=np.array(a.ids),
                                 clinical=a.clinical,genomic=a.genomic,
                                 anchor=a.anchor,aspects=a.aspects,
                                 time_months=t,event=e)
    if args.out:
        import joblib
        joblib.dump(fb,Path(args.out)/'feature_builder.joblib')
        (Path(args.out)/'matrix_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--manifest',help='Actual split_id,patient_id,partition CSV')
    p.add_argument('--split-id',default='0')
    p.add_argument('--seed',type=int,default=11,help='Random seed for reproducible partitioning')
    p.add_argument('--out',help='Optional directory for matrices')
    print(json.dumps(build(p.parse_args()),indent=2))
