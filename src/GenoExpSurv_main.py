"""Training, evaluation, model persistence and command-line interface."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import argparse
import yaml

from seed import set_seed
from GenoExpSurv_models import GenoExpSurv, SemanticMapper, mapper_loss
from util import (
    Cohort, FeatureBuilder, generate_demo, _tensor, RidgeCox, BreslowBaseline,
    cox_loss, harrell_c, evaluate_survival, METHODS, run_baseline,
    audit_cohort, run_portable, plot_semantics, plot_calibration, plot_km,
    schoenfeld_residuals, schoenfeld_correlations, subgroup_bootstrap,
    plot_schoenfeld, plot_subgroups,
)


# ====== Split and two-stage fitting ======
DEFAULTS = {
 'seeds':[11,22,33,44,55,66,77,88,99,110],
 'ridge_lambdas':[0.01,0.1,1.,10.],
 'predictor_epochs':160,'predictor_patience':10,
 'predictor_lr':0.0015,'predictor_weight_decay':0.0003,
 'residual_penalty':0.001,'dropout':0.20,
 'mapper_epochs':200,'mapper_patience':15,
 'mapper_lr':0.001,'mapper_weight_decay':0.0001,
 'mapper_entropy':0.0001,'device':'cpu','threads':2,
 'validation_metric':'c_index'
}


def _allocate(n,total,weight):
    # Preserve train/val/test exact total; generic event-stratified allocation.
    raw = n*np.asarray(weight,float)/sum(weight)
    nums = np.floor(raw).astype(int)
    for i in np.argsort(-(raw-nums))[:n-nums.sum()]: nums[i] += 1
    return nums


def make_split(cohort,seed):
    """Fixed patient-level event-stratified 2/3,1/6,1/6 partitions.

    For the 1661-patient/832-event study cohort, the allocation uses
    1107 training, 277 validation, and 277 test patients.
    Alternatively, supply a named patient-level split manifest.
    """
    if 'event' not in cohort.patients.columns:
        raise ValueError('Cannot split prediction-only data')
    df = cohort.patients
    rng = np.random.default_rng(seed)
    groups = {}
    for status in (1,0):
        ids = df.index[df.event==status].to_numpy(copy=True)
        rng.shuffle(ids)
        if len(df)==1661 and int(df.event.sum())==832:
            sizes = [554,139,139] if status==1 else [553,138,138]
        else:
            sizes = _allocate(len(ids),len(ids),[2/3,1/6,1/6])
        groups[status] = np.split(ids,np.cumsum(sizes)[:-1])
    out = {}
    for i,part in enumerate(('train','val','test')):
        ids = np.concatenate([groups[1][i],groups[0][i]])
        rng.shuffle(ids)
        out[part] = list(map(str,ids))
    assert len(set(out['train'])|set(out['val'])|set(out['test']))==len(df)
    assert all(set(out[a]).isdisjoint(out[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    return out


def read_manifest(path,cohort,split_id):
    df = pd.read_csv(path,dtype={'patient_id':str})
    if not {'split_id','patient_id','partition'}.issubset(df.columns):
        raise ValueError('manifest.csv needs split_id,patient_id,partition')
    df = df[df.split_id.astype(str)==str(split_id)]
    if df.empty: raise ValueError(f'No rows for split_id={split_id}')
    if df.patient_id.duplicated().any(): raise ValueError('Duplicate patient within manifest split')
    if set(df.patient_id)!=set(cohort.patients.index):
        raise ValueError('Manifest must partition every patient exactly once')
    if not set(df.partition).issubset({'train','val','test'}):
        raise ValueError('Invalid partition name')
    out = {p:df.loc[df.partition==p,'patient_id'].tolist() for p in ('train','val','test')}
    if min(map(len,out.values()))==0: raise ValueError('Empty partition')
    return out


def write_manifest(splits,path):
    rows=[]
    for sid, split in splits.items():
        for part,ids in split.items():
            rows.extend({'split_id':sid,'patient_id':pid,'partition':part} for pid in ids)
    pd.DataFrame(rows).to_csv(path,index=False)


def _forward_np(model,arrays,anchor,device='cpu'):
    model.eval()
    with torch.no_grad():
        out = model(_tensor(arrays.clinical,device),_tensor(arrays.genomic,device),
                    _tensor(anchor.predict(arrays.anchor),device))
    return {k:v.cpu().numpy() for k,v in out.items()}


def train_predictor(cohort,split,config,variant="full"):
    device=config['device']
    if variant in ('w/o SV','w/o SNV/Indel'):
        # Remove raw event calls BEFORE rebuilding all pathway and burden features.
        cohort=copy.copy(cohort)
        if variant=='w/o SV':
            cohort.sv=cohort.sv.iloc[:0].copy()
        else:
            cohort.mutations=cohort.mutations.iloc[:0].copy()
    builder = FeatureBuilder().fit(cohort,split['train'])
    arrays = {part:builder.transform(cohort,ids) for part,ids in split.items()}
    if variant.startswith('w/o ') and variant not in ('w/o SV','w/o SNV/Indel'):
        for arr in arrays.values():
            if variant=='w/o pathway':
                arr.genomic[:,3*len(builder.genes)+3:]=0.
            elif variant in ('w/o GENE_PANEL','w/o DRUG_TYPE','w/o CANCER_TYPE'):
                name=variant[4:]
                ix=[i for i,col in enumerate(builder.clinical_names) if col.startswith(name+'_')]
                arr.clinical[:,ix]=0.;arr.anchor[:,ix]=0.
            elif variant=='w/o TMB / mutation burden':
                g=3*len(builder.genes)
                arr.genomic[:,g:g+2]=0.
                arr.anchor[:,-3:-1]=0.
            elif variant!='w/o balanced fusion':
                raise ValueError('Unknown ablation: '+variant)
    labels = {part:cohort.labels(ids) for part,ids in split.items()}
    train,val=arrays['train'],arrays['val']
    t,e=labels['train']; vt,ve=labels['val']
    # Stage I-A: select anchor regularization on VALIDATION only.
    choices=[]
    for lam in config['ridge_lambdas']:
        anchor = RidgeCox(l2=float(lam)).fit(train.anchor,t,e)
        c = harrell_c(vt,ve,anchor.predict(val.anchor))
        choices.append((c,-float(lam),anchor))
    anchor = max(choices,key=lambda item:(np.nan_to_num(item[0],nan=-1),item[1]))[2]
    model=GenoExpSurv(train.clinical.shape[1],train.genomic.shape[1],
                      dropout=config['dropout'],
                      gate=(variant!='w/o balanced fusion'),
                      clinical_only=(variant=='Clinical residual'),
                      genomic_only=(variant=='Genomic residual')).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['predictor_lr'],
                          weight_decay=config['predictor_weight_decay'])
    xc,xg,ar=(_tensor(train.clinical,device),_tensor(train.genomic,device),
              _tensor(anchor.predict(train.anchor),device))
    xt,xe=_tensor(t,device),_tensor(e,device)
    best=-np.inf; best_state=None; best_epoch=0; patience=0; history=[]
    for epoch in range(1,config['predictor_epochs']+1):
        model.train(); opt.zero_grad()
        pred=model(xc,xg,ar)
        loss=cox_loss(pred['risk'],xt,xe)+config['residual_penalty']*pred['residual'].square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        opt.step()
        valrisk=_forward_np(model,val,anchor,device)['risk']
        score=harrell_c(vt,ve,valrisk)
        history.append({'epoch':epoch,'train_loss':float(loss.detach()),'val_c_index':score})
        if np.isfinite(score) and score>best+1e-7:
            best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());patience=0
        else:
            patience+=1
        if patience>=config['predictor_patience']:break
    if best_state is None: raise RuntimeError('No valid validation checkpoint')
    model.load_state_dict(best_state)
    # Stage I-B: estimate baseline ON TRAINING ONLY using selected model.
    trisk=_forward_np(model,train,anchor,device)['risk']
    baseline=BreslowBaseline.fit(t,e,trisk)
    return builder,anchor,model,baseline,arrays,labels,history,{'epoch':best_epoch,'val_c_index':best,'ridge_lambda':anchor.l2}


def train_mapper(model,anchor,arrays,config):
    """Stage II: all Stage I parameters frozen and not in mapper optimizer."""
    device=config['device']
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    frozen={part:_forward_np(model,a,anchor,device) for part,a in arrays.items()}
    trainrisk=frozen['train']['risk']
    mean=float(trainrisk.mean());std=max(float(trainrisk.std()),1e-8)
    mapper=SemanticMapper(arrays['train'].aspects.shape[1]).to(device)
    opt=torch.optim.AdamW(mapper.parameters(),lr=config['mapper_lr'],
                          weight_decay=config['mapper_weight_decay'])
    best=float('inf');best_state=None;best_epoch=0;patience=0;history=[]
    def data(part):
        return (_tensor(frozen[part]['latent'],device),_tensor(arrays[part].aspects,device),
                _tensor((frozen[part]['risk']-mean)/std,device))
    ht,zt,rt=data('train');hv,zv,rv=data('val')
    for epoch in range(1,config['mapper_epochs']+1):
        mapper.train();opt.zero_grad()
        output=mapper(ht,zt)
        loss,components=mapper_loss(output,ht,rt,config['mapper_entropy'])
        loss.backward();torch.nn.utils.clip_grad_norm_(mapper.parameters(),5.0);opt.step()
        mapper.eval()
        with torch.no_grad():
            vloss,_=mapper_loss(mapper(hv,zv),hv,rv,config['mapper_entropy'])
        history.append({'epoch':epoch,'train_loss':float(loss.detach()),'val_loss':float(vloss)})
        if float(vloss)<best-1e-6:
            best=float(vloss);best_state=copy.deepcopy(mapper.state_dict());best_epoch=epoch;patience=0
        else:patience+=1
        if patience>=config['mapper_patience']:break
    if best_state is None: raise RuntimeError('No valid mapper checkpoint')
    mapper.load_state_dict(best_state);mapper.eval()
    return mapper,history,{'epoch':best_epoch,'val_loss':best,'risk_mean':mean,'risk_std':std}


def predict_arrays(builder,anchor,model,baseline,mapper,arrays,device='cpu',horizons=(12.,24.)):
    out=_forward_np(model,arrays,anchor,device)
    result={'patient_id':arrays.ids,'log_risk':out['risk'],
            'anchor_risk':anchor.predict(arrays.anchor),
            'residual':out['residual'],'gate_clinical':out['gate'][:,0],
            'gate_genomic':out['gate'][:,1]}
    survival=baseline.survival(out['risk'],horizons)
    for i,h in enumerate(horizons):result[f'survival_{h:g}m']=survival[:,i]
    frame=pd.DataFrame(result)
    if mapper is not None:
        mapper.eval()
        with torch.no_grad():
            x=mapper(_tensor(out['latent'],device),_tensor(arrays.aspects,device))
        weights=x['relevance'].cpu().numpy()
        weighted=x['weighted'].cpu().numpy()
        # Keep full arrays separately; large per-patient aspect columns not in summary CSV.
        return frame,weights,weighted
    return frame,None,None


def save_bundle(folder,builder,anchor,model,baseline,mapper,config,extra):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    joblib.dump({'builder':builder,'anchor':anchor,'baseline':baseline},folder/'preprocessing.joblib')
    torch.save({'predictor':model.state_dict(),'mapper':mapper.state_dict() if mapper else None},folder/'weights.pt')
    with open(folder/'metadata.json','w',encoding='utf-8') as f:
        json.dump({'n_clinical':len(builder.clinical_names),'n_genomic':len(builder.genomic_names),
                   'n_aspects':len(builder.aspect_names),'config':config,'extra':extra},f,indent=2)


def load_bundle(folder,device='cpu'):
    folder=Path(folder)
    obj=joblib.load(folder/'preprocessing.joblib')
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    model=GenoExpSurv(meta['n_clinical'],meta['n_genomic'],
                      dropout=meta['config']['dropout']).to(device)
    # Load ONLY trusted model files: pickle/joblib and weights_only=False are unsafe for untrusted inputs.
    weights=torch.load(folder/'weights.pt',map_location=device,weights_only=True)
    model.load_state_dict(weights['predictor']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    mapper=None
    if weights['mapper'] is not None:
        mapper=SemanticMapper(meta['n_aspects']).to(device)
        mapper.load_state_dict(weights['mapper']);mapper.eval()
    return obj['builder'],obj['anchor'],model,obj['baseline'],mapper,meta


def run_split(cohort,split,config,outdir,split_id='0',seed=11):
    set_seed(seed,config['threads'])
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    builder,anchor,model,baseline,arrays,labels,h1,meta1=train_predictor(cohort,split,config)
    # Assert frozen risk is identical before and after Stage II.
    risk_before=_forward_np(model,arrays['test'],anchor,config['device'])['risk'].copy()
    mapper,h2,meta2=train_mapper(model,anchor,arrays,config)
    risk_after=_forward_np(model,arrays['test'],anchor,config['device'])['risk']
    if not np.array_equal(risk_before,risk_after):
        raise AssertionError('Stage II unexpectedly modified the survival predictor')
    trt,tre=labels['train'];tet,tee=labels['test']
    prediction,weights,weighted=predict_arrays(builder,anchor,model,baseline,mapper,
                                              arrays['test'],config['device'])
    metrics=evaluate_survival(trt,tre,tet,tee,risk_after,
                              lambda horizons:baseline.survival(risk_after,horizons))
    metrics.update({'split_id':str(split_id),'seed':int(seed),
                    'n_train':len(split['train']),'n_val':len(split['val']),'n_test':len(split['test'])})
    save_bundle(outdir/'model',builder,anchor,model,baseline,mapper,config,
                {'stage1':meta1,'stage2':meta2})
    prediction.to_csv(outdir/'test_predictions.csv',index=False)
    pd.DataFrame(h1).to_csv(outdir/'stage1_history.csv',index=False)
    pd.DataFrame(h2).to_csv(outdir/'stage2_history.csv',index=False)
    pd.DataFrame(weights,columns=builder.aspect_names,index=arrays['test'].ids).to_csv(outdir/'semantic_relevance.csv',index_label='patient_id')
    pd.DataFrame(weighted,columns=builder.aspect_names,index=arrays['test'].ids).to_csv(outdir/'semantic_weighted.csv',index_label='patient_id')
    plot_semantics(builder.aspect_names,risk_after,weights,weighted,outdir/'semantic_heatmap.png')
    plot_calibration(tet,tee,risk_after,baseline,outdir/'calibration.png')
    plot_km(tet,tee,risk_after,outdir/'km_risk_groups.png')
    anchor_names=builder.clinical_names+['log1p_TMB','mutation_burden','sv_burden']
    sr=schoenfeld_residuals(arrays['train'].anchor,trt,tre,
                            anchor.predict(arrays['train'].anchor),anchor_names)
    selected=([n for n in anchor_names if n.startswith('AGE_GROUP_')][:1] +
              ['log1p_TMB','mutation_burden','sv_burden'])
    schoenfeld_correlations(sr,selected).to_csv(outdir/'schoenfeld_exploratory.csv',index=False)
    plot_schoenfeld(sr,selected,outdir/'schoenfeld_exploratory.png')
    subgroup=subgroup_bootstrap(cohort.patients.loc[split['test']],tet,tee,risk_after,seed)
    subgroup.to_csv(outdir/'subgroup_metrics.csv',index=False)
    plot_subgroups(subgroup,outdir/'subgroup_forest.png')
    (outdir/'metrics.json').write_text(json.dumps({k:(float(v) if np.isfinite(v) else None)
        if isinstance(v,(float,np.floating)) else v for k,v in metrics.items()},
        indent=2,allow_nan=False),encoding='utf-8')
    return metrics


# ====== Feature and structural ablation studies ======
VARIANTS=['Full','w/o pathway','w/o SV','w/o GENE_PANEL',
          'Clinical residual','Anchor only','Genomic residual',
          'w/o balanced fusion','w/o DRUG_TYPE','w/o CANCER_TYPE',
          'w/o TMB / mutation burden','w/o SNV/Indel']


def run_ablation(cohort,split,config,variants,seed=11):
    rows=[]
    for variant in variants:
        if variant not in VARIANTS:raise ValueError('Unknown ablation: '+variant)
        set_seed(seed,config['threads'])
        # For anchor-only, Stage I still fits the anchor and does not use the residual.
        builder,anchor,model,baseline,arrays,labels,history,meta=train_predictor(
            cohort,split,config,variant='full' if variant in ('Full','Anchor only') else variant)
        if variant=='Anchor only':
            risk_train=anchor.predict(arrays['train'].anchor)
            risk_test=anchor.predict(arrays['test'].anchor)
            baseline=BreslowBaseline.fit(*labels['train'],risk_train)
        else:
            risk_test=_forward_np(model,arrays['test'],anchor,config['device'])['risk']
        scores=evaluate_survival(*labels['train'],*labels['test'],risk_test,
                                 lambda t:baseline.survival(risk_test,t))
        rows.append({'variant':variant,**scores})
    return pd.DataFrame(rows)


# ====== User commands ======
def config_from_file(path):
    config=DEFAULTS.copy()
    if path:
        with open(path,encoding='utf-8') as f:
            overrides=yaml.safe_load(f) or {}
        unknown=set(overrides)-set(DEFAULTS)
        if unknown:raise ValueError('Unknown configuration keys: '+str(sorted(unknown)))
        config.update(overrides)
    return config


def add_common(parser):
    parser.add_argument('--data',required=True,help='Folder containing patients.csv, panel_coverage.csv, etc.')
    parser.add_argument('--out',default='outputs/experiment')
    parser.add_argument('--config',default=None,help='YAML overrides')
    parser.add_argument('--manifest',default=None,help='Optional split_id/patient_id/partition CSV')
    parser.add_argument('--split',default='0',help='Split index 0..9, or manifest split_id')
    parser.add_argument('--all-splits',action='store_true',help='Run ten independent internal splits')


def seed_for_split(config, split_id):
    """Resolve a deterministic seed for numbered or named split manifests."""
    import hashlib
    seeds = config['seeds']
    if not seeds:
        raise ValueError('Configuration must contain at least one seed')
    split_id = str(split_id)
    if split_id.isdecimal() and int(split_id) < len(seeds):
        return int(seeds[int(split_id)])
    digest = hashlib.sha256(split_id.encode('utf-8')).digest()
    return int(seeds[int.from_bytes(digest[:8], 'big') % len(seeds)])


def get_splits(cohort,args,config):
    ids=[str(i) for i in range(10)] if args.all_splits else [str(args.split)]
    out={}
    for sid in ids:
        out[sid] = (read_manifest(args.manifest,cohort,sid) if args.manifest
                    else make_split(cohort,seed_for_split(config,sid)))
    return out


def aggregate(df,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(outdir/'all_split_metrics.csv',index=False)
    columns=[c for c in df.select_dtypes(include='number').columns if c not in ('seed','n_train','n_val','n_test')]
    summary={k:{'mean':float(df[k].mean()) if pd.notna(df[k].mean()) else None,
                'std':float(df[k].std(ddof=1)) if len(df)>1 and pd.notna(df[k].std(ddof=1)) else None}
             for k in columns}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def cmd_train(args):
    config=config_from_file(args.config)
    cohort=Cohort.load(args.data)
    splits=get_splits(cohort,args,config)
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    write_manifest(splits,outdir/'used_split_manifest.csv')
    rows=[]
    for sid,split in splits.items():
        seed=seed_for_split(config,sid)
        print(f'Fitting split={sid} seed={seed} n={list(map(len,split.values()))}',flush=True)
        metrics=run_split(cohort,split,config,outdir/f'split_{sid}',sid,seed)
        rows.append(metrics)
        print('  test:',{k:round(v,4) for k,v in metrics.items() if k in ('c_index','ibs_1_24')},flush=True)
    summary=aggregate(pd.DataFrame(rows),outdir)
    print(json.dumps(summary,indent=2,allow_nan=False))


def cmd_demo(args):
    data=generate_demo(args.data,n=args.n,seed=args.seed)
    print(f'SYNTHETIC demo dataset written to {data}',flush=True)
    train_args=argparse.Namespace(data=str(data),out=args.out,config=args.config,
                                  manifest=None,split='0',all_splits=False)
    cmd_train(train_args)
    print('Demo completed using generated example patients and gene sets.')


def cmd_predict(args):
    cohort=Cohort.load(args.data)
    builder,anchor,model,baseline,mapper,meta=load_bundle(args.model)
    arrays=builder.transform(cohort,cohort.patients.index)
    frame,relevance,weighted=predict_arrays(builder,anchor,model,baseline,mapper,arrays,
                                            horizons=args.horizons)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out,index=False)
    if relevance is not None:
        pd.DataFrame(relevance,columns=builder.aspect_names,index=arrays.ids).to_csv(
            out.with_name(out.stem+'_semantic_relevance.csv'),index_label='patient_id')
        pd.DataFrame(weighted,columns=builder.aspect_names,index=arrays.ids).to_csv(
            out.with_name(out.stem+'_semantic_weighted.csv'),index_label='patient_id')
    print(f'Saved {len(frame)} patient predictions to {out}')


def cmd_benchmark(args):
    config=config_from_file(args.config)
    cohort=Cohort.load(args.data)
    splits=get_splits(cohort,args,config)
    methods=METHODS if args.methods=='all' else [x.strip() for x in args.methods.split(',')]
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sid,split in splits.items():
        seed=seed_for_split(config,sid);set_seed(seed,config['threads'])
        builder=FeatureBuilder().fit(cohort,split['train'])
        arrays={part:builder.transform(cohort,ids) for part,ids in split.items()}
        labels={part:cohort.labels(ids) for part,ids in split.items()}
        for name in methods:
            print('Benchmark:',sid,name,flush=True)
            try:
                risk,survival,extra=run_baseline(name,arrays,labels,config,seed,quick=args.quick)
                scores=evaluate_survival(*labels['train'],*labels['test'],risk,survival)
                rows.append({'split_id':sid,'model':name,**scores,**extra})
            except (ImportError,ValueError,RuntimeError) as exc:
                if args.strict:raise
                print(f'  SKIPPED {name}: {exc}',flush=True)
                rows.append({'split_id':sid,'model':name,'error':str(exc)})
    pd.DataFrame(rows).to_csv(outdir/'baseline_metrics.csv',index=False)
    print(f'Saved baseline metrics to {outdir}')


def cmd_ablate(args):
    config=config_from_file(args.config)
    cohort=Cohort.load(args.data)
    splits=get_splits(cohort,args,config)
    variants=VARIANTS if args.variants=='all' else [x.strip() for x in args.variants.split(',')]
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sid,split in splits.items():
        seed=seed_for_split(config,sid)
        frame=run_ablation(cohort,split,config,variants,seed)
        frame.insert(0,'split_id',sid)
        rows.append(frame)
    pd.concat(rows,ignore_index=True).to_csv(outdir/'ablation_metrics.csv',index=False)
    print(f'Saved ablation metrics to {outdir}')


def cmd_validate(args):
    result=audit_cohort(Cohort.load(args.data))
    if args.out:
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def cmd_portable(args):
    result=run_portable(args.internal,args.external,args.out,args.seed,args.bootstraps)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description='GenoExpSurv survival modeling and interpretation pipeline')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('demo',help='Generate synthetic data and run complete two-stage pipeline')
    p.add_argument('--data',default='data/demo');p.add_argument('--out',default='outputs/demo')
    p.add_argument('--n',type=int,default=160);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--config',default='configs/quick.yaml');p.set_defaults(func=cmd_demo)
    p=sub.add_parser('train',help='Fit and evaluate real-data model (1 or 10 splits)')
    add_common(p);p.set_defaults(func=cmd_train)
    p=sub.add_parser('predict',help='Apply frozen model to unseen patients; no outcome needed')
    p.add_argument('--data',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',default='outputs/new_predictions.csv')
    p.add_argument('--horizons',nargs='+',type=float,default=[12.,24.])
    p.set_defaults(func=cmd_predict)
    p=sub.add_parser('benchmark',help='Run eight survival baseline families')
    add_common(p);p.add_argument('--methods',default='all')
    p.add_argument('--quick',action='store_true');p.add_argument('--strict',action='store_true')
    p.set_defaults(func=cmd_benchmark)
    p=sub.add_parser('ablate',help='Run Table 3 architectural and feature ablations')
    add_common(p);p.add_argument('--variants',default='all')
    p.set_defaults(func=cmd_ablate)
    p=sub.add_parser('validate',help='Audit panel coverage, events, Hallmark overlap, and labels')
    p.add_argument('--data',required=True)
    p.add_argument('--out',default=None)
    p.set_defaults(func=cmd_validate)
    p=sub.add_parser('portable',help='External-cohort analysis with reduced-feature survival models')
    p.add_argument('--internal',required=True,help='Internal patients.csv')
    p.add_argument('--external',required=True,help='External patients.csv')
    p.add_argument('--out',default='outputs/portable')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--bootstraps',type=int,default=1000)
    p.set_defaults(func=cmd_portable)
    p=sub.add_parser('self-test',help='Run invariant and synthetic integration tests; temporary files only')
    p.set_defaults(func=cmd_self_test)
    args=parser.parse_args()
    args.func(args)



def cmd_self_test(args):
    """Run deterministic invariant and end-to-end pipeline checks.

    Any synthetic patients, training outputs, and checkpoints are placed in a
    temporary directory and removed automatically when the test ends.
    """
    import tempfile
    from types import SimpleNamespace
    from scipy.optimize import check_grad
    from util import (_cox_numpy_objective, ipcw_brier, _tensor)
    from GenoExpSurv_models import PatientRepresentation, SurvivalPrediction

    # Breslow Cox objective: numerical gradient and loss agree with PyTorch.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 4))
    t = np.array([1,2,2,3,3,4,4,5,6,6,7,8.], float)
    e = np.array([1,1,0,1,1,0,1,0,1,0,1,0], float)
    beta = rng.normal(size=4)*.1
    err = check_grad(lambda b:_cox_numpy_objective(b,x,t,e,.1)[0],
                     lambda b:_cox_numpy_objective(b,x,t,e,.1)[1],beta)
    assert err < 1e-5, f'Cox numerical gradient mismatch {err}'
    score = torch.tensor(x@beta, dtype=torch.float64, requires_grad=True)
    loss = cox_loss(score, torch.tensor(t), torch.tensor(e))
    assert np.isclose(float(loss.detach()), _cox_numpy_objective(beta,x,t,e,0)[0], atol=1e-8)

    # Survival probabilities are non-increasing and order-preserving.
    times = np.array([2.,4.,5.,7.,9.])
    events = np.array([1,1,0,1,0])
    risks = np.array([0.,.5,1.,-1.,0.])
    baseline = BreslowBaseline.fit(times,events,risks)
    s = baseline.survival(np.array([-.5,1.]),[0,2,4,8,12])
    assert np.allclose(s[:,0],1)
    assert np.all(np.diff(s,axis=1) <= 1e-12)
    assert np.all(s[1] <= s[0])
    assert np.isfinite(ipcw_brier(times,events,times,events,baseline.survival(risks,[4])[:,0],4))

    # The three scientific modules implement 32+64+32=128 dimensions.
    rep = PatientRepresentation(6,22,dropout=0).eval()
    survivor = SurvivalPrediction(dropout=0).eval()
    mapper0 = SemanticMapper(10).eval()
    xc, xg, ar = torch.randn(5,6), torch.randn(5,22), torch.randn(5)
    z = rep(xc,xg)
    assert z['latent'].shape == (5,128)
    assert torch.allclose(z['gate'].sum(dim=1),torch.ones(5),atol=1e-6)
    pred = survivor(z['latent'],ar)
    assert torch.all(torch.abs(pred['residual'])<.5)
    assert torch.allclose(pred['risk'],ar+pred['residual'])
    mapped = mapper0(z['latent'],torch.randn(5,10))
    assert mapped['relevance'].shape==(5,10)
    assert mapped['reconstruction'].shape==(5,128)

    # Event-stratified paper cohort counts; no patient data modifications.
    df = pd.DataFrame({'event':[1]*832+[0]*829},
                      index=[f'p{i}' for i in range(1661)])
    parts=make_split(SimpleNamespace(patients=df),seed=11)
    assert [len(parts[x]) for x in ('train','val','test')]==[1107,277,277]
    assert [int(df.loc[parts[x],'event'].sum()) for x in ('train','val','test')]==[554,139,139]

    with tempfile.TemporaryDirectory(prefix='genoexpsurv_selftest_') as tmp:
        tmp = Path(tmp)
        synthetic = generate_demo(tmp/'synthetic',n=72,genes=18,seed=8)
        cohort=Cohort.load(synthetic)
        fb=FeatureBuilder().fit(cohort,make_split(cohort,11)['train'])
        a=fb.transform(cohort,cohort.patients.index)
        g=len(fb.genes)
        cov=a.genomic[:,2*g:3*g]
        assert a.genomic.shape[1] == 3*g+3+4*50
        assert np.all(a.genomic[:,:g]<=cov)
        assert np.all(a.genomic[:,g:2*g]<=cov)
        assert np.any(cov == 0) and np.any(cov == 1)
        # End-to-end fit, frozen mapper, checkpoint reload and prediction.
        config=DEFAULTS | {'predictor_epochs':4,'predictor_patience':3,
                           'mapper_epochs':5,'mapper_patience':3,
                           'ridge_lambdas':[.1],'threads':2}
        split=make_split(cohort,11)
        metrics=run_split(cohort,split,config,tmp/'test_output',split_id='0',seed=11)
        saved=tmp/'test_output/model'
        assert (saved/'weights.pt').is_file()
        fb,anchor,model,baseline,mapper,meta=load_bundle(saved)
        samples=fb.transform(cohort,split['test'])
        pred,weights,weighted=predict_arrays(fb,anchor,model,baseline,mapper,samples)
        assert len(pred)==len(split['test'])
        assert weights.shape==weighted.shape==(len(pred),len(fb.aspect_names))
        assert np.allclose(weights.sum(axis=1),1,atol=1e-5)
        assert np.all((pred['survival_12m']>=0)&(pred['survival_12m']<=1))
        assert np.all(pred['survival_24m']<=pred['survival_12m']+1e-10)
        assert 'c_index' in metrics
    print('PASS: loss, survival, 3 model modules, patient splits, panel masks,')
    print('      synthetic two-stage training, frozen mapper, reload and inference.')
    print('All temporary synthetic data and test outputs have been deleted.')


if __name__ == "__main__":
    main()
