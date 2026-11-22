"""Clinical and genomic cohort data preprocessing."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd

FUNCTIONAL_CLASSES = frozenset({
    'Missense_Mutation', 'Nonsense_Mutation', 'Nonstop_Mutation',
    'Splice_Site', 'Frame_Shift_Del', 'Frame_Shift_Ins',
    'In_Frame_Del', 'In_Frame_Ins', 'Translation_Start_Site',
    'Start_Codon_SNP', 'Start_Codon_Del', 'Start_Codon_Ins',
})
CLINICAL = ['SEX','AGE_GROUP','CANCER_TYPE','SAMPLE_TYPE','DRUG_TYPE','GENE_PANEL']


def table(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sep='\t' if path.suffix.lower() in {'.tsv','.txt','.maf'} else ','
    return pd.read_csv(path, sep=sep, comment='#', dtype=str, low_memory=False)


def require(df, cols, name):
    missing=set(cols)-set(df)
    if missing:
        raise ValueError(f'{name}: missing required columns {sorted(missing)}')


def read_death_status(series):
    """Return 0/1, accepting raw MSK strings (e.g. 1:DECEASED)."""
    s=series.fillna('').astype(str).str.strip().str.upper()
    s=s.str.replace(r'^\s*([01])\s*:.+$',r'\1',regex=True)
    s=s.replace({'DECEASED':'1','DEAD':'1','LIVING':'0','ALIVE':'0',
                 'TRUE':'1','FALSE':'0'})
    bad=~s.isin(['0','1'])
    if bad.any():
        raise ValueError(f'Unrecognized OS_STATUS examples: {s[bad].head(5).tolist()}')
    return s.astype(int)


def load_patients(patient_tsv, sample_tsv, sample_policy='error', zero_time='error',
                  panel_assignment=None, panel_source='error'):
    cp=table(patient_tsv);cs=table(sample_tsv)
    require(cp,['PATIENT_ID','OS_MONTHS','OS_STATUS','SEX','AGE_GROUP','DRUG_TYPE'], 'clinical_patient')
    require(cs,['PATIENT_ID','SAMPLE_ID','CANCER_TYPE','SAMPLE_TYPE',
                'GENE_PANEL','TMB_NONSYNONYMOUS'], 'clinical_sample')
    cp=cp.copy(); cs=cs.copy()
    for df,kind in ((cp,'patient'),(cs,'sample')):
        df['PATIENT_ID']=df['PATIENT_ID'].astype('string').str.strip()
        if df['PATIENT_ID'].isna().any() or (df['PATIENT_ID']=='').any():
            raise ValueError(f'{kind} table contains empty PATIENT_ID')
    if cp['PATIENT_ID'].duplicated().any():
        raise ValueError('clinical_patient has duplicated PATIENT_ID; reconcile source records first')
    cs['SAMPLE_ID']=cs['SAMPLE_ID'].astype('string').str.strip()
    if cs.SAMPLE_ID.isna().any() or cs.SAMPLE_ID.duplicated().any():
        raise ValueError('SAMPLE_ID must be nonmissing and unique in clinical_sample')
    multiples=cs.loc[cs.PATIENT_ID.duplicated(keep=False),'PATIENT_ID'].nunique()
    if multiples and sample_policy=='error':
        raise ValueError(f'{multiples} patients have multiple samples; provide --sample-policy first ONLY if justified')
    if sample_policy=='first':
        cs=cs.drop_duplicates('PATIENT_ID',keep='first')
    panel_disagreements=0
    if panel_assignment is not None:
        assayed=table(panel_assignment)
        require(assayed,['SAMPLE_ID','mutations'],'sample-to-panel assignment')
        assayed=assayed[['SAMPLE_ID','mutations']].rename(columns={'mutations':'ASSAY_PANEL'})
        if assayed.SAMPLE_ID.duplicated().any():
            raise ValueError('Duplicate SAMPLE_ID in sample-to-panel assignment')
        cs=cs.merge(assayed,on='SAMPLE_ID',how='left',validate='one_to_one')
        if cs.ASSAY_PANEL.isna().any():
            raise ValueError('Some selected samples have no mutation assay-panel assignment')
        disagreement=(cs.GENE_PANEL.astype(str)!=cs.ASSAY_PANEL.astype(str))
        panel_disagreements=int(disagreement.sum())
        if panel_disagreements:
            if panel_source=='error':
                raise ValueError(f'{panel_disagreements} samples have conflicting clinical vs assay panel names; set --panel-source clinical or assay only after checking source provenance')
            if panel_source=='assay':
                cs.loc[disagreement,'GENE_PANEL']=cs.loc[disagreement,'ASSAY_PANEL']
            elif panel_source!='clinical':
                raise ValueError('Unknown panel source policy')
    patient=cp.merge(cs,on='PATIENT_ID',how='inner',validate='one_to_one',suffixes=('','_sample'))
    if len(patient)!=len(cp):
        raise ValueError('Some clinical patients have no unambiguous matched clinical sample')
    patient['time_months']=pd.to_numeric(patient.OS_MONTHS,errors='coerce')
    patient['event']=read_death_status(patient.OS_STATUS)
    if patient.time_months.isna().any() or (patient.time_months < 0).any():
        raise ValueError('OS_MONTHS must be numeric and nonnegative')
    zero_n=int((patient.time_months==0).sum())
    if zero_n:
        if zero_time=='error':
            raise ValueError(f'{zero_n} patients have zero-month follow-up; original protocol must specify how to handle them. You may explicitly use --zero-time exclude, but that changes N.')
        if zero_time=='exclude':
            patient=patient[patient.time_months>0].copy()
        elif zero_time=='keep':
            pass  # Preserve source-reported zero-month times exactly; audit records count.
        else:
            raise ValueError('Unsupported zero_time policy')
    patient=patient.rename(columns={'PATIENT_ID':'patient_id','TMB_NONSYNONYMOUS':'TMB'})
    for c in CLINICAL:
        if c not in patient:raise ValueError(f'Missing merged clinical attribute {c}')
    if patient.GENE_PANEL.isna().any():raise ValueError('GENE_PANEL missing')
    if pd.to_numeric(patient.TMB,errors='coerce').notna().sum()==0:
        raise ValueError('TMB values could not be parsed; check raw input')
    keep=['patient_id',*CLINICAL,'TMB','time_months','event']
    out=patient[keep].copy()
    out['patient_id']=out['patient_id'].astype(str)
    sample_map=patient[['SAMPLE_ID','patient_id']].copy()
    return out,sample_map,{'raw_patients':len(cp),'samples':len(cs),'zero_month_patients':zero_n,
                           'after_policy_patients':len(out),'multi_sample_patients':int(multiples),
                           'panel_assignment_conflicts':panel_disagreements,
                           'panel_assignment_policy':panel_source if panel_assignment else 'not checked'}


def load_mutations(path,sample_map,strict=True):
    src=table(path)
    require(src,['Tumor_Sample_Barcode','Hugo_Symbol','Variant_Classification'],'mutation MAF')
    src=src[src['Variant_Classification'].isin(FUNCTIONAL_CLASSES)].copy()
    src['Tumor_Sample_Barcode']=src.Tumor_Sample_Barcode.astype(str).str.strip()
    match=src.merge(sample_map,left_on='Tumor_Sample_Barcode',right_on='SAMPLE_ID',how='left',validate='many_to_one')
    not_found=int(match.patient_id.isna().sum())
    if strict and not_found:
        raise ValueError(f'{not_found} retained functional mutation rows could not be linked to selected clinical sample IDs')
    result=match[['patient_id','Hugo_Symbol']].rename(columns={'Hugo_Symbol':'gene'}).dropna().drop_duplicates()
    result['gene']=result.gene.astype(str).str.strip()
    result=result[result.gene.ne('')]
    return result,{'maf_retained_rows':len(src),'maf_unmapped_rows':not_found}


def load_sv(path,sample_map,strict=True,allowed_status=None):
    src=table(path)
    require(src,['Sample_Id','Site1_Hugo_Symbol','Site2_Hugo_Symbol'],'structural variants')
    if allowed_status is not None:
        if 'SV_Status' not in src:
            raise ValueError('SV_Status missing but --sv-status is requested')
        src=src[src.SV_Status.isin(allowed_status)].copy()
    src['Sample_Id']=src.Sample_Id.astype(str).str.strip()
    match=src.merge(sample_map,left_on='Sample_Id',right_on='SAMPLE_ID',how='left',validate='many_to_one')
    not_found=int(match.patient_id.isna().sum())
    if strict and not_found:
        raise ValueError(f'{not_found} SV rows could not be linked to selected clinical sample IDs')
    items=[]
    for col in ('Site1_Hugo_Symbol','Site2_Hugo_Symbol'):
        a=match[['patient_id',col]].rename(columns={col:'gene'}).dropna()
        a['gene']=a['gene'].astype(str).str.strip()
        a=a[~a.gene.isin(['','nan','.','None'])]
        items.append(a)
    result=pd.concat(items,ignore_index=True).drop_duplicates() if items else pd.DataFrame(columns=['patient_id','gene'])
    return result,{'sv_input_rows':len(src),'sv_unmapped_rows':not_found}


def export_reference_mappings(patient_tsv, sample_tsv, maf_path, out):
    """Create shareable NON-PATIENT metadata from locally downloaded source data.

    This catalog is descriptive. It is NEVER used to learn one-hot vocabularies
    for train/validation/test partitions, to avoid full-cohort feature leakage.
    """
    import pickle
    cp, cs, maf = table(patient_tsv), table(sample_tsv), table(maf_path)
    require(cp,['SEX','AGE_GROUP','DRUG_TYPE','OS_STATUS'],'patient table')
    require(cs,['CANCER_TYPE','SAMPLE_TYPE','GENE_PANEL','TMB_NONSYNONYMOUS'],'sample table')
    require(maf,['Variant_Classification'],'mutation MAF')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    categories={}
    for name in CLINICAL:
        source=cp if name in cp else cs
        categories[name]=sorted(map(str,source[name].dropna().unique()))
    mappings={
        'SEX':('data_clinical_patient.txt','SEX','categorical'),
        'AGE_GROUP':('data_clinical_patient.txt','AGE_GROUP','categorical'),
        'DRUG_TYPE':('data_clinical_patient.txt','DRUG_TYPE','categorical'),
        'CANCER_TYPE':('data_clinical_sample.txt','CANCER_TYPE','categorical'),
        'SAMPLE_TYPE':('data_clinical_sample.txt','SAMPLE_TYPE','categorical'),
        'GENE_PANEL':('data_clinical_sample.txt','GENE_PANEL','categorical'),
        'TMB':('data_clinical_sample.txt','TMB_NONSYNONYMOUS','float'),
        'time_months':('data_clinical_patient.txt','OS_MONTHS','float'),
        'event':('data_clinical_patient.txt','OS_STATUS','binary_status'),
        'patient_id':('data_clinical_patient.txt','PATIENT_ID','identifier'),
    }
    field_rows=[{'output_name':k,'input_file':v[0],'input_column':v[1],
                 'value_kind':v[2]} for k,v in mappings.items()]
    category_rows=[{'field':field,'category':value,'reference_index':i}
                   for field,values in categories.items()
                   for i,value in enumerate(values)]
    classes=sorted(map(str,maf.Variant_Classification.dropna().unique()))
    variant_rows=[{'variant_classification':v,
                   'retained_as_functional':int(v in FUNCTIONAL_CLASSES)} for v in classes]
    status_rows=[]
    for v in sorted(cp.OS_STATUS.dropna().unique()):
        event=int(read_death_status(pd.Series([v])).iloc[0])
        status_rows.append({'source_OS_STATUS':v,'event':event})
    base='https://media.githubusercontent.com/media/cBioPortal/datahub/master/reference_data/gene_panels/'
    panels=[{'panel':p,'reference_file':f'data_gene_panel_{p.lower()}.txt',
             'download_url':base+f'data_gene_panel_{p.lower()}.txt',
             'contains_assayed_gene_list':False} for p in sorted(categories['GENE_PANEL'])]
    tables={
        'clinical_field_mapping.csv':field_rows,
        'clinical_category_reference.csv':category_rows,
        'mutation_class_mapping.csv':variant_rows,
        'survival_status_mapping.csv':status_rows,
        'panel_source_mapping.csv':panels,
    }
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(out/name,index=False)
    objects={
        'clinical_category_reference.pkl':{
            'description':'Cohort-wide descriptive categories; NEVER train-fitted vocabulary',
            'source':'tmb_mskcc_2018 clinical tables','categories':categories},
        'clinical_field_mapping.pkl':{
            'description':'Raw field mapping; no patient data',
            'fields':{k:{'source_file':v[0],'source_column':v[1],'kind':v[2]}
                      for k,v in mappings.items()}},
        'mutation_class_mapping.pkl':{
            'description':'Functional event classification rules',
            'observed_classes':variant_rows,'included_classes':sorted(FUNCTIONAL_CLASSES)},
        'panel_reference.pkl':{
            'description':'Candidate DataHub reference URLs only; not actual panel target gene lists',
            'panels':panels},
        'feature_contract.pkl':{
            'description':'Model input feature contract; no patient feature matrices',
            'clinical':CLINICAL,
            'raw_features':['TMB','mutations','sv','coverage'],
            'pathway_scores':['mut_fraction','sv_fraction','either_fraction','assayed_fraction'],
            'hallmark_pathways_expected':50},
    }
    for name,content in objects.items():
        with (out/name).open('wb') as f:pickle.dump(content,f,protocol=4)
    manifest={'asset_type':'public-non-patient-level-reference-mappings',
              'status':'derived from caller-supplied public cohort',
              'source_study':'tmb_mskcc_2018',
              'source_files':['data_clinical_patient.txt','data_clinical_sample.txt',
                              'data_mutations.txt'],
              'reference_pkl_files':sorted(objects),
              'reference_mapping_csv_files':sorted(tables),
              'does_not_contain_patient_records':True,
              'does_not_contain_split_fitted_feature_vocabulary':True,
              'does_not_contain_assay_gene_target_lists':True,
              'does_not_contain_official_hallmark_genes':True,
              'does_not_contain_experiment_results':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return {'public_reference_pkl_files':sorted(objects),
            'public_reference_csv_files':sorted(tables)}


def prepare(args):
    out=Path(args.out)
    # Validate independently supplied reference files BEFORE making output directories.
    cov=pd.read_csv(args.panel_genes,dtype=str)
    require(cov,['panel','gene'],'panel-gene coverage')
    cov=cov[['panel','gene']].dropna().drop_duplicates()
    if cov.empty:raise ValueError('Panel-gene coverage mapping is empty')
    with open(args.hallmark_gmt,encoding='utf8') as f:
        sets=[line.split('\t',1)[0] for line in f if line.strip()]
    if len(sets)!=50 or len(set(sets))!=50:
        raise ValueError('Expected 50 unique official Hallmark gene sets')
    if not all(x.startswith('HALLMARK_') for x in sets):
        raise ValueError('The provided GMT is not named as an MSigDB Hallmark collection')
    patients,smap,audit=load_patients(args.clinical_patient,args.clinical_sample,
                                      args.sample_policy,args.zero_time,args.panel_assignment,args.panel_source)
    unknown_panels=set(patients.GENE_PANEL)-set(cov.panel)
    if unknown_panels:raise ValueError(f'Unmapped patient panels: {sorted(unknown_panels)}')
    mut,am=load_mutations(args.mutation_maf,smap,strict=not args.ignore_unmapped)
    sv,asv=load_sv(args.sv_calls,smap,strict=not args.ignore_unmapped,
                   allowed_status=args.sv_status)
    audit.update(am);audit.update(asv)
    audit.update({'mutations_patient_gene':len(mut),'sv_patient_gene':len(sv),
                  'panel_definitions':int(cov.panel.nunique()),'hallmark_sets':50,
                  'provenance':'Inputs are caller-supplied. No source rows or panel coverage inferred.'})
    out.mkdir(parents=True,exist_ok=True)
    patients.to_csv(out/'patients.csv',index=False)
    cov.to_csv(out/'panel_coverage.csv',index=False)
    mut.to_csv(out/'mutations.csv',index=False)
    sv.to_csv(out/'sv.csv',index=False)
    shutil.copyfile(args.hallmark_gmt,out/'hallmark.gmt')
    if getattr(args,'write_local_pkl',False):
        # Patient-level caches are written ONLY on the reader's local machine
        # after downloading source datasets. They are excluded from GitHub ZIP.
        import pickle
        local = {
            'patients_processed.pkl':patients,
            'mutation_events.pkl':mut,
            'sv_events.pkl':sv,
            'panel_coverage.pkl':cov,
        }
        pathways={}
        for line in open(args.hallmark_gmt,encoding='utf8'):
            fields=line.rstrip('\n').split('\t')
            if len(fields)>=3 and fields[0]:
                pathways[fields[0]]=fields[2:]
        local['hallmark_sets.pkl']=pathways
        local['data_schema.pkl']={'patients':list(patients.columns),
                                  'mutations':list(mut.columns),
                                  'sv':list(sv.columns),
                                  'panel_coverage':list(cov.columns),
                                  'hallmark_set_count':len(pathways)}
        for filename, obj in local.items():
            with (out/filename).open('wb') as fp:
                pickle.dump(obj,fp,protocol=4)
        audit['local_preprocessed_cache_files']=sorted(local)
    (out/'conversion_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clinical-patient',required=True)
    p.add_argument('--clinical-sample',required=True)
    p.add_argument('--mutation-maf',required=True)
    p.add_argument('--sv-calls',help='Original SV table; required for patient processing')
    p.add_argument('--panel-genes',help='Verified CSV with panel,gene; required for patient processing')
    p.add_argument('--hallmark-gmt',help='Licensed official 50-set GMT; required for patient processing')
    p.add_argument('--out',required=True)
    p.add_argument('--build-reference',action='store_true',help='Regenerate shareable, non-patient mapping CSV/PKL reference objects')
    p.add_argument('--write-local-pkl',action='store_true',help='Write processed PKL caches to the local output directory')
    p.add_argument('--sample-policy',choices=['error','first'],default='error')
    p.add_argument('--panel-assignment',help='Optional sample-to-mutation-panel table: SAMPLE_ID, mutations')
    p.add_argument('--panel-source',choices=['error','clinical','assay'],default='error',
                   help='Explicit resolution of conflicting panel metadata')
    p.add_argument('--zero-time',choices=['error','keep','exclude'],default='error')
    p.add_argument('--ignore-unmapped',action='store_true',help='Explicitly discard event rows without selected clinical sample match')
    p.add_argument('--sv-status',nargs='+',help='Optional exact allowed SV_Status values, verified from source documentation')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.build_reference:
        report=export_reference_mappings(args.clinical_patient,args.clinical_sample,
                                         args.mutation_maf,args.out)
    else:
        missing=[k for k in ('sv_calls','panel_genes','hallmark_gmt') if not getattr(args,k)]
        if missing:
            parser().error('Standard processing requires '+', '.join('--'+k.replace('_','-') for k in missing))
        report=prepare(args)
    print(json.dumps(report,ensure_ascii=False,indent=2))
