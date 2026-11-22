"""Data processing, survival analysis, evaluation and visualization utilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapezoid
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import json
import copy
from scipy.stats import norm
from torch import nn
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def _tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# Survival analysis and estimation
def cox_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Breslow partial log likelihood, normalized by number of events."""
    log_risk = log_risk.reshape(-1)
    time, event = time.reshape(-1), event.reshape(-1)
    order = torch.argsort(time, descending=True, stable=True)
    r, t, e = log_risk[order], time[order], event[order]
    _, counts = torch.unique_consecutive(t, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    log_denom = torch.logcumsumexp(r, dim=0)[ends]
    per_patient_denom = torch.repeat_interleave(log_denom, counts)
    n_events = torch.sum(e)
    if n_events.item() == 0:
        raise ValueError('Cox loss requires at least one observed event.')
    return -torch.sum(e * (r - per_patient_denom)) / n_events


def _cox_numpy_objective(beta, X, time, event, l2=0.0):
    """Exact objective/gradient for Breslow ties; stable log-risk shifting."""
    order = np.argsort(-time, kind='stable')
    x, t, e = X[order], time[order], event[order]
    r = x @ beta
    shift = float(np.max(r))
    ex = np.exp(r - shift)
    den = np.cumsum(ex)
    xden = np.cumsum(ex[:, None] * x, axis=0)
    ends = np.r_[np.flatnonzero(np.diff(t) != 0), len(t)-1]
    starts = np.r_[0, ends[:-1]+1]
    d = np.array([e[a:b+1].sum() for a,b in zip(starts,ends)], dtype=float)
    ne = e.sum()
    if ne == 0:
        raise ValueError('No observed events.')
    ll = np.sum(d * (np.log(den[ends]) + shift)) - np.dot(e, r)
    grad = np.sum((d / den[ends])[:,None] * xden[ends], axis=0) - e @ x
    return float(ll / ne + .5*l2*np.dot(beta,beta)), grad/ne + l2*beta


@dataclass
class RidgeCox:
    beta: np.ndarray | None = None
    l2: float = 0.1

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=np.float64)
        result = minimize(_cox_numpy_objective, np.zeros(X.shape[1]),
                          args=(X, time, event, self.l2), method='L-BFGS-B',
                          jac=True, options={'maxiter': 350, 'ftol': 1e-10})
        if not result.success and np.linalg.norm(result.jac) > 1e-3:
            raise RuntimeError('Ridge-Cox optimization failed: '+result.message)
        self.beta = result.x
        return self

    def predict(self, X):
        if self.beta is None:
            raise RuntimeError('Ridge-Cox has not been fitted')
        return np.asarray(X, dtype=float) @ self.beta


@dataclass
class BreslowBaseline:
    event_times: np.ndarray
    cumulative_hazard: np.ndarray

    @classmethod
    def fit(cls, time, event, log_risk):
        time = np.asarray(time, float)
        event = np.asarray(event, bool)
        risk = np.asarray(log_risk, float)
        unique = np.unique(time[event])
        if len(unique) == 0:
            raise ValueError('Baseline hazard requires observed deaths')
        # Stable, shifted exp prevents overflow of exp(risk).
        shift = float(risk.max())
        exp_shift = np.exp(risk - shift)
        increments = []
        for t in unique:
            d = np.count_nonzero((time == t) & event)
            denom = exp_shift[time >= t].sum()
            increments.append(d * np.exp(-shift) / denom)
        return cls(unique, np.cumsum(increments))

    def hazard_at(self, horizons):
        horizons = np.asarray(horizons, float)
        idx = np.searchsorted(self.event_times, horizons, side='right') - 1
        return np.where(idx >= 0, self.cumulative_hazard[np.maximum(idx, 0)], 0.)

    def survival(self, log_risk, horizons):
        r = np.asarray(log_risk, float).reshape(-1)
        t = np.asarray(horizons, float).reshape(-1)
        # H0 is derived ONLY from the training cohort.
        return np.exp(-np.exp(np.clip(r[:,None], -40, 40)) * self.hazard_at(t)[None,:])


@dataclass
class KaplanMeier:
    event_times: np.ndarray
    survival_values: np.ndarray

    @classmethod
    def fit(cls, time, event):
        time = np.asarray(time, float)
        event = np.asarray(event, bool)
        unique = np.unique(time)
        n = len(time)
        vals = []
        surv = 1.
        for t in unique:
            deaths = np.count_nonzero((time == t) & event)
            at_risk = np.count_nonzero(time >= t)
            if at_risk:
                surv *= 1. - deaths/at_risk
            vals.append(surv)
        return cls(unique, np.array(vals))

    def at(self, t, left=False):
        x = np.asarray(t, float)
        idx = np.searchsorted(self.event_times, x, side='left' if left else 'right') - 1
        return np.where(idx >= 0, self.survival_values[np.maximum(idx,0)], 1.)


def harrell_c(time, event, risk):
    """Harrell concordance; strict time ordering, tied risks receive half credit.

    Equal event times are excluded. The same tie convention is applied
    consistently across evaluations in this implementation.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    risk = np.asarray(risk, float)
    ti, tj = time[:,None], time[None,:]
    comparable = (ti < tj) & event[:,None]
    n = int(comparable.sum())
    if n == 0:
        return float('nan')
    delta = risk[:,None] - risk[None,:]
    return float(((delta > 0) * comparable).sum() + .5*((delta == 0)*comparable).sum()) / n


def ipcw_brier(train_time, train_event, test_time, test_event, survival, horizon):
    """IPCW Brier using censoring KM from training partition, not test data."""
    G = KaplanMeier.fit(train_time, ~np.asarray(train_event, bool))
    t, e = np.asarray(test_time, float), np.asarray(test_event, bool)
    s = np.asarray(survival, float).reshape(-1)
    before = (t <= horizon) & e
    after = t > horizon
    w1 = 1. / np.clip(G.at(t, left=True), 1e-8, None)
    w2 = 1. / max(float(G.at(horizon)), 1e-8)
    return float((np.sum(before * w1 * s**2) + np.sum(after * w2 * (1-s)**2)) / len(t))


def dynamic_auc(train_time, train_event, test_time, test_event, risk, horizon):
    """Cumulative/dynamic AUC with training-derived IPCW for cases."""
    t, e, risk = np.asarray(test_time,float), np.asarray(test_event,bool), np.asarray(risk,float)
    case = (t <= horizon) & e
    control = t > horizon
    if not case.any() or not control.any():
        return float('nan')
    G = KaplanMeier.fit(train_time, ~np.asarray(train_event,bool))
    weights = 1./np.clip(G.at(t[case], left=True), 1e-8, None)
    delta = risk[case,None] - risk[None,control]
    return float(np.sum(weights[:,None] * ((delta > 0) + .5*(delta == 0))) /
                 (np.sum(weights)*control.sum()))


def km_ece(test_time, test_event, survival, horizon, n_bins=5):
    """KM-bin expected calibration error; binning is a documented implementation choice."""
    pred = np.asarray(survival,float)
    t, e = np.asarray(test_time,float), np.asarray(test_event,bool)
    # Equal-sized predicted survival quantile bins.
    bins = np.array_split(np.argsort(pred), min(n_bins,len(pred)))
    obs, estimated, sizes = [], [], []
    for idx in bins:
        if len(idx) == 0: continue
        obs.append(float(KaplanMeier.fit(t[idx],e[idx]).at(horizon)))
        estimated.append(float(pred[idx].mean()))
        sizes.append(len(idx))
    ece = float(np.average(np.abs(np.asarray(obs)-np.asarray(estimated)), weights=sizes))
    return ece, {'observed':obs,'predicted':estimated,'count':sizes}


def evaluate_survival(train_time, train_event, test_time, test_event, risk, survival_fn,
                      horizons=(12.,24.), ibs_grid=None):
    if ibs_grid is None:
        ibs_grid = np.linspace(1.,24.,24)
    out = {'c_index':harrell_c(test_time,test_event,risk)}
    for h in horizons:
        s = survival_fn([h])[:,0]
        out[f'auc_{int(h)}'] = dynamic_auc(train_time,train_event,test_time,test_event,risk,h)
        out[f'brier_{int(h)}'] = ipcw_brier(train_time,train_event,test_time,test_event,s,h)
        out[f'ece_{int(h)}'], _ = km_ece(test_time,test_event,s,h)
    bs = [ipcw_brier(train_time,train_event,test_time,test_event,survival_fn([h])[:,0],h)
          for h in ibs_grid]
    out['ibs_1_24'] = float(trapezoid(bs,ibs_grid)/(ibs_grid[-1]-ibs_grid[0]))
    return out



# Clinical and genomic feature processing
CLINICAL = ['SEX','AGE_GROUP','CANCER_TYPE','SAMPLE_TYPE','DRUG_TYPE','GENE_PANEL']


REQUIRED = ['patient_id',*CLINICAL,'TMB']


def read_gmt(path):
    sets = {}
    with open(path,encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 3: continue
            if fields[0] in sets:
                raise ValueError('Duplicate Hallmark set: '+fields[0])
            sets[fields[0]] = set(fields[2:])
    if not sets:
        raise ValueError('GMT must contain at least one pathway')
    return sets


def _read_events(path, filename):
    p = Path(path)/filename
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Supply an empty CSV with header patient_id,gene if no calls.')
    df = pd.read_csv(p, dtype={'patient_id':str,'gene':str})
    if not {'patient_id','gene'}.issubset(df.columns):
        raise ValueError(f'{filename} must contain patient_id,gene')
    return df[['patient_id','gene']].dropna().drop_duplicates().reset_index(drop=True)


@dataclass
class Cohort:
    patients: pd.DataFrame
    panel_coverage: pd.DataFrame
    mutations: pd.DataFrame
    sv: pd.DataFrame
    hallmark: dict

    @classmethod
    def load(cls, folder, require_hallmark_50=True):
        folder = Path(folder)
        patients = pd.read_csv(folder/'patients.csv', dtype={'patient_id':str})
        missing = set(REQUIRED)-set(patients.columns)
        if missing:
            raise ValueError(f'patients.csv missing columns: {sorted(missing)}')
        if (patients.patient_id.isna().any() or
                patients.patient_id.astype(str).str.strip().eq('').any() or
                patients.patient_id.duplicated().any()):
            raise ValueError('patient_id must be unique and nonempty')
        patients = patients.set_index('patient_id',drop=False)
        for col in CLINICAL:
            patients[col] = patients[col].fillna('__MISSING__').astype(str)
        patients['TMB'] = pd.to_numeric(patients['TMB'],errors='coerce')
        if (patients['TMB'].dropna() < 0).any():
            raise ValueError('TMB must be nonnegative; check input data')
        if {'time_months','event'}.issubset(patients.columns):
            patients['time_months'] = pd.to_numeric(patients['time_months'],errors='coerce')
            patients['event'] = pd.to_numeric(patients['event'],errors='coerce')
            if patients.time_months.isna().any() or (patients.time_months<0).any():
                raise ValueError('time_months must be nonnegative and nonmissing; zero-month records require documented handling')
            if not patients.event.isin([0,1]).all():
                raise ValueError('event must be 0 (censored) or 1 (death)')
        coverage = pd.read_csv(folder/'panel_coverage.csv',dtype=str)
        if not {'panel','gene'}.issubset(coverage.columns):
            raise ValueError('panel_coverage.csv needs panel,gene columns')
        coverage = coverage[['panel','gene']].dropna().drop_duplicates()
        if (coverage['panel'].str.strip().eq('').any() or
                coverage['gene'].str.strip().eq('').any()):
            raise ValueError('Panel coverage contains empty panel or gene labels')
        missing_panels = set(patients.GENE_PANEL)-set(coverage.panel)
        if missing_panels:
            raise ValueError('No coverage definition for panels: '+str(sorted(missing_panels)))
        mutations = _read_events(folder,'mutations.csv')
        sv = _read_events(folder,'sv.csv')
        valid = set(patients.index)
        for name, df in [('mutations',mutations),('sv',sv)]:
            unknown = set(df.patient_id)-valid
            if unknown: raise ValueError(f'{name} contains unknown patient IDs: {list(unknown)[:5]}')
        hallmark = read_gmt(folder/'hallmark.gmt')
        if require_hallmark_50 and len(hallmark)!=50:
            raise ValueError(f'Paper specifies 50 Hallmark pathways; got {len(hallmark)}')
        return cls(patients,coverage,mutations,sv,hallmark)

    def labels(self, ids):
        if not {'time_months','event'}.issubset(self.patients.columns):
            raise ValueError('Cohort does not contain survival outcomes')
        df = self.patients.loc[list(ids)]
        return df.time_months.to_numpy(float), df.event.to_numpy(int)


@dataclass
class Arrays:
    clinical: np.ndarray
    genomic: np.ndarray
    anchor: np.ndarray
    aspects: np.ndarray
    ids: list


class FeatureBuilder:
    """Train-fitted, leakage-safe transform; panel map is assay metadata.

    Binary gene channels are MUT, SV, COV. Pathway order per gene set:
    SNV fraction, SV fraction, either-event fraction, coverage fraction.
    """
    def __init__(self):
        self.fitted = False

    def _setup_genes(self, cohort):
        self.genes = sorted(set(cohort.panel_coverage.gene))
        self.pathways = sorted(cohort.hallmark)
        self.hallmark_definitions = {k:set(v) for k,v in cohort.hallmark.items()}
        self.gene_to_idx = {g:i for i,g in enumerate(self.genes)}
        self.panels = {}
        for panel, df in cohort.panel_coverage.groupby('panel'):
            cov = np.zeros(len(self.genes),dtype=np.float32)
            for g in df.gene:
                cov[self.gene_to_idx[g]] = 1.
            self.panels[panel] = cov
        self.pathway_matrix = np.array([[float(g in cohort.hallmark[p]) for p in self.pathways]
                                         for g in self.genes],dtype=np.float32)
        # Eq. (8) denominator is the FULL Hallmark set, including genes never
        # measured by any supplied panel. Do not restrict to gene vocabulary.
        self.pathway_size = np.array([max(1,len(cohort.hallmark[p])) for p in self.pathways],
                                     dtype=np.float32)
        self.genomic_names = ([f'MUT:{g}' for g in self.genes] +
                              [f'SV:{g}' for g in self.genes] +
                              [f'COV:{g}' for g in self.genes] +
                              ['log1p_TMB','mutation_burden','sv_burden'] +
                              [f'{p}:{kind}' for p in self.pathways
                               for kind in ('mutation','sv','either','coverage')])

    def _raw(self, cohort, ids):
        ids = [str(i) for i in ids]
        p = cohort.patients.loc[ids]
        n, g = len(ids), len(self.genes)
        cov = np.stack([self.panels[str(panel)] for panel in p.GENE_PANEL],axis=0)
        mut = np.zeros((n,g),dtype=np.float32)
        sv = np.zeros((n,g),dtype=np.float32)
        id_to_row = {id_:i for i,id_ in enumerate(ids)}
        for events, target in [(cohort.mutations,mut),(cohort.sv,sv)]:
            for pid,gene in events[['patient_id','gene']].itertuples(index=False,name=None):
                if pid in id_to_row and gene in self.gene_to_idx:
                    i,j = id_to_row[pid],self.gene_to_idx[gene]
                    # Paper Eq. (4): calls outside the assay are NOT observed.
                    target[i,j] = cov[i,j]
        mut *= cov
        sv *= cov
        either = np.maximum(mut,sv)
        tmb = np.log1p(np.maximum(p.TMB.to_numpy(float),0))
        tmb[np.isnan(p.TMB.to_numpy(float))] = np.nan
        burden = np.stack([tmb,mut.sum(axis=1),sv.sum(axis=1)],axis=1)
        den = np.maximum(cov @ self.pathway_matrix,1.)
        p_mut = (mut @ self.pathway_matrix)/den
        p_sv = (sv @ self.pathway_matrix)/den
        p_either = (either @ self.pathway_matrix)/den
        p_cov = (cov @ self.pathway_matrix)/self.pathway_size[None,:]
        pathway = np.stack([p_mut,p_sv,p_either,p_cov],axis=-1).reshape(n,-1)
        numeric = np.concatenate([burden,pathway],axis=1)
        return p, mut, sv, cov, numeric, p_either, p_cov

    def fit(self, cohort, train_ids):
        self._setup_genes(cohort)
        p, mut, sv, cov, numeric, p_either, p_cov = self._raw(cohort,train_ids)
        self.encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        self.encoder.fit(p[CLINICAL])
        self.imputer = SimpleImputer(strategy='median',keep_empty_features=True)
        self.imputer.fit(numeric)
        self.scaler = StandardScaler()
        self.scaler.fit(self.imputer.transform(numeric))
        self.clinical_names = list(self.encoder.get_feature_names_out(CLINICAL))
        self.aspect_names = ([f'clinical:{x}' for x in self.clinical_names] +
                             ['burden:log1p_TMB','burden:mutation','burden:sv'] +
                             [f'pathway:{x}:either' for x in self.pathways] +
                             [f'pathway:{x}:coverage' for x in self.pathways] +
                             ['panel:covered_fraction'])
        raw_aspects = self._aspect_raw(p, numeric, p_either, p_cov, cov)
        self.aspect_scaler = StandardScaler().fit(raw_aspects)
        self.fitted = True
        return self

    def _aspect_raw(self, p, numeric, p_either, p_cov, cov):
        clinical = self.encoder.transform(p[CLINICAL])
        filled = self.imputer.transform(numeric)
        return np.concatenate([clinical,filled[:,:3],p_either,p_cov,
                               cov.mean(axis=1,keepdims=True)],axis=1)

    def transform(self, cohort, ids):
        if not self.fitted: raise RuntimeError('Call fit() on training patients first')
        if {k:set(v) for k,v in cohort.hallmark.items()} != self.hallmark_definitions:
            raise ValueError('Hallmark definitions differ from training: refusing incompatible inference')
        for panel in cohort.patients.loc[list(ids),'GENE_PANEL'].unique():
            if panel not in self.panels:
                raise ValueError(f'Unseen sequencing panel {panel}; its gene coverage is not fitted')
            current=set(cohort.panel_coverage.loc[cohort.panel_coverage.panel==panel,'gene'])
            expected={g for g,j in self.gene_to_idx.items() if self.panels[panel][j]}
            if current!=expected:
                raise ValueError(f'Coverage definition for {panel} differs from training')
        p,mut,sv,cov,numeric,p_either,p_cov = self._raw(cohort,ids)
        clinical = self.encoder.transform(p[CLINICAL]).astype(np.float32)
        continuous = self.scaler.transform(self.imputer.transform(numeric)).astype(np.float32)
        genomic = np.concatenate([mut,sv,cov,continuous],axis=1).astype(np.float32)
        anchor = np.concatenate([clinical,continuous[:,:3]],axis=1).astype(np.float32)
        aspects = self.aspect_scaler.transform(self._aspect_raw(p,numeric,p_either,p_cov,cov)).astype(np.float32)
        assert genomic.shape[1] == len(self.genomic_names)
        assert aspects.shape[1] == len(self.aspect_names)
        return Arrays(clinical,genomic,anchor,aspects,list(map(str,ids)))



# Synthetic data for functional tests
def generate_demo(folder,n=160,genes=40,seed=2026):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    names=[f'DEMO_GENE_{i:03d}' for i in range(genes)]
    panels={'DEMO_P1':names[:genes//2],
            'DEMO_P2':names[:int(genes*.75)],'DEMO_P3':names}
    pd.DataFrame([{'panel':p,'gene':g} for p,gl in panels.items() for g in gl]).to_csv(folder/'panel_coverage.csv',index=False)
    with open(folder/'hallmark.gmt','w',encoding='utf-8') as f:
        for i in range(50):
            sampled=rng.choice(names,size=min(9,genes),replace=False)
            f.write(f'DEMO_HALLMARK_{i+1:02d}\tsynthetic_not_msigdb\t'+'\t'.join(sampled)+'\n')
    records=[];muts=[];svs=[]
    for i in range(n):
        pid=f'DEMO_{i:04d}'
        sex=rng.choice(['Male','Female']);age=rng.choice(['<65','>=65'])
        cancer=rng.choice(['Melanoma','NSCLC','Other'])
        drug=rng.choice(['anti-PD1','anti-CTLA4','combination'])
        panel=rng.choice(list(panels))
        sample=rng.choice(['Primary','Metastatic'])
        tmb=float(rng.lognormal(1.4,.7))
        cov=panels[panel]
        n_mut=int(rng.poisson(1.5+tmb/8))
        n_sv=int(rng.poisson(.5))
        mutated=rng.choice(cov,size=min(n_mut,len(cov)),replace=False)
        altered=rng.choice(cov,size=min(n_sv,len(cov)),replace=False)
        muts.extend({'patient_id':pid,'gene':g} for g in mutated)
        svs.extend({'patient_id':pid,'gene':g} for g in altered)
        log_hazard = (.5*(age=='>=65')+.35*(cancer=='Other')+.2*(drug=='anti-CTLA4')+
                      .12*np.log1p(tmb)+.06*len(mutated)-.15*(sex=='Female'))
        # Synthetic right-censored OS in months.
        true_time=float(rng.exponential(scale=42*np.exp(-log_hazard)))+.1
        censor=float(rng.uniform(6,50))
        records.append({'patient_id':pid,'SEX':sex,'AGE_GROUP':age,'CANCER_TYPE':cancer,
                        'SAMPLE_TYPE':sample,'DRUG_TYPE':drug,'GENE_PANEL':panel,
                        'TMB':tmb,'time_months':min(true_time,censor),'event':int(true_time<=censor)})
    pd.DataFrame(records).to_csv(folder/'patients.csv',index=False)
    pd.DataFrame(muts,columns=['patient_id','gene']).to_csv(folder/'mutations.csv',index=False)
    pd.DataFrame(svs,columns=['patient_id','gene']).to_csv(folder/'sv.csv',index=False)
    return folder



# Dataset validation
def audit_cohort(cohort):
    p=cohort.patients
    coverage={name:set(df.gene) for name,df in cohort.panel_coverage.groupby('panel')}
    panel_of=p.GENE_PANEL.to_dict()
    warnings=[];offpanel={}
    for name,df in [('mutations',cohort.mutations),('sv',cohort.sv)]:
        invalid=[(pid,gene) for pid,gene in df[['patient_id','gene']].itertuples(index=False,name=None)
                 if gene not in coverage.get(panel_of[pid],set())]
        offpanel[name]=len(invalid)
        if invalid:
            warnings.append(f'{len(invalid)} {name} calls not in patient panel coverage; Eq. (4) will mask these calls')
    all_covered=set().union(*coverage.values()) if coverage else set()
    in_sets=set().union(*cohort.hallmark.values())
    overlap=len(all_covered&in_sets)
    if overlap==0:warnings.append('No gene symbols overlap between panels and Hallmark GMT: check gene ID harmonization')
    if p.TMB.isna().sum():warnings.append(f'{int(p.TMB.isna().sum())} missing TMB values will be imputed using training-only median')
    result={'patients':len(p),'deaths':int(p.event.sum()) if 'event' in p else None,
            'censored':int((p.event==0).sum()) if 'event' in p else None,
            'panels':{k:len(v) for k,v in coverage.items()},
            'panel_union_genes':len(all_covered),'hallmark_sets':len(cohort.hallmark),
            'panel_hallmark_gene_overlap':overlap,
            'mutation_calls':len(cohort.mutations),'sv_calls':len(cohort.sv),
            'offpanel_calls':offpanel,'missing_tmb':int(p.TMB.isna().sum()),
            'warnings':warnings}
    return result



# Survival baseline models
METHODS=['CoxPH','Ridge-Cox','LASSO-Cox','Elastic-Net Cox',
         'Random Survival Forest','DeepSurv','XGBoost-Cox','XGBoost-AFT']


def _stack(a):
    return np.concatenate([a.clinical,a.genomic],axis=1)


def _select_ridge(Xtr,Xval,train_labels,val_labels,lambdas):
    tt,te=train_labels;vt,ve=val_labels
    fitted=[]
    for lam in lambdas:
        model=RidgeCox(l2=lam).fit(Xtr,tt,te)
        fitted.append((harrell_c(vt,ve,model.predict(Xval)),-lam,model))
    return max(fitted,key=lambda v:(np.nan_to_num(v[0],nan=-1),v[1]))[2]


def _coxnet(Xtr,Xval,train_labels,val_labels,l1_ratio):
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
    except ImportError as e:
        raise ImportError('LASSO/Elastic-Net requires optional scikit-survival: pip install -r requirements-optional.txt') from e
    tt,te=train_labels;vt,ve=val_labels
    y=np.array(list(zip(te.astype(bool),tt)),dtype=[('event','?'),('time','<f8')])
    model=CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,n_alphas=35,alpha_min_ratio=.02,
                                 normalize=False,max_iter=100000)
    model.fit(Xtr,y)
    candidates=[]
    for alpha in model.alphas_:
        risk=model.predict(Xval,alpha=alpha)
        score=harrell_c(vt,ve,risk)
        candidates.append((score,alpha))
    best=max(candidates,key=lambda v:np.nan_to_num(v[0],nan=-1))[1]
    return model,best


def _rsf(Xtr,train_labels,seed):
    try:
        from sksurv.ensemble import RandomSurvivalForest
    except ImportError as e:
        raise ImportError('Random Survival Forest requires scikit-survival; install optional requirements') from e
    tt,te=train_labels
    y=np.array(list(zip(te.astype(bool),tt)),dtype=[('event','?'),('time','<f8')])
    model=RandomSurvivalForest(n_estimators=200,min_samples_split=10,
                               min_samples_leaf=5,max_features='sqrt',n_jobs=2,random_state=seed)
    model.fit(Xtr,y)
    return model


def _deepsurv(Xtr,Xval,train_labels,val_labels,config,seed):
    torch.manual_seed(seed)
    device=config['device']
    model=nn.Sequential(nn.Linear(Xtr.shape[1],128),nn.GELU(),nn.LayerNorm(128),nn.Dropout(.2),
                        nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64),nn.Dropout(.2),
                        nn.Linear(64,1)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['predictor_lr'],weight_decay=config['predictor_weight_decay'])
    x=_tensor(Xtr,device);t=_tensor(train_labels[0],device);e=_tensor(train_labels[1],device)
    best=-np.inf;state=None;patience=0
    for epoch in range(config['predictor_epochs']):
        model.train();opt.zero_grad()
        loss=cox_loss(model(x).flatten(),t,e)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval()
        with torch.no_grad():valrisk=model(_tensor(Xval,device)).flatten().cpu().numpy()
        score=harrell_c(*val_labels,valrisk)
        if np.isfinite(score) and score>best+1e-7:
            best=score;state=copy.deepcopy(model.state_dict());patience=0
        else:patience+=1
        if patience>=config['predictor_patience']:break
    if state is None:raise RuntimeError('DeepSurv no valid checkpoint')
    model.load_state_dict(state);model.eval()
    return model


def _xgb_fit(method,Xtr,Xval,train_labels,val_labels,seed,quick=False):
    try: import xgboost as xgb
    except ImportError as e:raise ImportError('XGBoost baselines require xgboost') from e
    tt,te=train_labels;vt,ve=val_labels
    if method=='XGBoost-Cox':
        train=xgb.DMatrix(Xtr,label=np.where(te==1,tt,-tt))
        valid=xgb.DMatrix(Xval,label=np.where(ve==1,vt,-vt))
        params={'objective':'survival:cox','eval_metric':'cox-nloglik','eta':.04,
                'max_depth':3,'subsample':.85,'colsample_bytree':.85,
                'lambda':1.,'seed':seed,'nthread':2}
    else:
        train=xgb.DMatrix(Xtr);valid=xgb.DMatrix(Xval)
        for dm,t,e in ((train,tt,te),(valid,vt,ve)):
            dm.set_float_info('label_lower_bound',t)
            dm.set_float_info('label_upper_bound',np.where(e==1,t,np.inf))
        params={'objective':'survival:aft','eval_metric':'aft-nloglik',
                'aft_loss_distribution':'normal','aft_loss_distribution_scale':1.,
                'eta':.04,'max_depth':3,'subsample':.85,'colsample_bytree':.85,
                'lambda':1.,'seed':seed,'nthread':2}
    booster=xgb.train(params,train,num_boost_round=50 if quick else 400,
                      evals=[(valid,'validation')],early_stopping_rounds=12 if quick else 25,
                      verbose_eval=False)
    return booster


def run_baseline(name,arrays,labels,config,seed=11,quick=False):
    """Return (risk_test, survival_fn(horizons), metadata)."""
    if name not in METHODS:raise ValueError(name)
    tr,va,te=(arrays[x] for x in ('train','val','test'))
    trlab,vlab=labels['train'],labels['val']
    Xtr,Xval,Xte=(_stack(a) for a in (tr,va,te))
    extra={}
    if name in ('CoxPH','Ridge-Cox'):
        model=_select_ridge(tr.anchor,va.anchor,trlab,vlab,
                            [0.] if name=='CoxPH' else config['ridge_lambdas'])
        risk_train=model.predict(tr.anchor)
        risk_test=model.predict(te.anchor)
        extra['l2']=model.l2
    elif name in ('LASSO-Cox','Elastic-Net Cox'):
        model,alpha=_coxnet(tr.anchor,va.anchor,trlab,vlab,1. if name=='LASSO-Cox' else .5)
        risk_train=model.predict(tr.anchor,alpha=alpha)
        risk_test=model.predict(te.anchor,alpha=alpha)
        extra['alpha']=float(alpha)
    elif name=='Random Survival Forest':
        model=_rsf(Xtr,trlab,seed)
        risk_test=model.predict(Xte)
        functions=model.predict_survival_function(Xte)
        def survival_fn(horizons):
            # RSF survival functions are right-continuous step functions.
            ts=np.asarray(horizons,float)
            result=[]
            for fn in functions:
                idx=np.searchsorted(fn.x,ts,side='right')-1
                result.append(np.where(idx>=0,fn.y[np.maximum(idx,0)],1.))
            return np.stack(result)
        return risk_test,survival_fn,extra
    elif name=='DeepSurv':
        model=_deepsurv(Xtr,Xval,trlab,vlab,config,seed)
        with torch.no_grad():
            risk_train=model(_tensor(Xtr,config['device'])).flatten().cpu().numpy()
            risk_test=model(_tensor(Xte,config['device'])).flatten().cpu().numpy()
    else:
        booster=_xgb_fit(name,Xtr,Xval,trlab,vlab,seed,quick)
        import xgboost as xgb
        margin_tr=booster.predict(xgb.DMatrix(Xtr),output_margin=True,iteration_range=(0,booster.best_iteration+1))
        margin_te=booster.predict(xgb.DMatrix(Xte),output_margin=True,iteration_range=(0,booster.best_iteration+1))
        extra['best_iteration']=int(booster.best_iteration)
        if name=='XGBoost-AFT':
            # Normal AFT: predicted margin is log survival time; risk = -margin.
            risk_test=-margin_te
            def survival_fn(horizons):
                ts=np.asarray(horizons,float)
                return norm.sf(np.log(np.maximum(ts[None,:],1e-8))-margin_te[:,None])
            return risk_test,survival_fn,extra
        risk_train=margin_tr
        risk_test=margin_te
    baseline=BreslowBaseline.fit(*trlab,risk_train)
    return risk_test,lambda horizons:baseline.survival(risk_test,horizons),extra



# Reduced-feature cohort analysis
PORTABLE_CLINICAL=['SEX','AGE_GROUP','CANCER_TYPE','SAMPLE_TYPE','DRUG_TYPE']


def _prep(train,valid,external):
    for df in (train,valid,external):
        for col in PORTABLE_CLINICAL:
            if col not in df:df[col]='__MISSING__'
            df[col]=df[col].fillna('__MISSING__').astype(str)
        if 'TMB' not in df:df['TMB']=np.nan
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    encoder.fit(train[PORTABLE_CLINICAL])
    def numeric(df):
        tmb=pd.to_numeric(df.TMB,errors='coerce').to_numpy(float)
        return np.log1p(np.maximum(tmb,0)).reshape(-1,1)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(numeric(train))
    scaler=StandardScaler().fit(imputer.transform(numeric(train)))
    def transform(df):
        z=scaler.transform(imputer.transform(numeric(df)))
        return np.concatenate([encoder.transform(df[PORTABLE_CLINICAL]),z],axis=1),z
    return transform(train),transform(valid),transform(external)


def run_portable(internal_path,external_path,outdir,seed=11,bootstraps=1000):
    import xgboost as xgb
    internal=pd.read_csv(internal_path);external=pd.read_csv(external_path)
    for name,df in (('internal',internal),('external',external)):
        for col in ('time_months','event'):
            if col not in df:raise ValueError(f'{name} missing {col}')
        if not df.event.isin([0,1]).all():raise ValueError('event must be 0 or 1')
    tr,va=train_test_split(internal,test_size=.20,stratify=internal.event,random_state=seed)
    tr=tr.copy();va=va.copy();external=external.copy()
    (Xtr,Ztr),(Xva,Zva),(Xex,Zex)=_prep(tr,va,external)
    tt,te=tr.time_months.to_numpy(float),tr.event.to_numpy(int)
    vt,ve=va.time_months.to_numpy(float),va.event.to_numpy(int)
    et,ee=external.time_months.to_numpy(float),external.event.to_numpy(int)
    def ridge_best(X,Z,XX):
        fitted=[]
        for lam in (.01,.1,1.,10.):
            model=RidgeCox(l2=lam).fit(X,tt,te)
            fitted.append((harrell_c(vt,ve,model.predict(Z)),-lam,model))
        return max(fitted,key=lambda a:a[0])[2].predict(XX)
    risk_ridge=ridge_best(Xtr,Xva,Xex)
    risk_tmb=ridge_best(Ztr,Zva,Zex)
    dtr=xgb.DMatrix(Xtr,label=np.where(te==1,tt,-tt))
    dva=xgb.DMatrix(Xva,label=np.where(ve==1,vt,-vt))
    params={'objective':'survival:cox','eval_metric':'cox-nloglik','eta':.04,
            'max_depth':3,'subsample':.85,'colsample_bytree':.85,'lambda':1.,
            'seed':seed,'nthread':2}
    model=xgb.train(params,dtr,num_boost_round=400,evals=[(dva,'val')],
                    early_stopping_rounds=25,verbose_eval=False)
    risk_xgb=model.predict(xgb.DMatrix(Xex),output_margin=True,iteration_range=(0,model.best_iteration+1))
    estimates={'Portable XGBoost-Cox':harrell_c(et,ee,risk_xgb),
               'Core Ridge-Cox':harrell_c(et,ee,risk_ridge),
               'TMB-only Ridge-Cox':harrell_c(et,ee,risk_tmb)}
    rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(bootstraps):
        ids=rng.integers(0,len(et),len(et))
        a=harrell_c(et[ids],ee[ids],risk_xgb[ids]);b=harrell_c(et[ids],ee[ids],risk_ridge[ids])
        if np.isfinite(a) and np.isfinite(b):deltas.append(a-b)
    ci=list(map(float,np.percentile(deltas,[2.5,97.5]))) if deltas else [None,None]
    result={'n_external':len(et),'deaths_external':int(ee.sum()),
            'c_index':estimates,'paired_bootstrap_xgb_minus_ridge_ci95':ci,
            'note':'External-cohort evaluation using reduced-feature survival models.'}
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'portable_external_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame({'patient_id':external.patient_id if 'patient_id' in external else np.arange(len(et)),
                  'xgboost_cox_risk':risk_xgb,'ridge_cox_risk':risk_ridge,
                  'tmb_only_risk':risk_tmb}).to_csv(outdir/'portable_external_risks.csv',index=False)
    return result



# Figures and visualization


def logrank_p(time,event,group):
    time=np.asarray(time,float);event=np.asarray(event,bool);group=np.asarray(group,bool)
    O=E=V=0.
    for t in np.unique(time[event]):
        at=time>=t
        n1=np.sum(at & group);n0=np.sum(at & ~group)
        d1=np.sum((time==t)&event&group);d0=np.sum((time==t)&event&~group)
        n=n1+n0;d=d1+d0
        if n<=1:continue
        O+=d1;E+=d*n1/n
        V+=(n1*n0*d*(n-d))/(n*n*(n-1))
    return float(chi2.sf((O-E)**2/V,1)) if V>0 else float('nan')


def plot_km(time,event,risk,path):
    order=np.argsort(risk)
    group=np.zeros(len(risk),bool);group[order[len(risk)//2:]]=True
    fig,ax=plt.subplots(figsize=(7,4.5))
    for selected,label in [(~group,'Lower predicted risk'),(group,'Higher predicted risk')]:
        km=KaplanMeier.fit(np.asarray(time)[selected],np.asarray(event)[selected])
        ax.step(np.r_[0,km.event_times],np.r_[1,km.survival_values],where='post',
                label=f'{label} (n={selected.sum()})')
    ax.set(xlabel='Months from first ICI',ylabel='KM overall survival',ylim=(0,1.02),
           title=f'Predicted-risk stratification (log-rank p={logrank_p(time,event,group):.3g})')
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def plot_calibration(time,event,risk,baseline,path):
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,h in zip(axes,(12.,24.)):
        s=baseline.survival(risk,[h])[:,0]
        ece,curve=km_ece(time,event,s,h)
        ax.plot([0,1],[0,1],linestyle='--',color='gray',label='Perfect calibration')
        ax.plot(curve['predicted'],curve['observed'],'o-',label='KM observed')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel=f'Predicted survival at {int(h)} months',
               ylabel='Observed KM survival',title=f'{int(h)} months | KM-bin ECE={ece:.3f}')
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def plot_semantics(names,risk,relevance,weighted,path,top=8):
    """Paper Eqs. (29)-(30): mean w and mean |w*z| per predicted-risk quintile."""
    names=np.asarray(names)
    quint=np.array_split(np.argsort(risk),5)
    rel=np.stack([relevance[idx].mean(axis=0) for idx in quint])
    mag=np.stack([np.abs(weighted[idx]).mean(axis=0) for idx in quint])
    clinical=np.array([s.startswith('clinical:') for s in names])
    fig,axes=plt.subplots(2,2,figsize=(15,8))
    for j,mask in enumerate((clinical,~clinical)):
        candidates=np.where(mask)[0]
        selected=candidates[np.argsort(-rel[:,candidates].mean(axis=0))[:top]]
        for k,values in enumerate((rel,mag)):
            ax=axes[j,k]
            a=values[:,selected].T
            im=ax.imshow(a,aspect='auto',cmap='viridis')
            ax.set_yticks(np.arange(len(selected)),labels=[names[i].replace('clinical:','').replace('pathway:','') for i in selected])
            ax.set_xticks(range(5),labels=[f'Q{i}' for i in range(1,6)])
            ax.set_xlabel('Predicted mortality-risk quintile (increasing)')
            ax.set_title(('Clinical' if j==0 else 'Genomic / pathway')+' — '+
                         ('mean relevance' if k==0 else 'mean |weighted value|'))
            fig.colorbar(im,ax=ax,fraction=.04)
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)



# Model diagnostics
def schoenfeld_residuals(X,time,event,log_risk,names):
    X=np.asarray(X,float);time=np.asarray(time,float)
    event=np.asarray(event,bool);log_risk=np.asarray(log_risk,float)
    rows=[]
    for i in np.flatnonzero(event):
        at_risk=time>=time[i]
        score=log_risk[at_risk]
        w=np.exp(score-score.max());w/=w.sum()
        residual=X[i]-w @ X[at_risk]
        rows.append(residual)
    return pd.DataFrame(rows,columns=names).assign(log_event_time=np.log(time[event]))


def schoenfeld_correlations(frame,selected=None):
    if selected is None:
        selected=[c for c in frame.columns if c!='log_event_time']
    rows=[]
    for name in selected:
        if name not in frame.columns:continue
        r,p=spearmanr(frame['log_event_time'],frame[name])
        rows.append({'covariate':name,'spearman_r':float(r),'exploratory_p':float(p)})
    return pd.DataFrame(rows)


def subgroup_bootstrap(patients,time,event,risk,seed=11,replicates=250):
    rng=np.random.default_rng(seed)
    rows=[]
    for col in ('CANCER_TYPE','DRUG_TYPE','AGE_GROUP','SEX'):
        for value,group in patients.groupby(col,sort=True):
            positions=patients.index.get_indexer(group.index)
            t,e,r=time[positions],event[positions],risk[positions]
            c=harrell_c(t,e,r)
            values=[]
            for _ in range(replicates):
                ix=rng.integers(0,len(t),len(t))
                score=harrell_c(t[ix],e[ix],r[ix])
                if np.isfinite(score):values.append(score)
            low,high=(np.percentile(values,[2.5,97.5]) if len(values)>20 else [np.nan,np.nan])
            rows.append({'category':col,'subgroup':str(value),'n':len(t),'deaths':int(e.sum()),
                         'c_index':c,'ci_lower':low,'ci_upper':high})
    return pd.DataFrame(rows)


def plot_schoenfeld(residuals,selected,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[name for name in selected if name in residuals.columns][:4]
    if not chosen:return
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,name in zip(axes.flat,chosen):
        ax.scatter(residuals.log_event_time,residuals[name],alpha=.35,s=12)
        r,p=spearmanr(residuals.log_event_time,residuals[name])
        ax.axhline(0,color='gray',linestyle='--')
        ax.set(xlabel='log(event month)',ylabel='Schoenfeld residual',
               title=f'{name} (exploratory Spearman p={p:.3g})')
    for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def plot_subgroups(frame,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=frame.dropna(subset=['c_index','ci_lower','ci_upper']).copy()
    if f.empty:return
    fig,ax=plt.subplots(figsize=(9,max(4,len(f)*.42)))
    y=np.arange(len(f))
    ax.errorbar(f.c_index,y,xerr=[np.maximum(0,f.c_index-f.ci_lower),np.maximum(0,f.ci_upper-f.c_index)],
                fmt='o',capsize=3,ecolor='gray')
    ax.set_yticks(y,labels=[f'{r.category}: {r.subgroup} (n={r.n})' for r in f.itertuples()])
    ax.axvline(.5,linestyle='--',color='gray')
    ax.invert_yaxis();ax.set(xlabel='Harrell C-index (patient-bootstrap 95% CI)',
                             title='Held-out subgroup discrimination')
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
