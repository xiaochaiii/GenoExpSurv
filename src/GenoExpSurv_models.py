"""Neural modules for GenoExpSurv survival prediction and interpretation."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def block(in_dim:int,out_dim:int,dropout:float)->nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim,out_dim),nn.GELU(),
                         nn.LayerNorm(out_dim),nn.Dropout(dropout))


class PatientRepresentation(nn.Module):
    """Patient representation: clinical 32d + genomic 64d + interaction 32d.

    The inputs have already been constructed from measured-negative, event,
    and not-covered states using ``FeatureBuilder``. The interaction follows
    Eq.(13): elementwise product of the *ungated* two embeddings.
    """
    def __init__(self,n_clinical:int,n_genomic:int,dropout:float=.20,
                 gate:bool=True,clinical_only:bool=False,genomic_only:bool=False):
        super().__init__()
        self.clinical_encoder=block(n_clinical,32,dropout)
        self.genomic_encoder=block(n_genomic,64,dropout)
        self.gate_enabled=gate
        self.clinical_only=clinical_only
        self.genomic_only=genomic_only
        self.gate=nn.Linear(96,2)

    def forward(self,clinical:torch.Tensor,genomic:torch.Tensor)->dict:
        hc=self.clinical_encoder(clinical)
        hg=self.genomic_encoder(genomic)
        if self.gate_enabled:
            alpha=torch.softmax(self.gate(torch.cat([hc,hg],dim=1)),dim=1)
        else:
            alpha=torch.full((len(hc),2),.5,device=hc.device,dtype=hc.dtype)
        interaction=hc*hg[:,:32]  # 32d, ungated according to manuscript Eq.(13)
        if self.clinical_only:
            hg=torch.zeros_like(hg)
            interaction=torch.zeros_like(interaction)
            alpha=torch.stack([torch.ones_like(alpha[:,0]),torch.zeros_like(alpha[:,1])],dim=1)
        if self.genomic_only:
            hc=torch.zeros_like(hc)
            interaction=torch.zeros_like(interaction)
            alpha=torch.stack([torch.zeros_like(alpha[:,0]),torch.ones_like(alpha[:,1])],dim=1)
        latent=torch.cat([alpha[:,0:1]*hc,alpha[:,1:2]*hg,interaction],dim=1)
        return {'latent':latent,'gate':alpha}


class SurvivalPrediction(nn.Module):
    """Relative survival risk with a fixed external Ridge-Cox reference.

    ``anchor_risk`` is supplied from the pre-fitted and frozen Ridge-Cox model;
    it is not re-estimated here. Time-specific survival estimates subsequently
    use the training-only Breslow baseline from ``util.BreslowBaseline``.
    """
    def __init__(self,latent_dim:int=128,dropout:float=.20):
        super().__init__()
        self.residual=nn.Sequential(block(latent_dim,128,dropout),
                                    block(128,64,dropout),nn.Linear(64,1))
        self.raw_gamma=nn.Parameter(torch.tensor(0.0))

    def forward(self,latent:torch.Tensor,anchor_risk:torch.Tensor)->dict:
        gamma=.5*torch.sigmoid(self.raw_gamma)
        correction=gamma*torch.tanh(self.residual(latent).flatten())
        risk=anchor_risk.flatten()+correction
        return {'risk':risk,'residual':correction,'gamma':gamma}


class GenoExpSurv(nn.Module):
    """Stage I prediction model (does NOT train the explanatory mapper)."""
    def __init__(self,n_clinical:int,n_genomic:int,dropout:float=.20,
                 gate:bool=True,clinical_only:bool=False,genomic_only:bool=False):
        super().__init__()
        self.representation=PatientRepresentation(n_clinical,n_genomic,dropout,
                                gate=gate,clinical_only=clinical_only,
                                genomic_only=genomic_only)
        self.survival=SurvivalPrediction(128,dropout)

    def forward(self,clinical:torch.Tensor,genomic:torch.Tensor,anchor_risk:torch.Tensor)->dict:
        encoded=self.representation(clinical,genomic)
        predicted=self.survival(encoded['latent'],anchor_risk)
        return {**encoded,**predicted}


class SemanticMapper(nn.Module):
    """Stage II: softmax relevance -> weighted aspects -> latent/risk surrogate.

    This module must be trained AFTER the predictor is frozen. Its surrogate
    risk is used only for a consistency loss, never for survival prediction.
    """
    def __init__(self,n_aspects:int,latent_dim:int=128,hidden:int=128):
        super().__init__()
        self.relevance=nn.Sequential(nn.Linear(latent_dim,hidden),nn.GELU(),
                                     nn.Linear(hidden,n_aspects))
        self.aspect_basis=nn.Linear(n_aspects,latent_dim,bias=False)
        self.surrogate=nn.Sequential(nn.Linear(n_aspects,hidden),nn.GELU(),
                                     nn.Linear(hidden,1))

    def forward(self,latent:torch.Tensor,aspects:torch.Tensor)->dict:
        weights=torch.softmax(self.relevance(latent),dim=1)
        weighted=weights*aspects
        reconstruction=self.aspect_basis(weighted)
        surrogate=self.surrogate(weighted).flatten()
        return {'relevance':weights,'weighted':weighted,
                'reconstruction':reconstruction,'surrogate_risk':surrogate}


def mapper_loss(outputs:dict,latent:torch.Tensor,risk_standardized:torch.Tensor,
                entropy_weight:float=1e-4):
    cosine=(1.-F.cosine_similarity(outputs['reconstruction'],latent,dim=1)).mean()
    mse=F.mse_loss(outputs['surrogate_risk'],risk_standardized.flatten())
    weights=outputs['relevance'].clamp_min(1e-10)
    entropy=-(weights*torch.log(weights)).sum(dim=1).mean()
    return cosine+mse+entropy_weight*entropy, {'cosine':float(cosine.detach()),
                                               'risk_mse':float(mse.detach()),
                                               'entropy':float(entropy.detach())}
