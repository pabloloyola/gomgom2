#!/usr/bin/env python3
"""Evaluate frozen public-food persona selections on the untouched Task-4 holdout.

Run this only after the three-seed search artifacts have been frozen. Unlike the search
stage, this script intentionally reads Q9.5 and produces the one-shot final public
benchmark summary. It never feeds Task-4 information back into candidate selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eipg.econ import MNLConfig, MultinomialLogitModel
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig

ALTS=("organic","conventional","circular","none")


def load_params(path: Path) -> MixtureGeneratorParams:
    d=json.loads(path.read_text(encoding="utf-8"))
    return MixtureGeneratorParams(
        weights=np.asarray(d["weights"],dtype=float), means=np.asarray(d["means"],dtype=float),
        features=tuple(d["features"]), within_component_std=float(d["within_component_std"]),
        segment_labels=tuple(d["segment_labels"]),
    )


def init_params(cfg: dict[str,Any]) -> MixtureGeneratorParams:
    p=cfg["persona_generator"]
    return MixtureGeneratorParams(
        weights=np.asarray(p["initial_weights"],dtype=float), means=np.asarray(p["initial_means"],dtype=float),
        features=tuple(p["features"]), within_component_std=float(p["within_component_std"]),
        segment_labels=tuple(p["segment_labels"]),
    )


def contexts(cfg: dict[str,Any], tasks: list[int]) -> pd.DataFrame:
    rows=[]
    for t in tasks:
        prices=cfg["tasks"][t]["prices"]
        for a in ALTS:
            price=float(prices[a])
            rows.append({"context_id":f"bread_task_{t}","task_id":t,"alternative_id":a,
                "price_sensitivity":price,"organic_affinity":float(a=="organic"),
                "conventional_affinity":float(a=="conventional"),"circular_affinity":float(a=="circular"),
                "opt_out_propensity":float(a=="none"),"price":price,
                "asc_organic":float(a=="organic"),"asc_conventional":float(a=="conventional"),
                "asc_circular":float(a=="circular")})
    return pd.DataFrame(rows)


def softmax(x):
    x=np.asarray(x,float); x=x-x.max(); e=np.exp(x); return e/e.sum()


def predict(fit, ctx: pd.DataFrame) -> dict[int,dict[str,float]]:
    out={}
    for t,g0 in ctx.groupby("task_id"):
        g=g0.set_index("alternative_id").loc[list(ALTS)].reset_index()
        p=softmax(g[list(fit.features)].to_numpy(float) @ fit.beta)
        out[int(t)]={a:float(p[i]) for i,a in enumerate(ALTS)}
    return out


def fit_candidate(params,cfg,cal_ctx,seed):
    p=cfg["persona_generator"]
    pers=MixturePersonaGenerator(params,seed=seed+100_000).sample(int(p["n_personas"]))
    d=RandomUtilityChoiceSimulator(SyntheticSimulatorConfig(),seed=seed+110_000).simulate_long_dataset(
        contexts=cal_ctx,personas=pers,n_observations=int(p["simulation_observations"]),dataset_label="public_final_eval")
    im=cfg["inner_model"]
    return MultinomialLogitModel(MNLConfig(features=tuple(im["features"]),l2=float(im["l2"]),max_iter=int(im["max_iter"]))).fit(d)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=Path('configs/public_food_benchmark.yaml'))
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--selected-dir',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(a.config.read_text())
    raw=pd.read_csv(a.csv,sep=';',low_memory=False)
    ds=cfg['dataset']; codes={int(v):k for k,v in ds['choice_codes'].items()}
    cal_cols=[cfg['tasks'][int(t)]['column'] for t in cfg['split']['calibration_tasks']]
    final_t=int(cfg['split']['final_holdout_tasks'][0]); final_col=cfg['tasks'][final_t]['column']
    mask=pd.to_numeric(raw[ds['country_column']],errors='coerce').eq(int(ds['country_code']))
    mask &= pd.to_numeric(raw[ds['purchase_column']],errors='coerce').eq(int(ds['purchase_required_value']))
    for c in [*cal_cols,final_col]: mask &= pd.to_numeric(raw[c],errors='coerce').isin(set(codes))
    panel=raw.loc[mask].copy()
    if len(panel)!=int(ds['expected_complete_panels']): raise RuntimeError('final population differs from frozen population')

    # First and only outcome read from the frozen Task-4 holdout.
    final_choices=pd.to_numeric(panel[final_col],errors='raise').astype(int).map(codes)
    human_final=np.asarray([float((final_choices==x).mean()) for x in ALTS])
    anchor_t=int(cfg['split']['anchor_tasks'][0]); anchor_col=cfg['tasks'][anchor_t]['column']
    anchor_choices=pd.to_numeric(panel[anchor_col],errors='raise').astype(int).map(codes)
    human_anchor=np.asarray([float((anchor_choices==x).mean()) for x in ALTS])
    human_response=human_final-human_anchor

    cal_ctx=contexts(cfg,[int(t) for t in cfg['split']['calibration_tasks']])
    eval_ctx=contexts(cfg,[anchor_t,final_t])
    rows=[]
    seeds=[int(s) for s in cfg['search']['search_seeds']]
    methods=['initial_static','anchor_only_eipg','intervention_rich_eipg']
    for seed in seeds:
        for method in methods:
            params=init_params(cfg) if method=='initial_static' else load_params(a.selected_dir/f'selected_{method}_seed_{seed}.json')
            fit=fit_candidate(params,cfg,cal_ctx,seed)
            pr=predict(fit,eval_ctx)
            pred_final=np.asarray([pr[final_t][x] for x in ALTS])
            pred_anchor=np.asarray([pr[anchor_t][x] for x in ALTS])
            response=pred_final-pred_anchor
            nll=float(-np.mean([np.log(max(pr[final_t][x],1e-12)) for x in final_choices]))
            rows.append({'method':method,'search_seed':seed,
                'final_choice_share_l2':float(np.linalg.norm(pred_final-human_final)),
                'final_negative_log_likelihood':nll,
                'final_response_moment_l2':float(np.linalg.norm(response-human_response)),
                **{f'pred_final_share_{x}':float(pr[final_t][x]) for x in ALTS},
                'mnl_price':float(fit.beta_by_feature['price'])})

    # Human MNL reference: fit real Tasks 1--3, then predict Task 4.
    hr=[]
    for t in cfg['split']['calibration_tasks']:
        t=int(t); col=cfg['tasks'][t]['column']; prices=cfg['tasks'][t]['prices']
        for _,r in panel.iterrows():
            chosen=codes[int(r[col])]; oid=f"{r[ds['respondent_id']]}::human::{t}"
            for alt in ALTS:
                hr.append({'observation_id':oid,'alternative_id':alt,'chosen':int(alt==chosen),'price':float(prices[alt]),
                    'asc_organic':float(alt=='organic'),'asc_conventional':float(alt=='conventional'),'asc_circular':float(alt=='circular')})
    im=cfg['inner_model']; hfit=MultinomialLogitModel(MNLConfig(features=tuple(im['features']),l2=float(im['l2']),max_iter=int(im['max_iter']))).fit(pd.DataFrame(hr))
    hpr=predict(hfit,eval_ctx); hf=np.asarray([hpr[final_t][x] for x in ALTS]); ha=np.asarray([hpr[anchor_t][x] for x in ALTS])
    hnll=float(-np.mean([np.log(max(hpr[final_t][x],1e-12)) for x in final_choices]))
    rows.append({'method':'direct_human_mnl_reference','search_seed':-1,
        'final_choice_share_l2':float(np.linalg.norm(hf-human_final)),'final_negative_log_likelihood':hnll,
        'final_response_moment_l2':float(np.linalg.norm((hf-ha)-human_response)),
        **{f'pred_final_share_{x}':float(hpr[final_t][x]) for x in ALTS},'mnl_price':float(hfit.beta_by_feature['price'])})

    res=pd.DataFrame(rows); res.to_csv(a.outdir/'final_public_results.csv',index=False)
    summ=res.groupby('method',as_index=False).agg(
        n=('final_choice_share_l2','size'),mean_share_l2=('final_choice_share_l2','mean'),
        mean_nll=('final_negative_log_likelihood','mean'),mean_response_l2=('final_response_moment_l2','mean'))
    summ.to_csv(a.outdir/'final_public_summary.csv',index=False)
    manifest={'protocol':cfg['protocol']['name'],'final_holdout_task':final_t,'n_final_respondents':len(panel),
        'human_final_shares':{x:float(human_final[i]) for i,x in enumerate(ALTS)},
        'final_holdout_consumed_now':True,'selection_reopened_after_final':False}
    (a.outdir/'final_public_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(summ.to_string(index=False)); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
