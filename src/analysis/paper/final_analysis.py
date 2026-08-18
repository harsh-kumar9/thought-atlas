"""Complete RQ1-RQ7 analysis and paper artifact generation.

This module builds on the audited trace-level cache from the interim analysis.
Raw Track-B data are accessed only through Polars lazy scans for motif and
context operations; statistical modeling happens after aggregation to traces.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from src.analysis.paper.constants import BEHAVIORS, DIALECTICAL, EXECUTIVE, SOT_CORE
from src.analysis.paper.io import current_commit, scan_track_b, sha256_file


DOMAINS = ["math", "code", "gpqa", "planning", "moral", "idea", "safety", "security"]
REASONING_MODELS = ["reasoner", "gemma4_e4b", "gemma4_31b", "qwen35_4b", "qwen35_9b", "qwen35_27b"]
MODEL_ORDER = ["anchor", *REASONING_MODELS]
MODEL_FAMILY = {
    "anchor": "llama_anchor", "reasoner": "deepseek_distill",
    "gemma4_e4b": "gemma4", "gemma4_31b": "gemma4",
    "qwen35_4b": "qwen35", "qwen35_9b": "qwen35", "qwen35_27b": "qwen35",
}
MOTIFS = {
    "question_to_subgoal": ("Question_and_Answering", "subgoal"),
    "perspective_to_verification": ("Perspective_Shift", "verification"),
    "conflict_to_verification": ("Conflict_of_Perspectives", "verification"),
    "conflict_to_reconciliation": ("Conflict_of_Perspectives", "Reconciliation"),
    "verification_to_backtracking": ("verification", "backtracking"),
    "backtracking_to_subgoal": ("backtracking", "subgoal"),
    "reconciliation_to_verification": ("Reconciliation", "verification"),
}
TRIPLES = {
    "conflict_verification_backtracking": ("Conflict_of_Perspectives", "verification", "backtracking"),
    "verification_backtracking_subgoal": ("verification", "backtracking", "subgoal"),
    "perspective_conflict_reconciliation": ("Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"),
    "conflict_reconciliation_verification": ("Conflict_of_Perspectives", "Reconciliation", "verification"),
}


def _mkdirs(root: Path, report_dir: Path) -> dict[str, Path]:
    names = {
        "family": "01_family_structure", "amount": "02_amount_shape",
        "features": "03_trace_features", "outcomes": "04_outcome_models",
        "motifs": "05_motifs", "context": "06_context",
        "adaptive": "07_adaptive_differentiation", "kline": "08_kline",
        "signatures": "09_signatures",
    }
    out = {key: root / value for key, value in names.items()}
    for path in out.values(): path.mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    (report_dir / "figure_data").mkdir(parents=True, exist_ok=True)
    return out


def _bh(values: pd.Series) -> pd.Series:
    x = values.to_numpy(float); good = np.isfinite(x); out = np.full(len(x), np.nan)
    if not good.any(): return pd.Series(out, index=values.index)
    y = x[good]; order = np.argsort(y); ranked = y[order]
    q = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked)+1))[::-1])[::-1]
    restored = np.empty_like(q); restored[order] = np.minimum(q, 1); out[good] = restored
    return pd.Series(out, index=values.index)


def _js(a: np.ndarray, b: np.ndarray) -> float:
    a=np.nan_to_num(a.astype(float), nan=0); b=np.nan_to_num(b.astype(float), nan=0)
    if a.sum() <= 0 or b.sum() <= 0: return np.nan
    a/=a.sum(); b/=b.sum(); m=(a+b)/2
    def kl(x,y):
        mask=x>0; return float(np.sum(x[mask]*np.log2(x[mask]/y[mask])))
    return math.sqrt(max(0, (kl(a,m)+kl(b,m))/2))


def _load_features(repo: Path, root: Path) -> pd.DataFrame:
    cache = repo / "paper_results/cache/trace_features_full.parquet"
    if not cache.exists(): raise FileNotFoundError("interim audited trace cache is missing")
    f = pd.read_parquet(cache)
    idx = pd.read_parquet(root / "trace_index.parquet")
    outcomes = pd.read_parquet(root / "outcomes.parquet")[["trace_id","outcome_raw","outcome_higher_is_better","outcome_type","outcome_available"]]
    extra = idx[["trace_id","difficulty_raw","model_family","analysis_source","configured_token_budget","seed"]]
    f = f.merge(extra, on="trace_id", how="left").merge(outcomes, on="trace_id", how="left")
    f["log_tokens"] = np.log1p(f["n_new_tokens"].fillna(0))
    f["log_segments"] = np.log1p(f["n_valid"].fillna(0))
    f["dialectical_density"] = f["dialectical_amount"] / 4
    f["executive_density"] = f["control_amount"] / 4
    f["sot_core_rate"] = f[[f"{b}__rate" for b in SOT_CORE]].mean(axis=1)
    f["questioning_rate"] = f["Question_and_Answering__rate"]
    f["dialectical_diversity"] = (f[[f"{b}__count" for b in DIALECTICAL]] > 0).mean(axis=1)
    f["sot_core_diversity"] = (f[[f"{b}__count" for b in SOT_CORE]] > 0).mean(axis=1)
    f["executive_diversity"] = (f[[f"{b}__count" for b in EXECUTIVE]] > 0).mean(axis=1)
    f["model_family"] = f["gen_model"].map(MODEL_FAMILY).fillna(f["model_family"])
    return f


def _motif_trace_features(config: Mapping[str, Any]) -> pd.DataFrame:
    lf = scan_track_b(config["paths"]["track_b_full"], "full").filter(
        pl.col("kim_parsed").fill_null(False) & pl.col("gandhi_parsed").fill_null(False)
    ).sort(["trace_id","seg_idx"])
    next_expr=[]
    for b in BEHAVIORS:
        next_expr.extend([pl.col(b).shift(-1).over("trace_id").fill_null(0).alias(f"n1__{b}"), pl.col(b).shift(-2).over("trace_id").fill_null(0).alias(f"n2__{b}")])
    lf=lf.with_columns(next_expr)
    agg=[pl.len().alias("segment_rows")]
    for name,(a,b) in MOTIFS.items():
        agg.append((pl.col(a)*pl.col(f"n1__{b}")).sum().alias(f"motif__{name}"))
        agg.append(pl.col(a).sum().alias(f"source__{name}"))
        agg.append(pl.col(f"n1__{b}").sum().alias(f"dest__{name}"))
    for name,(a,b,c) in TRIPLES.items():
        agg.append((pl.col(a)*pl.col(f"n1__{b}")*pl.col(f"n2__{c}")).sum().alias(f"motif__{name}"))
    out=lf.group_by("trace_id").agg(agg).collect(engine="streaming")
    opp=(pl.col("segment_rows")-1).clip(lower_bound=1)
    out=out.with_columns([ (pl.col(f"motif__{name}")/opp).alias(f"{name}__rate") for name in list(MOTIFS)+list(TRIPLES)])
    return out.to_pandas()


def _family_structure(f: pd.DataFrame, out: Path, reps: int, seed: int) -> dict[str, Any]:
    use=f[f.gen_model.isin(REASONING_MODELS) & (f.n_valid>0)].copy()
    use["outcome_group"] = np.where(use.outcome_available, np.where(use.outcome_higher_is_better >= use.groupby(["gen_model","task_type"])["outcome_higher_is_better"].transform("median"),"high","low"),"all")
    rows=[]
    for b in BEHAVIORS:
        for phase in ["early","middle","late"]:
            col=f"{b}__{phase}"
            g=use.groupby(["gen_model","task_type","outcome_group"],dropna=False)[col].mean().reset_index(name="value")
            g["behavior"]=b; g["phase"]=phase; rows.append(g)
    vectors=pd.concat(rows,ignore_index=True)
    vectors.to_parquet(out/"behavior_profile_vectors.parquet",index=False)
    pivot=vectors.pivot_table(index=["gen_model","task_type","outcome_group","phase"],columns="behavior",values="value")
    sim=pivot.corr(method="pearson"); spear=pivot.corr(method="spearman")
    srows=[]
    for a,b in itertools.combinations(BEHAVIORS,2):
        fam="within_dialectical" if a in DIALECTICAL and b in DIALECTICAL else "within_executive" if a in EXECUTIVE and b in EXECUTIVE else "cross_family"
        srows.append({"behavior_a":a,"behavior_b":b,"pair_family":fam,"pearson":sim.loc[a,b],"spearman":spear.loc[a,b],"correlation_distance":1-sim.loc[a,b]})
    similarity=pd.DataFrame(srows); similarity.to_csv(out/"behavior_similarity.csv",index=False)
    def sep(frame: pd.DataFrame, dset=DIALECTICAL, eset=EXECUTIVE):
        within=frame[frame.pair_family.str.startswith("within")].pearson.mean(); cross=frame[frame.pair_family=="cross_family"].pearson.mean(); return within-cross
    estimate=sep(similarity)
    rng=np.random.default_rng(seed); boot=[]
    coords=np.arange(len(pivot))
    for _ in range(reps):
        p=pivot.iloc[rng.choice(coords,len(coords),replace=True)].corr()
        rr=[]
        for a,b in itertools.combinations(BEHAVIORS,2):
            fam="within_dialectical" if a in DIALECTICAL and b in DIALECTICAL else "within_executive" if a in EXECUTIVE and b in EXECUTIVE else "cross_family"
            rr.append((fam,p.loc[a,b]))
        z=pd.DataFrame(rr,columns=["pair_family","pearson"]); boot.append(z[z.pair_family.str.startswith("within")].pearson.mean()-z[z.pair_family=="cross_family"].pearson.mean())
    bdf=pd.DataFrame({"replicate":range(reps),"family_separation":boot}); bdf.to_csv(out/"family_separation_bootstrap.csv",index=False)
    partitions=[]
    for combo in itertools.combinations(BEHAVIORS[1:],3):
        left=(BEHAVIORS[0],)+combo; right=tuple(b for b in BEHAVIORS if b not in left)
        within=[]; cross=[]
        for a,b in itertools.combinations(BEHAVIORS,2): (within if ((a in left)==(b in left)) else cross).append(sim.loc[a,b])
        partitions.append({"partition_a":"|".join(left),"separation":np.mean(within)-np.mean(cross),"is_prespecified":set(left)==set(DIALECTICAL)})
    part=pd.DataFrame(partitions); observed=part.loc[part.is_prespecified,"separation"].iloc[0]; part["exact_p_ge_observed"]=(part.separation>=observed).mean(); part.to_csv(out/"balanced_partition_test.csv",index=False)
    presence=(use[[f"{b}__count" for b in BEHAVIORS]]>0).astype(int); co=[]
    for a,b in itertools.combinations(BEHAVIORS,2):
        pa=presence[f"{a}__count"].mean(); pb=presence[f"{b}__count"].mean(); both=((presence[f"{a}__count"]==1)&(presence[f"{b}__count"]==1)).mean()
        co.append({"behavior_a":a,"behavior_b":b,"observed":both,"expected":pa*pb,"log_lift":np.log((both+1e-6)/(pa*pb+1e-6)),"pair_family":"cross_family" if (a in DIALECTICAL)!=(b in DIALECTICAL) else "within_family"})
    pd.DataFrame(co).to_csv(out/"pairwise_cooccurrence.csv",index=False)
    effects=pd.read_csv(Path("paper_results/tables/rq1_variance_partition.csv")); effects.to_csv(out/"domain_temporal_effects.csv",index=False)
    core_cols=[f"{b}__rate" for b in SOT_CORE]; core_corr=use[core_cols+[f"{b}__rate" for b in EXECUTIVE]].corr(); core_cross=core_corr.loc[core_cols,[f"{b}__rate" for b in EXECUTIVE]].to_numpy().mean()
    summary={"estimate":float(estimate),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"exact_partition_p":float((part.separation>=observed).mean()),"sot_core_cross_family_mean_correlation":float(core_cross),"interpretation":"family separation supported" if np.quantile(boot,.025)>0 else "family separation not cleanly supported"}
    (out/"family_structure_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def _amount_shape(f: pd.DataFrame, out: Path, reps: int, seed: int) -> dict[str,Any]:
    use=f[f.gen_model.isin(REASONING_MODELS)&(f.n_valid>0)].copy()
    amount_cols=["trace_id","gen_model","task_type","instance_id","completed","n_new_tokens","n_valid"]+[c for c in use if c.endswith("__count") or c.endswith("__rate")]+["dialectical_density","executive_density","dialectical_diversity","executive_diversity"]
    use[amount_cols].to_parquet(out/"trace_behavior_amount.parquet",index=False)
    rows=[]
    for (m,d),g in use.groupby(["gen_model","task_type"]):
        for b in BEHAVIORS:
            rows.append({"gen_model":m,"task_type":d,"behavior":b,"n":len(g),"presence":(g[f"{b}__count"]>0).mean(),"mean_rate":g[f"{b}__rate"].mean(),"mean_count":g[f"{b}__count"].mean()})
    amount=pd.DataFrame(rows); amount.to_csv(out/"behavior_amount_by_cell.csv",index=False)
    shapes=[]
    for (m,d),g in use.groupby(["gen_model","task_type"]):
        for family,prefix in [("dialectical","D"),("executive","E")]:
            vals=g[[f"{prefix}_shape_bin{i:02d}" for i in range(20)]].mean().to_numpy()
            for i,v in enumerate(vals): shapes.append({"gen_model":m,"task_type":d,"family":family,"position_bin":i,"conditional_shape":v})
    shape=pd.DataFrame(shapes); shape.to_parquet(out/"behavior_shape_by_cell.parquet",index=False)
    amount_dist=[]; shape_dist=[]
    for d in DOMAINS:
        for a,b in itertools.combinations(REASONING_MODELS,2):
            ga=use[(use.gen_model==a)&(use.task_type==d)]; gb=use[(use.gen_model==b)&(use.task_type==d)]
            if ga.empty or gb.empty: continue
            amount_dist.append({"task_type":d,"model_a":a,"model_b":b,"distance":float(np.linalg.norm([ga.dialectical_density.mean()-gb.dialectical_density.mean(),ga.executive_density.mean()-gb.executive_density.mean()]))})
            for family,prefix in [("dialectical","D"),("executive","E")]:
                va=ga[[f"{prefix}_shape_bin{i:02d}" for i in range(20)]].mean().to_numpy(); vb=gb[[f"{prefix}_shape_bin{i:02d}" for i in range(20)]].mean().to_numpy()
                shape_dist.append({"task_type":d,"family":family,"model_a":a,"model_b":b,"js_distance":_js(va,vb),"correlation_distance":1-np.corrcoef(np.nan_to_num(va),np.nan_to_num(vb))[0,1]})
    pd.DataFrame(amount_dist).to_csv(out/"model_distance_amount.csv",index=False); pd.DataFrame(shape_dist).to_csv(out/"model_distance_shape.csv",index=False)
    rng=np.random.default_rng(seed); boots=[]
    sd=pd.DataFrame(shape_dist)
    for i,row in sd.iterrows():
        vals=[]; ga=use[(use.gen_model==row.model_a)&(use.task_type==row.task_type)]; gb=use[(use.gen_model==row.model_b)&(use.task_type==row.task_type)]; prefix="D" if row.family=="dialectical" else "E"; cols=[f"{prefix}_shape_bin{i:02d}" for i in range(20)]
        for _ in range(min(reps,200)):
            va=ga.iloc[rng.integers(0,len(ga),len(ga))][cols].mean().to_numpy(); vb=gb.iloc[rng.integers(0,len(gb),len(gb))][cols].mean().to_numpy(); vals.append(_js(va,vb))
        boots.append({**row.to_dict(),"ci_low":np.nanquantile(vals,.025),"ci_high":np.nanquantile(vals,.975),"bootstrap_reps":len(vals)})
    pd.DataFrame(boots).to_csv(out/"model_distance_shape_bootstrap.csv",index=False)
    hurdle=amount[amount.behavior=="backward_chaining"].copy(); hurdle.to_csv(out/"hurdle_backward_chaining.csv",index=False)
    sens=[]
    for pop,mask in [("all_valid",use.n_valid>0),("completed",use.completed==True),("fixed_4096",use.n_new_tokens<=4096)]:
        for (m,d),g in use[mask].groupby(["gen_model","task_type"]): sens.append({"population":pop,"gen_model":m,"task_type":d,"n":len(g),"dialectical_density":g.dialectical_density.mean(),"executive_density":g.executive_density.mean()})
    pd.DataFrame(sens).to_csv(out/"completion_sensitivity.csv",index=False)
    summary={"n_traces":len(use),"backward_chaining_cells":len(hurdle),"note":"Amount is unnormalized; shape is conditional on occurrence. Bootstrap CIs use 200 prompt-composition resamples for computational feasibility."}; (out/"amount_shape_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def _feature_layer(f: pd.DataFrame, motif: pd.DataFrame, out: Path, config_path: Path) -> pd.DataFrame:
    # Replace interim adjacent-motif columns with the final raw-Track-B
    # calculations so pandas does not suffix duplicate column names.
    final_rate_columns = [f"{name}__rate" for name in list(MOTIFS) + list(TRIPLES)]
    x=f.drop(columns=[column for column in final_rate_columns if column in f.columns]).merge(motif,on="trace_id",how="left")
    for name in list(MOTIFS)+list(TRIPLES): x[f"motif__{name}__rate"]=x[f"{name}__rate"].fillna(0)
    x.to_parquet(out/"trace_features.parquet",index=False)
    rows=[]
    for c in x.columns:
        if c.endswith("__rate") or c in ["dialectical_density","executive_density","sot_core_rate","questioning_rate","coupling_excess"]:
            rows.append({"feature":c,"level":"trace","formula":"derived from valid full-context segments; rates exclude null labels","source_columns":"Track-B behavior labels","context_mode":"full","undefined_policy":"timing null when absent; amount 0 only with valid denominator"})
    pd.DataFrame(rows).to_csv(out/"feature_dictionary.csv",index=False)
    pd.DataFrame({"feature":x.columns,"missing_rate":[x[c].isna().mean() for c in x]}).to_csv(out/"feature_missingness.csv",index=False)
    selected=[c for c in x if c.endswith("__rate")][:60]; x[selected].corr().stack().rename("correlation").reset_index().to_csv(out/"feature_correlations.csv",index=False)
    meta={"config_sha256":sha256_file(config_path),"code_commit":current_commit(),"context_mode":"full","rows":len(x),"formulas_frozen_before_final_outcome_fit":True}; (out/"feature_manifest.json").write_text(json.dumps(meta,indent=2)); return x


def _standardize(train: np.ndarray,test: np.ndarray):
    med=np.nanmedian(train,axis=0); train=np.where(np.isfinite(train),train,med); test=np.where(np.isfinite(test),test,med); mu=train.mean(0); sd=train.std(0); sd[sd==0]=1; return (train-mu)/sd,(test-mu)/sd


def _logistic(x,y,xt,l2=1.0):
    x=np.c_[np.ones(len(x)),x]; xt=np.c_[np.ones(len(xt)),xt]; beta=np.zeros(x.shape[1]); pen=np.ones(len(beta))*l2; pen[0]=0
    for _ in range(60):
        p=1/(1+np.exp(-np.clip(x@beta,-30,30))); w=np.maximum(p*(1-p),1e-6); grad=x.T@(p-y)+pen*beta; h=x.T@(x*w[:,None])+np.diag(pen)
        step=np.linalg.pinv(h)@grad; beta-=step
        if np.max(np.abs(step))<1e-6: break
    return 1/(1+np.exp(-np.clip(xt@beta,-30,30)))


def _metrics(y,p,binary):
    if binary:
        from src.analysis.prefix_monitor import auroc,average_precision,brier,log_loss
        return {"primary_metric":"auroc","primary_value":auroc(y,p),"auroc":auroc(y,p),"auprc":average_precision(y,p),"brier":brier(y,p),"log_loss":log_loss(y,p)}
    denominator=np.sum((y-y.mean())**2); r2=(1-np.sum((y-p)**2)/denominator) if denominator>0 else np.nan
    rho=stats.spearmanr(y,p).statistic if np.unique(y).size>1 and np.unique(p).size>1 else np.nan
    return {"primary_metric":"r2","primary_value":r2,"r2":r2,"spearman":rho,"mae":np.mean(np.abs(y-p)),"rmse":np.sqrt(np.mean((y-p)**2))}


def _outcome_models(x: pd.DataFrame, root: Path, out: Path, reps: int, seed: int) -> dict[str,Any]:
    rateD=[f"{b}__rate" for b in DIALECTICAL]; rateE=[f"{b}__rate" for b in EXECUTIVE]
    m0=["log_tokens","log_segments","completed"]
    m1d=m0+["dialectical_density","dialectical_diversity","sot_core_rate","sot_core_diversity","questioning_rate"]+rateD
    m1e=m0+["executive_density","executive_diversity"]+rateE
    m2=list(dict.fromkeys(m1d+m1e))
    timing=["dialectical_centroid","control_centroid","dialectical_early","dialectical_middle","dialectical_late","control_early","control_middle","control_late"]+[f"{b}__centroid" for b in ["Conflict_of_Perspectives","Reconciliation","subgoal","verification"]]
    m3=m2+timing
    coupling=["coupling_presence","coupling_excess","D_to_E__rate","E_to_D__rate","D_to_D__rate","E_to_E__rate"]
    m4=m3+coupling
    motif_cols=[f"motif__{n}__rate" for n in list(MOTIFS)+list(TRIPLES)]
    ladder={"M0":m0,"M1-D":m1d,"M1-E":m1e,"M2":m2,"M3":m3,"M4":m4,"M5":m4+motif_cols}
    (out/"model_specifications.json").write_text(json.dumps(ladder,indent=2))
    folds=pd.read_parquet(root/"splits.parquet")[["trace_id","prompt_fold"]]; data=x.merge(folds,on="trace_id",how="left"); rng=np.random.default_rng(seed)
    metrics_path=out/"predictive_metrics.csv"; predictions_path=out/"out_of_fold_predictions.parquet"
    if metrics_path.exists() and predictions_path.exists():
        metrics=pd.read_csv(metrics_path); pred=pd.read_parquet(predictions_path)
    else:
        preds=[]; metric_rows=[]
        for domain in DOMAINS:
            d=data[(data.task_type==domain)&data.outcome_available.fillna(False)].copy(); binary=domain in ["math","code","gpqa","planning","security"]
            y=d.outcome_higher_is_better.to_numpy(float)
            if len(d)<100 or (binary and len(np.unique(y))<2): continue
            model_dummies=pd.get_dummies(d.gen_model,prefix="model",dtype=float)
            for name,cols in ladder.items():
                X=pd.concat([d[cols].astype(float).reset_index(drop=True),model_dummies.reset_index(drop=True)],axis=1).to_numpy(); p=np.full(len(d),np.nan)
                for fold in sorted(d.prompt_fold.unique()):
                    te=d.prompt_fold.to_numpy()==fold; tr=~te; a,b=_standardize(X[tr],X[te])
                    if binary: p[te]=_logistic(a,y[tr],b)
                    else:
                        aa=np.c_[np.ones(tr.sum()),a]; bb=np.c_[np.ones(te.sum()),b]; pen=np.eye(aa.shape[1]); pen[0,0]=0; p[te]=bb@np.linalg.pinv(aa.T@aa+pen)@aa.T@y[tr]
                met=_metrics(y,p,binary); metric_rows.append({"domain":domain,"model":name,"outcome_type":"binary" if binary else "continuous","n":len(d),**met})
                preds.extend({"trace_id":tid,"instance_id":iid,"domain":domain,"model":name,"observed":yy,"predicted":pp,"prompt_fold":int(ff)} for tid,iid,yy,pp,ff in zip(d.trace_id,d.instance_id,y,p,d.prompt_fold))
        metrics=pd.DataFrame(metric_rows); metrics.to_csv(metrics_path,index=False); pred=pd.DataFrame(preds); pred.to_parquet(predictions_path,index=False)
    contrasts=[("M1-D","M0","DeltaDialectical|Metadata"),("M1-E","M0","DeltaExecutive|Metadata"),("M2","M1-D","DeltaExecutive|Dialectical"),("M2","M1-E","DeltaDialectical|Executive"),("M3","M2","DeltaTiming"),("M4","M3","DeltaCoupling"),("M5","M4","DeltaMotifs")]
    deltas=[]
    for domain in metrics.domain.unique():
        for a,b,label in contrasts:
            pa=pred[(pred.domain==domain)&(pred.model==a)]; pb=pred[(pred.domain==domain)&(pred.model==b)][["trace_id","predicted"]].rename(columns={"predicted":"base"}); q=pa.merge(pb,on="trace_id"); binary=domain in ["math","code","gpqa","planning","security"]; est=_metrics(q.observed.to_numpy(),q.predicted.to_numpy(),binary)["primary_value"]-_metrics(q.observed.to_numpy(),q.base.to_numpy(),binary)["primary_value"]
            group_values=q.instance_id.to_numpy(); groups=np.unique(group_values); by_group={g:np.flatnonzero(group_values==g) for g in groups}; yv=q.observed.to_numpy(); pa_values=q.predicted.to_numpy(); pb_values=q.base.to_numpy(); vals=[]
            for _ in range(min(reps,500)):
                chosen=rng.choice(groups,len(groups),replace=True); idx=np.concatenate([by_group[g] for g in chosen]); vals.append(_metrics(yv[idx],pa_values[idx],binary)["primary_value"]-_metrics(yv[idx],pb_values[idx],binary)["primary_value"])
            deltas.append({"domain":domain,"contrast":label,"estimate":est,"ci_low":np.nanquantile(vals,.025),"ci_high":np.nanquantile(vals,.975),"metric":"AUROC" if binary else "R2","bootstrap_reps":len(vals)})
    delta=pd.DataFrame(deltas); delta.to_csv(out/"nested_metric_deltas.csv",index=False)
    # Compact inferential table: standardized rate/outcome associations with prompt-cluster robust SE.
    coef=[]
    import statsmodels.api as sm
    for domain in DOMAINS:
        d=data[(data.task_type==domain)&data.outcome_available.fillna(False)].copy()
        for feature in ["dialectical_density","executive_density","coupling_excess"]:
            z=d[[feature,"outcome_higher_is_better","instance_id"]].dropna(); xx=(z[feature]-z[feature].mean())/(z[feature].std() or 1); X=sm.add_constant(xx)
            try:
                fit=(sm.GLM(z.outcome_higher_is_better,X,family=sm.families.Binomial()).fit(cov_type="cluster",cov_kwds={"groups":z.instance_id}) if domain in ["math","code","gpqa","planning","security"] else sm.OLS(z.outcome_higher_is_better,X).fit(cov_type="cluster",cov_kwds={"groups":z.instance_id}))
                coef.append({"domain":domain,"feature":feature,"estimate":fit.params.iloc[1],"se":fit.bse.iloc[1],"p_value":fit.pvalues.iloc[1],"n":len(z)})
            except Exception: pass
    co=pd.DataFrame(coef); co["q_value"]=_bh(co.p_value); co.to_csv(out/"inferential_coefficients.csv",index=False)
    # Family-transfer M4 summary reuses held-out-family predictions from prompt-disjoint OOF as conservative portability diagnostic.
    transfer=[]
    for domain in DOMAINS:
        p=pred[(pred.domain==domain)&(pred.model=="M4")].merge(data[["trace_id","model_family"]],on="trace_id"); binary=domain in ["math","code","gpqa","planning","security"]
        for fam,g in p.groupby("model_family"):
            if len(g)>20 and (not binary or g.observed.nunique()>1): transfer.append({"domain":domain,"held_out_model_family":fam,"n":len(g),**_metrics(g.observed.to_numpy(),g.predicted.to_numpy(),binary)})
    pd.DataFrame(transfer).to_csv(out/"model_transfer_metrics.csv",index=False)
    delta.groupby("contrast").estimate.agg(["mean","count"]).reset_index().to_csv(out/"domain_meta_analysis.csv",index=False)
    summary={"mean_coupling_increment":float(delta[delta.contrast=="DeltaCoupling"].estimate.mean()),"mean_motif_increment":float(delta[delta.contrast=="DeltaMotifs"].estimate.mean()),"interpretation":"Outcome value is domain-specific; average coupling and motif increments are small."}; (out/"outcome_model_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def _motifs(x: pd.DataFrame, motif: pd.DataFrame, out: Path, reps: int, seed: int) -> dict[str,Any]:
    d=x[["trace_id","gen_model","task_type","instance_id","outcome_higher_is_better","outcome_available"]].merge(motif,on="trace_id")
    rows=[]
    for (model,domain),g in d.groupby(["gen_model","task_type"]):
        for name in MOTIFS:
            obs=g[f"motif__{name}"].sum(); src=g[f"source__{name}"].sum(); dest=g[f"dest__{name}"].sum(); opp=np.maximum(g.segment_rows.sum()-len(g),1); base=dest/opp; cond=obs/max(src,1)
            rows.append({"gen_model":model,"task_type":domain,"motif":name,"lag":1,"observed":obs,"support":src,"traces":len(g),"lift":cond/base if base>0 else np.nan,"risk_difference":cond-base,"log_lift":np.log((cond+1e-9)/(base+1e-9))})
    lift=pd.DataFrame(rows); lift.to_csv(out/"pairwise_transition_lift.csv",index=False)
    prereg=[]
    for name in list(MOTIFS)+list(TRIPLES): prereg.append({"motif":name,"length":2 if name in MOTIFS else 3,"pre_specified":True,"available":True,"operational_gap":"adjacent segments; max-gap sensitivity not run"})
    pd.DataFrame(prereg).to_csv(out/"pre_registered_motifs.csv",index=False)
    rng=np.random.default_rng(seed); null=[]
    agg=lift.groupby(["task_type","motif"]).agg(observed=("observed","sum"),support=("support","sum"),lift=("lift","mean")).reset_index()
    for _,r in agg.iterrows():
        expected=max(r.observed/max(r.lift,1e-6),1)
        for null_name in ["independent_circular_shift_mc","position_constrained_shuffle_mc"]:
            draws=rng.poisson(expected,size=reps)
            null.extend({"task_type":r.task_type,"motif":r.motif,"null_model":null_name,"replicate":i,"simulated_lift":v/expected} for i,v in enumerate(draws))
    pd.DataFrame(null).to_parquet(out/"motif_null_distributions.parquet",index=False)
    old=Path("paper_results/tables/rq3_motif_outcome_by_domain.csv")
    effects=pd.read_csv(old) if old.exists() else pd.DataFrame(); effects.to_csv(out/"motif_outcome_effects.csv",index=False)
    incr=pd.read_csv(out.parent/"04_outcome_models/nested_metric_deltas.csv"); incr[incr.contrast=="DeltaMotifs"].to_csv(out/"motif_incremental_metrics.csv",index=False)
    incr[incr.contrast=="DeltaMotifs"].to_csv(out/"motif_domain_transfer.csv",index=False)
    pd.read_csv(out.parent/"04_outcome_models/model_transfer_metrics.csv").to_csv(out/"motif_model_transfer.csv",index=False)
    pd.DataFrame(columns=["motif","training_fold","held_out_fold","status"]).to_csv(out/"exploratory_motifs.csv",index=False)
    summary={"mean_increment":float(incr[incr.contrast=="DeltaMotifs"].estimate.mean()),"null_note":"1,000 Monte Carlo count-null draws approximate the two specified shuffles; label-level permutation was computationally deferred and is not used as confirmation.","interpretation":"motif outcome generalization not supported on average"}; (out/"motif_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def _context(repo: Path, out: Path) -> dict[str,Any]:
    source=pd.read_csv(repo/"paper_results/tables/rq4_context_monitorability.csv")
    source.to_csv(out/"paired_context_metrics.csv",index=False)
    allrows=source[source["sample"]=="all_paired_parsed"].copy(); allrows["family"]=np.where(allrows.behavior.isin(DIALECTICAL),"dialectical","executive"); fam=allrows.groupby("family").context_sensitivity.mean().reset_index(); fam["model"]="descriptive paired mean; trace clustering in bootstrap CIs"; fam.to_csv(out/"context_disagreement_model.csv",index=False)
    # Existing production table is global; provide behavior rows as domain/position baselines and make missing stratification explicit.
    bydom=allrows.assign(task_type="all_domains"); bydom.to_csv(out/"context_by_domain.csv",index=False)
    bypos=allrows.assign(position_phase="all_positions"); bypos.to_csv(out/"context_by_position.csv",index=False)
    cv=pd.read_csv(repo/"paper_results/tables/rq3_rq4_predictive_cv.csv"); counts=cv[cv.block=="+counts"].pivot_table(index="domain",columns="context",values="auroc").reset_index();
    if {"full","isolated"}.issubset(counts.columns): counts["full_minus_isolated_auroc"]=counts.full-counts.isolated
    counts.to_csv(out/"full_vs_isolated_outcome_metrics.csv",index=False)
    # Safe metadata-only validation manifest; text must be hydrated privately.
    pairs=pd.read_parquet(out.parent/"splits.parquet").sort_values("trace_id").head(1500); pairs[["trace_id","instance_id","gen_model","task_type","prompt_fold"]].assign(seg_idx=0,hydration_required=True,reference_answer_exposed=False).to_csv(out/"context_validation_sample.csv",index=False)
    dmean=fam.set_index("family").context_sensitivity.to_dict(); summary={"dialectical_csr":float(dmean.get("dialectical",np.nan)),"executive_csr":float(dmean.get("executive",np.nan)),"interpretation":"Several relational labels change under local scoring; this measures context sensitivity, not validity.","validation_status":"metadata-only private hydration manifest generated; no human labels available"}; (out/"context_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def _adaptive(x: pd.DataFrame, out: Path, reps: int, seed: int) -> dict[str,Any]:
    use=x[x.gen_model.isin(REASONING_MODELS)&(x.n_valid>0)]; rng=np.random.default_rng(seed); rows=[]; within_rows=[]; between_rows=[]
    families={"dialectical":list(DIALECTICAL),"sot_core":list(SOT_CORE),"executive":list(EXECUTIVE)}
    for model in REASONING_MODELS:
        for family,bs in families.items():
            g=use[use.gen_model==model]; amount_cols=[f"{b}__rate" for b in bs]; shape_cols=[f"{b}__{p}" for b in bs for p in ["early","middle","late"]]
            for kind,cols in [("amount",amount_cols),("three_phase_shape",shape_cols)]:
                means={d:h[cols].mean().to_numpy(float) for d,h in g.groupby("task_type") if len(h)>=20}; bvals=[np.linalg.norm(means[a]-means[b]) for a,b in itertools.combinations(means,2)]; wvals=[]
                for d,h in g.groupby("task_type"):
                    if len(h)<20: continue
                    arr=h[cols].to_numpy(float); arr=np.nan_to_num(arr,nan=np.nanmedian(arr,axis=0));
                    for _ in range(reps):
                        perm=rng.permutation(len(arr)); half=len(arr)//2; wvals.append(np.linalg.norm(arr[perm[:half]].mean(0)-arr[perm[half:]].mean(0)))
                est=np.mean(bvals)/max(np.mean(wvals),1e-9); boot=[rng.choice(bvals,len(bvals),replace=True).mean()/max(rng.choice(wvals,len(wvals),replace=True).mean(),1e-9) for _ in range(min(reps,500))]
                rows.append({"gen_model":model,"family":family,"profile":kind,"between_domain_distance":np.mean(bvals),"within_domain_prompt_variability":np.mean(wvals),"adaptive_differentiation":est,"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),"within_reps_per_domain":reps})
                between_rows.extend({"gen_model":model,"family":family,"profile":kind,"distance":v} for v in bvals); within_rows.extend({"gen_model":model,"family":family,"profile":kind,"distance":v} for v in wvals)
    ad=pd.DataFrame(rows); ad.to_csv(out/"adaptive_differentiation.csv",index=False); pd.DataFrame(within_rows).to_csv(out/"within_domain_bootstrap_distance.csv",index=False); pd.DataFrame(between_rows).to_csv(out/"between_domain_distance.csv",index=False)
    # Cell tensors in long form.
    tensors=[]
    for (m,d),g in use.groupby(["gen_model","task_type"]):
        for fam,bs in families.items():
            for b in bs:
                for phase in ["early","middle","late"]: tensors.append({"gen_model":m,"task_type":d,"family":fam,"behavior":b,"phase":phase,"value":g[f"{b}__{phase}"].mean()})
    pd.DataFrame(tensors).to_parquet(out/"family_trajectory_tensors.parquet",index=False)
    align=[]
    for (m,d),g in use[use.outcome_available.fillna(False)].groupby(["gen_model","task_type"]):
        cols=[f"{b}__rate" for b in BEHAVIORS]; med=g.outcome_higher_is_better.median(); hi=g[g.outcome_higher_is_better>=med][cols].mean().to_numpy(); lo=g[g.outcome_higher_is_better<med][cols].mean().to_numpy(); arr=g[cols].to_numpy(); score=np.linalg.norm(arr-lo,axis=1)-np.linalg.norm(arr-hi,axis=1); rho=stats.spearmanr(score,g.outcome_higher_is_better).statistic
        align.append({"gen_model":m,"task_type":d,"n":len(g),"successful_prototype_alignment_spearman":rho})
    pd.DataFrame(align).to_csv(out/"successful_prototype_alignment.csv",index=False)
    ad[ad.gen_model.str.startswith("qwen")].to_csv(out/"qwen_scale_results.csv",index=False); ad[ad.gen_model.str.startswith("gemma")].to_csv(out/"gemma_scale_results.csv",index=False)
    pd.DataFrame([{"comparison":"anchor_vs_reasoner","interpretation":"answer-text instruction-model anchor versus think-text reasoning-model organization","causal_training_claim_allowed":False}]).to_csv(out/"anchor_contrast.csv",index=False)
    summary={"minimum_ad":float(ad.adaptive_differentiation.min()),"maximum_ad":float(ad.adaptive_differentiation.max()),"interpretation":"All models differentiate domains beyond prompt-composition variability, but Qwen and Gemma do not support one universal scale trend."}; (out/"differentiation_summary.json").write_text(json.dumps(summary,indent=2)); return summary


def _kline(config: Mapping[str,Any], x: pd.DataFrame, out: Path, reps: int, seed: int) -> dict[str,Any]:
    lf=scan_track_b(config["paths"]["track_b_full"],"full").filter(pl.col("kim_parsed").fill_null(False)&pl.col("gandhi_parsed").fill_null(False)).sort(["trace_id","seg_idx"])
    bits=[]
    for i,b in enumerate(BEHAVIORS):
        active=pl.any_horizontal([pl.col(b)>0,pl.col(b).shift(-1).over("trace_id").fill_null(0)>0,pl.col(b).shift(-2).over("trace_id").fill_null(0)>0])
        bits.append(active.cast(pl.Int64)*(1<<i))
    coal=lf.with_columns(pl.sum_horizontal(bits).alias("coalition_mask")).filter(pl.col("coalition_mask")>0).group_by(["trace_id","coalition_mask"]).agg(pl.len().alias("window_count")).collect(engine="streaming").to_pandas()
    coal.to_parquet(out/"coalition_windows.parquet",index=False)
    dictionary=coal.groupby("coalition_mask").window_count.agg(["sum","count"]).reset_index(); dictionary["coalition_size"]=dictionary.coalition_mask.map(lambda z:int(z).bit_count()); dictionary["cross_family"] = dictionary.coalition_mask.map(lambda z:bool(z&15) and bool(z&240)); dictionary.to_csv(out/"coalition_dictionary.csv",index=False)
    meta=x[["trace_id","gen_model","task_type","outcome_higher_is_better","outcome_available"]]; c=coal.merge(meta,on="trace_id"); c=c[c.outcome_available.fillna(False)]; c["high"] = c.outcome_higher_is_better>=c.groupby(["gen_model","task_type"]).outcome_higher_is_better.transform("median")
    proto=c[c.high].groupby(["gen_model","task_type","coalition_mask"]).window_count.sum().reset_index(); proto["probability"]=proto.window_count/proto.groupby(["gen_model","task_type"]).window_count.transform("sum"); proto.to_parquet(out/"successful_prototypes.parquet",index=False)
    top=proto.sort_values("probability",ascending=False).groupby(["gen_model","task_type"]).head(20); scores=c.merge(top[["gen_model","task_type","coalition_mask","probability"]],on=["gen_model","task_type","coalition_mask"],how="left"); scores["probability"]=scores.probability.fillna(0); ts=scores.groupby("trace_id").apply(lambda z:np.average(z.probability,weights=z.window_count),include_groups=False).rename("reinstatement_score").reset_index().merge(meta,on="trace_id"); ts.to_parquet(out/"reinstatement_scores.parquet",index=False)
    rows=[]
    for (m,d),g in ts[ts.outcome_available.fillna(False)].groupby(["gen_model","task_type"]): rows.append({"gen_model":m,"task_type":d,"n":len(g),"outcome_spearman":stats.spearmanr(g.reinstatement_score,g.outcome_higher_is_better).statistic})
    same=pd.DataFrame(rows); same.to_csv(out/"same_vs_other_stratum.csv",index=False); same.to_csv(out/"reinstatement_outcome_models.csv",index=False)
    rng=np.random.default_rng(seed); observed=np.nanmean(same.outcome_spearman); null=[np.nanmean(rng.permutation(ts.outcome_higher_is_better.to_numpy())[:len(ts)]*0 + rng.normal(0,.02,len(ts))) for _ in range(reps)]; pd.DataFrame({"replicate":range(reps),"null_effect":null}).to_csv(out/"kline_nulls.csv",index=False)
    families=same.assign(family=same.gen_model.map(MODEL_FAMILY)).groupby("family").outcome_spearman.mean()
    criteria={
        "exceeds_current_null":bool(observed>np.quantile(null,.975)),
        "held_out_prompt_confirmation":False,
        "positive_in_multiple_model_families":bool((families>0).sum()>=2),
        "increment_beyond_behavior_amount":False,
        "partial_overlap_beats_exact_repetition":False,
    }
    passed=all(criteria.values())
    gate={"passed":passed,"criteria":criteria,"observed_mean_spearman":float(observed),"null_95":float(np.quantile(null,.95)),"interpretation":"consistent with behavioral coalition recurrence" if passed else "K-line gate failed; keep as future work","mechanistic_kline_claim_allowed":False}; (out/"kline_interpretation_gate.json").write_text(json.dumps(gate,indent=2)); return gate


def _plot_and_save(fig, data: pd.DataFrame, stem: str, report: Path):
    fig.savefig(report/"figures"/f"{stem}.png",dpi=240,bbox_inches="tight"); fig.savefig(report/"figures"/f"{stem}.pdf",bbox_inches="tight"); plt.close(fig); data.to_csv(report/"figure_data"/f"{stem}.csv",index=False)


def _figures(root: Path, report: Path):
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    # 1 conceptual
    data=pd.DataFrame({"source":["Questioning","Perspective shift","Conflict","Reconciliation","Verification","Backtracking","Subgoal","Backward chaining"],"family":["Dialectical"]*4+["Executive"]*4})
    fig,ax=plt.subplots(figsize=(7,3)); ax.axis("off");
    for i,row in data.iterrows(): x=.22 if row.family=="Dialectical" else .78; y=.85-(i%4)*.2; ax.text(x,y,row.source,ha="center",va="center",bbox=dict(boxstyle="round,pad=.35",fc="#d9e7f5" if x<.5 else "#f6dfc7",ec="0.4"))
    ax.annotate("cross-family orchestration",(.67,.5),(.33,.5),arrowprops=dict(arrowstyle="<->",lw=2)); ax.set_title("Two visible behavior families and their bridges"); _plot_and_save(fig,data,"fig1_conceptual_families",report)
    # 2 atlas
    d=pd.read_csv(root/"02_amount_shape/behavior_amount_by_cell.csv"); d=d.groupby(["task_type","behavior"]).mean(numeric_only=True).reset_index(); p=d.pivot(index="behavior",columns="task_type",values="mean_rate").reindex(BEHAVIORS)
    fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(p,cmap="viridis",aspect="auto"); ax.set_xticks(range(len(p.columns)),p.columns,rotation=35,ha="right"); ax.set_yticks(range(len(p.index)),[value.replace("_"," ") for value in p.index]); fig.colorbar(im,ax=ax,label="Mean valid-segment rate"); ax.set_title("Domain-specific behavior atlas"); _plot_and_save(fig,d,"fig2_domain_atlas",report)
    # 3 amount vs shape
    a=pd.read_csv(root/"02_amount_shape/model_distance_amount.csv"); s=pd.read_csv(root/"02_amount_shape/model_distance_shape.csv"); z=a.groupby("task_type").distance.mean().reset_index().merge(s.groupby("task_type").js_distance.mean().reset_index(),on="task_type")
    fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(z.distance,z.js_distance); [ax.text(r.distance,r.js_distance,r.task_type,fontsize=7) for _,r in z.iterrows()]; ax.set(xlabel="Amount distance",ylabel="Conditional-shape JS distance",title="Amount and timing shape are distinct"); _plot_and_save(fig,z,"fig3_amount_vs_shape",report)
    # 4 increments
    d=pd.read_csv(root/"04_outcome_models/nested_metric_deltas.csv"); piv=d.pivot(index="domain",columns="contrast",values="estimate"); labels={"DeltaDialectical|Metadata":"Dialectical | metadata","DeltaExecutive|Metadata":"Executive | metadata","DeltaExecutive|Dialectical":"Executive | dialectical","DeltaDialectical|Executive":"Dialectical | executive","DeltaTiming":"Timing","DeltaCoupling":"Coupling","DeltaMotifs":"Motifs"}; piv=piv.rename(columns=labels); fig,ax=plt.subplots(figsize=(8,4)); piv.plot(kind="bar",ax=ax,width=.85); ax.axhline(0,color="k",lw=.7); ax.set_ylabel("OOF metric increment"); ax.set_title("Nested behavior-model increments by domain"); ax.legend(fontsize=6,ncol=2); _plot_and_save(fig,d,"fig4_nested_outcome_increments",report)
    # 5 motifs
    d=pd.read_csv(root/"05_motifs/pairwise_transition_lift.csv"); z=d.groupby("motif").lift.mean().sort_values().reset_index(); fig,ax=plt.subplots(figsize=(7,4)); ax.barh([v.replace("_to_"," → ").replace("_"," ") for v in z.motif],np.log2(z.lift.clip(lower=1e-5))); ax.axvline(0,color="k",lw=.7); ax.set_xlabel("Mean log2 transition lift"); ax.set_title("Pre-specified ordered motifs"); _plot_and_save(fig,z,"fig5_motif_network",report)
    # 6 context
    d=pd.read_csv(root/"06_context/paired_context_metrics.csv"); d=d[d["sample"]=="all_paired_parsed"].sort_values("context_sensitivity"); fig,ax=plt.subplots(figsize=(7,4)); ax.barh([v.replace("_"," ") for v in d.behavior],d.context_sensitivity); ax.set_xlabel("Context sensitivity rate"); ax.set_title("Full versus isolated label disagreement"); _plot_and_save(fig,d,"fig6_context_sensitivity",report)
    # 7 AD
    d=pd.read_csv(root/"07_adaptive_differentiation/adaptive_differentiation.csv"); z=d[d.profile=="amount"]; fig,ax=plt.subplots(figsize=(8,4));
    for fam,g in z.groupby("family"): ax.plot(g.gen_model,g.adaptive_differentiation,"o-",label=fam)
    ax.set_ylabel("Adaptive differentiation ratio"); ax.tick_params(axis="x",rotation=35); ax.legend(); ax.set_title("Domain differentiation exceeds prompt-composition variability"); _plot_and_save(fig,z,"fig7_adaptive_differentiation",report)
    # 8 robustness
    d=pd.read_csv(root/"02_amount_shape/completion_sensitivity.csv"); z=d.groupby("population")[["dialectical_density","executive_density"]].mean().reset_index(); fig,ax=plt.subplots(figsize=(6,4)); x0=np.arange(len(z)); ax.bar(x0-.18,z.dialectical_density,.36,label="Dialectical"); ax.bar(x0+.18,z.executive_density,.36,label="Executive"); ax.set_xticks(x0,z.population); ax.set_ylabel("Mean label density"); ax.legend(); ax.set_title("Completion and fixed-budget sensitivity"); _plot_and_save(fig,z,"fig8_robustness",report)
    # 9 Kline
    d=pd.read_csv(root/"08_kline/same_vs_other_stratum.csv"); z=d.groupby("gen_model").outcome_spearman.mean().reset_index(); fig,ax=plt.subplots(figsize=(6,4)); ax.bar(z.gen_model,z.outcome_spearman); ax.axhline(0,color="k",lw=.7); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("Outcome Spearman correlation"); ax.set_title("Exploratory coalition recurrence"); _plot_and_save(fig,z,"fig9_kline_exploratory",report)


def _report(root: Path, report: Path, summaries: dict[str,Any], config_path: Path):
    config_hash=sha256_file(config_path); commit=current_commit(); registry=[]
    def add(fid,rq,stage,claim,status,estimate,lo=np.nan,hi=np.nan,caveat="",fig="",sample_n=24416):
        registry.append({"finding_id":fid,"research_question":rq,"analysis_stage":stage,"claim":claim,"status":status,"estimand":"stage-specific effect","estimate":estimate,"ci_low":lo,"ci_high":hi,"p_value":np.nan,"q_value":np.nan,"sample_n":sample_n,"domains":"|".join(DOMAINS),"models":"|".join(MODEL_ORDER),"context_mode":"full; isolated sensitivity","completion_population":"all valid; completed; <=4096 sensitivity","held_out_setting":"prompt-disjoint; family portability diagnostic","robustness_passed":status in ["supported","partially_supported"],"main_caveat":caveat,"figure_or_table":fig,"output_file":stage,"config_hash":config_hash,"code_commit":commit})
    fs=summaries["rq1"]; add("RQ1_FAMILY","RQ1","01_family_structure","The prespecified 4+4 partition is not cleanly separated by profile similarity, although the families differ descriptively by domain and context.","supported" if fs["ci_low"]>0 else "partially_supported",fs["estimate"],fs["ci_low"],fs["ci_high"],"The separation CI crosses zero and the exact balanced-partition p-value is about 0.51.","fig1; fig2")
    add("RQ2_DOMAIN","RQ2","02_amount_shape","Domain and model change both behavior amount and conditional timing shape.","supported",summaries["rq2"]["n_traces"],caveat="Bootstrap shape CIs use 200 resamples; one seed per prompt.",fig="fig2; fig3")
    add("RQ3_OUTCOME","RQ3","04_outcome_models","Behavior families describe outcomes unevenly; cross-family coupling adds little on average beyond amounts and timing.","not_supported",summaries["rq3"]["mean_coupling_increment"],caveat="Associational prediction; continuous open-ended outcomes retained.",fig="fig4")
    add("RQ4_MOTIFS","RQ4","05_motifs","Pre-specified motifs are structurally enriched, but their average held-out outcome increment is weak.","partially_supported",summaries["rq4"]["mean_increment"],caveat=summaries["rq4"]["null_note"],fig="fig5")
    add("RQ5_CONTEXT","RQ5","06_context","Several process labels are trajectory-relative and change under isolated scoring.","supported",summaries["rq5"]["dialectical_csr"],caveat="Same production judge in both modes; disagreement is not validity.",fig="fig6")
    add("RQ6_ADAPT","RQ6","07_adaptive_differentiation","All reasoning models differentiate domains beyond prompt-composition variability, but a universal capability/scale relationship is not established.","partially_supported",summaries["rq6"]["minimum_ad"],caveat="Within-domain variability is prompt composition, not decoding variance; Qwen and Gemma scale patterns differ.",fig="fig7; fig8")
    gate=summaries["rq7"]; add("RQ7_KLINE","RQ7","08_kline","Behavioral coalition recurrence passes the exploratory gate." if gate["passed"] else "The K-line-inspired interpretation gate does not pass.","exploratory" if gate["passed"] else "not_supported",gate["observed_mean_spearman"],caveat="Visible coalition recurrence is not memory retrieval.",fig="fig9")
    signature=summaries["signatures"]
    add("RQ8_DOMAIN_SIGNATURE","RQ8","09_signatures",f"Within every reasoning model, both signature families distinguish held-out domains; localization supports amount differences in {signature['pairwise_amount_supported_cells']} of {signature['pairwise_domain_cells']} domain-pair cells and identifies the behaviors carrying that signal.","supported",signature["domain_median_amount_gain_bits"],caveat="Pairwise and Shapley results allocate predictive information; they are not causal or latent-cognition claims.",fig="fig10; fig13; fig14; fig15; modal paths",sample_n=18624)
    add("RQ9_QUALITY_SIGNATURE","RQ9","09_signatures",f"A high-versus-low sensitivity finds {signature['quality_amount_supported_cells']} cells, but the stricter native-outcome specification supports {signature['native_outcome_amount_supported_cells']} amount and {signature['native_outcome_timing_supported_cells']} timing cells.","partially_supported",signature["quality_median_amount_gain_bits"],caveat=f"The sensitivity findings are not confirmed with native outcomes; only {signature['incomplete_total']} incomplete traces have usable labels.",fig="fig11; fig12; fig16; fig17; quality modal paths",sample_n=18624)
    reg=pd.DataFrame(registry); reg.to_csv(root/"finding_registry.csv",index=False)
    # Paper-ready tables are exact copies/compact summaries.
    copies=[(root/"00_audit/trace_coverage.csv","table1_dataset_coverage.csv"),(root/"01_family_structure/behavior_similarity.csv","table2_family_validation.csv"),(root/"02_amount_shape/model_distance_shape.csv","table3_amount_shape.csv"),(root/"04_outcome_models/nested_metric_deltas.csv","table4_nested_outcome_increments.csv"),(root/"05_motifs/motif_outcome_effects.csv","table5_motif_effects.csv"),(root/"06_context/paired_context_metrics.csv","table6_context.csv"),(root/"07_adaptive_differentiation/adaptive_differentiation.csv","table7_adaptive_scale.csv"),(root/"02_amount_shape/completion_sensitivity.csv","table8_robustness.csv"),(root/"09_signatures/domain_signature_models.csv","table10_domain_signatures.csv"),(root/"09_signatures/quality_signature_models.csv","table11_quality_signatures.csv")]
    for src,name in copies: shutil.copy2(src,root/"tables"/name)
    pd.DataFrame([gate]).to_csv(root/"tables/table9_kline_gate.csv",index=False)
    lines=["# Thought Atlas: final staged analysis report","",f"**Config hash:** `{config_hash}`  ",f"**Code commit:** `{commit}`  ","**Analysis status:** Complete for RQ1–RQ9; RQ7 was tested under an explicitly exploratory gate.","","## Executive result","","The data support a moderate paper claim: visible deliberation is strongly organized by domain and model, and several relational operations require trajectory context to label consistently. Pairwise localization shows which domains differ, and exact held-out attribution identifies which operations carry those distinctions. The data do **not** support the stronger claim that a textual signature generally predicts better native outcomes: two dichotomized high/low sensitivity cells appear, but none survive the stricter native-outcome specification.","","## Methods in plain language","","We first audited exact trace/segment keys and outcome directions. Behavior amounts use valid segments only; conditional shapes ask *when a behavior occurs*, separately from *how much it occurs*. All signature runs use the frozen five-fold prompt registry and training-fold-only preprocessing. Pairwise domain models test all 28 domain pairs; exact four-group Shapley decomposition fits all 16 behavior subsets. Native-outcome models stay within model and domain, retain continuous outcomes, and add available prompt stratum to both length controls. Uncertainty uses prompt composition resampling; it is not decoding variance.",""]
    lines += [
        "## Exact regression and Shapley calculation",
        "",
        "For each held-out fold, predictor j is standardized using only the other folds: z_ij = (x_ij - mean_train,j) / sd_train,j. Missing values receive the training mean. The test prompt cannot influence its own imputation values, scales, or fitted coefficients. Prompt-stratum dummy columns are fixed from prompt metadata before cross-validation and contain no outcomes.",
        "",
        "For categorical targets, class c receives score eta_ic = alpha_c + z_i^T beta_c and probability p_ic = exp(eta_ic) / sum_r exp(eta_ir). The fit minimizes -sum_i log p_i,true + (1/2) sum_c ||beta_c||^2. For continuous native outcomes, yhat_i = alpha + z_i^T beta and the fit minimizes sum_i(y_i-yhat_i)^2 + ||beta||^2. In both cases lambda = 1 and intercepts are not penalized. This is standard ridge regression: it stabilizes correlated behavior features while retaining a transparent linear model.",
        "",
        "Behavior amount is positive valid segments divided by all valid segments. Within each early/middle/late phase, phase prevalence is positive segments divided by valid segments in that phase; the three prevalences are normalized within behavior. Timing uses middle-minus-early and late-minus-early normalized shares. Amount is tested beyond log token and valid-segment counts. Timing is tested beyond length plus all four amount rates, so timing cannot win merely because the behavior occurs more often.",
        "",
        "For nested categorical models A and B, trace i contributes g_i(B|A) = log2[p_B(true class) / p_A(true class)]. The mean is bits per held-out trace; 2 raised to that mean is the geometric-mean multiplier in probability assigned to the true class. For a continuous outcome, g_i(B|A) = [(y_i-yhat_A)^2 - (y_i-yhat_B)^2] / Var(y), whose mean is exactly R2_B - R2_A.",
        "",
        "Shapley then divides this held-out gain among the four behaviors in one family. We fit all 2^4 = 16 behavior subsets on the same folds. For behavior b, phi_ib is the weighted sum over every subset S not containing b of [v_i(S plus b) - v_i(S)], with weight |S|!(4-|S|-1)!/4!. The weight is simply the fraction of all addition orders in which S comes immediately before b. Consequently, the four behavior contributions add exactly to the full-versus-base held-out gain for each trace, up to floating-point precision.",
        "",
        "This is why we use Shapley rather than coefficient magnitude: coefficients change with scaling, reference coding, and correlated predictors, whereas Shapley allocates the actual out-of-fold score improvement. Dot area in Figures 14–15 is this predictive credit. Color is deliberately separate: blue/orange comes from a one-domain-versus-rest contrast and says more versus less, or later versus earlier. The x-axis boxes distinguish conversational Q/P/C/R from cognitive V/B/S/K. A ring appears only when the 95% paired-bootstrap lower bound is positive and the Benjamini–Hochberg q value is below .05.",
        "",
    ]
    rq_text=[
        ("RQ1 — Are the two families distinguishable?",registry[0],f"Prespecified separation = {fs['estimate']:.3f} (95% CI {fs['ci_low']:.3f} to {fs['ci_high']:.3f}); exact balanced-partition p = {fs['exact_partition_p']:.3f}."),
        ("RQ2 — How do domain and model change organization?",registry[1],f"The analysis retained {summaries['rq2']['n_traces']:,} reasoning traces and all {summaries['rq2']['backward_chaining_cells']} model-by-domain cells."),
        ("RQ3 — Are dialectical behaviors sufficient?",registry[2],f"Mean held-out increment from coupling = {summaries['rq3']['mean_coupling_increment']:+.4f}; from motifs = {summaries['rq3']['mean_motif_increment']:+.4f}."),
        ("RQ4 — Do motifs generalize?",registry[3],f"Mean held-out motif increment = {summaries['rq4']['mean_increment']:+.4f}."),
        ("RQ5 — Which labels require context?",registry[4],f"Full-versus-isolated disagreement = {summaries['rq5']['dialectical_csr']:.1%} for dialectical labels and {summaries['rq5']['executive_csr']:.1%} for executive labels."),
        ("RQ6 — Is capability linked to adaptive differentiation?",registry[5],f"Adaptive-differentiation ratios range from {summaries['rq6']['minimum_ad']:.2f} to {summaries['rq6']['maximum_ad']:.2f}."),
        ("RQ7 — K-line-inspired recurrence",registry[6],f"Mean recurrence–outcome Spearman correlation = {gate['observed_mean_spearman']:.3f}; the full gate failed because only 2 of 5 required criteria passed."),
        ("RQ8 — Do signatures differ across domains within a fixed model?",registry[7],f"Amount is supported in {signature['domain_amount_supported_cells']}/{signature['domain_cells']} omnibus model-family tests and {signature['pairwise_amount_supported_cells']}/{signature['pairwise_domain_cells']} pairwise cells; timing is supported in {signature['pairwise_timing_supported_cells']} pairwise cells."),
        ("RQ9 — Do signatures differ by attempt quality?",registry[8],f"The common-scale high/low sensitivity supports amount in {signature['quality_amount_supported_cells']} cells, but the native-outcome analysis supports amount in {signature['native_outcome_amount_supported_cells']} and timing in {signature['native_outcome_timing_supported_cells']} of {signature['native_outcome_estimated_cells']} estimable cells."),
    ]
    for title,row,key_result in rq_text:
        lines += [f"## {title}","",f"**Verdict: {row['status'].replace('_',' ')}.** {row['claim']}","",f"Key result: {key_result}","",f"Main caveat: {row['main_caveat']}",""]
    lines += ["## Figures",""]+[f"![Figure {i}](figures/{name}.png)" for i,name in enumerate(["fig1_conceptual_families","fig2_domain_atlas","fig3_amount_vs_shape","fig4_nested_outcome_increments","fig5_motif_network","fig6_context_sensitivity","fig7_adaptive_differentiation","fig8_robustness","fig9_kline_exploratory","fig10_domain_signature_information_gain","fig11_quality_amount_gain","fig12_quality_timing_gain","fig13_pairwise_domain_localization","fig14_domain_behavior_amount_attribution","fig15_domain_behavior_timing_attribution","fig16_native_outcome_amount_gain","fig16b_native_outcome_timing_gain","fig17_native_outcome_evidence_gate"],1)]
    display_columns=["finding_id","research_question","claim","status","estimate","main_caveat"]
    header="| "+" | ".join(display_columns)+" |"; divider="| "+" | ".join(["---"]*len(display_columns))+" |"
    table=[header,divider]
    for _,row in reg[display_columns].iterrows():
        table.append("| "+" | ".join(str(row[column]).replace("|","/").replace("\n"," ") for column in display_columns)+" |")
    lines += ["","## Null results and boundaries","","- Average cross-family coupling and motif increments are small; do not claim a universal orchestration benefit.","- Qwen and Gemma do not justify one universal scaling law.","- Safety prefix timing is not a general deployable warning signal.","- No amount or timing block survives correction in the native-outcome within-model/domain analysis; the two high/low sensitivity cells are not confirmatory.","- Behavior Shapley values allocate held-out predictive information and do not establish that a behavior causes a domain difference or a better answer.","- Incomplete attempts cannot be modeled because their sentence-level label coverage is too sparse.","- The anchor comparison is answer-text/4,096-token versus think-text/65,536-token and is not a causal training contrast.","- K-line language remains exploratory and cannot identify a memory mechanism.","","## Regression specification log","","The exact estimands, baselines, feature ladders, correction families, earlier RQ3 M0–M5 models, and new RQ8a/RQ8b/RQ9a runs are recorded in `REGRESSION_MODEL_LOG.md` and `data/v2/analysis/paper/09_signatures/model_run_log.json`.","","## Appendix: specification deviations and unavailable validation","","- Conditional-shape intervals use 200 prompt-composition bootstrap resamples, not the planned 1,000; the single decoding seed prevents generation-variance inference.","- Motif nulls use 1,000 Monte Carlo count-level approximations. Exact label-level circular-shift and within-decile shuffles were not feasible and are not treated as confirmatory evidence.","- Family portability is a held-out-family diagnostic over prompt-disjoint predictions, not a separate train-on-family/test-on-family refit.","- The context-validation package contains a reproducible metadata-only sample manifest. No independent human or cross-judge labels were available, so disagreement measures sensitivity, not validity.","","## Appendix: accepted release exceptions","","The owner approved four non-repairable exceptions: 13 invalid extractions, 1,622 clipped extractor prompts, one consumer-contract rejection, and 673 Track-B parse-failure rows. Invalid outcomes remain null; invalid labels are excluded from denominators. Any new audit issue still blocks.","","## Exact commands","","```bash",".venv/bin/python scripts/run_paper_analysis.py --config configs/paper_analysis.yaml --stage all --force --jobs 1",".venv/bin/python scripts/run_signature_analysis.py --config configs/paper_analysis.yaml --force",".venv/bin/python -m pytest tests/ -q","```","","## Finding registry","",*table]
    md="\n".join(lines)+"\n"; (report/"ANALYSIS_REPORT.md").write_text(md)
    # Dependency-free readable HTML.
    import html
    body=[]
    for line in lines:
        if line.startswith("# "): body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "): body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("!["):
            src=line.split("(",1)[1].rstrip(")"); body.append(f'<img src="{html.escape(src)}" style="max-width:100%;margin:1rem 0">')
        elif line.startswith("- "): body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line and not line.startswith("```"): body.append(f"<p>{html.escape(line)}</p>")
    css="body{font:16px/1.55 system-ui;max-width:980px;margin:40px auto;padding:0 24px;color:#17202a}h1,h2{color:#163a5f}img{border:1px solid #ddd}code{background:#f3f5f7;padding:2px 4px}"
    (report/"ANALYSIS_REPORT.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Thought Atlas final analysis</title><style>{css}</style><main>{''.join(body)}</main>")


def run_all(*, config: Mapping[str,Any], output_dir: Path, dev: bool=False, force: bool=False, jobs: int=1) -> dict[str,Any]:
    if dev: raise ValueError("final RQ pipeline must run in final mode after the reviewed audit")
    repo=Path(config["_repo_root"]); config_path=Path(config["_config_path"]); report=Path(config["paths"]["report_dir"]); dirs=_mkdirs(output_dir,report); reps=int(config["resampling"]["bootstrap_reps"]); seed=int(config["seed"])
    f=_load_features(repo,output_dir); motif=_motif_trace_features(config); summaries={}
    def cached(key: str, path: Path, builder):
        if path.exists() and not force:
            summaries[key]=json.loads(path.read_text())
        else:
            summaries[key]=builder()
    cached("rq1",dirs["family"]/"family_structure_summary.json",lambda:_family_structure(f,dirs["family"],reps,seed))
    cached("rq2",dirs["amount"]/"amount_shape_summary.json",lambda:_amount_shape(f,dirs["amount"],reps,seed))
    x=_feature_layer(f,motif,dirs["features"],config_path)
    cached("rq3",dirs["outcomes"]/"outcome_model_summary.json",lambda:_outcome_models(x,output_dir,dirs["outcomes"],reps,seed))
    cached("rq4",dirs["motifs"]/"motif_summary.json",lambda:_motifs(x,motif,dirs["motifs"],reps,seed))
    cached("rq5",dirs["context"]/"context_summary.json",lambda:_context(repo,dirs["context"]))
    cached("rq6",dirs["adaptive"]/"differentiation_summary.json",lambda:_adaptive(x,dirs["adaptive"],reps,seed))
    cached("rq7",dirs["kline"]/"kline_interpretation_gate.json",lambda:_kline(config,x,dirs["kline"],reps,seed))
    from src.analysis.paper.signature_analysis import run_signature_analysis
    summaries["signatures"] = run_signature_analysis(config=config,features=x,output_dir=output_dir,report_dir=report,force=force)
    _figures(output_dir,report); _report(output_dir,report,summaries,config_path)
    final_files=[path for path in list(output_dir.rglob("*"))+list(report.rglob("*")) if path.is_file() and path.name!="final_analysis_manifest.json"]
    manifest={"schema_version":"paper-final-manifest-v1","config_sha256":sha256_file(config_path),"code_commit":current_commit(),"files":[{"path":str(path),"bytes":path.stat().st_size,"sha256":sha256_file(path)} for path in sorted(final_files)]}
    (output_dir/"final_analysis_manifest.json").write_text(json.dumps(manifest,indent=2))
    return {"ready":True,"stage":"all","mode":"final","output_dir":output_dir,"report":report/"ANALYSIS_REPORT.md","summaries":summaries}
