"""Paper figure/table exports from corrected results only; every figure records its scope."""
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import atomic_json


def save_figure(fig,path):
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def plot_history(history,path,dataset,render_only=False):
    candidates=['train_render','train_loss','val_rmse','selection_score'] if render_only else ['train_loss','selection_score','val_rmse','train_glvp_cosine']
    keys=[k for k in candidates if k in history]
    if not keys:raise ValueError('Training history missing figure inputs')
    fig,axes=plt.subplots(1,len(keys),figsize=(4*len(keys),3.2),squeeze=False)
    for ax,key in zip(axes[0],keys):
        ax.plot(history.global_epoch,history[key],marker='o',markersize=3)
        for boundary in np.flatnonzero(history.stage.ne(history.stage.shift()).to_numpy())[1:]:
            ax.axvline(history.iloc[boundary].global_epoch-.5,color='grey',linestyle='--',linewidth=.8)
        ax.set(title=key,xlabel='Completed epoch');ax.grid(alpha=.25)
    fig.suptitle(f'{dataset.upper()} — corrected run, actual recorded history')
    fig.tight_layout();save_figure(fig,path)


def qualitative(record,path,dataset):
    rgb=record['rgb'].permute(0,2,3,1).numpy()
    pred=record['depth'][:,0].numpy();gt=record['reference_depth'][:,0].numpy();valid=record['valid'][:,0].numpy()>.5
    K=len(rgb);fig,axes=plt.subplots(6,K,figsize=(3*K,15),squeeze=False,layout='constrained')
    lo,hi=np.quantile(gt[valid],[.01,.99])
    if hi-lo<1e-6:
        margin=max(.001,abs(float(lo))*.01);lo-=margin;hi+=margin
    error=np.abs(pred-gt);limit=max(.01,float(np.quantile(error[valid],.95)))
    rows=[('RGB',rgb,None),('Reference depth',np.where(valid,gt,np.nan),'magma'),('Predicted depth',pred,'magma'),
          ('Absolute depth error',np.where(valid,error,np.nan),'inferno'),('Confidence',record['confidence'][:,0].numpy(),'viridis'),
          ('Predicted support',record['foreground'][:,0].numpy(),'gray')]
    for row,(title,data,cmap) in enumerate(rows):
        artist=None
        for k in range(K):
            options={} if row==0 else dict(cmap=cmap)
            if row in [1,2]:options.update(vmin=lo,vmax=hi)
            if row==3:options.update(vmin=0,vmax=limit)
            if row in [4,5]:options.update(vmin=0,vmax=1)
            artist=axes[row,k].imshow(data[k],**options);axes[row,k].set_axis_off()
            axes[row,k].set_title(f'{title}\nview {k}',fontsize=9)
        if row:fig.colorbar(artist,ax=axes[row].tolist(),fraction=.013,pad=.01,
                           extend='both' if row in [1,2] else 'neither')
    fig.suptitle(f'{dataset.upper()} — {K} input views; fixed depth scale shared across views')
    save_figure(fig,path)


def trajectory(record,path):
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,(a,b,label) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        for key,name in [('reference_pose','Reference'),('initial_pose','RGB initial'),('predicted_pose','Refined')]:
            poses=record[key].numpy()
            ax.plot(poses[:,a,3],poses[:,b,3],marker='o',label=name)
        ax.set_title(label);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('One window in its reference camera frame; no fitted trajectory alignment')
    save_figure(fig,path)


def cloud_views(record,path,threshold=None):
    pts=record['points'].permute(0,2,3,1).reshape(-1,3).numpy()
    colors=record['rgb'][:,:,::8,::8].permute(0,2,3,1).reshape(-1,3).numpy()
    if threshold is not None:
        confidence=record['confidence'][:,0,::8,::8].reshape(-1).numpy()
        support=record['foreground'][:,0,::8,::8].reshape(-1).numpy()>.5
        keep=(confidence>threshold)&support;pts,colors=pts[keep],colors[keep]
    take=np.arange(len(pts))[::max(1,len(pts)//12000)]
    pts,colors=pts[take],colors[take]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,(a,b,label) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        gt=record['reference_points'].permute(0,2,3,1).reshape(-1,3).numpy()
        gt=gt[record['valid'][:,0,::8,::8].reshape(-1).numpy()>.5]
        gt=gt[::max(1,len(gt)//6000)]
        if len(gt):ax.scatter(gt[:,a],gt[:,b],c='black',s=.15,alpha=.15,label='Fixed reference')
        if len(pts):ax.scatter(pts[:,a],pts[:,b],c=colors.clip(0,1),s=.2)
        ax.set_title(label);ax.set_aspect('equal',adjustable='datalim')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'One physical window; predicted coordinates; reference overlay; confidence={threshold}')
    save_figure(fig,path)


def threshold_clouds(record,path,thresholds):
    pts=record['points'].permute(0,2,3,1).reshape(-1,3).numpy()
    confidence=record['confidence'][:,0,::8,::8].reshape(-1).numpy()
    support=record['foreground'][:,0,::8,::8].reshape(-1).numpy()>.5
    fig,axes=plt.subplots(len(thresholds),3,figsize=(12,3*len(thresholds)),squeeze=False)
    for row,threshold in enumerate(thresholds):
        selected=pts[(confidence>threshold)&support]
        selected=selected[::max(1,len(selected)//8000)]
        for ax,(a,b,label) in zip(axes[row],[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
            if len(selected):ax.scatter(selected[:,a],selected[:,b],s=.3)
            ax.set_title(f'{label}; confidence > {threshold}; n={len(selected)}')
            ax.set_aspect('equal',adjustable='datalim')
    fig.suptitle('Same physical window and coordinate frame at every threshold')
    fig.tight_layout();save_figure(fig,path)


def confidence_figure(run,c,path):
    sources={c['dataset']:Path(run)}
    companion=Path(c['companion_run']) if c.get('companion_run') else None
    other='tum' if c['dataset']=='co3d' else 'co3d'
    if companion is None:
        candidate=Path(c['output_root'])/other/c['study']/c['run_name']
        if (candidate/'test_provenance.json').exists():companion=candidate
    if companion and (companion/'test_provenance.json').exists():
        provenance=json.loads((companion/'test_provenance.json').read_text())
        if provenance.get('reference_support')!='fixed_independent_of_prediction':
            raise ValueError('Companion result uses an incompatible surface protocol')
        current=json.loads((Path(run)/'test_provenance.json').read_text())
        if provenance.get('source_hash')!=current['source_hash'] or provenance.get('synthetic')!=current['synthetic']:
            raise ValueError('Companion results must use the same corrected source and evidence type')
        other_config=json.loads((companion/'resolved_config.json').read_text())
        if other_config['dataset']!=other or any(other_config[k]!=c[k] for k in ['study','variant','seed','eval_views']):
            raise ValueError('Companion must be the other dataset with the same study, variant, seed and evaluation K')
        sources[other]=companion
    fig,axes=plt.subplots(2,len(sources),figsize=(6*len(sources),7),squeeze=False)
    for col,(dataset,directory) in enumerate(sorted(sources.items())):
        df=pd.read_csv(directory/'paper_tables/test_surface_summary.csv')
        df=df[(df.alignment=='raw')&(df.filtering=='conf_plus_mask')].sort_values('confidence_threshold')
        for row,(key,label) in enumerate([('chamfer_l1','Per-window mean Chamfer'),('fscore_0.20','F@0.20')]):
            axes[row,col].plot(df.confidence_threshold,df[key],marker='o')
            axes[row,col].set(title=f'{dataset.upper()} — {label}',xlabel='Confidence threshold');axes[row,col].grid(alpha=.25)
    fig.suptitle('Fixed-reference, per-window evaluation; test sweeps are descriptive')
    fig.tight_layout();save_figure(fig,path)
    return list(sorted(sources))


def export_report(run,c,manifest,threshold,provenance):
    run=Path(run);figures=run/'paper_figures';figures.mkdir(parents=True,exist_ok=True)
    hist=pd.read_csv(run/'training_history.csv')
    dense=pd.read_csv(run/'paper_tables/test_window_metrics.csv')
    surfaces=pd.read_csv(run/'paper_tables/test_surface_summary.csv')
    sources=sorted((run/'qualitative').glob('window_*.pt'))
    if not sources:raise ValueError('No corrected qualitative exports were created')
    record=torch.load(sources[0],map_location='cpu',weights_only=True)
    scope={}
    def register(name,description):
        scope[name]=dict(description=description,**provenance)
        return figures/name
    if c['dataset']=='co3d':
        history_name='figure02_co3d_training_history.png'
        qualitative_name='figure05_co3d_qualitative_predictions.png'
        cloud_name='figure04_co3d_aggregate_cloud_views.png'
        sweep_name='appendix_co3d_raw_conf_plus_mask.png'
    else:
        history_name='figure06_tum_training_validation.png'
        qualitative_name='figure10_tum_qualitative_five_views.png'
        cloud_name='tum_single_window_cloud_views.png'
        sweep_name='appendix_tum_raw_conf_plus_mask.png'
    plot_history(hist,register(history_name,'Actual completed epoch history, including stage boundaries'),c['dataset'])
    qualitative(record,register(qualitative_name,'Configured evaluation K, reference depth and absolute errors on shared scales'),c['dataset'])
    cloud_views(record,register(cloud_name,'Single physical window; replaces the invalid mixed-gauge pooled visualization'),threshold)
    threshold_clouds(record,register(sweep_name,'Same single window under each confidence threshold'),c['confidence_thresholds'])
    trajectory(record,register('camera_trajectory_reference_vs_prediction.png','Reference, RGB initial and refined camera centers; common window frame'))
    datasets=confidence_figure(run,c,register('confidence_sweep_comparison.png','Corrected test threshold curves; cross-dataset panel requires both new runs'))
    if c['dataset']=='tum':
        plot_history(hist,register('appendix_tum_gs_training_curves.png',
            'NEW scope: corrected RGB-only final-model render/training history; not the old GT-noise experiment'),c['dataset'],render_only=True)
        fig,ax=plt.subplots(figsize=(5,3.5))
        ax.bar(['RGB initial','Refined'],[dense.initial_ate.mean(),dense.ate.mean()])
        ax.set(ylabel='Mean window ATE (m)',title='RGB-predicted cameras: initial versus refined')
        save_figure(fig,register('appendix_tum_gs_pose_audit_bars.png',
            'NEW scope: RGB-predicted initial versus refined camera ATE; no externally initialized poses'))
    main=surfaces[(surfaces.confidence_threshold==threshold)&(surfaces.filtering=='conf_plus_mask')]
    main.to_csv(run/'paper_tables/table_main_reconstruction.csv',index=False)
    dense.select_dtypes('number').agg(['mean','std','median','min','max']).to_csv(run/'paper_tables/table_dense_distribution.csv')
    summary=dense.select_dtypes('number').mean().to_dict();summary['n_eval_windows']=len(dense)
    pd.DataFrame([summary]).to_csv(run/'paper_tables/table_primary_results.csv',index=False)
    for name in scope:
        if not (figures/name).is_file():raise FileNotFoundError(name)
    atomic_json(run/'figure_manifest.json',dict(figures=scope,confidence_panel_datasets=datasets,
        cross_dataset_panel_complete=len(datasets)==2,
        historical_caption_changes_required=['figure04_co3d_aggregate_cloud_views.png','appendix_tum_gs_training_curves.png','appendix_tum_gs_pose_audit_bars.png'],
        old_results_reused=False))
    atomic_json(run/'result_summary.json',dict(dataset=c['dataset'],study=c['study'],variant=c['variant'],
        seed=c['seed'],epoch_schedule=c['epochs'],validation_confidence_threshold=threshold,primary=summary,**provenance))
    # Full checkpoints remain separate downloadable files; figures/tables ZIP stays small.
    archive=run/'paper_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['paper_tables','paper_figures']:
            for p in sorted((run/folder).glob('*')):
                if p.is_file():z.write(p,p.relative_to(run))
        for p in run.glob('*.json'):
            if p.name!='run_state.json':z.write(p,p.name)
        z.write(run/'training_history.csv','training_history.csv')


def collect_studies(root,dataset):
    """Aggregate only completed corrected runs; preserve per-run and across-seed rows."""
    root=Path(root);rows=[]
    for path in sorted((root/dataset).glob('*/*/result_summary.json')):
        run=path.parent
        state=json.loads((run/'run_state.json').read_text())
        result=json.loads(path.read_text())
        if state['status']!='complete' or result.get('synthetic'):continue
        rows.append(dict(study=result['study'],variant=result['variant'],seed=result['seed'],
            source_hash=result['source_hash'],data_hash=result['data_hash'],
            checkpoint_sha256=result['checkpoint_sha256'],run=str(run),**result['primary']))
    if not rows:raise ValueError('No completed corrected research runs found')
    frame=pd.DataFrame(rows)
    if frame[['study','variant','seed']].duplicated().any():
        raise ValueError('Duplicate study/variant/seed; choose one intended run explicitly')
    # Different scientific settings require separate tables, not an accidental pooled average.
    if frame.source_hash.nunique()!=1 or frame.data_hash.nunique()!=1:
        raise ValueError('Study collection contains different code/data fingerprints')
    out=root/dataset/'study_comparison';out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'completed_runs.csv',index=False)
    columns=[x for x in ['rmse','abs_rel','ate','rpe_translation','rpe_rotation_deg'] if x in frame]
    frame.groupby(['study','variant'])[columns].agg(['count','mean','std']).to_csv(out/'across_seed_summary.csv')
    fig,axes=plt.subplots(1,len(columns),figsize=(4*len(columns),4),squeeze=False)
    for ax,key in zip(axes[0],columns):
        for _,r in frame.iterrows():ax.scatter(f"{r['study']}/{r['variant']}",r[key],s=18)
        ax.tick_params(axis='x',rotation=90);ax.set_title(key);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save_figure(fig,out/'study_comparison.png')
    atomic_json(out/'coverage.json',dict(completed_run_count=len(rows),
        observed=frame[['study','variant','seed']].to_dict('records'),
        required_glvp_variants=['A0_no_glvp','A1_glvp_0025','A2_no_conditioning','A3_glvp_0010','A4_glvp_0050'],
        missing_experiments_are_not_imputed=True))
    return out
