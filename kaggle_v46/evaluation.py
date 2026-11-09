"""Per-window metrics with fixed references and validation-only threshold selection."""
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader,Subset

from . import model as M
from .data import WindowDataset,even_indices,slice_views,atomic_json
from .losses import rotation_angle,gaussian_render,camera_match_loss,glvp_terms


def to_device(batch,device):
    return {key:value.to(device,non_blocking=True) if torch.is_tensor(value) else value for key,value in batch.items()}


def paired_sim3(source,target):
    p,g=np.asarray(source,np.float64),np.asarray(target,np.float64)
    if len(p)<6 or p.shape!=g.shape or not np.isfinite(p).all():
        return None
    pmean,gmean=p.mean(0),g.mean(0)
    x,y=p-pmean,g-gmean
    variance=np.square(x).sum(1).mean()
    if variance<1e-12:return None
    U,S,V=np.linalg.svd(y.T @ x/len(x))
    D=np.eye(3);D[-1,-1]=np.sign(np.linalg.det(U @ V))
    rotation=U @ D @ V
    scale=float((S*np.diag(D)).sum()/variance)
    if not np.isfinite(scale) or scale<=0:return None
    translation=gmean-scale*rotation @ pmean
    return scale,rotation,translation


def transform(points,fit):
    scale,rotation,translation=fit
    return scale*(points @ rotation.T)+translation


def sample_points(points,maximum,seed):
    if len(points)<=maximum:return points
    return points[np.random.default_rng(seed).choice(len(points),maximum,replace=False)]


def cloud_metrics(predicted,reference,c):
    p,g=np.asarray(predicted,np.float64).reshape(-1,3),np.asarray(reference,np.float64).reshape(-1,3)
    if not len(g):raise ValueError('Empty fixed reference surface')
    invalid=not len(p) or not np.isfinite(p).all()
    if invalid:
        # Empty predictions remain failures in aggregate means; never NaN/drop this window.
        result=dict(accuracy=2*c['max_depth'],completeness=2*c['max_depth'],
                    chamfer_l1=4*c['max_depth'],failed=1,n_pred=len(p),n_gt=len(g))
        for threshold in c['fscore_thresholds']:
            label=f'{threshold:.2f}'
            result.update({f'precision_{label}':0.,f'recall_{label}':0.,f'fscore_{label}':0.})
        return result
    dp=cKDTree(g).query(p,k=1,workers=1)[0]
    dg=cKDTree(p).query(g,k=1,workers=1)[0]
    result=dict(accuracy=float(dp.mean()),completeness=float(dg.mean()),
                chamfer_l1=float(dp.mean()+dg.mean()),failed=0,n_pred=len(p),n_gt=len(g))
    for threshold in c['fscore_thresholds']:
        precision,recall=float((dp<threshold).mean()),float((dg<threshold).mean())
        label=f'{threshold:.2f}'
        result.update({f'precision_{label}':precision,f'recall_{label}':recall,
                       f'fscore_{label}':2*precision*recall/(precision+recall+1e-12)})
    return result


def confidence_bins(pred,batch,c):
    valid=batch['valids'].bool()
    confidence=pred['confidence'].float()[valid]
    success=((pred['depth'].float()[valid]-batch['depths'][valid]).abs()/batch['depths'][valid]
             <=c['confidence_relative_tolerance']).float()
    rows=[];ece=confidence.new_zeros(())
    for index in range(10):
        keep=(confidence>=index/10)&((confidence<(index+1)/10) if index<9 else (confidence<=1))
        count=int(keep.sum())
        predicted_sum=float(confidence[keep].sum());correct_sum=float(success[keep].sum())
        rows.append(dict(bin_index=index,lower=index/10,upper=(index+1)/10,count=count,
                         confidence_sum=predicted_sum,correct_sum=correct_sum))
        if count:ece+=abs(predicted_sum-correct_sum)/max(len(confidence),1)
    return rows,dict(confidence_brier=float((confidence-success).square().mean()),
                     confidence_ece=float(ece),depth_success_fraction=float(success.mean()))


def dense_metrics(pred,batch,c):
    valid=batch['valids'].bool()
    p,g=pred['depth'].float()[valid],batch['depths'].float()[valid]
    if not len(g):raise ValueError('No reference depth')
    if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite predicted depth')
    ratio=torch.maximum(p/g,g/p.clamp_min(1e-6))
    cam,world,relative=M.gt_local_world_points(batch)
    poses=pred['poses_pred_c2w'].float()
    result=dict(rmse=float((p-g).square().mean().sqrt()),abs_rel=float(((p-g).abs()/g).mean()),
                delta1=float((ratio<1.25).float().mean()),delta2=float((ratio<1.25**2).float().mean()),
                delta3=float((ratio<1.25**3).float().mean()),
                ate=float((poses[...,:3,3]-relative[...,:3,3]).square().sum(-1).mean().sqrt()),
                intrinsics_mae=float((M.normalize_K_params(pred['intrinsics_pred'])-M.normalize_K_params(batch['intrinsics'])).abs().mean()),
                pointmap_rmse=float((pred['pointmap_cam']-cam)[valid.expand_as(cam)].square().mean().sqrt()))
    if poses.shape[1]>1:
        pr=M.invert_se3(poses[:,:-1]) @ poses[:,1:]
        gr=M.invert_se3(relative[:,:-1]) @ relative[:,1:]
        error=M.invert_se3(gr) @ pr
        result['rpe_translation']=float(error[...,:3,3].square().sum(-1).mean().sqrt())
        result['rpe_rotation_deg']=float(torch.rad2deg(rotation_angle(pr[...,:3,:3],gr[...,:3,:3])).square().mean().sqrt())
    else:result.update(rpe_translation=0.,rpe_rotation_deg=0.)
    v=valid.expand_as(pred['pointmap_world'])
    source=pred['pointmap_world'].permute(0,1,3,4,2)[valid.squeeze(2)].cpu().numpy()
    target=world.permute(0,1,3,4,2)[valid.squeeze(2)].cpu().numpy()
    if len(source)>c['max_fit_points']:
        take=np.random.default_rng(c['eval_seed']).choice(len(source),c['max_fit_points'],replace=False)
        source,target=source[take],target[take]
    fit=paired_sim3(source,target)
    result['aligned_point_euclidean_mean']=float(np.linalg.norm(transform(source,fit)-target,axis=1).mean()) if fit else 2*c['max_depth']
    result['fitted_scale']=float(fit[0]) if fit else 1.
    # Initial-vs-refined camera audit comes from RGB-only predictions, never synthetic GT poses.
    initial=pred.get('poses_initial_c2w',poses)
    result['initial_ate']=float((initial[...,:3,3]-relative[...,:3,3]).square().sum(-1).mean().sqrt())
    identity_ate=relative[...,:3,3].square().sum(-1).mean().sqrt()
    # A training-depth-derived floor keeps stationary validation windows finite.
    floor=max(float(M.CFG.DEPTH_NEAR)*.001,1e-6)
    result['identity_ate']=float(identity_ate)
    result['ate_over_identity']=result['ate']/max(float(identity_ate),floor)
    if poses.shape[1]>2:
        result['pred_nonreference_spread']=float(poses[:,1:,:3,3].std(dim=1,unbiased=False).norm(dim=-1).mean())
        result['gt_nonreference_spread']=float(relative[:,1:,:3,3].std(dim=1,unbiased=False).norm(dim=-1).mean())
    return result,world,relative


def surface_rows(pred,batch,world,c,window_id,sequence,export=None,operating_threshold=None):
    stride=c['cloud_stride']
    predicted=pred['pointmap_world'][0,:,:,::stride,::stride].permute(0,2,3,1).cpu().numpy()
    reference=world[0,:,:,::stride,::stride].permute(0,2,3,1).cpu().numpy()
    valid=batch['valids'][0,:,0,::stride,::stride].bool().cpu().numpy()
    confidence=pred['confidence'][0,:,0,::stride,::stride].float().cpu().numpy()
    foreground=pred.get('foreground_mask')
    mask=foreground[0,:,0,::stride,::stride].float().cpu().numpy()>.5 if foreground is not None else np.ones_like(valid)
    image_domain=np.ones_like(valid)
    if c['dataset']=='tum':
        H=batch['images'].shape[-2]
        yy=np.arange(0,H,stride)
        image_domain &= ((yy>=H//8)&(yy<7*H//8))[None,:,None]
    fixed_reference=sample_points(reference[valid],c['max_metric_points'],c['eval_seed'])
    fit_p,fit_g=predicted[valid],reference[valid]
    if len(fit_p)>c['max_fit_points']:
        take=np.random.default_rng(c['eval_seed']).choice(len(fit_p),c['max_fit_points'],replace=False)
        fit_p,fit_g=fit_p[take],fit_g[take]
    fit=paired_sim3(fit_p,fit_g)
    if export is not None:
        if operating_threshold is None:raise ValueError('Cloud export requires a validation-selected threshold')
        select=(confidence>operating_threshold)&image_domain&mask
        points=sample_points(predicted[select],c['max_metric_points'],c['eval_seed']+1)
        # A separate file for every independent physical window; never concatenate gauges.
        export=Path(export);export.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(export/(window_id.replace('/','_')+'.npz'),
            predicted_raw=points,predicted_sim3=transform(points,fit) if fit else points,
            reference=fixed_reference,confidence_threshold=operating_threshold,
            sim3_valid=fit is not None,sequence=sequence,window_id=window_id)
    rows=[]
    for threshold in c['confidence_thresholds']:
        for filtering in ['conf_only','conf_plus_mask']:
            select=(confidence>threshold)&image_domain
            if filtering=='conf_plus_mask':select &= mask
            selected=sample_points(predicted[select],c['max_metric_points'],c['eval_seed']+1)
            for alignment in ['raw','sim3']:
                cloud=transform(selected,fit) if alignment=='sim3' and fit is not None else selected
                metrics=cloud_metrics(cloud,fixed_reference,c)
                metrics['alignment_failed']=int(alignment=='sim3' and fit is None)
                rows.append(dict(window_id=window_id,sequence=sequence,k_views=batch['images'].shape[1],
                    confidence_threshold=threshold,filtering=filtering,alignment=alignment,
                    selected_fraction=float(select.sum()/max(1,image_domain.sum())),
                    reference_points_before_sampling=int(valid.sum()),fitted_scale=float(fit[0]) if fit else 1.,**metrics))
    return rows


def cpu_record(pred,batch,relative,world):
    return dict(rgb=batch['images_raw'][0].detach().cpu(),depth=pred['depth'][0].float().cpu(),
                reference_depth=batch['depths'][0].cpu(),valid=batch['valids'][0].cpu(),
                confidence=pred['confidence'][0].float().cpu(),
                foreground=pred.get('foreground_mask',batch['valids'])[0].float().cpu(),
                predicted_pose=pred['poses_pred_c2w'][0].cpu(),reference_pose=relative[0].cpu(),
                initial_pose=pred.get('poses_initial_c2w',pred['poses_pred_c2w'])[0].cpu(),
                reference_points=world[0,:,:,::8,::8].cpu(),
                points=pred['pointmap_world'][0,:,:,::8,::8].cpu())


@torch.no_grad()
def evaluate(network,manifest,split,c,device,rank=0,world_size=1,full=False,output=None,operating_threshold=None):
    network.set_stage('eval');network.eval()
    teacher=getattr(network,'_evaluation_teacher',None) if split=='val' else None
    dataset=WindowDataset(manifest,split,c,teacher)
    n=len(dataset) if not c['max_eval_windows'] else min(len(dataset),c['max_eval_windows'])
    ids=list(range(rank,n,world_size))
    loader=DataLoader(Subset(dataset,ids),batch_size=1,shuffle=False,num_workers=c['workers_per_gpu'],pin_memory=device.type=='cuda')
    dense,surfaces,renders,calibration=[],[],[],[]
    for index,batch in zip(ids,loader):
        batch=to_device(batch,device)
        batch=slice_views(batch,even_indices(batch['images'].shape[1],c['eval_views']))
        t=time.perf_counter()
        with torch.amp.autocast(device.type,enabled=c['eval_amp'] and device.type=='cuda'):
            pred=network(batch['images'],batch['images_raw'])
        metrics,world,relative=dense_metrics(pred,batch,c)
        bins,calibration_metrics=confidence_bins(pred,batch,c);metrics.update(calibration_metrics)
        _,match_metrics=camera_match_loss(pred,batch,c,world,relative);metrics.update(match_metrics)
        if c['glvp_weight']>0 and 'teacher' in batch:
            _,glvp_metrics=glvp_terms(pred['crossjepa_pred_embeddings'],batch['teacher'],c)
            metrics.update(glvp_metrics)
        wid=dataset.windows[index]['id'];seq=dataset.windows[index]['seq']
        if full:calibration.extend(dict(window_id=wid,sequence=seq,**row) for row in bins)
        dense.append(dict(window_id=wid,sequence=seq,k_views=c['eval_views'],**metrics,
                          evaluation_seconds=time.perf_counter()-t))
        if full:
            export=Path(output)/'window_clouds' if output and c['export_points'] else None
            surfaces.extend(surface_rows(pred,batch,world,c,wid,seq,export,operating_threshold))
            if c['use_render'] and c['eval_views']>1:
                loss,parts=gaussian_render(pred,batch,c)
                renders.append(dict(window_id=wid,sequence=seq,k_views=c['eval_views'],render_total=float(loss),**parts))
            if output and index in [0,5,10]:
                Path(output,'qualitative').mkdir(parents=True,exist_ok=True)
                torch.save(cpu_record(pred,batch,relative,world),Path(output,'qualitative',f'window_{index:05d}.pt'))
        del pred
    return dict(dense=dense,surfaces=surfaces,renders=renders,calibration=calibration)


def combine_results(parts):
    return {key:[row for part in parts for row in part[key]] for key in ['dense','surfaces','renders','calibration']}


def mean_metrics(rows):
    if not rows:raise ValueError('No evaluation windows')
    df=pd.DataFrame(rows)
    return {key:float(value) for key,value in df.select_dtypes('number').mean().items()}


def selected_threshold(validation_surfaces):
    df=pd.DataFrame(validation_surfaces)
    df=df[(df.alignment=='raw')&(df.filtering=='conf_plus_mask')]
    scores=df.groupby('confidence_threshold').chamfer_l1.mean()
    if not len(scores):raise ValueError('Validation sweep missing')
    return float(scores.idxmin())


def save_results(results,directory,split,c,provenance):
    p=Path(directory)/'paper_tables';p.mkdir(parents=True,exist_ok=True)
    names={'dense':f'{split}_window_metrics.csv','surfaces':f'{split}_surface_metrics.csv','renders':f'{split}_render_metrics.csv',
           'calibration':f'{split}_confidence_reliability.csv'}
    for key,name in names.items():
        frame=pd.DataFrame(results[key]);frame.to_csv(p/name,index=False)
        if key=='surfaces' and len(frame):
            group=['confidence_threshold','filtering','alignment']
            summary=frame.groupby(group).mean(numeric_only=True).reset_index()
            summary.to_csv(p/f'{split}_surface_summary.csv',index=False)
            frame.groupby(['sequence']+group).mean(numeric_only=True).reset_index().to_csv(p/f'{split}_sequence_surface_summary.csv',index=False)
    atomic_json(Path(directory)/f'{split}_provenance.json',provenance)


@torch.no_grad()
def simple_baselines(manifest,c,device,rank=0,world_size=1):
    """Depth and constant-calibration/identity-pose controls, trained only via train median."""
    rows=[]
    dataset=WindowDataset(manifest,'test',c)
    median=manifest['training_depth_statistics']['median']
    for index in range(rank,len(dataset),world_size):
        item=dataset[index]
        selected=even_indices(len(item['depths']),c['eval_views'])
        item={key:value[selected] if torch.is_tensor(value) and value.ndim>0 and value.shape[0]==c['window_views'] else value for key,value in item.items()}
        valid=item['valids'].bool();gt=item['depths'][valid]
        rows.append(dict(window_id=dataset.windows[index]['id'],sequence=dataset.windows[index]['seq'],
            method='training_median_depth',rmse=float(((gt-median)**2).mean().sqrt()),
            abs_rel=float(((gt-median).abs()/gt).mean())))
        relative=M.relative_c2w_to_first(item['poses_c2w'][None].to(device))
        rows.append(dict(window_id=dataset.windows[index]['id'],sequence=dataset.windows[index]['seq'],
            method='identity_pose',ate=float(relative[...,:3,3].square().sum(-1).mean().sqrt())))
        constant=torch.tensor(manifest['training_median_K'])
        rows.append(dict(window_id=dataset.windows[index]['id'],sequence=dataset.windows[index]['seq'],
            method='training_median_intrinsics',intrinsics_mae=float((M.normalize_K_params(item['intrinsics'])-M.normalize_K_params(constant)).abs().mean())))
    return rows


@torch.no_grad()
def ten_view_trajectories(network,manifest,c,device,rank=0,world_size=1):
    network.set_stage('eval');network.eval()
    dataset=WindowDataset(manifest,'test',c)
    count=len(dataset) if not c['max_eval_windows'] else min(len(dataset),c['max_eval_windows'])
    rows=[]
    for index in range(rank,count,world_size):
        item=dataset[index]
        batch=to_device({key:value[None] if torch.is_tensor(value) else value for key,value in item.items()},device)
        prediction=network(batch['images'],batch['images_raw'])
        p=prediction['poses_pred_c2w'][0].float()
        g=M.relative_c2w_to_first(batch['poses_c2w'])[0]
        pc,gc=p[:,:3,3].cpu().numpy(),g[:,:3,3].cpu().numpy()
        x,y=pc-pc.mean(0),gc-gc.mean(0)
        U,_,V=np.linalg.svd(x.T @ y)
        D=np.eye(3);D[-1,-1]=np.sign(np.linalg.det(U @ V))
        rotation=U @ D @ V
        aligned=x @ rotation+gc.mean(0)
        error=M.invert_se3(M.invert_se3(g[:-1]) @ g[1:]) @ (M.invert_se3(p[:-1]) @ p[1:])
        rows.append(dict(window_id=dataset.windows[index]['id'],sequence=dataset.windows[index]['seq'],
            k_views=len(p),alignment='rigid_per_window_no_scale',
            ate_rmse_m=float(np.sqrt(np.square(aligned-gc).sum(1).mean())),
            rpe_translation=float(error[:,:3,3].square().sum(-1).mean().sqrt()),
            rpe_rotation_deg=float(torch.rad2deg(rotation_angle(error[:,:3,:3],torch.eye(3,device=device))).square().mean().sqrt())))
    return rows
