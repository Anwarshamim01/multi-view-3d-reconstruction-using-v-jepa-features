"""Run with torchrun --standalone --nproc_per_node=2 -m kaggle_v44.train --config ..."""
import argparse
from contextlib import nullcontext
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler

from .config import read_config,validate
from .data import (prepare_manifest,WindowDataset,load_rgb,file_hash,json_hash,atomic_json)
from . import model as M
from .losses import TrainingStep
from .evaluation import (evaluate,combine_results,mean_metrics,selected_threshold,save_results,
                         simple_baselines,to_device,ten_view_trajectories)

STAGES=['stage1_frozen','stage2_finetune','stage3_align_refine']


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def gather_objects(value,world):
    if world==1:return [value]
    output=[None]*world
    dist.all_gather_object(output,value)
    return output


def barrier(world):
    if world>1:dist.barrier()


def source_fingerprint():
    root=Path(__file__).parent
    return json_hash({p.name:file_hash(p) for p in sorted(root.glob('*.py'))} | {'model_defaults.json':file_hash(root/'model_defaults.json')})


def scientific_config(c):
    ignored={'mode','resume_checkpoint','evaluate_checkpoint','output_root','asset_root','data_root',
             'allow_download','max_hours','companion_run','workers_per_gpu','require_two_gpus','backend'}
    return {key:value for key,value in c.items() if key not in ignored}


def atomic_checkpoint(path,payload):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def load_checkpoint(network,path,identity,evaluation=False):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!='vjepa_geo_v44':
        raise ValueError('Legacy checkpoints cannot resume corrected experiments; start a new run')
    for key in ['source_hash','data_hash','config_hash']:
        if payload['identity'][key]!=identity[key]:
            raise ValueError(f'Checkpoint {key} mismatch; refusing mixed experiment artifacts')
    network.load_state_dict(payload['model'],strict=True)
    if evaluation:
        network.set_stage('eval');network.eval()
    return payload


@torch.no_grad()
def teacher_cache(network,manifest,c,device,rank,world,identity):
    if c['glvp_weight']<=0:return {}
    key=json_hash(dict(data_hash=identity['data_hash'],encoder_revision=c['upstream_revision'],
                       encoder_sha256=c.get('resolved_encoder_sha256','synthetic'),
                       image_size=c['image_size'],tiny=c['tiny_backbone'],source='frozen_pretrained_encoder'))
    root=Path(c['output_root'])/c['dataset']/'teacher_cache';root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.pt')
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['key']!=key:raise ValueError('Teacher cache mismatch')
        return payload['embeddings']
    # Only train and validation targets; test images never create training targets.
    ids=sorted(set(fid for split in ['train','val'] for window in manifest['splits'][split] for fid in window['frames']))
    if c.get('backbone_initialization')=='random':
        old=M.RUN_CONFIG
        M.RUN_CONFIG=dict(c,backbone_initialization='pretrained')
        encoder=M.load_vjepa_backbone().to(device).eval()
        M.RUN_CONFIG=old
    else:encoder=network.backbone
    encoder.eval()
    local={}
    for fid in ids[rank::world]:
        image,_=load_rgb(manifest['frames'][fid]['rgb'],c)
        with torch.amp.autocast(device.type,enabled=c['amp'] and device.type=='cuda'):
            output=encoder(M.make_vjepa_input(image[None].to(device)))
        tensor=M.select_feature_tensor(output)
        feature=M.tokens_to_feature_map(tensor,M.CFG.FEAT_DIM,c['image_size'],16)
        local[fid]=torch.nn.functional.normalize(feature.float().mean((-1,-2))[0],dim=0).half().cpu()
    embeddings={k:v for shard in gather_objects(local,world) for k,v in shard.items()}
    if rank==0:atomic_checkpoint(path,dict(key=key,embeddings=embeddings))
    barrier(world)
    if encoder is not network.backbone:del encoder
    return embeddings


def make_optimizer(network,c):
    backbone=[p for p in network.backbone.parameters() if p.requires_grad]
    head=[p for name,p in network.named_parameters() if not name.startswith('backbone.') and p.requires_grad]
    groups=[dict(params=head,lr=c['lr_heads'])]
    if backbone:groups.append(dict(params=backbone,lr=c['lr_backbone']))
    return torch.optim.AdamW(groups,weight_decay=c['weight_decay'])


def capture_rng(device):
    return dict(python=random.getstate(),numpy=torch.tensor(np.random.get_state()[1].astype(np.int64)),
                numpy_pos=int(np.random.get_state()[2]),numpy_gauss=int(np.random.get_state()[3]),
                numpy_cached=float(np.random.get_state()[4]),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state,device):
    random.setstate(state['python'])
    np.random.set_state(('MT19937',state['numpy'].numpy().astype(np.uint32),state['numpy_pos'],state['numpy_gauss'],state['numpy_cached']))
    torch.set_rng_state(state['torch'])
    if device.type=='cuda' and state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],device)


def train(network,manifest,c,device,rank,world,run,identity):
    accumulation=c['global_effective_batch']//world
    # Build targets BEFORE loading a trained checkpoint so the teacher stays pretrained.
    teacher=teacher_cache(network,manifest,c,device,rank,world,identity)
    seed_everything(c['seed']+rank)  # Cache creation cannot perturb the training random stream.
    dataset=WindowDataset(manifest,'train',c,teacher)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=c['seed'],drop_last=False)
    loader=DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=c['workers_per_gpu'],
                      pin_memory=device.type=='cuda',persistent_workers=False)
    start=0;best=float('inf');history=[];resume=None
    if c['resume_checkpoint']:
        resume=load_checkpoint(network,c['resume_checkpoint'],identity)
        if resume['world_size']!=world:raise ValueError('Exact continuation requires the original GPU world size')
        start=resume['next_global_epoch'];best=resume['best_score'];history=resume['history']
        restore_rng(resume['rng_by_rank'][rank],device)
        # best.pt may live in an attached previous output; copy its validated sibling.
        previous=Path(c['resume_checkpoint']).parent/'best.pt'
        if not previous.exists() and not (run/'best.pt').exists():
            raise FileNotFoundError('Resume needs both last.pt and its sibling best.pt from the previous session')
        if rank==0 and previous.exists() and previous.resolve()!=(run/'best.pt').resolve():
            shutil.copy2(previous,run/'best.pt')
        barrier(world)
    boundaries=np.cumsum([0]+c['epochs']).tolist()
    wall=time.monotonic()
    for stage in range(3):
        if boundaries[stage+1]<=start:continue
        network.set_stage(STAGES[stage])
        step_module=TrainingStep(network,c,stage).to(device)
        wrapped=DDP(step_module,device_ids=[device.index] if device.type=='cuda' else None,
                    find_unused_parameters=True,broadcast_buffers=False) if world>1 else step_module
        optimizer=make_optimizer(network,c)
        scaler=torch.amp.GradScaler('cuda',enabled=c['amp'] and device.type=='cuda')
        if resume and resume['stage_index']==stage:
            optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
        for epoch in range(max(start,boundaries[stage]),boundaries[stage+1]):
            network.set_stage(STAGES[stage]);wrapped.train()
            if stage!=1 or not c['trainable_backbone_blocks']:network.backbone.eval()
            sampler.set_epoch(epoch)
            dataset.training_stage=stage;dataset.training_epoch=epoch
            optimizer.zero_grad(set_to_none=True)
            rows=[];epoch_start=time.monotonic()
            steps=len(loader) if not c['max_train_steps'] else min(len(loader),c['max_train_steps'])
            for index,batch in enumerate(loader):
                if index>=steps:break
                batch=to_device(batch,device)
                offset=index%accumulation
                group_size=min(accumulation,steps-(index-offset))
                synchronize=(offset==group_size-1)
                context=wrapped.no_sync() if world>1 and not synchronize else nullcontext()
                with context:
                    with torch.amp.autocast(device.type,enabled=c['amp'] and device.type=='cuda',dtype=torch.float16 if c['amp_dtype']=='float16' else torch.bfloat16):
                        loss,stats=wrapped(batch)
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, rank {rank}')
                    scaler.scale(loss/group_size).backward()
                if synchronize:
                    scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(network.parameters(),c['grad_clip'])
                    if not torch.isfinite(norm) and not scaler.is_enabled():
                        raise FloatingPointError('Nonfinite gradient norm without AMP scaling')
                    # FP16 overflow is expected during scale discovery: GradScaler skips the
                    # update and lowers its scale. Do not abort before it can recover.
                    old_scale=scaler.get_scale()
                    scaler.step(optimizer);scaler.update()
                    stats['optimizer_update_skipped']=float(scaler.get_scale()<old_scale)
                    optimizer.zero_grad(set_to_none=True)
                rows.append(stats)
                if rank==0 and index%25==0:
                    print(f"epoch {epoch+1}/{sum(c['epochs'])} {STAGES[stage]} step {index+1}/{steps} loss {stats['loss']:.5f} K={batch['images'].shape[1]} effective_batch={c['global_effective_batch']}",flush=True)
            train_stats=mean_metrics([r for shard in gather_objects(rows,world) for r in shard])
            validation=combine_results(gather_objects(evaluate(network,manifest,'val',c,device,rank,world),world))
            metrics=mean_metrics(validation['dense'])
            score=metrics['abs_rel']  # Same fixed-support criterion for every training variant.
            history_row=dict(global_epoch=epoch+1,stage=STAGES[stage],stage_epoch=epoch-boundaries[stage]+1,
                seconds=time.monotonic()-epoch_start,selection_score=score,
                **{f'train_{k}':v for k,v in train_stats.items()},**{f'val_{k}':v for k,v in metrics.items()})
            history.append(history_row)
            is_best=score<best;best=min(best,score)
            rng_states=gather_objects(capture_rng(device),world)
            if rank==0:
                payload=dict(schema='vjepa_geo_v44',identity=identity,config=c,
                    model={k:v.detach().cpu() for k,v in network.state_dict().items()},
                    optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),next_global_epoch=epoch+1,
                    stage_index=stage,best_score=best,history=history,world_size=world,rng_by_rank=rng_states)
                atomic_checkpoint(run/'last.pt',payload)
                if is_best:atomic_checkpoint(run/'best.pt',payload)
                pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
                print('validation',json.dumps(dict(epoch=epoch+1,score=score,best=best)),flush=True)
            barrier(world)
            elapsed=torch.tensor(time.monotonic()-wall,device=device)
            if world>1:dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            if float(elapsed)>c['max_hours']*3600:
                if rank==0:
                    atomic_json(run/'run_state.json',dict(status='training_incomplete',completed_epochs=epoch+1,
                        required_epochs=sum(c['epochs']),resume_checkpoint=str(run/'last.pt')))
                return False
        del wrapped,step_module,optimizer
        if device.type=='cuda':torch.cuda.empty_cache()
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();c=read_config(args.config)
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    validate(c,world)
    if c['require_two_gpus'] and (world!=2 or torch.cuda.device_count()<2):
        raise RuntimeError('Select Kaggle GPU T4 ×2 and launch with torchrun --nproc_per_node=2')
    device=torch.device('cuda',local) if torch.cuda.is_available() and c['backend']=='nccl' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(local)
    if world>1:dist.init_process_group(c['backend'],timeout=datetime.timedelta(minutes=90))
    seed_everything(c['seed'])  # Identical initialization on every rank, including heads across variants.
    run=Path(c['output_root'])/c['dataset']/c['study']/c['run_name'];run.mkdir(parents=True,exist_ok=True)
    manifest_path=run/'data_manifest.json'
    if rank==0:
        if c['mode']=='train' and not c['resume_checkpoint'] and (run/'last.pt').exists():
            raise FileExistsError('Run already contains a checkpoint; use mode=resume with an explicit v44 last.pt or choose a new RUN_NAME')
        if c['mode'] not in ['evaluate','resume'] or not manifest_path.exists():
            prepare_manifest(c,manifest_path)
        if not c['tiny_backbone']:
            from .assets import prepare_assets
            prepare_assets(c)
    barrier(world)
    if not c['tiny_backbone']:
        c['resolved_encoder_sha256']=file_hash(Path(c['asset_root'])/c['encoder_checkpoint'])
    manifest=json.loads(manifest_path.read_text())
    M.configure(c,manifest['training_depth_statistics'])
    identity=dict(source_hash=source_fingerprint(),data_hash=manifest['fingerprint'],config_hash=json_hash(scientific_config(c)))
    if rank==0:
        atomic_json(run/'resolved_config.json',c)
        atomic_json(run/'identity.json',identity)
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            world_size=world,device=str(device),gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            global_effective_batch=c['global_effective_batch'],gradient_accumulation=c['global_effective_batch']//world,
            source_hash=identity['source_hash'],synthetic=c['tiny_backbone'])
        atomic_json(run/'runtime.json',runtime)
    network=M.VJepaPoseFreeReconSuperman().to(device)
    if c['mode'] in ['train','resume']:
        if c['mode']=='resume' and not c['resume_checkpoint']:raise ValueError('Set an explicit resume_checkpoint')
        if not train(network,manifest,c,device,rank,world,run,identity):
            barrier(world)
            if world>1:dist.destroy_process_group()
            return
    checkpoint=Path(c['evaluate_checkpoint']) if c['evaluate_checkpoint'] else run/'best.pt'
    selected_payload=load_checkpoint(network,checkpoint,identity,evaluation=True)
    if rank==0 and not (run/'training_history.csv').exists():
        pd.DataFrame(selected_payload['history']).to_csv(run/'training_history.csv',index=False)
    provenance=dict(**identity,checkpoint_sha256=file_hash(checkpoint),checkpoint=str(checkpoint),
                    epipolar_enabled=c['use_epipolar'],intrinsics='predicted',k_views=c['eval_views'],
                    reference_support='fixed_independent_of_prediction',aggregation='per_window_then_mean',
                    synthetic=c['tiny_backbone'])
    validation=combine_results(gather_objects(evaluate(network,manifest,'val',c,device,rank,world,full=True),world))
    threshold=selected_threshold(validation['surfaces'])
    if rank==0:
        atomic_json(run/'validation_operating_point.json',dict(confidence_threshold=threshold,
            chosen_using='validation_mean_raw_masked_chamfer',**identity))
        save_results(validation,run,'val',c,provenance)
    test=combine_results(gather_objects(evaluate(network,manifest,'test',c,device,rank,world,
        full=True,output=run,operating_threshold=threshold),world))
    baselines=[r for shard in gather_objects(simple_baselines(manifest,c,device,rank,world),world) for r in shard]
    trajectories=[]
    if c['dataset']=='tum' and c['ten_view_trajectory_diagnostic']:
        trajectories=[r for shard in gather_objects(ten_view_trajectories(network,manifest,c,device,rank,world),world) for r in shard]
    if rank==0:
        save_results(test,run,'test',c,dict(provenance,confidence_threshold=threshold))
        pd.DataFrame(baselines).to_csv(run/'paper_tables/simple_baselines.csv',index=False)
        if trajectories:
            frame=pd.DataFrame(trajectories)
            frame.to_csv(run/'paper_tables/table_tum_window_trajectory_metrics.csv',index=False)
            frame.select_dtypes('number').agg(['mean','std','median','min','max']).to_csv(run/'paper_tables/table_tum_trajectory_distribution.csv')
    barrier(world)
    if rank==0:
        from .reporting import export_report
        export_report(run,c,manifest,threshold,provenance)
        atomic_json(run/'run_state.json',dict(status='complete',required_epochs=sum(c['epochs']),
            synthetic=c['tiny_backbone'],evaluated_checkpoint=provenance['checkpoint_sha256']))
        import zipfile
        with zipfile.ZipFile(run/'paper_artifacts.zip','a',zipfile.ZIP_DEFLATED) as z:
            z.write(run/'run_state.json','run_state.json')
    barrier(world)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    main()
