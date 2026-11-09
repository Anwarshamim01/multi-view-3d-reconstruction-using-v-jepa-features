"""Generate tiny artificial data for software checks, never for paper results."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .config import default_config
from .data import atomic_json,json_hash


def create_fixture(directory,dataset='tum'):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    c=default_config(dataset)
    c.update(tiny_backbone=True,image_size=32,edge_size=32,window_views=3,eval_views=3,
             epochs=[1,1,1],stage_k=[[3],[3],[3]],stage_k_probabilities=[[1.],[1.],[1.]],
             view_dropout=0.,workers_per_gpu=0,require_two_gpus=False,backend='gloo',
             use_render=False,amp=False,depth_bins=4,depth_chunk=2,
             checkpoint_activations=False,max_metric_points=1000,max_fit_points=1000,
             max_hours=1.,output_root=str(root/'outputs'),run_name='synthetic_software_check',
             cloud_stride=4,ten_view_trajectory_diagnostic=False)
    manifest=dict(schema='v44_exact_frames',dataset=dataset,frames={},splits={},
        training_depth_statistics=dict(near=1.,far=2.,median=1.5))
    for split,number in [('train',4),('val',2),('test',2)]:
        seq='artificial_'+split;manifest['splits'][split]=[]
        for w in range(number):
            frame_ids=[]
            for k in range(3):
                key=f'{seq}:{w*3+k}';frame_ids.append(key)
                h,width=(48,64) if dataset=='tum' else (32,32)
                color=np.zeros((h,width,3),np.uint8);color[:,:,0]=np.arange(width)[None]*3
                color[:,:,1]=np.arange(h)[:,None]*4;color[:,:,2]=80+k*10
                rgb=root/f'{seq}_{w}_{k}.png';Image.fromarray(color).save(rgb)
                depth=root/f'{seq}_{w}_{k}_depth.png'
                encoded=np.full((h,width),7500,np.uint16) if dataset=='tum' else np.full((h,width),1.5,np.float16).view(np.uint16)
                Image.fromarray(encoded).save(depth)
                mask=root/f'{seq}_{w}_{k}_mask.png';Image.fromarray(np.full((h,width),255,np.uint8)).save(mask)
                pose=np.eye(4);pose[0,3]=.02*k
                K=[[26.25,0,15.75],[0,26.25,15.75],[0,0,1]]
                manifest['frames'][key]=dict(seq=seq,frame_number=w*3+k,timestamp=float(w*3+k),
                    rgb=str(rgb),depth=str(depth),mask=str(mask),depth_mask=None,depth_scale_adjustment=1.,
                    pose=pose.tolist(),K=K)
            manifest['splits'][split].append(dict(id=f'{split}:{w}',seq=seq,frames=frame_ids))
    manifest['training_median_K']=K
    manifest['fingerprint']=json_hash(manifest)
    atomic_json(root/'fixture_manifest.json',manifest)
    c['fixture_manifest']=str(root/'fixture_manifest.json')
    c['splits']={key:['artificial_'+key] for key in ['train','val','test']}
    atomic_json(root/'config.json',c)
    return root/'config.json'


if __name__=='__main__':
    import sys
    print(create_fixture(sys.argv[1]))
