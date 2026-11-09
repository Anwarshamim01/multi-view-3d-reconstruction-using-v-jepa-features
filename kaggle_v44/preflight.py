"""Fail early on unavailable GPUs, incompatible compilers, or broken CUDA rendering."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def find_cuda_toolkit():
    import torch
    if not torch.version.cuda:raise RuntimeError('A CUDA-enabled PyTorch build is required')
    version=torch.version.cuda
    roots=[os.environ.get('CUDA_HOME',''),f'/usr/local/cuda-{version}','/usr/local/cuda']
    roots += [str(p) for p in sorted(Path('/usr/local').glob('cuda-*'),reverse=True)]
    executable=shutil.which('nvcc')
    if executable:roots.append(str(Path(executable).resolve().parent.parent))
    for root in roots:
        if not root or not (Path(root)/'bin/nvcc').is_file():continue
        process=subprocess.run([str(Path(root)/'bin/nvcc'),'--version'],capture_output=True,text=True)
        match=re.search(r'release (\d+)\.(\d+)',process.stdout)
        if process.returncode==0 and match and match[1]==version.split('.')[0]:
            return str(Path(root).resolve())
    raise RuntimeError(f'PyTorch CUDA {version} needs nvcc with the same major version. '
                       'Choose a Kaggle CUDA image with a matching toolkit; rendering will not be silently disabled.')


def main():
    import torch
    if torch.cuda.device_count()!=2:raise RuntimeError('Select GPU T4 ×2 in Kaggle Settings')
    toolkit=find_cuda_toolkit()
    os.environ['CUDA_HOME']=toolkit
    os.environ['PATH']=toolkit+'/bin:'+os.environ.get('PATH','')
    from gsplat.rendering import rasterization
    devices=[]
    for index in range(2):
        device=torch.device('cuda',index)
        means=torch.tensor([[-.1,0,2.],[.1,0,2.],[0,.1,2.],[0,-.1,2.]],device=device,requires_grad=True)
        quats=torch.tensor([[1.,0,0,0]],device=device).repeat(4,1)
        cameras=torch.eye(4,device=device)[None].requires_grad_(True)
        rendered,alpha,_=rasterization(means=means,quats=quats,
            scales=torch.full((4,3),.08,device=device),opacities=torch.full((4,),.8,device=device),
            colors=torch.ones(4,3,device=device),viewmats=cameras,
            Ks=torch.tensor([[[16.,0,8],[0,16.,8],[0,0,1]]],device=device),
            width=16,height=16,render_mode='RGB+ED',distributed=False)
        (rendered.mean()+alpha.mean()).backward()
        if not torch.isfinite(means.grad).all() or cameras.grad is None or not torch.isfinite(cameras.grad).all():
            raise RuntimeError('CUDA rasterizer gradient check failed')
        devices.append(dict(index=index,name=torch.cuda.get_device_name(index),
                            memory_gib=torch.cuda.get_device_properties(index).total_memory/2**30))
        del means,quats,cameras,rendered,alpha
        torch.cuda.empty_cache()
    result=dict(torch=torch.__version__,cuda=torch.version.cuda,toolkit=toolkit,devices=devices,
                actual_cuda_render_backward='passed')
    Path('/kaggle/working/v44_preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
