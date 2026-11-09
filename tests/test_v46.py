import copy
import json
from pathlib import Path
import types
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from kaggle_v46.config import default_config,apply_study,validate
from kaggle_v46.data import decode_co3d_png,WindowDataset,even_indices,slice_views,train_indices
from kaggle_v46.evaluation import cloud_metrics,surface_rows,paired_sim3,transform
from kaggle_v46.losses import gaussian_render,normals_valid,visible_matches,supervised_loss
from kaggle_v46 import model as M
from kaggle_v46.smoke_fixture import create_fixture


@pytest.fixture
def fixture(tmp_path):
    path=create_fixture(tmp_path)
    c=json.loads(path.read_text());manifest=json.loads(Path(c['fixture_manifest']).read_text())
    M.configure(c,manifest['training_depth_statistics'])
    torch.set_num_threads(2)
    return c,manifest


def test_official_co3d_encoding(tmp_path):
    expected=np.array([[0.,.1,1.,2.,8.,16.,100.]],np.float16)
    p=tmp_path/'depth.png';Image.fromarray(expected.view(np.uint16)).save(p)
    np.testing.assert_allclose(decode_co3d_png(p,1.7),expected.astype(np.float32)*1.7)


def test_fixed_reference_cannot_hide_missing_geometry():
    c=default_config('tum')
    reference=np.array([[0.,0.,1.],[10.,0.,1.]])
    full=cloud_metrics(reference[:1],reference,c)
    assert full['recall_0.20']==.5
    empty=cloud_metrics(np.empty((0,3)),reference,c)
    assert empty['failed']==1 and empty['fscore_0.20']==0
    assert np.isfinite(empty['chamfer_l1']) and empty['chamfer_l1']>full['chamfer_l1']


def test_paired_alignment_recovers_known_transform():
    source=np.random.default_rng(123).normal(size=(100,3))
    target=source*2+np.array([1.,2.,3.])
    fit=paired_sim3(source,target)
    np.testing.assert_allclose(transform(source,fit),target,atol=1e-6)


def test_training_schedule_and_global_batch():
    c=default_config('co3d');assert c['epochs']==[5,10,5]
    validate(c,world_size=2)
    assert c['global_effective_batch']//2==2
    c['global_effective_batch']=3
    with pytest.raises(ValueError):validate(c,world_size=2)
    a=apply_study(dict(default_config('co3d'),study='glvp',variant='A1_glvp_0025'))
    assert a['epochs']==[3,6,3] and a['eval_views']==3


def test_sampler_preserves_reference_and_metadata(fixture):
    c,manifest=fixture
    c=copy.deepcopy(c);c['stage_k']=[[3,5,8],[3,5],[3,5,8]]
    c['stage_k_probabilities']=[[.45,.35,.2],[.65,.35],[.45,.35,.2]]
    for seed in range(20):
        indices=train_indices(10,0,c,np.random.default_rng(seed))
        assert indices[0]==0 and indices[-1]==9 and indices==sorted(set(indices))
    item=WindowDataset(manifest,'train',c)[0]
    batch={key:value[None] if torch.is_tensor(value) else value for key,value in item.items()}
    selected=slice_views(batch,[0,2])
    assert selected['images'].shape[1]==selected['poses_c2w'].shape[1]==selected['timestamps'].shape[1]==2
    assert even_indices(10,5)==[0,2,4,7,9]


def test_eval_enables_geometry_and_preserves_rigidity(fixture):
    c,_=fixture
    network=M.VJepaPoseFreeReconSuperman()
    network.set_stage('stage1_frozen')
    state=copy.deepcopy(network.state_dict())
    restored=M.VJepaPoseFreeReconSuperman();restored.load_state_dict(state,strict=True)
    restored.set_stage('eval');restored.eval()
    with torch.no_grad(),torch.amp.autocast('cpu',dtype=torch.bfloat16):
        pred=restored(torch.randn(1,3,3,32,32),torch.rand(1,3,3,32,32))
    assert restored.current_stage=='eval' and pred['depth_prior_lr'] is not None
    assert pred['poses_pred_c2w'].dtype==torch.float32
    rotation=pred['poses_pred_c2w'][...,:3,:3]
    torch.testing.assert_close(rotation.transpose(-1,-2) @ rotation,torch.eye(3).expand_as(rotation),atol=1e-6,rtol=1e-6)


def test_occlusion_rejects_false_positive_correspondences(fixture):
    c,_=fixture
    B,K,H,W=1,2,8,8
    depth=torch.ones(B,K,1,H,W);depth[:,1]=2
    intrinsic=torch.eye(3).repeat(B,K,1,1)
    pose=torch.eye(4).repeat(B,K,1,1)
    points=M.camera_points_from_depth(depth.reshape(B*K,1,H,W),intrinsic.reshape(B*K,3,3)).reshape(B,K,3,H,W)
    batch=dict(images=torch.zeros(B,K,3,H,W),depths=depth,valids=torch.ones_like(depth),intrinsics=intrinsic)
    _,visible=visible_matches(batch,points,pose,0,1,torch.tensor([3]),torch.tensor([3]),c)
    assert not visible.any()


def test_normals_reject_invalid_neighborhood():
    valid=torch.ones(1,1,1,8,8,dtype=torch.bool);valid[...,3,3]=False
    safe=normals_valid(valid)
    assert not safe[...,2:5,2:5].any()
    assert not safe[...,0,:].any() and safe[...,6,6].all()


def test_empty_render_is_penalized_and_only_predictions_supply_cameras(fixture,monkeypatch):
    c,manifest=fixture;c=copy.deepcopy(c);c['render_size']=8
    item=WindowDataset(manifest,'val',c)[0]
    batch={key:value[None] if torch.is_tensor(value) else value for key,value in item.items()}
    network=M.VJepaPoseFreeReconSuperman().eval();network.set_stage('eval')
    pred=network(batch['images'],batch['images_raw'])
    seen={}
    def rasterization(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1,8,8,4),torch.zeros(1,8,8,1),{}
    package=types.ModuleType('gsplat');rendering=types.ModuleType('gsplat.rendering')
    rendering.rasterization=rasterization;package.rendering=rendering
    monkeypatch.setitem(sys.modules,'gsplat',package);monkeypatch.setitem(sys.modules,'gsplat.rendering',rendering)
    batch['intrinsics']=batch['intrinsics']*20  # Evaluation targets must not alter rendered cameras.
    loss,metrics=gaussian_render(pred,batch,c)
    expected=pred['intrinsics_pred'][0,1].clone();expected[:2,:]*=8/32
    torch.testing.assert_close(seen['Ks'][0],expected)
    assert loss>0 and metrics['render_coverage']==0 and metrics['render_failed']==1


def test_surface_reference_is_constant_across_prediction_thresholds(fixture):
    c,manifest=fixture
    item=WindowDataset(manifest,'val',c)[0]
    batch={key:value[None] if torch.is_tensor(value) else value for key,value in item.items()}
    network=M.VJepaPoseFreeReconSuperman().eval();network.set_stage('eval')
    with torch.no_grad():
        pred=network(batch['images'],batch['images_raw'])
        _,world,_=M.gt_local_world_points(batch)
        pred['confidence'].fill_(.02)
        rows=surface_rows(pred,batch,world,c,'window','physical_sequence')
    assert len({r['n_gt'] for r in rows})==1
    assert {r['window_id'] for r in rows}=={'window'}
    empty=[r for r in rows if r['confidence_threshold']==.05]
    assert all(r['n_pred']==0 and r['failed']==1 for r in empty)


def test_real_training_graph_has_finite_gradients(fixture):
    c,manifest=fixture
    item=WindowDataset(manifest,'train',c)[0]
    batch={key:value[None] if torch.is_tensor(value) else value for key,value in item.items()}
    c=copy.deepcopy(c);c['glvp_weight']=0
    M.configure(c,manifest['training_depth_statistics'])
    network=M.VJepaPoseFreeReconSuperman();network.set_stage('stage2_finetune');network.train()
    pred=network(batch['images'],batch['images_raw'])
    loss,stats=supervised_loss(pred,batch,c,1);loss.backward()
    assert torch.isfinite(loss) and stats['loss']>0
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    assert any(p.grad is not None for p in network.backbone.parameters())
