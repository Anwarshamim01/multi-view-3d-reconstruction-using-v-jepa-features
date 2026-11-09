import json
from pathlib import Path
import pytest
import torch
from kaggle_v46 import model as M
from kaggle_v46.config import default_config,apply_study
from kaggle_v46.losses import relative_depth_terms,confidence_loss,glvp_terms
from kaggle_v46.evaluation import confidence_bins
from kaggle_v46.train import standardize_teacher
from kaggle_v46.smoke_fixture import create_fixture


def test_relative_depth_loss_is_invariant_to_scene_units():
    p=torch.tensor([1.1,2.4,8.]);g=torch.tensor([1.,2.,10.]);valid=torch.ones(3,dtype=torch.bool)
    a=relative_depth_terms(p,g,valid);b=relative_depth_terms(p*17,g*17,valid)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=2e-6,rtol=2e-6)


def test_confidence_uses_relative_success_and_limits_invalid_background():
    c=default_config('co3d')
    valid=torch.tensor([True,True,False]);p=torch.tensor([1.05,2.5,1.]);g=torch.tensor([1.,2.,1.])
    probability=torch.tensor([.9,.1,.1],requires_grad=True)
    loss=confidence_loss(probability,p,g,valid,c)
    scaled=confidence_loss(probability,p*100,g*100,valid,c)
    torch.testing.assert_close(loss,scaled)
    more=confidence_loss(torch.cat([probability[:2],probability[2:].repeat(100)]),
        torch.cat([p[:2],p[2:].repeat(100)]),torch.cat([g[:2],g[2:].repeat(100)]),
        torch.tensor([True,True]+[False]*100),c)
    torch.testing.assert_close(loss,more)
    loss.backward();assert probability.grad[0]<0 and probability.grad[1]>0


def test_calibration_does_not_receive_geometry_gradients(tmp_path):
    c=json.loads(create_fixture(tmp_path).read_text());M.configure(c,dict(near=1.,far=2.))
    torch.set_num_threads(1)
    net=M.VJepaPoseFreeReconSuperman();net.set_stage('stage1_frozen')
    torch.nn.init.normal_(net.camera_head.pose_net[-1].weight,std=.01)
    pred=net(torch.randn(1,3,3,32,32),torch.rand(1,3,3,32,32))
    assert not pred['intrinsics_pred'].requires_grad and pred['intrinsics_norm_pred'].requires_grad
    (pred['depth'].mean()+pred['pointmap_world'].mean()+pred['poses_pred_c2w'].mean()).backward(retain_graph=True)
    assert all(p.grad is None for p in net.camera_head.k_net.parameters())
    pred['intrinsics_norm_pred'].sum().backward()
    assert net.camera_head.k_net[-1].weight.grad.abs().sum()>0


def test_teacher_statistics_exclude_validation_frames():
    c=default_config('co3d')
    manifest=dict(splits=dict(train=[dict(frames=['a','b'])],val=[dict(frames=['v'])]))
    embeddings=dict(a=torch.tensor([1.,0.,.2]),b=torch.tensor([0.,1.,.3]),v=torch.tensor([.3,.4,.5]))
    a,stats_a=standardize_teacher(embeddings,manifest,c)
    b,stats_b=standardize_teacher(dict(embeddings,v=torch.ones(3)*1e6),manifest,c)
    for fid in ['a','b']:torch.testing.assert_close(a[fid],b[fid])
    torch.testing.assert_close(stats_a['mean'],stats_b['mean'])
    assert stats_a['training_frames']==2 and stats_a['fitted_using']=='unique_training_frames_only'


def test_glvp_retrieval_distinguishes_collapsed_predictions():
    c=default_config('co3d');target=torch.eye(4)[None]
    perfect=target[:,None].expand(-1,4,-1,-1).clone().requires_grad_()
    collapsed=torch.ones_like(perfect)/2
    loss,stats=glvp_terms(perfect,target,c);bad,bad_stats=glvp_terms(collapsed,target,c)
    assert stats['glvp_retrieval_accuracy']==1. and bad_stats['glvp_retrieval_accuracy']<=.5 and loss<bad
    loss.backward();assert torch.isfinite(perfect.grad).all()


def test_confidence_reliability_counts_and_errors():
    c=default_config('co3d')
    pred=dict(confidence=torch.tensor([.8,.2]).reshape(1,1,1,1,2),depth=torch.tensor([1.,2.]).reshape(1,1,1,1,2))
    batch=dict(depths=torch.ones(1,1,1,1,2),valids=torch.ones(1,1,1,1,2,dtype=torch.bool))
    bins,stats=confidence_bins(pred,batch,c)
    assert sum(row['count'] for row in bins)==2
    assert stats['confidence_brier']==pytest.approx(.04) and stats['confidence_ece']==pytest.approx(.2)


def test_paired_glvp_control_preserves_full_schedule():
    full=apply_study(dict(default_config('co3d'),study='paired',variant='full'))
    control=apply_study(dict(default_config('co3d'),study='paired',variant='no_glvp'))
    differences={k for k in full if full[k]!=control[k]}
    assert differences=={'variant','glvp_weight'}
    assert control['epochs']==[5,10,5] and control['eval_views']==5
