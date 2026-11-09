import json
from pathlib import Path
import pytest
import torch
from kaggle_v46 import model as M
from kaggle_v46.spatial_camera import SpatialCameraHead,correlation
from kaggle_v46.losses import camera_match_loss
from kaggle_v46.train import validation_score
from kaggle_v46.data import WindowDataset
from kaggle_v46.smoke_fixture import create_fixture


@pytest.fixture
def setup(tmp_path):
    c=json.loads(create_fixture(tmp_path).read_text())
    manifest=json.loads(Path(c['fixture_manifest']).read_text())
    M.configure(c,manifest['training_depth_statistics']);torch.set_num_threads(1)
    return c,manifest


def test_known_spatial_permutation_recovers_correspondences(setup):
    features=torch.eye(16)[None]
    permutation=torch.tensor([4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3])
    logits=correlation(features,features[:,permutation])
    assert torch.equal(logits.argmax(-1)[0],permutation.argsort())


def test_camera_preserves_anchor_and_nonreference_permutation(setup):
    torch.manual_seed(7)
    head=SpatialCameraHead(32,32,2).eval()
    torch.nn.init.normal_(head.pose_net[-1].weight,std=.01)
    features=torch.randn(1,4,32,2,2)
    original=head(features)
    order=[0,3,1,2];permuted=head(features[:,order])
    torch.testing.assert_close(original[1][:,order],permuted[1],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(original[1][:,0],torch.eye(4)[None])
    assert torch.linalg.det(original[1][...,:3,:3]).sub(1).abs().max()<1e-6
    # Mean features are identical, but an inconsistent patch shuffle changes
    # spatial motion evidence and should change the camera estimate.
    shuffled=features.clone();shuffled[:,1]=features[:,1].flatten(2).flip(-1).reshape_as(features[:,1])
    assert not torch.allclose(head(shuffled)[0][:,1],original[0][:,1],atol=1e-5)
    one=head(features[:,:1]);assert one[-1] is None
    torch.testing.assert_close(one[1][0,0],torch.eye(4))


def test_matching_supervision_rejects_invisible_geometry(setup):
    c,manifest=setup
    item=WindowDataset(manifest,'train',c)[0]
    batch={k:v[None] if torch.is_tensor(v) else v for k,v in item.items()}
    # Identical cameras and depths have identity coarse correspondences.
    batch['poses_c2w'][:]=torch.eye(4)
    _,world,relative=M.gt_local_world_points(batch)
    logits=(torch.eye(4).repeat(1,2,1,1)*10).requires_grad_()
    pred=dict(depth=batch['depths'],camera_matches=logits)
    loss,stats=camera_match_loss(pred,batch,c,world,relative)
    assert stats['camera_match_count']==16 and stats['camera_match_accuracy']==1
    loss.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    batch['depths'][:,1:]=3.
    _,world,relative=M.gt_local_world_points(batch)
    loss,stats=camera_match_loss(pred,batch,c,world,relative)
    assert stats['camera_match_count']==0 and loss==0


def test_validation_balances_pose_and_depth(setup):
    c,_=setup
    collapsed=dict(abs_rel=.25,ate_over_identity=1.6,rpe_rotation_deg=30.)
    balanced=dict(abs_rel=.27,ate_over_identity=.8,rpe_rotation_deg=15.)
    assert validation_score(balanced,c)<validation_score(collapsed,c)
    with pytest.raises(FloatingPointError):
        validation_score(dict(balanced,abs_rel=float('nan')),c)


def test_match_labels_follow_known_camera_translation(setup):
    c,_=setup
    depth=torch.full((1,2,1,32,32),1.5)
    poses=torch.eye(4).repeat(1,2,1,1);poses[:,1,0,3]=1.
    intrinsics=torch.tensor([[24.,0,15.5],[0,24.,15.5],[0,0,1]]).repeat(1,2,1,1)
    batch=dict(images=torch.zeros(1,2,3,32,32),depths=depth,valids=torch.ones_like(depth),
               poses_c2w=poses,intrinsics=intrinsics)
    _,world,relative=M.gt_local_world_points(batch)
    # +1 camera translation at z=1.5, fx=24 projects 16 px to the left.
    score=torch.full((1,1,4,4),-10.)
    score[0,0,1,0]=10.;score[0,0,3,2]=10.
    pred=dict(depth=depth,camera_matches=score)
    _,stats=camera_match_loss(pred,batch,c,world,relative)
    assert stats['camera_match_count']==4 and stats['camera_match_accuracy']==1.


def test_camera_mixed_precision_gradient_reaches_match_projection(setup):
    head=SpatialCameraHead(32,32,2)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        outputs=head(torch.randn(1,3,32,2,2))
        logits=outputs[-1]
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,2),torch.arange(4).repeat(2))
    loss.backward()
    assert torch.isfinite(head.project.weight.grad).all()
    assert head.project.weight.grad.abs().sum()>0
