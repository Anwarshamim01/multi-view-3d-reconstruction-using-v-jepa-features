"""Regression tests for the real TUM duplicate-timestamp Slerp crash."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation,Slerp

from kaggle_v46.config import default_config
from kaggle_v46.data import (read_tum_groundtruth,read_tum_frames,interpolate_tum_pose,
                            prepare_manifest,WindowDataset)


def pose_row(timestamp,position,angle=0.):
    q=Rotation.from_euler('z',angle).as_quat()
    return ' '.join(map(str,[timestamp,position,0.,0.,*q]))+'\n'


def test_groundtruth_keeps_pose_association_when_sorting_and_removing_duplicates(tmp_path):
    path=tmp_path/'groundtruth.txt'
    path.write_text(pose_row(10.02,.04,.2)+pose_row(10.,0.)+pose_row(10.01,.02,.1)+
                    pose_row(10.01,999.,2.)+'10.015 0 0 0 0 0 0 0\n'+
                    'nan 0 0 0 0 0 0 1\n'+'bad row\n')
    times,translations,rotations,audit=read_tum_groundtruth(path)
    np.testing.assert_allclose(times,[10.,10.01,10.02])
    np.testing.assert_allclose(translations[:,0],[0.,.02,.04])
    np.testing.assert_allclose(rotations.as_euler('xyz')[:,2],[0.,.1,.2])
    assert audit['duplicate_timestamps_removed']==1
    assert audit['malformed_rows']==audit['nonfinite_rows']==audit['invalid_quaternion_rows']==1
    assert audit['retained_poses']==3 and audit['out_of_order_transitions']==1
    # This constructor previously crashed on the duplicated/unsorted input.
    interpolator=Slerp(times,rotations)
    middle=interpolate_tum_pose(10.005,times,translations,rotations,interpolator)
    np.testing.assert_allclose(middle[0,3],.01,atol=1e-12)
    np.testing.assert_allclose(Rotation.from_matrix(middle[:3,:3]).as_euler('xyz')[2],.05,atol=1e-12)


def test_exact_observations_at_boundaries_or_next_to_gap_are_retained():
    times=np.array([10.,10.01,10.2]);translations=np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
    rotations=Rotation.from_quat([[0.,0,0,1]]*3);interpolator=Slerp(times,rotations)
    for index,timestamp in enumerate(times):
        pose=interpolate_tum_pose(timestamp,times,translations,rotations,interpolator)
        np.testing.assert_allclose(pose[:3,3],translations[index])
    for timestamp in [9.99,10.21,10.1]:
        assert interpolate_tum_pose(timestamp,times,translations,rotations,interpolator) is None


def test_not_enough_distinct_valid_poses_has_actionable_error(tmp_path):
    path=tmp_path/'groundtruth.txt';path.write_text(pose_row(1.,0.)+pose_row(1.,1.))
    with pytest.raises(ValueError,match='at least two distinct finite ground-truth timestamps'):
        read_tum_groundtruth(path)


def test_image_timestamps_are_sorted_and_deduplicated_before_association(tmp_path):
    path=tmp_path/'depth.txt'
    path.write_text('2 depth/two.png\n1 depth/one.png\n1 depth/duplicate.png\nnan ignored.png\nbad\n')
    rows,audit=read_tum_frames(path)
    assert rows==[(1.,'depth/one.png'),(2.,'depth/two.png')]
    assert audit['duplicate_timestamps_removed']==1 and audit['invalid_rows']==2


def test_complete_tum_preparation_uses_cleaned_indices_and_audits(tmp_path):
    c=default_config('tum')
    c.update(data_root=str(tmp_path),image_size=32,window_views=3,frame_stride=1,window_step=1,
             allow_download=False,splits={s:['artificial_'+s] for s in ['train','val','test']})
    for sequences in c['splits'].values():
        p=tmp_path/sequences[0]
        (p/'rgb').mkdir(parents=True);(p/'depth').mkdir()
        for index in range(3):
            Image.fromarray(np.full((24,32,3),100,np.uint8)).save(p/'rgb'/f'{index}.png')
            Image.fromarray(np.full((24,32),7500,np.uint16)).save(p/'depth'/f'{index}.png')
        (p/'rgb.txt').write_text('10.02 rgb/2.png\n10 rgb/0.png\n10.01 rgb/1.png\n')
        (p/'depth.txt').write_text('10.01 depth/1.png\n10.02 depth/2.png\n10 depth/0.png\n')
        (p/'groundtruth.txt').write_text(pose_row(10.02,.04)+pose_row(10.,0.)+
            pose_row(10.01,.02)+pose_row(10.01,999.))
    manifest=prepare_manifest(c,tmp_path/'data_manifest.json')
    for split in ['train','val','test']:
        assert len(manifest['splits'][split])==1
        sample=WindowDataset(manifest,split,c)[0]
        np.testing.assert_allclose(sample['timestamps'].numpy(),[10.,10.01,10.02])
        np.testing.assert_allclose(sample['poses_c2w'][:,0,3].numpy(),[0.,.02,.04])
        assert manifest['input_timestamp_audit']['artificial_'+split]['groundtruth']['duplicate_timestamps_removed']==1
