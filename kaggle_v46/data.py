"""Strict sequence splits and documented CO3D/TUM decoding; no debug fallbacks."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile
import urllib.request

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import torch
from torch.utils.data import Dataset

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
CO3D_DEFAULT_ROOT = Path('/kaggle/temp/co3d_data')
TUM_DEFAULT_ROOT = Path('/kaggle/temp/datasets/tum_rgbd')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def decode_co3d_png(path, scale_adjustment):
    # Official load_16big_png_depth: reinterpret bits, do not divide integers by 1000.
    raw = np.asarray(Image.open(path), dtype=np.uint16)
    if raw.ndim != 2:
        raise ValueError(f"Expected a single-channel float16-bit PNG: {path}")
    depth = np.ascontiguousarray(raw).view(np.float16).astype(np.float32)
    depth *= float(scale_adjustment)
    depth[~np.isfinite(depth)] = 0
    return depth


def letterbox(array, size, interpolation, fill=0):
    h, w = array.shape[:2]
    scale = min(size/w, size/h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    px, py = (size-nw)//2, (size-nh)//2
    resized = cv2.resize(array, (nw, nh), interpolation=interpolation)
    out = np.full((size, size) + array.shape[2:], fill, dtype=resized.dtype)
    out[py:py+nh, px:px+nw] = resized
    return out, (scale, px, py)


def resize_array(array, c, depth=False):
    interp = cv2.INTER_NEAREST if depth else cv2.INTER_AREA
    if c["dataset"] == "tum":
        return letterbox(array, c["image_size"], interp)[0]
    return cv2.resize(array, (c["image_size"], c["image_size"]), interpolation=interp)


def load_rgb(path, c):
    raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)/255.
    raw = resize_array(raw, c).astype(np.float32)
    return torch.from_numpy((raw-MEAN)/STD).permute(2,0,1), torch.from_numpy(raw).permute(2,0,1)


def load_geometry(row, c):
    if c["dataset"] == "co3d":
        depth = decode_co3d_png(row["depth"], row["depth_scale_adjustment"])
    else:
        depth = np.asarray(Image.open(row["depth"]), dtype=np.float32)/5000.
    depth = resize_array(depth, c, depth=True)
    valid = np.isfinite(depth) & (depth > c["min_depth"]) & (depth < c["max_depth"])
    if c["dataset"] == "co3d" and row.get("depth_mask"):
        dm = np.asarray(Image.open(row["depth_mask"]).convert("L"), dtype=np.float32)/255.
        valid &= resize_array(dm, c, depth=True) > .5
    if c["dataset"] == "co3d":
        if not row.get("mask"):
            raise FileNotFoundError("CO3D object support is required; missing mask for " + row["rgb"])
        support = np.asarray(Image.open(row["mask"]).convert("L"), dtype=np.float32)/255.
        support = resize_array(support, c, depth=True) > .5
        # The benchmark target uses fixed reference object support, independent of predictions.
        valid &= support
    else:
        support = valid.copy()
    depth = np.where(valid, depth, 0).astype(np.float32)
    return depth, valid.astype(np.float32), support.astype(np.float32)


def co3d_camera(v, hw, size):
    h, w = hw
    fl = np.asarray(v["focal_length"], np.float64)
    pp = np.asarray(v["principal_point"], np.float64)
    scale = np.array([min(w,h), min(w,h)]) if v.get("intrinsics_format", "ndc_norm_image_bounds") == "ndc_isotropic" else np.array([w,h])
    focal = fl*scale/2
    center = np.array([w,h])/2-pp*scale/2
    focal *= np.array([size/w, size/h]); center *= np.array([size/w, size/h])
    K = [[focal[0],0,center[0]], [0,focal[1],center[1]], [0,0,1]]
    rot = np.asarray(v["R"], np.float64)
    T = np.eye(4)
    T[:3,:3] = rot
    T[:3,3] = -rot @ np.asarray(v["T"], np.float64)
    T = T @ np.diag([-1.,-1.,1.,1.])
    return np.asarray(K, np.float32), T.astype(np.float32)


def resolve_co3d(root, category, name):
    if name is None:
        return None
    p = Path(name)
    candidates = [p] if p.is_absolute() else [root/p, root/category/p, root.parent/p]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Missing CO3D file {name!r}, searched {candidates}")


def locate_root(c):
    if c["data_root"]:
        return Path(c["data_root"]).expanduser().resolve()
    return CO3D_DEFAULT_ROOT if c['dataset']=='co3d' else TUM_DEFAULT_ROOT


def co3d_category_ready(root,category,wanted):
    annotation=Path(root)/category/'frame_annotations.jgz'
    if not annotation.is_file():return False
    try:
        with gzip.open(annotation,'rt') as f:rows=json.load(f)
        found=set()
        for row in rows:
            seq=category+'/'+row['sequence_name']
            if seq not in wanted or not row.get('depth') or not row.get('viewpoint'):continue
            depth=row['depth'];mask=(row.get('mask') or {}).get('path')
            if not mask:return False
            for path in [row['image']['path'],depth['path'],mask,depth.get('mask_path')]:
                if path:resolve_co3d(Path(root),category,path)
            found.add(seq)
        return wanted<=found
    except (OSError,ValueError,KeyError,EOFError):return False


def co3d_category_root(root,category,c,wanted):
    if co3d_category_ready(root,category,wanted):
        print('Reusing complete CO3D category:',Path(root)/category,flush=True)
        return Path(root)
    cache=CO3D_DEFAULT_ROOT
    if cache!=Path(root) and co3d_category_ready(cache,category,wanted):return cache
    if not c['allow_download']:raise FileNotFoundError(f'Incomplete CO3D category {category}; automatic downloads are disabled')
    from .co3d_download import download_category
    download_category(cache,category)
    if not co3d_category_ready(cache,category,wanted):
        raise ValueError(f'CO3D download does not contain all configured sequences/files: {sorted(wanted)}')
    return cache


def prepare_co3d(root, c):
    frames = {}
    categories = sorted(set(seq.split('/')[0] for split in c["splits"].values() for seq in split))
    wanted = set(seq for split in c["splits"].values() for seq in split)
    dropped = {"no_depth": 0, "insufficient_valid_depth": 0}
    for cat in categories:
        category_root=co3d_category_root(root,cat,c,{s for s in wanted if s.startswith(cat+'/')})
        with gzip.open(category_root/cat/"frame_annotations.jgz", "rt") as f:
            annotations = json.load(f)
        for a in annotations:
            seq = cat+"/"+a["sequence_name"]
            if seq not in wanted:
                continue
            if not a.get("depth") or not a.get("viewpoint"):
                dropped["no_depth"] += 1
                continue
            K, T = co3d_camera(a["viewpoint"], a["image"]["size"], c["image_size"])
            dep = a["depth"]
            row = dict(seq=seq, frame_number=int(a["frame_number"]),
                       timestamp=float(a.get("frame_timestamp", a["frame_number"])),
                       rgb=resolve_co3d(category_root, cat, a["image"]["path"]),
                       depth=resolve_co3d(category_root, cat, dep["path"]),
                       depth_scale_adjustment=float(dep["scale_adjustment"]),
                       depth_mask=resolve_co3d(category_root, cat, dep.get("mask_path")),
                       mask=resolve_co3d(category_root, cat, (a.get("mask") or {}).get("path")),
                       K=K.tolist(), pose=T.tolist())
            depth, valid, _ = load_geometry(row, c)
            if valid.sum() < 128:
                dropped["insufficient_valid_depth"] += 1
                continue
            frames.setdefault(seq, []).append(row)
    print("Official float16-bit CO3D decoding. Filter counts:", dropped, flush=True)
    return frames


def read_index(path):
    return [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]


def read_tum_groundtruth(path):
    """Chronological, unique, finite poses; keep the first valid pose at a tied time."""
    source=read_index(path)
    audit=dict(input_rows=len(source),malformed_rows=0,nonfinite_rows=0,invalid_quaternion_rows=0)
    valid=[]
    for row in source:
        try:
            if len(row)<8:raise ValueError('short pose row')
            pose=np.asarray(row[:8],dtype=np.float64)
        except (ValueError,TypeError):
            audit['malformed_rows']+=1;continue
        if not np.isfinite(pose).all():audit['nonfinite_rows']+=1;continue
        with np.errstate(over='ignore',invalid='ignore'):
            norm=np.linalg.norm(pose[4:8])
        if not np.isfinite(norm) or norm<=1e-12:
            audit['invalid_quaternion_rows']+=1;continue
        pose[4:8]/=norm
        valid.append(pose)
    values=np.asarray(valid,dtype=np.float64).reshape(-1,8)
    audit['out_of_order_transitions']=int((np.diff(values[:,0])<0).sum())
    values=values[np.argsort(values[:,0],kind='stable')]
    keep=np.r_[True,np.diff(values[:,0])>0] if len(values) else np.zeros(0,dtype=bool)
    audit['duplicate_timestamps_removed']=int(len(values)-keep.sum())
    values=values[keep];audit['retained_poses']=len(values)
    audit.update(duplicate_policy='first_valid_row_in_file',source_sha256=file_hash(path))
    if len(values)<2:
        raise ValueError(f'{path}: need at least two distinct finite ground-truth timestamps after cleaning; {audit}')
    return values[:,0],values[:,1:4],Rotation.from_quat(values[:,4:8]),audit


def read_tum_frames(path):
    """Sort image indices before searchsorted association and resolve ties consistently."""
    source=read_index(path);valid=[];invalid=0
    for row in source:
        try:
            if len(row)<2:raise ValueError('short image row')
            timestamp=float(row[0])
            if not np.isfinite(timestamp):raise ValueError('nonfinite image timestamp')
        except (ValueError,TypeError):invalid+=1;continue
        valid.append((timestamp,row[1]))
    unordered=int(sum(valid[i][0]<valid[i-1][0] for i in range(1,len(valid))))
    valid.sort(key=lambda row:row[0])
    unique=[]
    for row in valid:
        if not unique or row[0]!=unique[-1][0]:unique.append(row)
    if not unique:raise ValueError(f'{path}: no finite image timestamps')
    audit=dict(input_rows=len(source),invalid_rows=invalid,out_of_order_transitions=unordered,
        duplicate_timestamps_removed=len(valid)-len(unique),retained_frames=len(unique),
        duplicate_policy='first_valid_row_in_file',source_sha256=file_hash(path))
    return unique,audit


def interpolate_tum_pose(timestamp,times,translations,rotations,interpolator,tolerance=.02):
    right=int(np.searchsorted(times,timestamp))
    if right<len(times) and times[right]==timestamp:
        # An exact observation remains valid at either boundary or next to a GT gap.
        rotation=rotations[right].as_matrix();translation=translations[right]
    else:
        if right==0 or right==len(times):return None
        left=right-1
        if max(timestamp-times[left],times[right]-timestamp)>tolerance:return None
        fraction=(timestamp-times[left])/(times[right]-times[left])
        rotation=interpolator([timestamp]).as_matrix()[0]
        translation=(1-fraction)*translations[left]+fraction*translations[right]
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=translation
    return pose


def unique_associations(first, second, tolerance=.02):
    """Greedy one-to-one nearest timestamp associations, as in the TUM tool."""
    second = np.asarray(second, np.float64)
    candidates = []
    for i, t in enumerate(first):
        lo, hi = np.searchsorted(second, [t-tolerance, t+tolerance], side='left')
        for j in range(lo, min(hi+1, len(second))):
            delta = abs(t-second[j])
            if delta <= tolerance:
                candidates.append((delta, i, j))
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            pairs.append((i,j)); used_a.add(i); used_b.add(j)
    return sorted(pairs)


def tum_sequence_ready(path):
    path=Path(path)
    if not all((path/name).is_file() for name in ['groundtruth.txt','rgb.txt','depth.txt']):return False
    if not (path/'rgb').is_dir() or not (path/'depth').is_dir():return False
    try:
        for name in ['rgb.txt','depth.txt']:
            rows=read_index(path/name)
            if not rows or any(len(row)<2 or not (path/row[1]).is_file() for row in rows):return False
        return bool(read_index(path/'groundtruth.txt'))
    except OSError:return False


def tum_sequence_root(root, seq, c):
    paths = [root/seq]
    mounted = Path('/kaggle/input')
    if mounted.exists():
        paths.extend(sorted(mounted.glob(f'**/{seq}')))
    for p in paths:
        if tum_sequence_ready(p):
            print('Reusing complete TUM sequence:',p,flush=True)
            return p
    if not c["allow_download"]:
        raise FileNotFoundError(f"Missing TUM sequence {seq}")
    root.mkdir(parents=True, exist_ok=True)
    family = seq.split('_')[2]
    archive = root/f'{seq}.tgz'
    url = f'https://cvg.cit.tum.de/rgbd/dataset/{family}/{seq}.tgz'
    print('Downloading', url, flush=True)
    from .co3d_download import download_file
    download_file(url,archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    archive.unlink()  # Only this newly downloaded temporary archive is removed.
    if not tum_sequence_ready(root/seq):raise ValueError(f'TUM download is incomplete after extraction: {seq}')
    return root/seq


def prepare_tum(root, c, audits=None):
    result = {}
    for seq in sorted(seq for split in c['splits'].values() for seq in split):
        p = tum_sequence_root(root, seq, c)
        rgbs,rgb_audit=read_tum_frames(p/'rgb.txt')
        depths,depth_audit=read_tum_frames(p/'depth.txt')
        times,translations,rotations,gt_audit=read_tum_groundtruth(p/'groundtruth.txt')
        interpolator = Slerp(times, rotations)
        audit=dict(rgb=rgb_audit,depth=depth_audit,groundtruth=gt_audit)
        print('TUM timestamp audit:',seq,json.dumps(audit),flush=True)
        if audits is not None:audits[seq]=audit
        # ROS default registered-depth convention recommended by the dataset authors.
        K = np.array([[525.,0,319.5],[0,525.,239.5],[0,0,1]], np.float32)
        s, px, py = min(c['image_size']/640,c['image_size']/480), 0, c['image_size']//8
        K[:2,:] *= s; K[0,2] += px; K[1,2] += py
        rows = []
        for i,j in unique_associations([r[0] for r in rgbs],[r[0] for r in depths]):
            t=rgbs[i][0]
            T=interpolate_tum_pose(t,times,translations,rotations,interpolator)
            if T is None:continue
            row = dict(seq=seq, timestamp=t, frame_number=i, rgb=str(p/rgbs[i][1]),
                       depth=str(p/depths[j][1]), K=K.tolist(), pose=T.tolist())
            dep, valid, _ = load_geometry(row,c)
            if valid.sum() >= 256:
                rows.append(row)
        result[seq] = rows
    return result


def prepare_manifest(c, destination):
    if c.get('fixture_manifest'):
        manifest = json.loads(Path(c['fixture_manifest']).read_text())
        atomic_json(destination, manifest)
        return manifest
    root = locate_root(c)
    audits={}
    frames = prepare_co3d(root,c) if c['dataset']=='co3d' else prepare_tum(root,c,audits=audits)
    manifest = {'schema': 'v46_exact_frames', 'dataset': c['dataset'], 'splits': {}, 'frames': {}}
    if audits:manifest['input_timestamp_audit']=audits
    for split, sequences in c['splits'].items():
        windows = []
        for seq in sequences:
            rows = sorted(frames.get(seq, []), key=lambda r:r['frame_number'])
            if len(rows) < (c['window_views']-1)*c['frame_stride']+1:
                raise ValueError(f"{seq}: too few usable frames for the configured window; no split fallback is permitted")
            for row in rows:
                key = seq+':'+str(row['frame_number'])
                manifest['frames'][key] = row
            for start in range(0, len(rows)-(c['window_views']-1)*c['frame_stride'], c['window_step']):
                ids = [seq+':'+str(rows[start+k*c['frame_stride']]['frame_number']) for k in range(c['window_views'])]
                windows.append(dict(id=f'{split}:{seq}:{start}',seq=seq,frames=ids))
        if not windows:
            raise ValueError(f'Empty {split} split')
        manifest['splits'][split] = windows
        print(split, 'sequences',len(sequences),'windows',len(windows),flush=True)
    # Training-only depth hypotheses and baseline statistics. Test data never chooses these.
    train_ids = sorted(set(i for w in manifest['splits']['train'] for i in w['frames']))
    samples = []
    for fid in train_ids[::max(1,len(train_ids)//128)]:
        dep, valid, _ = load_geometry(manifest['frames'][fid], c)
        values = dep[valid>0]
        samples.extend(values[::max(1,len(values)//1024)].tolist())
    near, far = np.quantile(samples,[.01,.99]).tolist()
    manifest['training_depth_statistics'] = dict(near=max(c['min_depth'],near),
        far=min(c['max_depth'],max(far,near*1.1)),median=float(np.median(samples)))
    manifest['training_median_K']=np.median(np.array([manifest['frames'][fid]['K'] for fid in train_ids]),axis=0).tolist()
    manifest['data_root'] = str(root)
    manifest['fingerprint'] = json_hash(manifest)
    atomic_json(destination, manifest)
    return manifest


class WindowDataset(Dataset):
    def __init__(self, manifest, split, c, teacher=None):
        self.manifest, self.c, self.teacher = manifest,c,teacher or {}
        self.windows = manifest['splits'][split]
        self.training_stage=None
        self.training_epoch=0
    def __len__(self):
        return len(self.windows)
    def __getitem__(self,index):
        window = self.windows[index]
        frame_ids=window['frames']
        if self.training_stage is not None:
            rng=np.random.default_rng(self.c['seed']+self.training_epoch*1000003+index)
            indices=train_indices(len(frame_ids),self.training_stage,self.c,rng)
            frame_ids=[frame_ids[i] for i in indices]
        images, raws, depths, valids, masks, poses, intrinsics, times = ([] for _ in range(8))
        for fid in frame_ids:
            row = self.manifest['frames'][fid]
            image, raw = load_rgb(row['rgb'],self.c)
            depth,valid,support = load_geometry(row,self.c)
            images.append(image); raws.append(raw)
            depths.append(torch.from_numpy(depth)[None]); valids.append(torch.from_numpy(valid)[None])
            masks.append(torch.from_numpy(support)[None]); poses.append(torch.tensor(row['pose'],dtype=torch.float32))
            intrinsics.append(torch.tensor(row['K'],dtype=torch.float32));times.append(row['timestamp'])
        batch = dict(images=torch.stack(images),images_raw=torch.stack(raws),depths=torch.stack(depths),
                     valids=torch.stack(valids),masks=torch.stack(masks),poses_c2w=torch.stack(poses),
                     intrinsics=torch.stack(intrinsics),timestamps=torch.tensor(times,dtype=torch.float64),
                     window_id=window['id'],sequence=window['seq'],frame_ids=frame_ids)
        if self.teacher:
            batch['teacher'] = torch.stack([self.teacher[fid].float() for fid in frame_ids])
        return batch


def slice_views(batch, indices):
    k = batch['images'].shape[1]
    for key,value in list(batch.items()):
        if torch.is_tensor(value) and value.ndim>=2 and value.shape[1]==k:
            batch[key] = value[:,indices].contiguous()
    batch['view_valid_mask'] = torch.ones(batch['images'].shape[:2], device=batch['images'].device)
    return batch


def even_indices(k, count):
    return np.linspace(0,k-1,min(k,count)).round().astype(int).tolist()


def train_indices(k,stage,c,rng):
    count = int(rng.choice(c['stage_k'][stage],p=c['stage_k_probabilities'][stage]))
    if count==1:
        return [0]
    interior = sorted(rng.choice(np.arange(1,k-1),count-2,replace=False).tolist()) if count>2 else []
    # Reference and last view are retained; dropout applies only to interior views.
    if count>3:
        kept = [i for i in interior if rng.random()>=c['view_dropout']]
        if len(kept)>=1:
            interior=kept
    return [0]+interior+[k-1]
