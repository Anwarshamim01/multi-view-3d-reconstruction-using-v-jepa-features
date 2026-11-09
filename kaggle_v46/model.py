"""V-JEPA-Geo corrected model primitives, derived from the supplied v43 notebooks.

Configuration is bound once per torchrun process by configure().
"""
import math, warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
CFG = SimpleNamespace()

def configure(config, depth_statistics=None):
    import json
    defaults = json.loads(Path(__file__).with_name('model_defaults.json').read_text())
    defaults.update(
        IMAGE_SIZE=config['image_size'], PATCH_SIZE=16,
        FEAT_DIM=32 if config.get('tiny_backbone') else 768,
        MIN_DEPTH=config['min_depth'], MAX_DEPTH=config['max_depth'],
        DEPTH_BINS=config['depth_bins'], DEPTH_CHUNK=config['depth_chunk'],
        DEPTH_NEAR=(depth_statistics or {}).get('near',config['min_depth']),
        DEPTH_FAR=(depth_statistics or {}).get('far',config['max_depth']),
        COST_TEMP=config['cost_temperature'],
        USE_EPIPOLAR_FUSION=config['use_epipolar'], POSEFREE_USE_PREDICTED_POSE_FOR_EPIPOLAR=True,
        USE_VARIABLE_K_SCENE_AGGREGATOR=config['use_scene_tokens'],
        USE_CAMERA_REFINEMENT_HEAD=config['use_refinement'],
        USE_FOREGROUND_MASK_HEAD=config['use_mask'],
        USE_TRACK_DESCRIPTOR_HEAD=config['use_descriptors'],
        USE_CROSSJEPA_AUX=config['glvp_weight']>0,
        CROSSJEPA_ZERO_VIEW_CONDITIONING=not config['glvp_conditioning'],
        CROSSJEPA_NORMALIZE_EMBEDDINGS=True,
        CROSSJEPA_PREDICTOR_HIDDEN=512, CROSSJEPA_POSE_INPUT_DIM=20,
        EDGE_ENCODER_IMAGE_SIZE=min(config['edge_size'],config['image_size']),
        EDGE_ENCODER_CHUNK_SIZE=2, EDGE_INPUT_SOURCE='raw_rgb', EDGE_RAW_CLAMP_01=True,
        USE_EDGE_GRAD_CHECKPOINT=config['checkpoint_activations'],
        USE_DECODER_GRAD_CHECKPOINT=config['checkpoint_activations'],
        USE_BACKBONE_GRAD_CHECKPOINT=config['checkpoint_activations'],
        USE_TRANSPORT_GRAD_CHECKPOINT=config['checkpoint_activations'],
        BACKBONE_ABLATION='vjepa_last2' if config['trainable_backbone_blocks'] else 'vjepa_frozen',
        VJEPA_FINETUNE_MODE='last_blocks' if config['trainable_backbone_blocks'] else 'none',
        POSEFREE_UNFREEZE_LAST_N_BLOCKS=config['trainable_backbone_blocks'],
        UNFREEZE_LAST_N_BLOCKS=config['trainable_backbone_blocks'],
        V43_SCENE_TOKENS=4, V43_VIEW_TRANSFORMER_LAYERS=2,
        V43_VIEW_TRANSFORMER_HEADS=8, V43_VIEW_TRANSFORMER_FF_MULT=2,
        CAMERA_REFINEMENT_MAX_TRANS=.08, CAMERA_REFINEMENT_MAX_ROT_RAD=.04,
        GRID_SAMPLE_ALIGN_CORNERS=True, USE_POINT_RESIDUAL=True, POINT_RESIDUAL_MAX=.1,
        NUM_FRAMES_FOR_VIDEO_REPEAT=2, FEATURE_TOKEN_FALLBACK_MODE='error',
        SIM3_ALIGNMENT_GRAD_MODE='detach', POSEFREE_SCALE_MIN=1e-6, POSEFREE_SCALE_MAX=1e6,
    )
    global CFG, RUN_CONFIG
    CFG=SimpleNamespace(**defaults)
    RUN_CONFIG=config
    return CFG


def stage_allows_predicted_pose_epipolar(stage):
    # Evaluation never inherits a training warm-up gate.
    return stage != 'stage1_frozen'


class TinyBackbone(nn.Module):
    """Synthetic smoke-test backbone; never permitted for research result claims."""
    def __init__(self):
        super().__init__()
        self.patch=nn.Conv2d(3,CFG.FEAT_DIM,16,stride=16)
        self.blocks=nn.ModuleList([nn.Conv2d(CFG.FEAT_DIM,CFG.FEAT_DIM,1)])
        self.norm=nn.GroupNorm(8,CFG.FEAT_DIM)
    def forward(self,x):
        x=self.patch(x[:,:,0])
        for block in self.blocks:
            x=x+F.gelu(block(x))
        return self.norm(x).flatten(2).transpose(1,2)


def load_vjepa_backbone():
    if RUN_CONFIG.get('tiny_backbone'):
        return TinyBackbone()
    import sys
    from .assets import prepare_assets
    repo, checkpoint = prepare_assets(RUN_CONFIG)
    sys.path.insert(0,str(repo))
    # Call the encoder builder directly: no hub URL lookup, no unused predictor.
    from app.vjepa_2_1.models import vision_transformer as vit
    encoder=vit.vit_base(img_size=(384,384),patch_size=16,num_frames=64,tubelet_size=2,
                         use_sdpa=True,use_SiLU=False,wide_SiLU=True,uniform_power=False,
                         use_rope=True,img_temporal_dim_size=1,interpolate_rope=True)
    if RUN_CONFIG.get('backbone_initialization','pretrained')=='random':
        return encoder
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'ema_encoder' not in payload:
        raise ValueError('Expected the V-JEPA 2.1 ema_encoder checkpoint key')
    state={}
    for key,value in payload['ema_encoder'].items():
        for prefix in ['module.','backbone.']:
            if key.startswith(prefix):
                key=key[len(prefix):]
        if key!='pos_embed':
            state[key]=value
    missing,unexpected=encoder.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise ValueError(f'Encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    return encoder


class PoseFreeCameraHead(nn.Module):
    """Positive focal lengths and SE3 poses without data-dependent hard pose bounds."""
    def __init__(self,ch):
        super().__init__()
        self.view_transformer=ViewSetTransformer(ch)
        self.pose_net=nn.Sequential(nn.Linear(ch*2,512),nn.GELU(),nn.Linear(512,256),nn.GELU(),nn.Linear(256,6))
        self.k_net=nn.Sequential(nn.Linear(ch*2,512),nn.GELU(),nn.Linear(512,256),nn.GELU(),nn.Linear(256,4))
        for net in [self.pose_net,self.k_net]:
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    def forward(self,feats):
        B,K,C,_,_=feats.shape
        pooled=self.view_transformer(feats.mean(dim=(-1,-2)))
        x=torch.cat([pooled,pooled[:,0:1].expand(-1,K,-1)],dim=-1)
        raw=self.pose_net(x).float()
        xi=torch.cat([torch.zeros_like(raw[:,0:1]),raw[:,1:]],dim=1)
        poses=se3_exp(xi.reshape(-1,6)).reshape(B,K,4,4)
        camera=self.k_net(x).float()
        focal=F.softplus(camera[...,:2]+math.log(math.expm1(.82)))+.001
        center=.5+.1*camera[...,2:]
        normalized=torch.cat([focal,center],dim=-1)
        return xi,poses,make_K_from_normalized(normalized,CFG.IMAGE_SIZE),normalized,pooled


class EpipolarCostFusion(nn.Module):
    """Streaming log-depth transport with cosine logits and checkpointed chunks."""
    def __init__(self,ch,depth_bins=24):
        super().__init__()
        self.depth_bins=depth_bins
        self.combine=nn.Sequential(nn.Conv2d(ch*2+1,ch,1),nn.GroupNorm(32,ch),nn.GELU(),
                                   nn.Conv2d(ch,ch,3,padding=1),nn.GroupNorm(32,ch),nn.GELU())
    def sample_volume(self,fs,pose_t,pose_s,Kt,Ks,depths):
        B,C,h,w=fs.shape
        with torch.amp.autocast(amp_device(fs),enabled=False):
            yy,xx=torch.meshgrid(torch.arange(h,device=fs.device,dtype=torch.float32),
                                 torch.arange(w,device=fs.device,dtype=torch.float32),indexing='ij')
            uv=torch.stack([xx.flatten(),yy.flatten(),torch.ones(h*w,device=fs.device)],dim=0)
            rays=torch.linalg.solve(Kt.float(),uv[None].expand(B,-1,-1))
            pc=rays[:,None]*depths[None,:,None,None]
            points=torch.cat([pc,torch.ones(B,len(depths),1,h*w,device=fs.device)],dim=2)
            transformed=(invert_se3(pose_s.float())[:,None] @ pose_t.float()[:,None] @ points)[:,:,:3]
            z=transformed[:,:,2]
            projection=Ks.float()[:,None] @ transformed
            u,v=projection[:,:,0]/z.clamp_min(1e-5),projection[:,:,1]/z.clamp_min(1e-5)
            gx,gy=2*u/max(w-1,1)-1,2*v/max(h-1,1)-1
            valid=(z>1e-5)&(gx>=-1)&(gx<=1)&(gy>=-1)&(gy<=1)
            grid=torch.stack([gx,gy],dim=-1).reshape(B*len(depths),h,w,2)
            source=fs.float()[:,None].expand(B,len(depths),C,h,w).reshape(B*len(depths),C,h,w)
            sampled=F.grid_sample(source,grid,align_corners=True,padding_mode='zeros')
            return sampled.reshape(B,len(depths),C,h,w),valid.reshape(B,len(depths),h,w)
    def transport(self,fs,ft,pt,ps,kt,ks,depths):
        sample,valid=self.sample_volume(fs,pt,ps,kt,ks,depths)
        similarity=(F.normalize(ft.float(),dim=1)[:,None]*F.normalize(sample,dim=2)).sum(dim=2)
        weight=torch.exp((similarity-1)/CFG.COST_TEMP)*valid.float()
        numerator=(sample*weight[:,:,None]).sum(dim=1)
        denominator=weight.sum(dim=1,keepdim=True)
        depth_num=(weight*depths[None,:,None,None]).sum(dim=1,keepdim=True)
        return numerator,denominator,depth_num
    def forward(self,features,poses,intrinsics):
        B,K,C,h,w=features.shape
        if K<2 or not CFG.USE_EPIPOLAR_FUSION:
            return features,None
        with torch.amp.autocast(amp_device(features),enabled=False):
            cameras=intrinsics.float().clone()
            cameras[...,:2,:]*=h/CFG.IMAGE_SIZE
            bins=torch.logspace(math.log10(CFG.DEPTH_NEAR),math.log10(CFG.DEPTH_FAR),
                                self.depth_bins,device=features.device)
        fused,priors=[],[]
        for target in range(K):
            numerator=features[:,target].float()*0
            denominator=torch.zeros(B,1,h,w,device=features.device)
            depth_num=torch.zeros_like(denominator)
            for source in range(K):
                if source==target:continue
                for depths in bins.split(CFG.DEPTH_CHUNK):
                    args=(features[:,source],features[:,target],poses[:,target],poses[:,source],cameras[:,target],cameras[:,source],depths)
                    if self.training and CFG.USE_TRANSPORT_GRAD_CHECKPOINT:
                        num,den,dnum=torch_checkpoint(self.transport,*args,use_reentrant=False)
                    else:
                        num,den,dnum=self.transport(*args)
                    numerator=numerator+num;denominator=denominator+den;depth_num=depth_num+dnum
            geo=numerator/denominator.clamp_min(1e-8)
            prior=depth_num/denominator.clamp_min(1e-8)
            fused.append(self.combine(torch.cat([features[:,target],geo.to(features.dtype),
                                                (prior/CFG.MAX_DEPTH).to(features.dtype)],dim=1)))
            priors.append(prior)
        return torch.stack(fused,dim=1),torch.stack(priors,dim=1)

def collect_tensors(obj):
    out = []
    if torch.is_tensor(obj):
        if obj.dim() >= 3:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values(): out.extend(collect_tensors(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj: out.extend(collect_tensors(v))
    return out

def select_feature_tensor(output):
    tensors = collect_tensors(output)
    if not tensors:
        raise RuntimeError(f"No tensor found in V-JEPA output: {type(output)}")
    return max(tensors, key=lambda t: t.numel())

def tokens_to_feature_map(x, feat_dim=768, image_size=384, patch_size=16):
    # v41_dynamic_grid_note: for 512 training, image_size=512 and patch_size=16 => 32x32=1024 tokens.
    """
    Convert a V-JEPA tensor into a dense [B, C, H, W] feature map.

    Expected token layouts:
      - [B, spatial_tokens, C]
      - [B, 1 + spatial_tokens, C] with a leading class/global token
      - [B, T * spatial_tokens, C] for repeated/video-like tokens
      - [B, 1 + T * spatial_tokens, C]

    Any other token count is suspicious because it may indicate that the
    upstream V-JEPA output format changed. By default this raises an error.
    Set CFG.FEATURE_TOKEN_FALLBACK_MODE="warn" only when you intentionally
    want the legacy last-spatial-tokens fallback.
    """
    side = image_size // patch_size
    spatial_tokens = side * side

    if x.dim() == 4:
        if x.shape[1] == feat_dim:
            return x
        if x.shape[-1] == feat_dim:
            return x.permute(0, 3, 1, 2).contiguous()

    if x.dim() == 5:
        if x.shape[1] == feat_dim:
            return x.mean(dim=2)
        if x.shape[-1] == feat_dim:
            return x.mean(dim=1).permute(0, 3, 1, 2).contiguous()

    if x.dim() == 3:
        original_shape = tuple(x.shape)
        B, N, C = x.shape

        if C != feat_dim and x.shape[1] == feat_dim:
            x = x.transpose(1, 2).contiguous()
            B, N, C = x.shape

        removed_leading_token = False
        if (N - 1) % spatial_tokens == 0:
            x = x[:, 1:, :]
            N -= 1
            removed_leading_token = True

        if N == spatial_tokens:
            return x.transpose(1, 2).reshape(B, C, side, side)

        if N % spatial_tokens == 0:
            T = N // spatial_tokens
            x = x.reshape(B, T, side, side, C).mean(dim=1)
            return x.permute(0, 3, 1, 2).contiguous()

        msg = (
            f"V-JEPA token count {N} after class/global-token handling does not match "
            f"expected spatial_tokens={spatial_tokens} or T*spatial_tokens. "
            f"original_shape={original_shape}, current_shape={tuple(x.shape)}, "
            f"removed_leading_token={removed_leading_token}, side={side}, feat_dim={feat_dim}."
        )

        if N < spatial_tokens:
            raise RuntimeError(msg + " Cannot use last-spatial-tokens fallback because N < spatial_tokens.")

        cfg = globals().get("CFG", None)
        mode = getattr(cfg, "FEATURE_TOKEN_FALLBACK_MODE", "error")
        if mode == "error":
            raise RuntimeError(
                msg
                + " Set CFG.FEATURE_TOKEN_FALLBACK_MODE='warn' to allow the legacy fallback."
            )
        if mode == "warn":
            warnings.warn(
                msg + " Using last spatial tokens fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif mode == "silent":
            pass
        else:
            raise ValueError(
                f"Unknown CFG.FEATURE_TOKEN_FALLBACK_MODE={mode!r}. "
                "Use 'error', 'warn', or 'silent'."
            )

        x = x[:, -spatial_tokens:, :]
        return x.transpose(1, 2).reshape(B, C, side, side)

    raise RuntimeError(f"Cannot convert feature shape {tuple(x.shape)}")

def amp_device(tensor):
    return "cuda" if tensor.device.type == "cuda" else "cpu"

def skew(w):
    """Return the skew-symmetric matrix [w]_x for vectors w=[wx,wy,wz]."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)

    return torch.stack(
        [
            torch.stack([O, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )

def _so3_coefficients(w, eps=1e-3):
    """
    Stable Rodrigues/left-Jacobian coefficients for SO(3).

    For theta = ||w|| and Omega=[w]_x:
        exp(Omega) = I + A Omega + B Omega^2
        J_l(w)     = I + B Omega + C Omega^2
    where
        A = sin(theta)/theta
        B = (1-cos(theta))/theta^2
        C = (theta-sin(theta))/theta^3

    Small-angle Taylor series avoid 0/0 and preserve gradients near identity.
    """
    theta2 = (w * w).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(eps * eps))
    small = theta2 < (eps * eps)

    theta4 = theta2 * theta2

    A_small = 1.0 - theta2 / 6.0 + theta4 / 120.0
    B_small = 0.5 - theta2 / 24.0 + theta4 / 720.0
    C_small = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0

    A_large = torch.sin(theta) / theta
    B_large = (1.0 - torch.cos(theta)) / theta2.clamp_min(eps * eps)
    C_large = (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(eps ** 3)

    A = torch.where(small, A_small, A_large)
    B = torch.where(small, B_small, B_large)
    C = torch.where(small, C_small, C_large)
    return A, B, C

def so3_exp(w):
    """
    Strict SO(3) exponential map using Rodrigues' formula.

    Args:
        w: [..., 3] rotation vector in radians.
    Returns:
        R: [..., 3, 3] rotation matrix.
    """
    with torch.amp.autocast(amp_device(w), enabled=False):
        w = w.float()
        Omega = skew(w)
        Omega2 = Omega @ Omega
        A, B, _ = _so3_coefficients(w)

        I = torch.eye(3, device=w.device, dtype=torch.float32).expand(w.shape[:-1] + (3, 3))
        R = I + A[..., None] * Omega + B[..., None] * Omega2
        return R

def so3_left_jacobian(w):
    """
    SO(3) left Jacobian J_l(w), used by the strict SE(3) exponential map.

    J_l(w) = I + (1-cos(theta))/theta^2 [w]_x
               + (theta-sin(theta))/theta^3 [w]_x^2
    """
    with torch.amp.autocast(amp_device(w), enabled=False):
        w = w.float()
        Omega = skew(w)
        Omega2 = Omega @ Omega
        _, B, C = _so3_coefficients(w)

        I = torch.eye(3, device=w.device, dtype=torch.float32).expand(w.shape[:-1] + (3, 3))
        J = I + B[..., None] * Omega + C[..., None] * Omega2
        return J

def se3_exp(xi):
    """
    Strict SE(3) exponential map.

    Convention used in this notebook:
        xi[..., :3] = rho   (translational tangent component)
        xi[..., 3:] = omega (rotation vector)

    The true matrix Lie exponential is:
        Exp_SE3([rho, omega]) = [[Exp_SO3(omega), J_l(omega) @ rho],
                                 [0,                 1          ]]
    where J_l is the SO(3) left Jacobian. This is different from the older
    simplified version that directly used t=rho without J_l(omega).
    """
    with torch.amp.autocast(amp_device(xi), enabled=False):
        xi = xi.float()
        rho = xi[..., :3]
        w = xi[..., 3:]

        Rm = so3_exp(w)
        J = so3_left_jacobian(w)
        t = (J @ rho[..., None]).squeeze(-1)

        T = torch.eye(4, device=xi.device, dtype=torch.float32).expand(xi.shape[:-1] + (4, 4)).clone()
        T[..., :3, :3] = Rm
        T[..., :3, 3] = t
        return T

def invert_se3(T):
    """
    Analytic inverse for rigid SE(3) transforms.

    For T = [[R, t], [0, 1]],
        T^{-1} = [[R^T, -R^T t], [0, 1]].

    This is faster and numerically cleaner than a generic 4x4 inverse for
    pose matrices, and avoids unnecessary linalg work inside hot loops.
    """
    with torch.amp.autocast(amp_device(T), enabled=False):
        T = T.float()
        R = T[..., :3, :3]
        t = T[..., :3, 3:4]
        Rt = R.transpose(-1, -2)
    
        Tinv = torch.zeros_like(T)
        Tinv[..., :3, :3] = Rt
        Tinv[..., :3, 3:4] = -Rt @ t
        Tinv[..., 3, 3] = 1.0
        return Tinv

def depth_to_normals(depth, K):
    """
    depth: B,1,H,W
    K:     B,3,3

    Returns:
        B,3,H,W normals in camera coordinates.

    This is geometry-heavy, so it uses float32 even under AMP.
    """
    with torch.amp.autocast(amp_device(depth), enabled=False):
        depth = depth.float()
        K = K.float()

        B, _, H, W = depth.shape
        device = depth.device

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        xs = xs[None].expand(B, -1, -1)
        ys = ys[None].expand(B, -1, -1)

        z = depth[:, 0]

        fx = K[:, 0, 0][:, None, None]
        fy = K[:, 1, 1][:, None, None]
        cx = K[:, 0, 2][:, None, None]
        cy = K[:, 1, 2][:, None, None]

        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy

        pts = torch.stack([X, Y, z], dim=1)

        dx = pts[:, :, :, 2:] - pts[:, :, :, :-2]
        dy = pts[:, :, 2:, :] - pts[:, :, :-2, :]

        dx = F.pad(dx, (1, 1, 0, 0), mode="replicate")
        dy = F.pad(dy, (0, 0, 1, 1), mode="replicate")

        n = torch.cross(dx, dy, dim=1)
        n = F.normalize(n, dim=1, eps=1e-6)

        # Orient roughly toward camera.
        n = torch.where(n[:, 2:3] > 0, -n, n)

    return n

def edge_aware_smoothness(depth, image_raw):
    dx_d = torch.abs(depth[..., :, 1:] - depth[..., :, :-1])
    dy_d = torch.abs(depth[..., 1:, :] - depth[..., :-1, :])

    dx_i = torch.mean(
        torch.abs(image_raw[..., :, 1:] - image_raw[..., :, :-1]),
        dim=1,
        keepdim=True,
    )

    dy_i = torch.mean(
        torch.abs(image_raw[..., 1:, :] - image_raw[..., :-1, :]),
        dim=1,
        keepdim=True,
    )

    return (dx_d * torch.exp(-dx_i)).mean() + (dy_d * torch.exp(-dy_i)).mean()

def freeze_all(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_last_blocks(backbone, n=2):
    freeze_all(backbone)
    blocks = None
    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
    elif hasattr(backbone, "module") and hasattr(backbone.module, "blocks"):
        blocks = backbone.module.blocks
    if blocks is None:
        print("WARNING: Could not find backbone.blocks; keeping V-JEPA frozen.")
        return []
    selected = list(blocks)[-n:]
    params = []
    for blk in selected:
        for p in blk.parameters():
            p.requires_grad = True
            params.append(p)
    # Usually keep final norm trainable too.
    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True
            params.append(p)
    print(f"Unfroze last {n} V-JEPA blocks; trainable backbone params:", sum(p.numel() for p in params))
    return params

def unfreeze_all_backbone(backbone):
    params = []
    for p in backbone.parameters():
        p.requires_grad = True
        params.append(p)
    print("Unfroze full V-JEPA backbone params:", sum(p.numel() for p in params))
    return params

def set_backbone_trainability(backbone, stage="frozen"):
    mode = CFG.BACKBONE_ABLATION
    if stage == "frozen" or mode == "vjepa_frozen" or CFG.VJEPA_FINETUNE_MODE == "none":
        freeze_all(backbone)
        backbone.eval()
        return []
    if mode in ("vjepa_last2", "random_vjepa_last2") or CFG.VJEPA_FINETUNE_MODE == "last_blocks":
        params = unfreeze_last_blocks(backbone, CFG.UNFREEZE_LAST_N_BLOCKS)
        backbone.train()
        return params
    if mode == "vjepa_full":
        params = unfreeze_all_backbone(backbone)
        backbone.train()
        return params
    raise ValueError(f"Unknown BACKBONE_ABLATION / VJEPA mode: {mode} / {CFG.VJEPA_FINETUNE_MODE}")

def make_vjepa_input(image):
    # image B,C,H,W -> B,C,T,H,W for V-JEPA hub PatchEmbed3D.
    return image.unsqueeze(2).repeat(1, 1, CFG.NUM_FRAMES_FOR_VIDEO_REPEAT, 1, 1)

def normalize_K_params(K, image_size=None):
    """K: [...,3,3] -> [...,4] = fx/W, fy/H, cx/W, cy/H."""
    if image_size is None:
        image_size = CFG.IMAGE_SIZE
    K = K.float()
    return torch.stack([
        K[..., 0, 0] / float(image_size),
        K[..., 1, 1] / float(image_size),
        K[..., 0, 2] / float(image_size),
        K[..., 1, 2] / float(image_size),
    ], dim=-1)

def make_K_from_normalized(params, image_size=None):
    if image_size is None:
        image_size = CFG.IMAGE_SIZE
    params = params.float()
    B, K = params.shape[:2]
    out = torch.zeros(B, K, 3, 3, device=params.device, dtype=torch.float32)
    out[..., 0, 0] = params[..., 0] * image_size
    out[..., 1, 1] = params[..., 1] * image_size
    out[..., 0, 2] = params[..., 2] * image_size
    out[..., 1, 2] = params[..., 3] * image_size
    out[..., 2, 2] = 1.0
    return out

def camera_points_from_depth(depth, K):
    """depth: B,1,H,W; K: B,3,3 -> B,3,H,W camera coordinates."""
    with torch.amp.autocast("cuda" if depth.device.type == "cuda" else "cpu", enabled=False):
        depth = depth.float()
        K = K.float()
        B, _, H, W = depth.shape
        device = depth.device
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xs = xs[None].expand(B, -1, -1)
        ys = ys[None].expand(B, -1, -1)
        z = depth[:, 0]
        fx = K[:, 0, 0][:, None, None].clamp_min(1e-6)
        fy = K[:, 1, 1][:, None, None].clamp_min(1e-6)
        cx = K[:, 0, 2][:, None, None]
        cy = K[:, 1, 2][:, None, None]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        return torch.stack([x, y, z], dim=1)

def transform_cam_to_world_points(points_cam, poses_c2w):
    """points_cam: B,K,3,H,W; poses_c2w: B,K,4,4 -> B,K,3,H,W."""
    with torch.amp.autocast("cuda" if points_cam.device.type == "cuda" else "cpu", enabled=False):
        points_cam = points_cam.float()
        poses_c2w = poses_c2w.float()
        B, K, _, H, W = points_cam.shape
        pts = points_cam.reshape(B, K, 3, -1)
        ones = torch.ones(B, K, 1, pts.shape[-1], device=points_cam.device, dtype=torch.float32)
        pts_h = torch.cat([pts, ones], dim=2)
        pts_w = torch.matmul(poses_c2w, pts_h)[:, :, :3]
        return pts_w.reshape(B, K, 3, H, W)

def relative_c2w_to_first(poses_c2w):
    """Absolute dataset c2w -> local reference where view 0 is identity."""
    with torch.amp.autocast("cuda" if poses_c2w.device.type == "cuda" else "cpu", enabled=False):
        poses = poses_c2w.float()
        T0_inv = invert_se3(poses[:, 0])[:, None]
        return torch.matmul(T0_inv, poses)

class ViewSetTransformer(nn.Module):
    """Small set transformer over per-view V-JEPA pooled tokens."""
    def __init__(self, ch, num_layers=2, num_heads=8):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ch, nhead=num_heads, dim_feedforward=ch * 2,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(ch)

    def forward(self, pooled):
        return self.norm(self.encoder(pooled))

class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )

    def forward(self, x):
        e1 = self.stem(x)      # H
        e2 = self.down2(e1)    # H/2
        e4 = self.down4(e2)    # H/4
        return e1, e2, e4

class PoseFreePointDecoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, 256, 1), nn.GroupNorm(32, 256), nn.GELU())
        self.up1 = self.block(256, 256)
        self.up2 = self.block(256 + 128, 128)
        self.up3 = self.block(128 + 64, 64)
        self.up4 = self.block(64 + 32, 32)
        self.final = nn.Conv2d(32, 8, 3, padding=1)  # depth, conf, normal xyz, point residual xyz

    def block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(max(8, cout // 8), cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(max(8, cout // 8), cout), nn.GELU(),
        )

    def forward(self, fmap, edge_feats):
        e1, e2, e4 = edge_feats
        x = self.proj(fmap)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        e4r = F.interpolate(e4, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([self.up1(x), e4r], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        e2r = F.interpolate(e2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, e2r], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        e1r = F.interpolate(e1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, e1r], dim=1))
        out = self.final(x)
        depth = CFG.MIN_DEPTH + torch.sigmoid(out[:, 0:1]) * (CFG.MAX_DEPTH - CFG.MIN_DEPTH)
        conf = torch.sigmoid(out[:, 1:2])
        normal = F.normalize(out[:, 2:5], dim=1, eps=1e-6)
        point_residual = torch.tanh(out[:, 5:8]) * getattr(CFG, "POINT_RESIDUAL_MAX", 0.12)
        if not getattr(CFG, "USE_POINT_RESIDUAL", True):
            point_residual = point_residual * 0.0
        return depth, conf, normal, point_residual

class PoseFreeTrackDescriptorHead(nn.Module):
    """Lightweight MASt3R/VGGT-inspired dense descriptor head.

    It predicts L2-normalized descriptors on the V-JEPA feature grid. The loss
    samples them sparsely using differentiable grid_sample, so it adds matching
    supervision without storing full 512x512 descriptor maps.
    """
    def __init__(self, in_ch, desc_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, desc_dim, 1),
        )

    def forward(self, fmap):
        desc = self.net(fmap)
        return F.normalize(desc, dim=1, eps=1e-6)

class CrossJEPAViewPredictionHead(nn.Module):
    """CrossJEPA-style view-conditioned latent predictor.

    Input: source fused V-JEPA/epipolar representation and predicted relative camera/view.
    Target: cached frozen image-teacher embedding of the target frame.

    This is deliberately small and auxiliary; it does not replace depth, pose, pointmap,
    rendering, Sim(3), reprojection, or PointTrack components.
    """
    def __init__(self, feat_dim=768, pose_dim=20, hidden=512):
        super().__init__()
        self.src_norm = nn.LayerNorm(feat_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden), nn.GELU(),
            nn.Linear(hidden, feat_dim), nn.LayerNorm(feat_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )

    def _pair_condition(self, poses_c2w, intrinsics_pred):
        B, K = poses_c2w.shape[:2]
        if getattr(CFG, "CROSSJEPA_ZERO_VIEW_CONDITIONING", False):
            return torch.zeros(B, K, K, 20, device=poses_c2w.device)
        with torch.amp.autocast("cuda" if poses_c2w.device.type == "cuda" else "cpu", enabled=False):
            T = poses_c2w.float()
            T_inv = invert_se3(T.reshape(B * K, 4, 4)).reshape(B, K, 4, 4)
            # rel[b,s,t] maps target-camera coordinates into source-camera coordinates.
            rel = torch.matmul(T_inv[:, :, None], T[:, None, :, :, :])
            R = rel[..., :3, :3].reshape(B, K, K, 9)
            t = rel[..., :3, 3]
            if intrinsics_pred is not None:
                k_norm = normalize_K_params(intrinsics_pred.float(), CFG.IMAGE_SIZE)
                k_src = k_norm[:, :, None, :].expand(B, K, K, 4)
                k_tgt = k_norm[:, None, :, :].expand(B, K, K, 4)
                cond = torch.cat([R, t, k_src, k_tgt], dim=-1)
            else:
                zeros = torch.zeros(B, K, K, 8, device=T.device, dtype=T.dtype)
                cond = torch.cat([R, t, zeros], dim=-1)
        return cond

    def forward(self, fused_feats, poses_c2w, intrinsics_pred=None):
        B, K, C, h, w = fused_feats.shape
        src = fused_feats.float().mean(dim=(-1, -2))
        src = self.src_norm(src)
        src_pair = src[:, :, None, :].expand(B, K, K, C)
        cond = self._pair_condition(poses_c2w, intrinsics_pred)
        pose_emb = self.pose_mlp(cond)
        pred = self.predictor(torch.cat([src_pair, pose_emb], dim=-1))
        if getattr(CFG, "CROSSJEPA_NORMALIZE_EMBEDDINGS", True):
            pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        return pred

class VariableKSceneAggregator(nn.Module):
    """VGGT-style set/sequence aggregator for variable K views.

    Inputs are dense V-JEPA/epipolar features. The module builds per-view tokens,
    adds lightweight camera conditioning, prepends learned global scene tokens,
    and returns context-enhanced per-view feature maps.
    """
    def __init__(self, feat_dim, scene_tokens=4, layers=2, heads=8, ff_mult=2):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.scene_tokens = int(scene_tokens)
        self.global_scene_tokens = nn.Parameter(torch.randn(1, self.scene_tokens, feat_dim) * 0.02)
        self.pose_mlp = nn.Sequential(
            nn.Linear(16, feat_dim), nn.GELU(), nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.LayerNorm(feat_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=int(heads),
            dim_feedforward=int(feat_dim * ff_mult),
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.out_norm = nn.LayerNorm(feat_dim)
        self.spatial_gate = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.Sigmoid())

    def _camera_condition(self, poses_c2w, intrinsics_pred):
        B, K = poses_c2w.shape[:2]
        with torch.amp.autocast("cuda" if poses_c2w.device.type == "cuda" else "cpu", enabled=False):
            T = poses_c2w.float()
            T_ref_inv = invert_se3(T[:, 0].float())
            rel = torch.matmul(T_ref_inv[:, None], T)  # B,K,4,4 relative to first view
            pose_vec = torch.cat([rel[..., :3, :3].reshape(B, K, 9), rel[..., :3, 3]], dim=-1)  # 12
            if intrinsics_pred is not None:
                k_norm = normalize_K_params(intrinsics_pred.float(), CFG.IMAGE_SIZE)
            else:
                k_norm = torch.zeros(B, K, 4, device=T.device, dtype=T.dtype)
            cond = torch.cat([pose_vec, k_norm], dim=-1)  # 16
        return cond

    def forward(self, feats, poses_c2w=None, intrinsics_pred=None, view_mask=None):
        B, K, C, h, w = feats.shape
        pooled = feats.float().mean(dim=(-1, -2))
        if poses_c2w is not None:
            cam = self.pose_mlp(self._camera_condition(poses_c2w, intrinsics_pred))
            view_tokens = pooled + cam
        else:
            view_tokens = pooled
        scene = self.global_scene_tokens.expand(B, -1, -1)
        seq = torch.cat([scene, view_tokens], dim=1)
        key_padding_mask = None
        if view_mask is not None:
            vm = view_mask.float() > 0.5
            scene_valid = torch.ones(B, self.scene_tokens, device=vm.device, dtype=torch.bool)
            valid = torch.cat([scene_valid, vm], dim=1)
            key_padding_mask = ~valid
        enc = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        scene_out = enc[:, :self.scene_tokens]
        view_out = self.out_norm(enc[:, self.scene_tokens:])
        gate = self.spatial_gate(view_out).view(B, K, C, 1, 1).to(feats.dtype)
        ctx = view_out.view(B, K, C, 1, 1).to(feats.dtype)
        out = feats + gate * ctx
        return out, scene_out, view_out

class CameraRefinementHead(nn.Module):
    """Small pose refinement after K-view aggregation."""
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fused_feats):
        B, K, C, h, w = fused_feats.shape
        pooled = fused_feats.float().mean(dim=(-1, -2))
        ref = pooled[:, 0:1].expand(-1, K, -1)
        raw = self.net(torch.cat([pooled, ref], dim=-1))
        trans = torch.tanh(raw[..., :3]) * float(getattr(CFG, "CAMERA_REFINEMENT_MAX_TRANS", 0.08))
        rot = torch.tanh(raw[..., 3:]) * float(getattr(CFG, "CAMERA_REFINEMENT_MAX_ROT_RAD", 0.04))
        delta = torch.cat([trans, rot], dim=-1)
        delta = torch.cat([torch.zeros_like(delta[:, 0:1]), delta[:, 1:]], dim=1)
        return delta

class ForegroundMaskHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, fmap, out_size):
        logits_lr = self.net(fmap.float())
        logits = F.interpolate(logits_lr, size=out_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits), logits

class VJepaPoseFreeReconSuperman(nn.Module):
    """v43 variable-K V-JEPA-Geo architecture.

    This overrides the v42 class while preserving all original heads/losses.
    It accepts images [B,K,3,H,W] for variable K and adds scene-level K-view
    reasoning before dense geometry decoding.
    """
    def __init__(self):
        super().__init__()
        self.backbone = load_vjepa_backbone()
        from .spatial_camera import SpatialCameraHead
        self.camera_head = SpatialCameraHead(CFG.FEAT_DIM, RUN_CONFIG['camera_dim'], RUN_CONFIG['camera_grid'])
        self.edge = EdgeEncoder()
        self.epi = EpipolarCostFusion(CFG.FEAT_DIM, CFG.DEPTH_BINS)
        self.mv_agg = VariableKSceneAggregator(
            CFG.FEAT_DIM,
            scene_tokens=int(getattr(CFG, "V43_SCENE_TOKENS", 4)),
            layers=int(getattr(CFG, "V43_VIEW_TRANSFORMER_LAYERS", 2)),
            heads=int(getattr(CFG, "V43_VIEW_TRANSFORMER_HEADS", 8)),
            ff_mult=int(getattr(CFG, "V43_VIEW_TRANSFORMER_FF_MULT", 2)),
        ) if getattr(CFG, "USE_VARIABLE_K_SCENE_AGGREGATOR", True) else None
        self.pose_refiner = CameraRefinementHead(CFG.FEAT_DIM) if getattr(CFG, "USE_CAMERA_REFINEMENT_HEAD", True) else None
        self.decoder = PoseFreePointDecoder(CFG.FEAT_DIM)
        self.mask_head = ForegroundMaskHead(CFG.FEAT_DIM) if getattr(CFG, "USE_FOREGROUND_MASK_HEAD", True) else None
        self.gaussian_head = None  # Removed the untrained, disconnected Gaussian attribute head.
        self.track_head = PoseFreeTrackDescriptorHead(CFG.FEAT_DIM, getattr(CFG, "TRACK_DESC_DIM", 64)) if getattr(CFG, "USE_TRACK_DESCRIPTOR_HEAD", False) else None
        self.crossjepa_head = CrossJEPAViewPredictionHead(
            feat_dim=CFG.FEAT_DIM,
            pose_dim=int(getattr(CFG, "CROSSJEPA_POSE_INPUT_DIM", 20)),
            hidden=int(getattr(CFG, "CROSSJEPA_PREDICTOR_HIDDEN", 512)),
        ) if getattr(CFG, "USE_CROSSJEPA_AUX", False) else None
        freeze_all(self.backbone)
        self.backbone.eval()
        self.backbone_stage = "frozen"
        self.current_stage = "stage1_frozen"

    def set_stage(self, stage):
        self.current_stage = stage
        if stage in ["stage1_frozen", "stage3_align_refine", "eval"]:
            set_backbone_trainability(self.backbone, "frozen")
            self.backbone.eval()
            self.backbone_stage = "frozen"
        elif stage == "stage2_finetune":
            old_n = CFG.UNFREEZE_LAST_N_BLOCKS if hasattr(CFG, "UNFREEZE_LAST_N_BLOCKS") else CFG.POSEFREE_UNFREEZE_LAST_N_BLOCKS
            CFG.UNFREEZE_LAST_N_BLOCKS = CFG.POSEFREE_UNFREEZE_LAST_N_BLOCKS
            set_backbone_trainability(self.backbone, "finetune")
            CFG.UNFREEZE_LAST_N_BLOCKS = old_n
            self.backbone.train()
            self.backbone_stage = "finetune"
        else:
            raise ValueError(stage)

    def extract_feats(self, images):
        B, K, C, H, W = images.shape
        x = images.reshape(B * K, C, H, W)
        x_in = make_vjepa_input(x)
        grad_enabled = self.backbone_stage == "finetune"
        with torch.set_grad_enabled(grad_enabled and torch.is_grad_enabled()):
            out = torch_checkpoint(self.backbone, x_in, use_reentrant=False) if grad_enabled and CFG.USE_BACKBONE_GRAD_CHECKPOINT else self.backbone(x_in)
        feat = select_feature_tensor(out)
        fmap = tokens_to_feature_map(feat, CFG.FEAT_DIM, CFG.IMAGE_SIZE, CFG.PATCH_SIZE)
        return fmap.reshape(B, K, CFG.FEAT_DIM, fmap.shape[-2], fmap.shape[-1])

    def forward(self, images, images_raw=None, view_valid_mask=None):
        B, K, C, H, W = images.shape
        feats = self.extract_feats(images)

        # Initial camera/intrinsics from raw V-JEPA features.
        pose_xi, poses_initial, K_pred, K_norm, view_tokens_initial, camera_matches = self.camera_head(feats)
        # Keep RGB-predicted values in all geometry paths. Only the supervised
        # normalized calibration output retains gradients into the K head.
        if RUN_CONFIG['isolate_intrinsics_gradients']:
            K_pred=K_pred.detach()

        # Global scene-token aggregation before geometry fusion/decoding.
        if self.mv_agg is not None:
            feats_global, scene_tokens, view_tokens_global = self.mv_agg(
                feats, poses_initial.float(), K_pred.float(), view_mask=view_valid_mask
            )
        else:
            feats_global, scene_tokens, view_tokens_global = feats, None, view_tokens_initial

        # Optional camera refinement after multi-view context.
        if self.pose_refiner is not None:
            delta_xi = self.pose_refiner(feats_global)
            delta_T = se3_exp(delta_xi.reshape(-1, 6)).reshape(B, K, 4, 4).float()
            with torch.amp.autocast(amp_device(delta_T), enabled=False):
                poses_pred = torch.matmul(poses_initial.float(), delta_T.float())
            eye = torch.eye(4, device=images.device, dtype=torch.float32).view(1, 1, 4, 4).expand(B, 1, 4, 4)
            poses_pred = torch.cat([eye, poses_pred[:, 1:]], dim=1)
        else:
            delta_xi = torch.zeros_like(pose_xi)
            poses_pred = poses_initial

        use_predicted_epipolar = (
            CFG.USE_EPIPOLAR_FUSION
            and CFG.POSEFREE_USE_PREDICTED_POSE_FOR_EPIPOLAR
            and stage_allows_predicted_pose_epipolar(getattr(self, "current_stage", "stage1_frozen"))
        )
        if use_predicted_epipolar:
            fused, depth_prior_lr = self.epi(feats_global, poses_pred.float(), K_pred.float())
        else:
            fused, depth_prior_lr = feats_global, None

        if self.crossjepa_head is not None:
            crossjepa_pred_embeddings = self.crossjepa_head(feats, poses_pred.float().detach(), K_pred.float().detach())
        else:
            crossjepa_pred_embeddings = None

        # Edge/local-detail branch.
        if getattr(CFG, "EDGE_INPUT_SOURCE", "raw_rgb") == "raw_rgb" and images_raw is not None:
            edge_images = images_raw.to(device=images.device, dtype=images.dtype)
            if getattr(CFG, "EDGE_RAW_CLAMP_01", True):
                edge_images = edge_images.clamp(0.0, 1.0)
        else:
            edge_images = images
        flat_edge_images = edge_images.reshape(B * K, C, H, W)

        # v43 OOM-safe edge branch: the dense edge pyramid is useful, but at
        # K=5-10 and 512px it can dominate activation memory during stage-2
        # backbone fine-tuning.  We preserve the branch and decoder interface,
        # but optionally compute edge features at a lower spatial resolution,
        # in chunks, and with activation checkpointing.
        edge_hw = int(getattr(CFG, "EDGE_ENCODER_IMAGE_SIZE", 0) or 0)
        if edge_hw > 0 and edge_hw < min(H, W):
            flat_edge_images = F.interpolate(
                flat_edge_images.float(),
                size=(edge_hw, edge_hw),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=edge_images.dtype)

        edge_chunk = int(getattr(CFG, "EDGE_ENCODER_CHUNK_SIZE", 0) or 0)
        use_edge_ckpt = bool(self.training and getattr(CFG, "USE_EDGE_GRAD_CHECKPOINT", False))
        if edge_chunk > 0 and flat_edge_images.shape[0] > edge_chunk:
            e1s, e2s, e4s = [], [], []
            for _edge_x in flat_edge_images.split(edge_chunk, dim=0):
                if use_edge_ckpt:
                    e1, e2, e4 = torch_checkpoint(lambda z: self.edge(z), _edge_x, use_reentrant=False)
                else:
                    e1, e2, e4 = self.edge(_edge_x)
                e1s.append(e1); e2s.append(e2); e4s.append(e4)
            edge_feats = (torch.cat(e1s, dim=0), torch.cat(e2s, dim=0), torch.cat(e4s, dim=0))
        else:
            if use_edge_ckpt:
                edge_feats = torch_checkpoint(lambda z: self.edge(z), flat_edge_images, use_reentrant=False)
            else:
                edge_feats = self.edge(flat_edge_images)

        flat_fused = fused.reshape(B * K, CFG.FEAT_DIM, fused.shape[-2], fused.shape[-1])

        if bool(self.training and getattr(CFG, "USE_DECODER_GRAD_CHECKPOINT", False)):
            def _decode(ff, e1, e2, e4):
                return self.decoder(ff, (e1, e2, e4))
            depth, conf, normal, point_residual = torch_checkpoint(
                _decode,
                flat_fused,
                edge_feats[0],
                edge_feats[1],
                edge_feats[2],
                use_reentrant=False,
            )
        else:
            depth, conf, normal, point_residual = self.decoder(flat_fused, edge_feats)
        if self.mask_head is not None:
            fg_mask, fg_logits = self.mask_head(flat_fused, out_size=(H, W))
        else:
            fg_mask, fg_logits = None, None
        if self.gaussian_head is not None:
            g_opacity, g_opacity_logits, g_scale, g_quat = self.gaussian_head(flat_fused, out_size=(H, W))
        else:
            g_opacity, g_opacity_logits, g_scale, g_quat = None, None, None, None

        if self.track_head is not None:
            track_desc_lr = self.track_head(flat_fused).reshape(B, K, -1, fused.shape[-2], fused.shape[-1])
        else:
            track_desc_lr = None

        K_flat = K_pred.reshape(B * K, 3, 3)
        points_proj = camera_points_from_depth(depth.float(), K_flat)
        points_cam = points_proj + point_residual.float()
        points_cam = points_cam.reshape(B, K, 3, H, W)
        points_proj = points_proj.reshape(B, K, 3, H, W)
        point_residual = point_residual.reshape(B, K, 3, H, W)
        points_world = transform_cam_to_world_points(points_cam.float(), poses_pred.float())

        out = {
            "camera_matches": camera_matches,
            "depth": depth.reshape(B, K, 1, H, W),
            "confidence": conf.reshape(B, K, 1, H, W),
            "normal": normal.reshape(B, K, 3, H, W),
            "pointmap_cam": points_cam,
            "pointmap_projective_cam": points_proj,
            "point_residual": point_residual,
            "pointmap_world": points_world,
            "poses_initial_c2w": poses_initial,
            "poses_pred_c2w": poses_pred,
            "poses_refined_c2w": poses_pred,
            "pose_xi": pose_xi,
            "pose_delta_xi": delta_xi,
            "intrinsics_pred": K_pred,
            "intrinsics_norm_pred": K_norm,
            "view_tokens": view_tokens_global,
            "scene_tokens": scene_tokens,
            "depth_prior_lr": depth_prior_lr,
            "track_desc_lr": track_desc_lr,
            "crossjepa_pred_embeddings": crossjepa_pred_embeddings,
        }
        if fg_mask is not None:
            out["foreground_mask"] = fg_mask.reshape(B, K, 1, H, W)
            out["foreground_logits"] = fg_logits.reshape(B, K, 1, H, W)
        if g_opacity is not None:
            out["gaussian_opacity"] = g_opacity.reshape(B, K, 1, H, W)
            out["gaussian_opacity_logits"] = g_opacity_logits.reshape(B, K, 1, H, W)
            out["gaussian_scale"] = g_scale.reshape(B, K, 3, H, W)
            out["gaussian_quat"] = g_quat.reshape(B, K, 4, H, W)
        return out

def normal_to_quat_wxyz(normals_w, eps=1e-6):
    """
    Convert world-space normals to quaternions that rotate the local +z axis
    onto each normal. Quaternion convention is wxyz, matching the existing
    identity quaternion [1,0,0,0] used by gsplat calls in this notebook.

    Because Gaussian x/y scales are equal, rotation around the normal is not
    important for covariance orientation. The shortest-arc z->normal rotation
    is stable and cheaper than constructing a full tangent frame.
    """
    n = F.normalize(normals_w.float(), dim=-1, eps=eps)
    a = torch.zeros_like(n)
    a[..., 2] = 1.0  # local +z

    dot = (a * n).sum(dim=-1, keepdim=True)
    v = torch.cross(a, n, dim=-1)
    w = 1.0 + dot
    q = torch.cat([w, v], dim=-1)

    # Antipodal case: normal is close to -z, so w≈0 and the shortest-arc
    # formula degenerates. Use 180 degrees around local x: [0,1,0,0].
    anti = dot.squeeze(-1) < (-1.0 + 1e-4)
    if anti.any():
        q_anti = torch.zeros_like(q[anti])
        q_anti[:, 1] = 1.0
        q = q.clone()
        q[anti] = q_anti

    return F.normalize(q, dim=-1, eps=eps).to(normals_w.dtype)

def sample_sparse_grid(H, W, stride, max_points, device):
    yy, xx = sample_grid_indices(H, W, stride, device)
    if len(xx) > max_points:
        take = torch.linspace(0, len(xx) - 1, max_points, device=device).long()
        yy, xx = yy[take], xx[take]
    return yy, xx

def project_world_points(Pw, pose_c2w, K, H, W):
    """Pw: B,3,N world; pose_c2w: B,4,4; K:B,3,3 -> normalized grid and depth."""
    with torch.amp.autocast("cuda" if Pw.device.type == "cuda" else "cpu", enabled=False):
        Pw = Pw.float(); pose_c2w = pose_c2w.float(); K = K.float()
        B, _, N = Pw.shape
        ones = torch.ones(B, 1, N, device=Pw.device, dtype=torch.float32)
        Pw_h = torch.cat([Pw, ones], dim=1)
        Tcw = invert_se3(pose_c2w)
        Pc = torch.matmul(Tcw, Pw_h)[:, :3]
        z = Pc[:, 2].clamp_min(1e-6)
        proj = torch.matmul(K, Pc)
        u = proj[:, 0] / z
        v = proj[:, 1] / z
        # Coordinate convention for align_corners=CFG.GRID_SAMPLE_ALIGN_CORNERS grid_sample.
        denom_w = max(W - 1, 1)
        denom_h = max(H - 1, 1)
        gx = (u / denom_w) * 2 - 1
        gy = (v / denom_h) * 2 - 1
        valid = (z > CFG.MIN_DEPTH) & (z < CFG.MAX_DEPTH) & (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
        grid = torch.stack([gx, gy], dim=-1)
        return grid, z, valid, Pc

def gt_local_world_points(batch):
    B, K, _, H, W = batch["depths"].shape
    gt_cam = camera_points_from_depth(
        batch["depths"].reshape(B * K, 1, H, W),
        batch["intrinsics"].reshape(B * K, 3, 3),
    ).reshape(B, K, 3, H, W)
    gt_rel = relative_c2w_to_first(batch["poses_c2w"])
    gt_world = transform_cam_to_world_points(gt_cam, gt_rel)
    return gt_cam, gt_world, gt_rel

def sim3_align_torch(src, dst, mask=None, eps=1e-6):
    """
    AMP-safe batched Sim(3) alignment using a small per-batch Umeyama/Kabsch loop.

    src and dst must be paired B,N,3 point sets. By default, the Sim(3)
    parameters are estimated under no_grad and then applied to src, so gradients
    flow to the predicted points but not through the SVD alignment itself. This
    avoids the unstable SVD-gradient case near repeated singular values.
    """
    device_type = "cuda" if src.device.type == "cuda" else "cpu"
    grad_mode = getattr(CFG, "SIM3_ALIGNMENT_GRAD_MODE", "detach")

    with torch.amp.autocast(device_type, enabled=False):
        src_f = src.float()
        dst_f = dst.float()

        B, N, _ = src_f.shape
        aligned = []
        scales = []

        for b in range(B):
            if mask is None:
                m = torch.ones(N, device=src_f.device, dtype=torch.bool)
            else:
                m = mask[b].bool()

            if int(m.sum().detach().cpu()) < 6:
                aligned.append(src_f[b])
                scales.append(torch.tensor(1.0, device=src_f.device, dtype=torch.float32))
                continue

            X_raw = src_f[b, m]
            Y_raw = dst_f[b, m]

            finite = torch.isfinite(X_raw).all(dim=-1) & torch.isfinite(Y_raw).all(dim=-1)
            if int(finite.sum().detach().cpu()) < 6:
                aligned.append(src_f[b])
                scales.append(torch.tensor(1.0, device=src_f.device, dtype=torch.float32))
                continue

            # Estimate Sim(3) on either detached or differentiable correspondences.
            X_est = X_raw[finite]
            Y_est = Y_raw[finite]
            if grad_mode == "detach":
                X_est = X_est.detach()
                Y_est = Y_est.detach()

            mu_x = X_est.mean(dim=0, keepdim=True)
            mu_y = Y_est.mean(dim=0, keepdim=True)
            Xc = X_est - mu_x
            Yc = Y_est - mu_y
            var_x = torch.mean(torch.sum(Xc * Xc, dim=-1)).clamp_min(eps)

            C = (Yc.transpose(0, 1) @ Xc) / float(X_est.shape[0])
            C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0).float()

            try:
                U, S, Vh = torch.linalg.svd(C, full_matrices=True)
            except RuntimeError:
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(C.detach().cpu(), full_matrices=True)
                U = U_cpu.to(src_f.device)
                S = S_cpu.to(src_f.device)
                Vh = Vh_cpu.to(src_f.device)

            D = torch.eye(3, device=src_f.device, dtype=torch.float32)
            det_val = torch.det(U @ Vh)
            if float(det_val.detach().cpu()) < 0.0:
                D[-1, -1] = -1.0

            R = U @ D @ Vh
            scale = (S * torch.diag(D)).sum() / var_x
            scale = torch.clamp(scale, min=CFG.POSEFREE_SCALE_MIN, max=CFG.POSEFREE_SCALE_MAX)
            t = mu_y.squeeze(0) - scale * (R @ mu_x.squeeze(0))

            # In detached mode, keep gradients through src_f only, not through SVD.
            if grad_mode == "detach":
                R = R.detach()
                scale = scale.detach()
                t = t.detach()

            aligned_b = (scale * (R @ src_f[b].transpose(0, 1))).transpose(0, 1) + t
            aligned_b = torch.nan_to_num(aligned_b, nan=0.0, posinf=0.0, neginf=0.0)
            aligned.append(aligned_b)
            scales.append(scale.float())

        return torch.stack(aligned, dim=0), torch.stack(scales, dim=0)
