"""Reference-to-view spatial matching for RGB-only camera estimation."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from . import model as M


def correlation(source, target, temperature=.1):
    # Normalization and softmax stay FP32 even during FP16 backbone training.
    with torch.amp.autocast(M.amp_device(source), enabled=False):
        return (F.normalize(source.float(),dim=-1) @
                F.normalize(target.float(),dim=-1).transpose(-1,-2))/temperature


class SpatialCameraHead(nn.Module):
    """Retain spatial flow, matching uncertainty and paired feature content.

    The first view anchors every camera. Nonreference views share parameters;
    their order does not encode the camera motion. No labels enter this head.
    """
    def __init__(self,ch,dim=128,grid=16):
        super().__init__()
        self.grid=grid
        self.project=nn.Conv2d(ch,dim,1)
        self.pair_tokens=nn.Sequential(nn.Linear(dim*3+7,dim),nn.GELU(),nn.LayerNorm(dim))
        layer=nn.TransformerEncoderLayer(dim,4,dim*2,dropout=0.,batch_first=True,norm_first=True)
        self.pair_transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        # Flattening preserves the fixed source-patch locations, rather than
        # making the motion descriptor invariant to arbitrary spatial shuffles.
        self.spatial=nn.Sequential(nn.Linear(grid*grid*5,256),nn.GELU(),nn.Linear(256,dim))
        self.pose_net=nn.Sequential(nn.Linear(dim*4+8,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
        self.calibration_context=M.ViewSetTransformer(ch)
        self.k_net=nn.Sequential(nn.Linear(ch*2,512),nn.GELU(),nn.Linear(512,256),nn.GELU(),nn.Linear(256,4))
        for net in [self.pose_net,self.k_net]:
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)

    def forward(self,feats):
        B,K,C,h,w=feats.shape
        pooled=feats.mean((-1,-2))
        calibration=self.calibration_context(pooled)
        camera=self.k_net(torch.cat([calibration,calibration[:,:1].expand(-1,K,-1)],-1)).float()
        focal=F.softplus(camera[...,:2]+math.log(math.expm1(.82)))+.001
        normalized=torch.cat([focal,.5+.1*camera[...,2:]],-1)
        g=self.grid
        projected=F.adaptive_avg_pool2d(self.project(feats.flatten(0,1)),(g,g))
        tokens=projected.flatten(2).transpose(1,2).reshape(B,K,g*g,-1)
        if K>1:
            source=tokens[:,:1].expand(-1,K-1,-1,-1).reshape(B*(K-1),g*g,-1)
            target=tokens[:,1:].reshape_as(source)
            logits=correlation(source,target)
            coord=(torch.arange(g,device=feats.device,dtype=torch.float32)+.5)/g
            yy,xx=torch.meshgrid(coord,coord,indexing='ij')
            xy=torch.stack([xx,yy],-1).reshape(1,g*g,2).expand(len(source),-1,-1)
            with torch.amp.autocast(M.amp_device(feats),enabled=False):
                weight=logits.softmax(-1)
                matched_xy=weight @ xy
                reverse=logits.transpose(-1,-2).softmax(-1)
                cycle_xy=weight @ (reverse @ xy)
                certainty=1-(-(weight*weight.clamp_min(1e-8).log()).sum(-1,keepdim=True)/max(math.log(g*g),1.))
                flow=matched_xy-xy
                spatial=torch.cat([flow,cycle_xy-xy,certainty],-1)
                paired=weight @ target.float()
            content=torch.cat([source,paired,paired-source,xy,matched_xy,cycle_xy,certainty],-1)
            context=self.pair_transformer(self.pair_tokens(content)).mean(1)
            location=self.spatial(spatial.flatten(1))
            global_source=source.mean(1);global_target=target.mean(1)
            # Calibration receives its explicit supervision rather than being
            # adjusted to compensate for camera/depth prediction errors.
            camera_for_pose=normalized.detach() if M.RUN_CONFIG['isolate_intrinsics_gradients'] else normalized
            nk=torch.cat([camera_for_pose[:,:1].expand(-1,K-1,-1),camera_for_pose[:,1:]],-1).flatten(0,1)
            raw=self.pose_net(torch.cat([context,location,global_source,global_target,nk],-1)).float()
            xi=torch.cat([raw.new_zeros(B,1,6),raw.reshape(B,K-1,6)],1)
            matches=logits.reshape(B,K-1,g*g,g*g)
        else:
            xi=feats.new_zeros(B,1,6,dtype=torch.float32)
            matches=None
        poses=M.se3_exp(xi.reshape(-1,6)).reshape(B,K,4,4)
        return xi,poses,M.make_K_from_normalized(normalized,M.CFG.IMAGE_SIZE),normalized,pooled,matches
