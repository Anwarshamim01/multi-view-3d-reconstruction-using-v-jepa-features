"""Fixed-support geometry supervision and a single predicted-camera render path."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from . import model as M


def zero(pred):
    return pred['depth'].float().sum()*0


def rotation_angle(a,b):
    with torch.amp.autocast(M.amp_device(a),enabled=False):
        error=b.float().transpose(-1,-2) @ a.float()
        vector=torch.stack([error[...,2,1]-error[...,1,2],error[...,0,2]-error[...,2,0],
                            error[...,1,0]-error[...,0,1]],dim=-1)*.5
        sine=torch.linalg.vector_norm(vector,dim=-1)
        cosine=(error.diagonal(dim1=-1,dim2=-2).sum(-1)-1)*.5
        return torch.atan2(sine,cosine)


def robust(x):
    return torch.sqrt(x.float().square()+1e-6)-.001


def sparse_grid(h,w,stride,maximum,device):
    y,x=torch.meshgrid(torch.arange(0,h,stride,device=device),torch.arange(0,w,stride,device=device),indexing='ij')
    y,x=y.flatten(),x.flatten()
    if len(x)>maximum:
        take=torch.linspace(0,len(x)-1,maximum,device=device).long()
        y,x=y[take],x[take]
    return y,x


def sample(image,grid,mode='bilinear'):
    return F.grid_sample(image.float(),grid.reshape(grid.shape[0],-1,1,2).float(),
                         align_corners=True,mode=mode,padding_mode='zeros').squeeze(-1)


def visible_matches(batch,gt_world,gt_relative,i,j,y,x,c):
    B,_,_,H,W=batch['images'].shape
    grid,z,inside,_=M.project_world_points(gt_world[:,i,:,y,x],gt_relative[:,j],batch['intrinsics'][:,j],H,W)
    target_d=sample(batch['depths'][:,j],grid,mode='nearest').squeeze(1)
    target_valid=sample(batch['valids'][:,j],grid,mode='nearest').squeeze(1)>.5
    visible=(target_d-z).abs() <= c['visibility_abs']+c['visibility_rel']*target_d
    mask=batch['valids'][:,i,0,y,x].bool() & target_valid & inside & visible
    return grid,mask


def normals_valid(valid):
    shape=valid.shape
    flat=valid.float().reshape(-1,1,*shape[-2:])
    # Invalid outside image, holes, and all finite-difference neighbors are excluded.
    eroded=F.conv2d(flat,torch.ones(1,1,3,3,device=flat.device),padding=1)==9
    return eroded.reshape(shape)


def sim3_loss(pred,batch,c):
    B,K,_,H,W=pred['pointmap_world'].shape
    y,x=sparse_grid(H,W,16,min(4096,c['max_fit_points']),pred['depth'].device)
    _,gt_world,_=M.gt_local_world_points(batch)
    p=pred['pointmap_world'][:,:,:,y,x].permute(0,1,3,2).reshape(B,-1,3)
    g=gt_world[:,:,:,y,x].permute(0,1,3,2).reshape(B,-1,3)
    valid=batch['valids'][:,:,0,y,x].reshape(B,-1).bool()
    aligned,scales=M.sim3_align_torch(p,g,valid)
    if not valid.any():
        return zero(pred),dict(sim3_active_fraction=0.,sim3_scale=1.)
    # Fixed valid support; confidence never erases points or multiplies reporting metrics.
    value=robust(aligned-g).sum(-1)[valid].mean()
    return value,dict(sim3_active_fraction=float(valid.float().mean().detach()),sim3_scale=float(scales.mean().detach()))


def correspondence_loss(pred,batch,c,gt_world,gt_relative):
    desc=pred.get('track_desc_lr')
    if desc is None:
        return zero(pred),zero(pred),dict(track_pairs=0)
    B,K,_,H,W=batch['images'].shape
    y,x=sparse_grid(H,W,16,256,desc.device)
    src_grid=torch.stack([2*x/max(W-1,1)-1,2*y/max(H-1,1)-1],dim=-1).float()[None].expand(B,-1,-1)
    attraction,contrast=[],[]
    for i in range(K):
        for j in range(i+1,K):
            grid,valid=visible_matches(batch,gt_world,gt_relative,i,j,y,x,c)
            di=F.normalize(sample(desc[:,i],src_grid),dim=1)
            dj=F.normalize(sample(desc[:,j],grid),dim=1)
            for b in range(B):
                ids=torch.nonzero(valid[b]).flatten()
                if ids.numel()<4:continue
                qi,qj=di[b,:,ids].t(),dj[b,:,ids].t()
                attraction.append((1-(qi*qj).sum(-1)).mean())
                logits=qi @ qj.t()/.07
                uv=grid[b,ids]
                near=torch.cdist(uv,uv)<4/max(H-1,W-1)
                near.fill_diagonal_(False)
                logits=logits.masked_fill(near,-1e4)
                labels=torch.arange(len(ids),device=desc.device)
                contrast.append((F.cross_entropy(logits,labels)+F.cross_entropy(logits.t(),labels))*.5)
    z=zero(pred)
    return (torch.stack(attraction).mean() if attraction else z,
            torch.stack(contrast).mean() if contrast else z,dict(track_pairs=len(contrast)))


def pairwise_loss(pred,batch,c,gt_world,gt_relative):
    _,K,_,H,W=batch['images'].shape
    y,x=sparse_grid(H,W,24,512,pred['depth'].device)
    consistency,cycle,rgb,depth,coverage=[],[],[],[],[]
    source_pixels=torch.stack([2*x/max(W-1,1)-1,2*y/max(H-1,1)-1],dim=-1).float()[None]
    active=0
    for i in range(K):
        for j in range(K):
            if i==j:continue
            _,eligible=visible_matches(batch,gt_world,gt_relative,i,j,y,x,c)
            if not eligible.any():continue
            grid,z,inside,_=M.project_world_points(pred['pointmap_world'][:,i,:,y,x],
                                                  pred['poses_pred_c2w'][:,j],pred['intrinsics_pred'][:,j],H,W)
            coverage.append(F.relu(grid.abs()-1)[eligible].mean())
            valid=eligible & inside
            if not valid.any():continue
            active+=1
            sampled_world=sample(pred['pointmap_world'][:,j],grid)
            # Coordinatewise robust penalty: signed coordinate cancellation is impossible.
            residual=robust(pred['pointmap_world'][:,i,:,y,x]-sampled_world).sum(1)
            if c['use_confidence_weighting']:
                confidence=pred['confidence'][:,i,0,y,x].detach().float().clamp(.1,1)
                consistency.append((residual[valid]*confidence[valid]).sum()/confidence[valid].sum())
            else:
                consistency.append(residual[valid].mean())
            back,_,_,_=M.project_world_points(sampled_world,pred['poses_pred_c2w'][:,i],pred['intrinsics_pred'][:,i],H,W)
            cycle.append(robust(back-source_pixels)[valid].mean())
            target_rgb=sample(batch['images_raw'][:,j],grid)
            rgb.append(robust(target_rgb-batch['images_raw'][:,i,:,y,x]).mean(1)[valid].mean())
            target_depth=sample(batch['depths'][:,j],grid,mode='nearest').squeeze(1)
            target_valid=sample(batch['valids'][:,j],grid,mode='nearest').squeeze(1)>.5
            use=valid & target_valid
            if use.any():
                depth.append(F.smooth_l1_loss(z[use],target_depth[use],beta=.08))
    z=zero(pred)
    mean=lambda values:torch.stack(values).mean() if values else z
    return dict(pair=mean(consistency),cycle=mean(cycle),reproj_rgb=mean(rgb),
                reproj_depth=mean(depth),projection_coverage=mean(coverage),active_pairs=active)


def gaussian_render(pred,batch,c,return_images=False):
    """Only predicted intrinsics/poses. RGB/depth and opacity use fixed GT-valid support."""
    if pred['depth'].shape[1]<2:
        return zero(pred),dict(render_available=0.,render_coverage=0.)
    from gsplat.rendering import rasterization
    rgb_losses,depth_losses,coverage_losses,coverages=[],[],[],[]
    images=None
    size=c['render_size']
    for b in range(pred['depth'].shape[0]):
        depth=pred['depth'][b,0,0].float()
        H,W=depth.shape
        y,x=sparse_grid(H,W,c['gaussian_stride'],c['max_gaussians'],depth.device)
        K=pred['intrinsics_pred'][b,0].float()
        z=depth[y,x]
        xyz=torch.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z],dim=-1)
        pose=pred['poses_pred_c2w'][b,0].float()
        means=(pose[:3,:3] @ xyz.t()).t()+pose[:3,3]
        normals=(pose[:3,:3] @ pred['normal'][b,0,:,y,x].float()).t()
        quats=M.normal_to_quat_wxyz(normals)
        scale=(z/((K[0,0]+K[1,1])*.5)*c['gaussian_stride']).clamp(.001,.2)
        scales=torch.stack([scale,scale,scale*.25],dim=-1)
        opacity=(pred['confidence'][b,0,0,y,x].detach().float().clamp(.02,.99) if c['use_confidence_weighting']
                 else torch.full_like(z,.8))
        colors=batch['images_raw'][b,0,:,y,x].float().t().contiguous()
        targetK=pred['intrinsics_pred'][b,1].float().clone()
        targetK[:2,:]*=size/H
        rendered,alpha,_=rasterization(means=means.float().contiguous(),quats=quats.float().contiguous(),
            scales=scales.float().contiguous(),opacities=opacity.contiguous(),colors=colors,
            viewmats=M.invert_se3(pred['poses_pred_c2w'][b,1])[None].contiguous(),
            Ks=targetK[None].contiguous(),width=size,height=size,render_mode='RGB+ED',distributed=False)
        rr=rendered[0,:,:,:3].permute(2,0,1)
        rd=rendered[0,:,:,3][None]
        ra=alpha[0].permute(2,0,1)
        target_rgb=F.interpolate(batch['images_raw'][b:b+1,1].float(),size=(size,size),mode='bilinear',align_corners=False)[0]
        target_depth=F.interpolate(batch['depths'][b:b+1,1].float(),size=(size,size),mode='nearest')[0]
        valid=F.interpolate(batch['valids'][b:b+1,1].float(),size=(size,size),mode='nearest')[0]>.5
        if not valid.any():
            raise ValueError('Rendering reference has no fixed valid support')
        rgb_losses.append((rr-target_rgb).abs()[valid.expand_as(rr)].mean())
        depth_losses.append((rd-target_depth).abs()[valid].mean())
        coverage_losses.append((1-ra)[valid].mean())
        coverages.append((ra[valid]>.05).float().mean())
        if return_images:
            images=dict(rgb=rr.detach().cpu(),depth=rd.detach().cpu(),alpha=ra.detach().cpu(),target=target_rgb.detach().cpu())
    lr,ld,lc=map(lambda v:torch.stack(v).mean(),[rgb_losses,depth_losses,coverage_losses])
    cov=torch.stack(coverages).mean()
    parts=dict(render_rgb=float(lr.detach()),render_depth=float(ld.detach()),render_coverage=float(cov.detach()),
               render_coverage_loss=float(lc.detach()),render_available=1.,render_failed=float(cov.detach()<.01))
    if return_images:parts['images']=images
    return .05*lr+.05*ld+.01*lc,parts


def camera_match_loss(pred,batch,c,world,relative):
    """Bidirectional coarse correspondence CE on visible GT geometry only.

    Labels supervise matching during training. They never condition forward,
    checkpoint selection or inference-time camera predictions.
    """
    logits=pred.get('camera_matches')
    if logits is None:return zero(pred),dict(camera_match_count=0,camera_match_accuracy=0.)
    B,_,_,H,W=batch['images'].shape
    g=c['camera_grid']
    ys=((torch.arange(g,device=logits.device)+.5)*H/g-.5).round().long().clamp(0,H-1)
    xs=((torch.arange(g,device=logits.device)+.5)*W/g-.5).round().long().clamp(0,W-1)
    y,x=torch.meshgrid(ys,xs,indexing='ij');y,x=y.flatten(),x.flatten()
    losses=[];correct=0;count=0
    for j in range(1,logits.shape[1]+1):
        for i,t,score in [(0,j,logits[:,j-1]),(j,0,logits[:,j-1].transpose(-1,-2))]:
            grid,eligible=visible_matches(batch,world,relative,i,t,y,x,c)
            u=(grid[...,0]+1)*(W-1)/2
            v=(grid[...,1]+1)*(H-1)/2
            labels=(((v+.5)*g/H).floor().long().clamp(0,g-1)*g+
                    ((u+.5)*g/W).floor().long().clamp(0,g-1))
            if eligible.any():
                chosen=score.float()[eligible];target=labels[eligible]
                losses.append(F.cross_entropy(chosen,target,label_smoothing=.05))
                correct+=int((chosen.argmax(-1)==target).sum().detach())
                count+=len(target)
    loss=torch.stack(losses).mean() if losses else logits.float().sum()*0
    return loss,dict(camera_match_count=count,camera_match_accuracy=correct/max(count,1))


def relative_depth_terms(depth,reference,valid):
    """Dimensionless terms stay unchanged under a joint scene-unit rescaling."""
    relative=(depth[valid]-reference[valid])/reference[valid].clamp_min(1e-6)
    log=depth[valid].clamp_min(1e-6).log()-reference[valid].clamp_min(1e-6).log()
    return dict(depth_relative=robust(relative).mean(),depth_log=F.smooth_l1_loss(log,torch.zeros_like(log),beta=.1))


def confidence_loss(confidence,depth,reference,valid,c):
    target=((depth.detach()-reference).abs()/reference.clamp_min(1e-6)<=c['confidence_relative_tolerance']).float()
    probability=confidence.float().clamp(1e-4,1-1e-4)
    loss=F.binary_cross_entropy(probability[valid],target[valid])
    # Background holes cannot dominate the valid-geometry confidence task.
    if (~valid).any():
        loss=loss+c['confidence_invalid_weight']*F.binary_cross_entropy(probability[~valid],torch.zeros_like(probability[~valid]))
    return loss


def glvp_terms(predicted,target,c):
    B,K,_,C=predicted.shape
    keep=~torch.eye(K,dtype=torch.bool,device=predicted.device)
    if K<2:
        return predicted.sum()*0,dict(glvp_cosine=0.,glvp_shuffled_cosine=0.,glvp_retrieval_accuracy=0.)
    target=F.normalize(target.float(),dim=-1)
    predicted=F.normalize(predicted.float(),dim=-1)
    cosine=(predicted*target[:,None]).sum(-1)
    logits=torch.einsum('bijc,blc->bijl',predicted,target)/c['glvp_temperature']
    labels=torch.arange(K,device=target.device)[None,None].expand(B,K,-1)
    retrieval=F.cross_entropy(logits[:,keep].reshape(-1,K),labels[:,keep].reshape(-1))
    value=(1-cosine[:,keep]).mean()+c['glvp_contrastive_weight']*retrieval
    return value,dict(glvp_cosine=float(cosine[:,keep].mean().detach()),
        glvp_shuffled_cosine=float((predicted*target.roll(1,1)[:,None]).sum(-1)[:,keep].mean().detach()),
        glvp_retrieval_accuracy=float((logits.argmax(-1)[:,keep]==labels[:,keep]).float().mean().detach()),
        glvp_contrastive_loss=float(retrieval.detach()),glvp_chance_accuracy=1./K)


def supervised_loss(pred,batch,c,stage):
    with torch.amp.autocast(M.amp_device(pred['depth']),enabled=False):
        valid=batch['valids'].bool()
        if not valid.any():raise ValueError('No valid geometry in training window')
        depth,gt=pred['depth'].float(),batch['depths'].float()
        log=(depth[valid].clamp_min(1e-5).log()-gt[valid].log())
        terms=dict(depth=F.smooth_l1_loss(depth[valid],gt[valid]),
                   silog=log.square().mean()-.85*log.mean().square())
        terms.update(relative_depth_terms(depth,gt,valid))
        terms['confidence']=confidence_loss(pred['confidence'],depth,gt,valid,c)
        target_normals=torch.stack([M.depth_to_normals(gt[:,k],batch['intrinsics'][:,k]) for k in range(gt.shape[1])],dim=1)
        vn=normals_valid(valid)
        error=1-(pred['normal'].float()*target_normals).sum(2,keepdim=True)
        terms['normal']=error[vn].mean() if vn.any() else zero(pred)
        terms['smooth']=M.edge_aware_smoothness(depth.flatten(0,1),batch['images_raw'].flatten(0,1))
        cam,world,relative=M.gt_local_world_points(batch)
        nk=pred['intrinsics_norm_pred'].float();target_k=M.normalize_K_params(batch['intrinsics'])
        terms['intrinsics']=(F.smooth_l1_loss(nk[...,:2].clamp_min(1e-6).log(),target_k[...,:2].clamp_min(1e-6).log(),beta=.1)
                             +F.smooth_l1_loss(nk[...,2:],target_k[...,2:],beta=.05))
        pose_scale=relative[:,1:,:3,3].square().sum(-1).mean(-1).sqrt() if relative.shape[1]>1 else relative.new_ones(relative.shape[0])
        pose_scale=pose_scale.clamp_min(max(M.CFG.DEPTH_NEAR*.01,1e-4))[:,None,None]
        terms['pose_t']=F.smooth_l1_loss(pred['poses_pred_c2w'][...,:3,3]/pose_scale,relative[...,:3,3]/pose_scale)
        terms['pose_r']=rotation_angle(pred['poses_pred_c2w'][...,:3,:3],relative[...,:3,:3]).mean()
        initial_t=F.smooth_l1_loss(pred['poses_initial_c2w'][...,:3,3]/pose_scale,relative[...,:3,3]/pose_scale)
        initial_r=rotation_angle(pred['poses_initial_c2w'][...,:3,:3],relative[...,:3,:3]).mean()
        terms['initial_pose']=initial_t+initial_r
        terms['refinement_guard']=F.relu(terms['pose_t']+terms['pose_r']-terms['initial_pose'].detach())
        graph=[]
        for i in range(depth.shape[1]):
            for j in range(i+1,depth.shape[1]):
                pp=M.invert_se3(pred['poses_pred_c2w'][:,i]) @ pred['poses_pred_c2w'][:,j]
                gg=M.invert_se3(relative[:,i]) @ relative[:,j]
                graph.append(F.smooth_l1_loss(pp[:,:3,3],gg[:,:3,3])+.5*rotation_angle(pp[:,:3,:3],gg[:,:3,:3]).mean())
        terms['pose_graph']=torch.stack(graph).mean() if graph else zero(pred)
        valid3=valid.expand_as(cam)
        terms['pointmap']=F.smooth_l1_loss(pred['pointmap_cam'][valid3],cam[valid3])
        terms['residual']=pred['point_residual'].float()[valid3].abs().mean()
        terms['sim3'],diagnostics=sim3_loss(pred,batch,c)
        terms['camera_match'],match_diagnostics=camera_match_loss(pred,batch,c,world,relative)
        diagnostics.update(match_diagnostics)
        if c['use_pairwise'] or c['use_reprojection']:
            pairs=pairwise_loss(pred,batch,c,world,relative)
        else:pairs={k:zero(pred) for k in ['pair','cycle','reproj_rgb','reproj_depth','projection_coverage']};pairs['active_pairs']=0
        terms.update({k:v for k,v in pairs.items() if torch.is_tensor(v)})
        diagnostics['active_pairs']=pairs['active_pairs']
        baseline=torch.linalg.vector_norm(pred['poses_pred_c2w'][...,:3,3],dim=-1)
        baseline_gt=torch.linalg.vector_norm(relative[...,:3,3],dim=-1)
        terms['scale']=F.smooth_l1_loss(baseline,baseline_gt)
        terms['track'],terms['infonce'],track_diag=correspondence_loss(pred,batch,c,world,relative)
        diagnostics.update(track_diag)
        if pred.get('foreground_mask') is not None:
            mask=batch['masks'].float(); probability=pred['foreground_mask'].float()
            bce=F.binary_cross_entropy_with_logits(pred['foreground_logits'].float(),mask)
            dice=1-(2*(probability*mask).sum()+1e-6)/(probability.sum()+mask.sum()+1e-6)
            terms['mask']=bce+dice
        else:terms['mask']=zero(pred)
        if c['glvp_weight']>0:
            target=F.normalize(batch['teacher'].float(),dim=-1)
            predicted=pred['crossjepa_pred_embeddings'].float()
            terms['glvp'],glvp_diagnostics=glvp_terms(predicted,target,c)
            diagnostics.update(glvp_diagnostics)
        else:terms['glvp']=zero(pred)
        if c['use_render'] and depth.shape[1]>1:
            terms['render'],rd=gaussian_render(pred,batch,c);diagnostics.update(rd)
        else:terms['render']=zero(pred)
        alignment=1.4 if stage==2 else 1.
        weights=dict(camera_match=c['camera_match_weight'] if c['use_camera_loss'] else 0.,
            depth=c['depth_absolute_weight'],depth_relative=c['depth_relative_weight'],depth_log=c['depth_log_weight'],
            silog=.1,confidence=.07 if c['use_confidence_loss'] else 0.,
            normal=.06,smooth=.04,intrinsics=c['intrinsics_loss_weight'] if c['use_camera_loss'] else 0.,
            pose_t=c['pose_normalized_weight'] if c['use_camera_loss'] else 0.,pose_r=.06 if c['use_camera_loss'] else 0.,
            initial_pose=c['initial_pose_weight'] if c['use_camera_loss'] else 0.,
            refinement_guard=c['refinement_guard_weight'] if c['use_camera_loss'] else 0.,
            pose_graph=alignment*.1 if c['use_camera_loss'] else 0.,
            pointmap=.48 if c['use_pointmap'] else 0.,residual=.025,
            sim3=alignment*.48 if c['use_sim3'] else 0.,
            pair=alignment*.32 if c['use_pairwise'] else 0.,cycle=alignment*.16 if c['use_pairwise'] else 0.,
            reproj_rgb=.035 if c['use_reprojection'] else 0.,reproj_depth=.055 if c['use_reprojection'] else 0.,
            projection_coverage=.01 if c['use_pairwise'] or c['use_reprojection'] else 0.,
            scale=.08 if c['use_scale_loss'] else 0.,track=.08,infonce=.02,mask=.06,
            glvp=c['glvp_weight']*[1.,.75,.5][stage],render=1.)
        total=sum(weights[k]*v for k,v in terms.items())
        stats={k:float(v.detach()) for k,v in terms.items()}
        stats.update(diagnostics);stats['loss']=float(total.detach())
        return total,stats


class TrainingStep(nn.Module):
    """DDP sees the scalar loss, so unused-head discovery follows the real backward graph."""
    def __init__(self,network,config,stage):
        super().__init__();self.network=network;self.config=config;self.stage=stage
    def forward(self,batch):
        prediction=self.network(batch['images'],batch['images_raw'])
        return supervised_loss(prediction,batch,self.config,self.stage)
