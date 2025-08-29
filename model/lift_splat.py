"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn

from data_osm.utils import gen_dx_bx
from .base import CamEncode_L, BevEncode


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class LiftSplat(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, instance_seg, embedded_dim):
        super(LiftSplat, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.reduce_to_3 = nn.Conv2d(64, 3, kernel_size=1)
        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        # D x H/downsample x D/downsample x 3
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode_L(self.D,self.camC,self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = False

    def create_frustum(self):
        # 原始图片大小， ogfH：128    ogfW：352
        ogfH, ogfW = self.data_aug_conf['final_dim']
 
        # 下采样16倍后图像大小，fH: 8   fW: 22
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
 
        # self.grid_conf['dbound'] = [4, 45, 1]
        # 在深度方向上划分网格，ds：D×fH×fW (41*8*22)
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        
        D, _, _ = ds.shape   # D:41  表示深度方向上网格的数量
        # 在0到351上划分22个格子  xs：D×fH×fW (41×8×22)
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        
        # 在0到127上划分8个格子  ys：D×fH×fW (41×8×22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
 
        # D x H x W x 3
        # 堆积起来形成网格坐标，frustum[i,j,k,0]就是(i,j)的位置
        # 深度为k的像素的宽度方向上的栅格坐标frustum：D×fH×fW×3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        B, N, _ = trans.shape

    # undo post-transformation
    # B x N x D x H x W x 3
    # 首先抵消因预处理带来的旋转和平移
  # undo post-transformation
    # B x N x D x H x W x 3
# **确保所有输入张量都在同一设备上**
        device = post_rots.device

# **先在 CPU 计算 `post_trans` 的变换**
        points = (self.frustum - post_trans.view(B, N, 1, 1, 1, 3)).cpu()

# **在 CPU 计算 `post_rots` 的逆矩阵**
        post_rots_inv = torch.inverse(post_rots.cpu()).view(B, N, 1, 1, 1, 3, 3)

# **在 CPU 进行第一个矩阵乘法** 
        points = post_rots_inv.matmul(points.unsqueeze(-1))

# **在 CPU 计算 `torch.cat()`**
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                    points[:, :, :, :, :, 2:3]
                   ), 5)

# **在 CPU 计算 `intrins` 的逆矩阵**
        intrins_inv = torch.inverse(intrins.cpu())

# **在 CPU 计算 `combine = rots.matmul(torch.inverse(intrins))`**
        combine = rots.cpu().matmul(intrins_inv)

# **在 CPU 进行 `combine` 变换**
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

# **在 CPU 计算 `trans` 平移**
        points += trans.view(B, N, 1, 1, 1, 3).cpu()

# **最后再移动回 GPU**
        points = points.to(device)


        return points


    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x
    # def cumsum_trick(x, geom_feats, ranks):
    #     x = x.cumsum(0)
    #     kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    #     kept[:-1] = (ranks[1:] != ranks[:-1])

    #     x, geom_feats = x[kept], geom_feats[kept]
    #     x = torch.cat((x[:1], x[1:] - x[:-1]))

    #     return x, geom_feats

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)：在ego坐标系下的坐标点；
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)：图像点云特征
 
        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 8  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime: 173184
 
        # flatten x
        x = x.reshape(Nprime, C)   # 将特征点云展平，一共有 B*N*D*H*W 个点
 
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将ego下的空间坐标[-50,50] [-10 10]的范围平移转换到体素坐标[0,100] [0,20]，计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将体素坐标同样展平  geom_feats: B*N*D*H*W x 3 (173184 x 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4(173184 x 4), geom_feats[:,3]表示batch_id
 
        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]  # x: 168648 x 64
        geom_feats = geom_feats[kept]
 
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648
 
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        # else:
        #     x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4
 
        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x 1 x 200 x 200
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中
 
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维
 
        return final  # final: 4 x 64 x 200 x 200
 

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # B x N x D x H/downsample x W/downsample x 3: (x,y,z) locations (in the ego frame)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # B x N x D x H/downsample x W/downsample x C: cam feats
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self,x, rots, trans, intrins, post_rots, post_trans,yaw_pitch_roll):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        # x = self.bevencode(x)
        return x
