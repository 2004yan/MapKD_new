import os
import numpy as np
import sys
import logging
import time
from tensorboardX import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from loss import SimpleLoss, DiscriminativeLoss
from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES, NUM_CLASSES_BD
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from eval import onehot_encoding, eval_iou

import warnings
warnings.filterwarnings("ignore")

import tqdm
import pdb
from PIL import Image
from model import get_model

from collections import OrderedDict
import torch.nn.functional as F
from sklearn import metrics
# Patch提取函数
def to_2d_patches(bev_features, patch_size=25):
    B, C, H, W = bev_features.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H/W必须能被patch_size整除"
    
    patches = bev_features.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size * patch_size)
    patches = patches.permute(0, 2, 1, 3).contiguous()  # [B, num_patches, C, patch_area]
    patch_seq = patches.view(B, -1, C * patch_size * patch_size)  # [B, N, D]
    return patch_seq

# 注意力计算
def compute_cross_modal_attention(features):
    return F.softmax(torch.matmul(features, features.transpose(1, 2)), dim=-1)  # [B, N, N]

# 蒸馏损失模块
class CrossDistillationLoss(nn.Module):
    def __init__(self, bev_channels=192, patch_size=25, embed_dim=512, temperature=1.0):
        super(CrossDistillationLoss, self).__init__()
        self.patch_size = patch_size
        self.temperature = temperature
        self.embed_dim = embed_dim
        device= torch.device(f'cuda:2')
        self.token_proj = nn.Linear(bev_channels, embed_dim).to(device)  # GAP token: C -> D
        self.patch_proj = nn.Linear(bev_channels * patch_size * patch_size, embed_dim).to(device)  # Patch: (C*patch_area) -> D

    def forward(self, student_camera_bev, teacher_fused_bev):
        B, C, H, W = student_camera_bev.shape
        device = student_camera_bev.device 
        # 1. 提取 patch 序列
        student_patches = to_2d_patches(student_camera_bev, patch_size=self.patch_size)
        teacher_patches = to_2d_patches(teacher_fused_bev, patch_size=self.patch_size)

        # 2. 投影 patch 到 embed_dim
        student_patches = self.patch_proj(student_patches).to(device)  # [B, N, D]
        teacher_patches = self.patch_proj(teacher_patches).to(device) 

        # 3. 提取 GAP token 并映射
        student_token = student_camera_bev.mean(dim=[2, 3])      # [B, C]
        teacher_token = teacher_fused_bev.mean(dim=[2, 3])       # [B, C]
        student_token = self.token_proj(student_token).unsqueeze(1).to(device)   # [B, 1, D]
        teacher_token = self.token_proj(teacher_token).unsqueeze(1).to(device) 

        # 4. 拼接 token + patch
        student_all = torch.cat([student_token, student_patches], dim=1)  # [B, 1+N, D]
        teacher_all = torch.cat([teacher_token, teacher_patches], dim=1)

        # 5. 计算跨模态注意力
        student_attn = compute_cross_modal_attention(student_all)
        teacher_attn = compute_cross_modal_attention(teacher_all)

        # 6. attention 蒸馏损失（KL）
        attn_loss = F.kl_div(
            F.log_softmax(student_attn / self.temperature, dim=-1),
            F.softmax(teacher_attn / self.temperature, dim=-1),
            reduction='batchmean'
        )

        # 7. 特征蒸馏损失（原始 BEV 空间）
        feat_loss = F.mse_loss(student_camera_bev, teacher_fused_bev)

        # 8. 总损失
        total_loss = attn_loss + feat_loss
        return total_loss
def distillation_loss_with_mask(student_output, teacher_output, ground_truth, alpha=0.5, temperature=1.0):
    tmp = ground_truth.permute(0, 2, 3, 1)
    mask = (tmp[:, :, :, 0] == 1)
    non_mask = ~mask 
    
    student_output_temp = student_output.permute(0, 2, 3, 1)
    teacher_output_temp = teacher_output.permute(0, 2, 3, 1)
    student_true = student_output_temp[non_mask]
    teacher_True = teacher_output_temp[non_mask]
    
    # 计算学生和教师输出的 softmax 概率分布
    teacher_probs = torch.sigmoid(teacher_True)
    loss1 = F.binary_cross_entropy_with_logits(student_true, teacher_probs)
    
    # 最终损失，结合蒸馏损失和学生损失
    return loss1

def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)
    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)

def train(args):
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    logname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    logging.basicConfig(filename=os.path.join(args.logdir, logname + '.log'),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    writer = SummaryWriter(logdir=args.logdir)

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'num_channels_bd': NUM_CLASSES_BD + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'patch_w': args.patch_w, 
        'patch_h': args.patch_h, 
        'mask_ratio': args.mask_ratio,
        'sd_thickness': args.sd_thickness,
        'mask_flag': args.mask_flag,
        'convbd': args.convbd,
        'is_onlybd': args.is_onlybd,
    }

    train_loader, val_loader = semantic_dataset(args.version, args.data_val, args.dataroot, data_conf, args.batch_size, args.nworkers)
    teacher_model = get_model(args.teacher_cfg, args, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    coach_model = get_model(args.coach_cfg, args, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)    

    # 设置默认的 GPU 设备为 cuda:2
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 确保模型在正确的设备上
    teacher_model.to(device)
    coach_model.to(device)

    if args.teacher_weight_path:
        logger.info(f"Loading teacher model weights from {args.teacher_weight_path}")
        teacher_state_dict = torch.load(args.teacher_weight_path, map_location=device)
        new_teacher_state_dict = OrderedDict()
        for k, v in teacher_state_dict.items():
            name = k[7:]   # Remove DataParallel prefix
            new_teacher_state_dict[name] = v
        teacher_model.load_state_dict(new_teacher_state_dict, strict=False)
        logger.info("Teacher model weights loaded successfully")

    for param in teacher_model.parameters():
        param.requires_grad = False

    # 确保优化器和损失函数在正确的设备上
    optimizer_coach = torch.optim.Adam(filter(lambda p: p.requires_grad, coach_model.parameters()), lr=args.lr)
    sched_coach = StepLR(optimizer_coach, step_size=args.steplr, gamma=0.1)

    teacher_model.eval()

    loss_fn = SimpleLoss(args.pos_weight).to(device)
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).to(device)
    direction_loss_fn = torch.nn.BCELoss(reduction='none').to(device)

    counter = 0
    last_idx = len(train_loader) - 1

    for epoch in range(args.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
            yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, osm_masks, osm_vectors, masks_bd_osm, mask_bd, timestamp, scene_id) in enumerate(train_loader):
            t0 = time.time()

            # 确保所有输入数据都在正确的设备上
            imgs = imgs.to(device)
            trans = trans.to(device)
            rots = rots.to(device)
            intrins = intrins.to(device)
            post_trans = post_trans.to(device)
            post_rots = post_rots.to(device)
            lidar_data = lidar_data.to(device)
            lidar_mask = lidar_mask.to(device)
            car_trans = car_trans.to(device)
            yaw_pitch_roll = yaw_pitch_roll.to(device)
            semantic_gt = semantic_gt.to(device)
            osm_masks = osm_masks.float().to(device)

            # 前向传播
            (teacher_semantic, teacher_embedding, teacher_direction), teacher_lidar_feature, teacher_bev1, teacher_fusion = teacher_model(
                imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_masks)

            (coach_semantic, coach_embedding, coach_direction), coach_camera, coach_lidar_feature, coach_bev1, coach_fusion = coach_model(
                imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_masks)

            distill_cross_fusion = CrossDistillationLoss()
            optimizer_coach.zero_grad() 
            student_bevdistill1 = distill_cross_fusion(coach_bev1, teacher_bev1)
            student_segdistill1 = distillation_loss_with_mask(coach_semantic, teacher_semantic, semantic_gt, alpha=0.5, temperature=1.0)

            semantic_pred = coach_semantic  # 使用学生模型的输出
            semantic_gt = semantic_gt.float()

            seg_loss = loss_fn(coach_semantic, semantic_gt)
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(coach_embedding, instance_gt.to(device))
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.to(device)
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(coach_direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(coach_direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = 1 * (seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction) + 0.8 * student_bevdistill1 + 0.8 * student_segdistill1
            final_loss.backward()

            torch.nn.utils.clip_grad_norm_(coach_model.parameters(), args.max_grad_norm)
            optimizer_coach.step()

            # 计算损失
            iou_intersects, iou_union = get_batch_iou(onehot_encoding(semantic_pred), semantic_gt)
            iou = iou_intersects / (iou_union + 1e-7)

            counter += 1
            t1 = time.time()

            if counter % 100 == 0:
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1 - t0:>7.4f}    "
                            f"Loss bev: {student_bevdistill1.item():>7.4f}    "
                            f"Loss seg: {student_segdistill1.item():>7.4f}    "
                            f"Final Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].cpu().numpy(), precision=3, floatmode='fixed')}")

                # 记录到TensorBoard
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/final_loss', final_loss.item(), counter)
                writer.add_scalar('train/iou_mean', iou[1:].mean().item(), counter)
                write_log(writer, iou, 'train', counter)

        # 保存模型
        model_save_path_coach = os.path.join(args.logdir, f"coach_model_epoch_{epoch}.pt")
        torch.save(coach_model.state_dict(), model_save_path_coach)
        logger.info(f"Coach model saved to {model_save_path_coach}")      

        # 验证
        iou = eval_iou(coach_model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
        write_log(writer, iou, 'eval', counter)
        
        coach_model.train()

        sched_coach.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./output/60*30-new-C_dis')
    # nuScenes config
    parser.add_argument('--dataset', type=str, default='/app/dataset/nuScenes/') 
    parser.add_argument('--dataroot', type=str, default='/app/dataset/nuScenes/') 
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--data_val', type=str, default='trainval', choices=['vis', 'trainval'])
    # model config
    parser.add_argument("--teacher_weight_path", type=str, default="/app/P-MapNet/new_60/P-MapNet-60m/output/hd60*30-T/model9.pt")
    parser.add_argument("--teacher_cfg", type=str, default='PMapNet_mae_head')
    parser.add_argument("--coach_cfg", type=str, default='PMapNet_sdmap_C')
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--convbd", action='store_true', help='conv the bd')
    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nworkers", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steplr", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)
    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--sd_thickness", type=int, default=5)
    parser.add_argument('--gpus', type=int, nargs='+', default=[2])
    parser.add_argument("--patch_w", type=int, default=2)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument('--mask_flag', action='store_true')
    # embedding config
    parser.add_argument('--instance_seg', action='store_false')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)
    # direction config
    parser.add_argument('--direction_pred', action='store_false')
    parser.add_argument('--angle_class', type=int, default=36)
    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=0.1)
    parser.add_argument("--scale_dist", type=float, default=0.1)
    parser.add_argument("--scale_direction", type=float, default=0.1)
    parser.add_argument('--is_onlybd', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'config.txt'), 'w') as f:
        argsDict = args.__dict__
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " : " + str(value) + "\n")
    train(args)