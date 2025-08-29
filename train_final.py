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
from eval_C import onehot_encoding, eval_iou

import warnings
warnings.filterwarnings("ignore")

import tqdm
import pdb
from PIL import Image
from model import get_model

from collections import OrderedDict
import torch.nn.functional as F
from sklearn import metrics

# # Define the 2D Patch conversion process
# def to_2d_patches(bev_features, patch_size=25):
#     batch_size, channels, height, width = bev_features.shape
#     assert height % patch_size == 0 and width % patch_size == 0, "Height and width must be divisible by patch size."

#     # Split the feature map into patches
#     patches = bev_features.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#     patches = patches.contiguous().view(batch_size, channels, -1, patch_size * patch_size)
#     patches = patches.permute(0, 2, 1, 3).contiguous()

#     return patches.view(batch_size, -1, patch_size * patch_size * channels)

# # Compute cross-modal attention (teacher input is fused BEV features)
# def compute_cross_modal_attention(teacher_bev):
#     attention = F.softmax(torch.matmul(teacher_bev, teacher_bev.transpose(1, 2)), dim=-1)
#     return attention

# class UncertaintyWeightingLoss(nn.Module):
#     def __init__(self, num_losses):
#         super(UncertaintyWeightingLoss, self).__init__()
#         self.log_vars = nn.Parameter(torch.zeros(num_losses))

#     def forward(self, losses):
#         total_loss = 0
#         for i in range(len(losses)):
#             precision = torch.exp(-self.log_vars[i])
#             total_loss += precision * losses[i] + self.log_vars[i]
#         return total_loss

# class CrossDistillationLoss(nn.Module):
#     def __init__(self, bev_channels=192, patch_size=25, embed_dim=512, temperature=1.0):
#         super(CrossDistillationLoss, self).__init__()
#         self.temperature = temperature
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         device= torch.device(f'cuda:0')
#         self.token_proj = nn.Linear(bev_channels, embed_dim).to(device)
#         self.patch_proj = nn.Linear(bev_channels * patch_size * patch_size, embed_dim).to(device)

#     def forward(self, student_camera_bev, coach_bev, teacher_fused_bev):
#         B, C, H, W = student_camera_bev.shape
#         device = student_camera_bev.device

#         # ====== Teacher 分支 ======
#         teacher_patches = to_2d_patches(teacher_fused_bev, patch_size=self.patch_size)
#         teacher_patches = self.patch_proj(teacher_patches).to(device)
#         teacher_token = teacher_fused_bev.mean(dim=[2, 3])
#         teacher_token = self.token_proj(teacher_token).unsqueeze(1).to(device)
#         teacher_all = torch.cat([teacher_token, teacher_patches], dim=1)

#         # ====== Coach 分支 ======
#         coach_patches = to_2d_patches(coach_bev, patch_size=self.patch_size)
#         coach_patches = self.patch_proj(coach_patches).to(device)
#         coach_token = coach_bev.mean(dim=[2, 3])
#         coach_token = self.token_proj(coach_token).unsqueeze(1).to(device)
#         coach_all = torch.cat([coach_token, coach_patches], dim=1)

#         # ====== Student 分支 ======
#         student_patches = to_2d_patches(student_camera_bev, patch_size=self.patch_size)
#         student_patches = self.patch_proj(student_patches).to(device)
#         student_token = student_camera_bev.mean(dim=[2, 3])
#         student_token = self.token_proj(student_token).unsqueeze(1).to(device)
#         student_all = torch.cat([student_token, student_patches], dim=1)

#         # ====== Attention 蒸馏 ======
#         student_attn = compute_cross_modal_attention(student_all)
#         teacher_attn = compute_cross_modal_attention(teacher_all)
#         coach_attn = compute_cross_modal_attention(coach_all)

#         attn_loss1 = F.kl_div(
#             F.log_softmax(student_attn / self.temperature, dim=-1),
#             F.softmax(teacher_attn / self.temperature, dim=-1),
#             reduction='batchmean'
#         )
#         feat_loss1 = F.mse_loss(student_camera_bev, teacher_fused_bev)
#         loss1 = attn_loss1 + 0.5*feat_loss1

#         attn_loss2 = F.kl_div(
#             F.log_softmax(student_attn / self.temperature, dim=-1),
#             F.softmax(coach_attn / self.temperature, dim=-1),
#             reduction='batchmean'
#         )
#         feat_loss2 = F.mse_loss(student_camera_bev, coach_bev)
#         loss2 = attn_loss2 + 0.5*feat_loss2

#         return loss1, loss2


# class TeacherCoachDistillationLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         super(TeacherCoachDistillationLoss, self).__init__()
#         self.temperature = temperature
        
#         self.conv_bev = nn.Sequential(
#         nn.Conv2d(128, out_channels=128, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(128, out_channels=128, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         )

#     def forward(self, coach_simlidar, teacher_lidar):
#         self.conv_bev = self.conv_bev.cuda()
#         coach_bev = self.conv_bev(coach_simlidar)
#         teacher_bev = self.conv_bev(teacher_lidar)
        
#         coach_simlidar_feat = to_2d_patches(coach_bev)
#         teacher_lidar_feat = to_2d_patches(teacher_bev) 
               
#         teacher_attention = compute_cross_modal_attention(teacher_lidar_feat)
#         coach_attention = compute_cross_modal_attention(coach_simlidar_feat)

#         attention_loss = F.kl_div(F.log_softmax(coach_attention / self.temperature, dim=-1),
#                                   F.softmax(teacher_attention / self.temperature, dim=-1), reduction='batchmean')

#         feature_loss = F.mse_loss(coach_bev, teacher_bev)
        
#         return attention_loss + feature_loss

def distillation_loss_with_mask(student_output, coach_output, teacher_output, ground_truth, alpha=0.5, temperature=1.0):
    tmp = ground_truth.permute(0, 2, 3, 1)
    mask = (tmp[:, :, :, 0] == 1)
    non_mask = ~mask 

    student_output_temp = student_output.permute(0, 2, 3, 1)
    coach_output_temp = coach_output.permute(0, 2, 3, 1)
    teacher_output_temp = teacher_output.permute(0, 2, 3, 1)
    
    student_true = student_output_temp[non_mask]
    coach_true = coach_output_temp[non_mask]
    teacher_True = teacher_output_temp[non_mask]
    
    teacher_probs = torch.sigmoid(teacher_True)
    loss1 = F.binary_cross_entropy_with_logits(student_true, teacher_probs)

    coach_probs = torch.sigmoid(coach_true)
    loss2 = F.binary_cross_entropy_with_logits(student_true, coach_probs)

    return loss1, loss2

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

    # Initialize the models
    teacher_model = get_model(args.teacher_cfg, args, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    coach_model = get_model(args.coach_cfg, args, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    student_model = get_model(args.student_cfg, args, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)

    # Set device to CUDA
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Move models to the correct device
    teacher_model.to(device)
    coach_model.to(device)
    student_model.to(device)

    # Load teacher model weights
    if args.teacher_weight_path:
        logger.info(f"Loading teacher model weights from {args.teacher_weight_path}")
        teacher_state_dict = torch.load(args.teacher_weight_path, map_location=device)
        new_teacher_state_dict = OrderedDict()
        for k, v in teacher_state_dict.items():
            name = k[7:]  # Remove DataParallel prefix
            new_teacher_state_dict[name] = v
        teacher_model.load_state_dict(new_teacher_state_dict, strict=False)
        logger.info("Teacher model weights loaded successfully")

    # Load coach model weights
    if args.coach_weight_path:
        logger.info(f"Loading coach model weights from {args.coach_weight_path}")
        coach_state_dict = torch.load(args.coach_weight_path, map_location=device)
        new_coach_state_dict = OrderedDict()
        for k, v in coach_state_dict.items():
            name = k[7:]  # Remove DataParallel prefix
            new_coach_state_dict[name] = v
        coach_model.load_state_dict(new_coach_state_dict, strict=False)
        logger.info("Coach model weights loaded successfully")

    # Freeze teacher and coach model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in coach_model.parameters():
        param.requires_grad = False

    # Set up optimizer for student model
    optimizer_student = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    sched_student = StepLR(optimizer_student, step_size=args.steplr, gamma=0.1)

    teacher_model.eval()

    # Loss functions
    loss_fn = SimpleLoss(args.pos_weight).to(device)
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).to(device)
    direction_loss_fn = torch.nn.BCELoss(reduction='none').to(device)

    counter = 0
    last_idx = len(train_loader) - 1

    for epoch in range(args.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
            yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, osm_masks, osm_vectors, masks_bd_osm, mask_bd, timestamp, scene_id) in enumerate(train_loader):
            t0 = time.time()

            # Move input data to correct device
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

            # Forward pass for teacher, coach, and student models
            (teacher_semantic, teacher_embedding, teacher_direction), teacher_lidar_feature, teacher_bev1, teacher_fusion = teacher_model(
                imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_masks)

            (coach_semantic, coach_embedding, coach_direction), coach_camera, coach_lidar_feature, coach_bev1, coach_fusion = coach_model(
                imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_masks)

            (student_semantic, student_embedding, student_direction), student_camera, student_bev_feature1, student_bev_feature2 = student_model(
                imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_masks)

            # Loss calculation
            semantic_pred = student_semantic
            semantic_gt = semantic_gt.cuda().float()

            seg_loss = loss_fn(student_semantic, semantic_gt)
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(student_embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(student_direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(student_direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            # distill_cross_fusion = CrossDistillationLoss()
            optimizer_student.zero_grad()
            # student_bevdistill1, student_bevdistill2 = distill_cross_fusion(student_bev_feature1, coach_bev1, teacher_bev1)
            student_segdistill1, student_segdistill2 = distillation_loss_with_mask(student_semantic, coach_semantic, teacher_semantic, semantic_gt, alpha=0.5, temperature=1.0)

            final_loss = seg_loss * 1.0 + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction +0.75 * student_segdistill1 + 0.001* student_segdistill2
            final_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer_student.step()

            iou_intersects, iou_union = get_batch_iou(onehot_encoding(semantic_pred), semantic_gt)
            iou = iou_intersects / (iou_union + 1e-7)

            counter += 1
            t1 = time.time()

            if counter % 100 == 0:
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1 - t0:>7.4f}    "
                            
                            f"Stu Segdis tea: {student_segdistill1.item():>7.4f}    "
                            f"Stu Segdis coa: {student_segdistill2.item():>7.4f}    "
                            f"Final Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].cpu().numpy(), precision=3, floatmode='fixed')}")

                # 记录到TensorBoard
                writer.add_scalar('train/step_time', t1 - t0, counter)
                
                writer.add_scalar('train/final_loss', final_loss.item(), counter)
                writer.add_scalar('train/iou_mean', iou[1:].mean().item(), counter)
                write_log(writer, iou, 'train', counter)

       

        # Save student model after each epoch
        model_save_path_student = os.path.join(args.logdir, f"student_model_epoch_{epoch}.pt")
        torch.save(student_model.state_dict(), model_save_path_student)
        logger.info(f"Student model saved to {model_save_path_student}")

        # Evaluate student model
        iou = eval_iou(student_model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]: IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
        write_log(writer, iou, 'eval', counter)

        student_model.train()
        sched_student.step()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./output/distill_seg')
    # nuScenes config
    parser.add_argument('--dataset', type=str, default='/app/dataset/nuScenes/') 
    parser.add_argument('--dataroot', type=str, default='/app/dataset/nuScenes/') 
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--data_val', type=str, default='trainval', choices=['vis', 'trainval'])
    # model config
    parser.add_argument("--teacher_weight_path", type=str, default="/app/P-MapNet/new_60/P-MapNet-60m/output/hd60*30-T/model9.pt")
    parser.add_argument("--coach_weight_path", type=str, default='/app/P-MapNet/new_60/P-MapNet-60m/output/hd60*30-C/model9.pt')
    parser.add_argument("--teacher_cfg", type=str, default='PMapNet_mae_head')

    
    parser.add_argument("--coach_cfg", type=str, default='PMapNet_sdmap_C')
    parser.add_argument("--student_cfg", type=str, default='PMapNet_sdmap_S')
    
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