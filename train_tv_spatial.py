import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import scipy.io as sio
import hdf5storage as hdf5 
from torch.utils.tensorboard import SummaryWriter

from inr import models
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================= 从 train_pu.py 移植的评价体系 =======================
def bandwise_psnr(img_real, img_fake, data_range=1.0):
    mse_bands = np.mean((img_real - img_fake) ** 2, axis=(0, 1))
    psnr_bands = np.zeros_like(mse_bands)
    zero_mask = (mse_bands == 0)
    non_zero_mask = ~zero_mask
    psnr_bands[non_zero_mask] = 10 * np.log10((data_range ** 2) / mse_bands[non_zero_mask])
    psnr_bands[zero_mask] = 100.0
    return np.mean(psnr_bands)

def pu_ergas(img_fake, img_real, scale_factor):
    img_fake, img_real = np.clip(img_fake, 0.0, 1.0), np.clip(img_real, 0.0, 1.0)
    channels = img_real.shape[2]
    inner_sum = sum(((np.sqrt(np.mean((img_real[:, :, i] - img_fake[:, :, i]) ** 2)) / np.mean(img_real[:, :, i])) ** 2) for i in range(channels) if np.mean(img_real[:, :, i]) != 0)
    return 100 / scale_factor * np.sqrt(inner_sum / channels)

def cross_correlation(img_fake, img_real):
    channels = img_real.shape[2]
    cc_val = 0
    for i in range(channels):
        v1, v2 = img_fake[:, :, i].flatten(), img_real[:, :, i].flatten()
        v1, v2 = v1 - np.mean(v1), v2 - np.mean(v2)
        den = np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2))
        if den != 0: cc_val += np.sum(v1 * v2) / den
    return cc_val / channels

def pu_sam(img1, img2):
    img1, img2 = img1.reshape(-1, img1.shape[-1]), img2.reshape(-1, img2.shape[-1])
    cos_theta = np.clip(np.sum(img1 * img2, axis=-1) / (np.linalg.norm(img1, axis=-1) * np.linalg.norm(img2, axis=-1) + 1e-8), -1, 1)
    return np.mean(np.arccos(cos_theta)) * 180 / np.pi

def quality_assessment(S_true, Z_pred, sf):
    Z_pred, S_true = np.clip(Z_pred, 0.0, 1.0), np.clip(S_true, 0.0, 1.0)
    return (bandwise_psnr(S_true, Z_pred, data_range=1.0), 
            structural_similarity(S_true, Z_pred, channel_axis=-1, data_range=1.0), 
            pu_sam(S_true, Z_pred), pu_ergas(Z_pred, S_true, sf), 
            cross_correlation(Z_pred, S_true), np.mean(np.abs(Z_pred - S_true) * 255.0))

# ======================= 从 train_pu.py 移植的绘图工具 =======================
def matlab_style_rgb(img_3d, bands):
    rgb = img_3d[:, :, bands].copy().astype(np.float32)
    for i in range(3):
        c_min, c_max = rgb[:, :, i].min(), rgb[:, :, i].max()
        rgb[:, :, i] = (rgb[:, :, i] - c_min) / (c_max - c_min + 1e-8)
    return rgb

def save_and_plot_spectral_curve(pred_hsi, gt_hsi, center_coords, window_size=3, save_dir='./spectral_results', method_name='CLoRF'):
    os.makedirs(save_dir, exist_ok=True)
    if pred_hsi.shape[0] < pred_hsi.shape[-1]:
        pred_hsi = np.transpose(pred_hsi, (1, 2, 0))
        gt_hsi = np.transpose(gt_hsi, (1, 2, 0))
    x, y = center_coords
    r = window_size // 2
    pred_region = pred_hsi[max(0, x-r):x+r+1, max(0, y-r):y+r+1, :]
    gt_region = gt_hsi[max(0, x-r):x+r+1, max(0, y-r):y+r+1, :]
    pred_spectrum = np.mean(pred_region, axis=(0, 1))
    gt_spectrum = np.mean(gt_region, axis=(0, 1))
    
    plt.figure(figsize=(8, 6))
    plt.plot(gt_spectrum, label='Ground Truth', color='black', linewidth=2)
    plt.plot(pred_spectrum, label=method_name, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Band Index')
    plt.ylabel('Reflectance')
    plt.title(f'Spectral Curve (Center: X={x}, Y={y})')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(save_dir, f'curve_{method_name}_{x}_{y}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    mat_dict = {'GT_spectrum': gt_spectrum, f'{method_name}_spectrum': pred_spectrum}
    sio.savemat(os.path.join(save_dir, f'spectrum_data_{x}_{y}.mat'), mat_dict)
    
    data_to_save = np.column_stack((np.arange(len(gt_spectrum)), gt_spectrum, pred_spectrum))
    np.savetxt(os.path.join(save_dir, f'spectrum_data_{x}_{y}.csv'), data_to_save, 
               delimiter=',', header=f'Band,GT,{method_name}', comments='')
# ==============================================================================


def main(args):
    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    set_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(log_dir=args.log_dir)
    save_yaml(args.log_dir, vars(args), 'config.yml')
    print_info(args)

    # 1. 加载数据
    try:
        mat_hsi = sio.loadmat(args.data_path)
        mat_r = sio.loadmat(args.r_path)
    except:
        mat_hsi = hdf5.loadmat(args.data_path)
        mat_r = hdf5.loadmat(args.r_path)

    X = np.float32(mat_hsi[args.mat_key])
    R = np.float32(mat_r[args.r_key])

    # [裁剪 256x256]
    if X.shape[0] >= 256 and X.shape[1] >= 256:
        X = X[:256, :256, :]
        print(f"[!] 已裁剪为左上角 256x256，当前形状: {X.shape}")
    else:
        raise ValueError("原始图像尺寸小于 256x256，无法裁剪！")

    # 2. R 矩阵对齐与归一化
    if R.shape[0] < R.shape[1]: R = R.T
    c_hsi = X.shape[2]
    if R.shape[0] > c_hsi:
        R = R[:c_hsi, :]
    col_sums = np.sum(R, axis=0)
    col_sums[col_sums == 0] = 1.0
    R = R / col_sums
    
    if X.max() > 1.0: X = X / X.max()
    h, w, c = X.shape
    factor = args.scale
    
    # 3. 显式统一设备到 GPU
    X_torch = torch.from_numpy(X).to(device, dtype=torch.float32)
    R_torch = torch.from_numpy(R).to(device, dtype=torch.float32)
    
    # [无噪音观测数据构造]
    MSI = hwc2chw(hsi2msi(X_torch, R_torch)) 
    msi = MSI.to(device)
    
    psf = get_hs_psf(N=5, n_channels=c, gaussian=True, sigma=1.)
    x = hwc2chw(X_torch)
    hsi_LR = Ax(x, psf, ratio=factor)
    hsi = hsi_LR.to(device)

    # 4. 构建 INR 模型
    spatial_params = {'nonlin': args.inr, 'in_features': 2, 'out_features': args.k,
                      'hidden_features': args.spatial_hidden_features,
                      'hidden_layers': args.spatial_hidden_layers, 'outermost_linear': True}
    if args.pos_enc: spatial_params['pos_encode'] = args.pos_enc
    spatial_model = models.get_INR(**spatial_params).to(device)

    temporal_params = {'nonlin': args.inr, 'in_features': 1, 'out_features': args.k,
                       'hidden_features': args.temporal_hidden_features,
                       'hidden_layers': args.temporal_hidden_layers, 'outermost_linear': True}
    if args.pos_enc: temporal_params['pos_encode'] = args.pos_enc
    temporal_model = models.get_INR(**temporal_params).to(device)

    spatial_coord = get_coordinate(h, w).to(device)
    temporal_coord = get_coordinate(c).to(device)

    optimizer = torch.optim.Adam([{'params': spatial_model.parameters()},
                                  {'params': temporal_model.parameters()}],
                                 lr=args.lr, weight_decay=args.wd)

    best_psnr = -np.inf 
    t0 = time.time()
    
    print("-" * 105)
    print(f"{'Iter':<8}{'Loss':<12}{'PSNR':<10}{'SSIM':<10}{'SAM':<10}{'ERGAS':<10}{'CC':<10}{'DD':<10}")
    print("-" * 105)

    for i in range(1, args.num_iters + 1):
        spatial = spatial_model(spatial_coord) 
        temporal = temporal_model(temporal_coord)
        output = (spatial @ temporal.T).reshape(h, w, c).permute(2, 0, 1) 
        output = output / 2 + 0.5 
        
        HP = Ax(output, psf, factor)
        MP = hsi2msi(output.permute(1, 2, 0), R_torch).permute(2, 0, 1)
        
        loss = F.mse_loss(HP, hsi) + args.lam * F.mse_loss(MP, msi)
        loss += tv_loss(spatial.T.view(1, args.k, h, w), args.alpha)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_decay: adjust_learning_rate(optimizer, args.lr, i, args.num_iters)

        # ============ 替换为你专属的评价体系 ============
        if i % args.eval_freq == 0:
            out_np = output.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
            
            psnr_v, ssim_v, sam_v, ergas_v, cc_v, dd_v = quality_assessment(X, out_np, factor)

            print(f"{i:<8}{loss.item():<12.6f}{psnr_v:<10.4f}{ssim_v:<10.4f}{sam_v:<10.4f}{ergas_v:<10.4f}{cc_v:<10.4f}{dd_v:<10.5f}")

            writer.add_scalar('loss', loss.item(), i)
            writer.add_scalar('psnr', psnr_v, i)
            writer.add_scalar('ssim', ssim_v, i)
            writer.add_scalar('ergas', ergas_v, i)
            writer.add_scalar('sam', sam_v, i)

            if psnr_v > best_psnr:
                best_psnr = psnr_v
                np.save(os.path.join(args.log_dir, 'best_psnr.npy'), best_psnr)
                np.save(os.path.join(args.log_dir, 'best_psnr_output.npy'), out_np)

    print('\nTraining finished, time_cost: ', time.time() - t0)
    writer.close()

    # ============================ 训练结束后的可视化与评估环节 ============================
    print(f"\n{'='*20} 启动最终评估与绘图 (最佳模型) {'='*20}")
    best_out = np.load(os.path.join(args.log_dir, 'best_psnr_output.npy'))
    psnr_v, ssim_v, sam_v, ergas_v, cc_v, dd_v = quality_assessment(X, best_out, factor)
    
    print(f"{'PSNR':<10} | {psnr_v:<10.4f}")
    print(f"{'SSIM':<10} | {ssim_v:<10.4f}")
    print(f"{'SAM':<10} | {sam_v:<10.4f}")
    print(f"{'ERGAS':<10} | {ergas_v:<10.4f}")
    print(f"{'CC':<10} | {cc_v:<10.4f}")
    print(f"{'DD':<10} | {dd_v:<10.5f}")
    print(f"{'='*54}\n")

    # 绘制并保存图像 (RGB 和 Error Map)
    RESULT_PATH = os.path.join(args.log_dir, 'visual_results')
    os.makedirs(RESULT_PATH, exist_ok=True)
    
    bands = [29, 19, 9] if c >= 30 else [c-1, c//2, 0] # 防越界保护
    gt_rgb = matlab_style_rgb(X, bands)
    pred_rgb = matlab_style_rgb(best_out, bands)
    err_map = np.mean(np.abs(X - best_out), axis=2)
    
    plt.imsave(os.path.join(RESULT_PATH, 'GT_RGB.png'), gt_rgb)
    plt.imsave(os.path.join(RESULT_PATH, 'Recon_RGB.png'), pred_rgb)
    plt.imsave(os.path.join(RESULT_PATH, 'Error_Map.png'), err_map, cmap='jet', vmin=0, vmax=0.05)
    
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1); plt.imshow(gt_rgb); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(pred_rgb); plt.title(f"Reconstruction (PSNR:{psnr_v:.2f})"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(err_map, cmap='jet', vmin=0, vmax=0.05); plt.title("Error Map"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'Comparison_Visual.png'), dpi=300)
    plt.close()
    print(f"✓ RGB与Error Map已保存至: {RESULT_PATH}")

    # 提取光谱曲线
    typical_regions = [(50, 50), (100, 120), (180, 200)]
    spec_dir = os.path.join(args.log_dir, 'spectral_results')
    for coords in typical_regions:
        save_and_plot_spectral_curve(best_out, X, coords, window_size=3, save_dir=spec_dir, method_name='CLoRF')
    print(f"✓ 光谱曲线及数据已保存至: {spec_dir}")

if __name__ == '__main__':
    parser.add_argument('--snr_hsi', type=float, default=30.0, help='SNR parameter for HSI (default: 30)')
    parser.add_argument('--snr_msi', type=float, default=35.0, help='SNR parameter for MSI (default: 35)')

    parser.add_argument('--data_path', type=str, required=True, help='HSI data .mat file path')
    parser.add_argument('--r_path', type=str, required=True, help='SRF data .mat file path')
    parser.add_argument('--mat_key', type=str, default='paviaU', help='Key of the HSI data')
    parser.add_argument('--r_key', type=str, default='R', help='Key of the SRF data')
    parser.add_argument('--scale', type=int, default=4, help='Downsampling scale factor')

    parser.add_argument('--snr', type=int, default=40, help='SNR parameter for adding noise (Disabled in code)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR', help='learning rate')
    parser.add_argument('--lr_decay', action='store_true', help='cosine learning rate decay')
    parser.add_argument('--wd', type=float, default=0., metavar='WD', help='weight decay')
    parser.add_argument('--num_iters', type=int, default=30000, metavar='NUM_ITERS', help='training iterations')
    parser.add_argument('--clip', action='store_true', help='clip data range in [0., 1.]')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--log_dir', default='./runs/PU_Experiment', type=str, metavar='PATH', help='where checkpoints and logs to be saved')
    parser.add_argument('--eval_freq', default=100, type=int, metavar='N', help='evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # INR Network params
    parser.add_argument('--inr', type=str, default='siren', help='inr model')
    parser.add_argument('--pos_enc', action='store_true', help='position encoding for ReLU')
    parser.add_argument('--spatial_hidden_features', type=int, default=512, help='hidden_features')
    parser.add_argument('--spatial_hidden_layers', type=int, default=5, help='hidden layers for spatial model')
    parser.add_argument('--temporal_hidden_features', type=int, default=128, help='hidden_features')
    parser.add_argument('--temporal_hidden_layers', type=int, default=6, help='hidden layers for temporal model')

    # Fusion Hyper-parameters
    parser.add_argument('--k', type=int, default=9, help='num of rank')
    parser.add_argument('--lam', type=float, default=1.25, help='lambda hyper-parameter')
    parser.add_argument('--alpha', type=float, default=0.0025, help='alpha hyper-parameter (TV loss weight)')
    
    args = parser.parse_args()
    main(args)
