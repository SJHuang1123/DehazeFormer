import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from HSI_dataset import HyperspectralDehazeDataset

FILE_NAME = 'HSID'
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()

import torch

def sam_metric(img1, img2, eps=1e-8, reduction='mean'):
    """
    Compute SAM (Spectral Angle Mapper) between two hyperspectral images.

    Args:
        img1: torch.Tensor of shape (B, C, H, W) - reference spectra
        img2: torch.Tensor of shape (B, C, H, W) - predicted spectra
        eps: float, small value to prevent division by zero
        reduction: 'mean' | 'none' | 'sum' - how to reduce the result

    Returns:
        sam: torch.Tensor
            - scalar if reduction='mean' or 'sum'
            - tensor of shape (B, H, W) if reduction='none'
    """
    # Flatten spectra across spectral dimension C
    dot_product = torch.sum(img1 * img2, dim=1)  # (B, H, W)
    norm1 = torch.norm(img1, dim=1)              # (B, H, W)
    norm2 = torch.norm(img2, dim=1)              # (B, H, W)

    # Cosine of spectral angle
    cos_theta = dot_product / (norm1 * norm2 + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical safety

    # SAM in radians
    angle = torch.acos(cos_theta)  # (B, H, W)

    if reduction == 'mean':
        return angle.mean() * (180.0 / torch.pi)  # convert to degrees
    elif reduction == 'sum':
        return angle.sum()
    elif reduction == 'none':
        return angle  # per-pixel SAM in radians
    else:
        raise ValueError("Invalid reduction type. Use 'mean', 'sum', or 'none'.")


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')
	sam_val_sum = 0.0
	for idx, batch in enumerate(test_loader):
		ground_truth = batch['gt'].cuda()
		trans = batch['trans'].cuda()
		hazy = ground_truth * trans + (1 -trans)

		with torch.no_grad():
			output = network(hazy)

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, ground_truth)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(ground_truth, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()
			sam_val = sam_metric(output, ground_truth).item()				
		sam_val_sum += sam_val
		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
			'SAM: {sam:.04f}'
			.format(idx, psnr=PSNR, ssim=SSIM, sam=sam_val))

		f_result.write('%s,%.02f,%.03f,%.04f\n' % (FILE_NAME, psnr_val, ssim_val, sam_val))

	f_result.close()

	os.rename(
		os.path.join(result_dir, 'results.csv'),
		os.path.join(result_dir, '%.02f | %.04f | %.05f.csv' % (
			PSNR.avg, SSIM.avg, sam_val_sum / len(test_loader)
		))
	)

if __name__ == '__main__':
	network = dehazeformer.HSIDehazeFormer(batch_size=1)
	network.eval()
	network.cuda()

	# Load pretrained weights
	ckp = torch.load('/home/q36131207/DehazeFormer/saved_models/indoor/QHSID.pth')
	network.load_state_dict(ckp, strict=False)

	dataset_dir = '/home/q36131207/HSID_dataset/AVIRIS'
	test_dataset = HyperspectralDehazeDataset(dataset_dir + '/qtest')
	test_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)