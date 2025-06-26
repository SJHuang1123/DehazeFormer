import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from utils import AverageMeter
from datasets.loader import PairLoader
import models.dehazeformer as models
from HSI_dataset import HyperspectralDehazeDataset

BATCH_SIZE = 2

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		ground_truth = batch['gt'].cuda()
		if ground_truth.size(0) < BATCH_SIZE:
			continue
		trans_map = batch['trans'].cuda()
		hazy_img = ground_truth * trans_map + (1 - trans_map)

		with autocast(args.no_autocast):
			output = network(hazy_img)
			loss = criterion(output, ground_truth)

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()
	import random
	magic = 10
	for i, batch in enumerate(val_loader):
		ground_truth = batch['gt'].cuda()
		if ground_truth.size(0) < BATCH_SIZE:
			continue
		trans_map = batch['trans'].cuda()
		hazy_img = ground_truth * trans_map + (1 - trans_map)

		with torch.no_grad():							# torch.no_grad() may cause warning
			output = network(hazy_img).clamp_(0, 1)		

		mse_loss = F.mse_loss(output, ground_truth, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), ground_truth.size(0))
		if i % magic == 0:
			img_to_save = torch.cat(
				[
					ground_truth[:,[25,15,6],:,:],
					hazy_img[:,[25,15,6],:,:],
					output[:,[25,15,6],:,:],
				],
				dim=0
			)
			save_image(img_to_save, 'result.jpg', nrow=BATCH_SIZE)
	return PSNR.avg


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = models.HSIDehazeFormer(pretrained=True, batch_size=BATCH_SIZE).cuda()
	
	# uncomment for resume

	# ckp = torch.load('/home/q36131207/DehazeFormer/saved_models/indoor/QHSID_trans.pth')
	# network.load_state_dict(ckp['state_dict'], strict=False)
	# network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	dataset_dir = '/home/q36131207/HSID_dataset/AVIRIS'
	train_dataset = HyperspectralDehazeDataset(dataset_dir + '/qtrain')
	train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = HyperspectralDehazeDataset(dataset_dir + '/qval')
	val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		print('==> Start training')
		# print(network)


		best_psnr = 0
		for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			print(f'Epoch [{epoch}/{setting["epochs"]}], Loss: {loss:.4f}')
			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr = valid(val_loader, network)
				
			if avg_psnr > best_psnr:
				print(f'==> New Best PSNR: {avg_psnr:.4f}, saving model...')
				best_psnr = avg_psnr
				torch.save(network.state_dict(), os.path.join(save_dir, 'QHSID' +'.pth'))

	else:
		print('==> Existing trained model')
		exit(1)
