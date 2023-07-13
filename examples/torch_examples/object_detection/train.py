from typing import Tuple, Dict
from pathlib import Path
from time import time
import contextlib
import torch
import json
import math
import sys
import os
import io


from utils import MetricLogger, SmoothedValue, reduce_dict
from evolly import compute_fitness, GetFlopsTorch
from dataloader import load_dataset
from create_model import get_model
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

# Disable torch and fvcore warnings
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('fvcore').setLevel(logging.CRITICAL)


def main() -> None:

	from cfg import cfg

	cfg.model.name = '0000_00001'

	# Which accelerator to use during training
	cfg.train.accelerators = ['cuda:0' if torch.cuda.is_available() else 'cpu']
	cfg.train.accelerator_type = 'GPU'

	train_wrapper(cfg)


def train_wrapper(cfg) -> None:

	try:
		# Checkpoint of the best epoch will be returned
		checkpoint, meta_data = train(cfg)

		# Compute fitness value
		meta_data['fitness'] = compute_fitness(
			val_metrics=meta_data['val_metric'],
			target_params=cfg.search.target,
			model_params=meta_data['parameters'],
			w=cfg.search.w,
			metric_op=cfg.val.metric_op
		)

		print(
			'{} ({:.2f}M / {:.2f}G) val mAP 05-95: '
			'{:.4f} - fit: {:.4f} - {:.2f} mins'
			.format(
				cfg.model.name,
				meta_data['parameters'] / 1e6,
				meta_data['flops'] / 1e9,
				max(meta_data['val_metric']),
				meta_data['fitness'],
				meta_data['training_time'] / 60
			)
		)

		# Save trained model to file
		cfg.model.name += f'_{meta_data["fitness"]:.5f}'
		Path(cfg.train.save_dir).mkdir(exist_ok=True)
		path = Path(cfg.train.save_dir, f'{cfg.model.name}.pth')
		torch.save(checkpoint, path)

		# Save metadata json
		metadata_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}_meta.json')
		save_json(metadata_path, meta_data)

	# Catch GPU out of memory errors in order not to crash evolutionary process.
	except Exception as e:
		if 'CUDA out of memory' in str(e):
			print(f'{cfg.model.name} | GPU out of memory')
		else:
			print(f'{cfg.model.name} | {str(e)}')


def train(cfg) -> Tuple[Dict, Dict]:

	device = cfg.train.accelerators[0]
	verbose = cfg.train.verbose

	start_time = time()

	meta_data = {'train_loss': [], 'val_metric': [], 'config': cfg}

	train_ds = load_dataset(cfg, split='train', augs_enabled=True, verbose=verbose)
	val_ds = load_dataset(cfg, split='val', batch_size=1, augs_enabled=False, verbose=verbose)

	# Build model from genotype
	model = get_model(cfg)
	model.to(device)
	parameters = [p for p in model.parameters() if p.requires_grad]

	model.eval()
	meta_data['parameters'] = int(sum(p.numel() for p in model.parameters()))
	meta_data['flops'] = int(
		GetFlopsTorch(model, [torch.rand(3, 256, 256).to(device)]).total()
	)
	model.train()

	# optimizer = torch.optim.SGD(
	# 	parameters,
	# 	lr=cfg.train.base_lr,
	# 	momentum=cfg.train.momentum,
	# 	weight_decay=cfg.train.weight_decay,
	# )
	optimizer = torch.optim.AdamW(parameters, lr=cfg.train.base_lr, weight_decay=cfg.train.weight_decay)

	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
	# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

	print(
		f'Start training {cfg.model.name} '
		f'({meta_data["parameters"] / 1e6:.2f}M / '
		f'{meta_data["flops"] / 1e9:.2f}G) '
		f'on {[acc for acc in cfg.train.accelerators]} '
		f'for {cfg.train.epochs} epochs'
	)

	checkpoint, best_val_metric, best_epoch_id = {}, 0.0, 0
	for epoch in range(cfg.train.epochs):

		_, loss = train_one_epoch(
			model, optimizer, train_ds, device, epoch,
			verbose=verbose
		)

		lr_scheduler.step()

		meta_data['train_loss'].append(loss)

		# evaluate after every epoch
		mAP_05, mAP_05_95 = evaluate(model, val_ds, device=device, verbose=verbose)
		val_metric = mAP_05_95
		meta_data['val_metric'].append(float(val_metric))

		if val_metric > best_val_metric:
			best_val_metric, best_epoch_id = val_metric, epoch
			checkpoint = {
				"model": model.state_dict(),
				"optimizer": optimizer.state_dict(),
				"lr_scheduler": lr_scheduler.state_dict(),
				"epoch": epoch,
			}

	meta_data['training_time'] = time() - start_time

	if verbose:
		print('Best val metric:', best_val_metric)
		print('Best epoch id:', best_epoch_id)
		print('Training time', meta_data['training_time'])

	return checkpoint, meta_data


def train_one_epoch(
		model,
		optimizer,
		data_loader,
		device,
		epoch,
		print_freq=100,
		verbose=True
):

	model.train()

	metric_logger = MetricLogger(delimiter="  ")
	metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
	header = f"Epoch: [{epoch}]"

	lr_scheduler = None
	if epoch == 0:
		warmup_factor = 1.0 / 1000
		warmup_iters = min(1000, len(data_loader) - 1)

		lr_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer, start_factor=warmup_factor, total_iters=warmup_iters
		)

	ds_loader = metric_logger.log_every(
		data_loader, print_freq, header
	) if verbose else data_loader

	losses_reduced = 0.0
	for images, targets in ds_loader:
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		with torch.cuda.amp.autocast(enabled=False):
			loss_dict = model(images, targets)
			losses = sum(loss for loss in loss_dict.values())

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = reduce_dict(loss_dict)
		losses_reduced = sum(loss for loss in loss_dict_reduced.values())

		loss_value = losses_reduced.item()

		if not math.isfinite(loss_value):
			print(f"Loss is {loss_value}, stopping training")
			print(loss_dict_reduced)
			sys.exit(1)

		optimizer.zero_grad()

		losses.backward()
		optimizer.step()

		if lr_scheduler is not None:
			lr_scheduler.step()

		metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
		metric_logger.update(lr=optimizer.param_groups[0]["lr"])

	return metric_logger, losses_reduced.item()


@torch.inference_mode()
def evaluate(
		model,
		data_loader,
		device,
		verbose=True
):
	n_threads = torch.get_num_threads()
	# FIXME remove this and make paste_masks_in_image run on the GPU
	torch.set_num_threads(1)
	cpu_device = torch.device("cpu")
	model.eval()
	metric_logger = MetricLogger(delimiter="  ")
	header = "Test:"

	coco = get_coco_api_from_dataset(data_loader.dataset)
	coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

	ds_loader = metric_logger.log_every(
		data_loader, 100, header
	) if verbose else data_loader

	for images, targets in ds_loader:
		images = list(img.to(device) for img in images)

		if torch.cuda.is_available():
			torch.cuda.synchronize()

		model_time = time()
		outputs = model(images)

		outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
		model_time = time() - model_time

		res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
		evaluator_time = time()
		coco_evaluator.update(res)
		evaluator_time = time() - evaluator_time
		metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	if verbose:
		print("Averaged stats:", metric_logger)
	coco_evaluator.synchronize_between_processes()

	# accumulate predictions from all images
	with contextlib.redirect_stdout(io.StringIO()) as stdout:
		coco_evaluator.accumulate()
		coco_evaluator.summarize()

	if verbose:
		print(stdout.getvalue())

	torch.set_num_threads(n_threads)

	metrics = coco_evaluator.coco_eval['bbox'].stats
	mAP_05 = metrics[1]
	mAP_05_95 = metrics[0]
	return mAP_05, mAP_05_95


def save_json(path, output_dict):
	with open(path, "w") as j:
		json.dump(output_dict, j, indent=2)


if __name__ == '__main__':
	main()
