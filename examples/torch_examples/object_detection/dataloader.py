import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import albumentations as alb
import contextlib
import io


def load_dataset(cfg, split, batch_size=None, augs_enabled=True, verbose=True):

	assert split in ['train', 'val']

	if batch_size is None:
		batch_size = cfg.dataset.batch_size

	if split == 'train':
		images_path = cfg.dataset.train_images
		annotations_path = cfg.dataset.train_anns
	else:
		images_path = cfg.dataset.val_images
		annotations_path = cfg.dataset.val_anns

	with contextlib.redirect_stdout(io.StringIO()) as stdout:
		dataset = CocoDetection(
			images_path, annotations_path,
			augs_enabled=augs_enabled
		)

	if verbose:
		print(stdout.getvalue())

	if split == 'train':
		dataset = _coco_remove_images_without_annotations(dataset)
		# dataset = torch.utils.data.Subset(dataset, list(range(100)))
		sampler = torch.utils.data.RandomSampler(dataset)
	else:
		# dataset = torch.utils.data.Subset(dataset, list(range(100)))
		sampler = torch.utils.data.SequentialSampler(dataset)

	sampler = torch.utils.data.BatchSampler(
		sampler, batch_size=batch_size, drop_last=True
	)

	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_sampler=sampler,
		num_workers=cfg.dataset.workers,
		collate_fn=collate_fn
	)

	return dataloader


def visualize_sample(image, sample_data):

	image = image.type(torch.uint8)

	annotated_image = draw_bounding_boxes(
		image,
		boxes=sample_data['boxes'],
		width=3
	)

	fig, ax = plt.subplots()
	ax.imshow(annotated_image.permute(1, 2, 0).numpy())
	ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
	fig.tight_layout()

	plt.show()


class CocoDetection(torchvision.datasets.CocoDetection):
	def __init__(
			self,
			img_folder,
			ann_file,
			target_height=256,
			target_width=256,
			augs_enabled=True
	):
		super().__init__(img_folder, ann_file)

		# List of transforms to apply for each sample (image + bbox)
		transforms = list()

		self._augs_enabled = augs_enabled

		# Augmentation transforms
		if augs_enabled:
			transforms.extend([
				alb.Flip(p=0.2),
				alb.RandomSizedBBoxSafeCrop(height=target_height, width=target_width, p=0.3),
				alb.ShiftScaleRotate(p=0.2),
				alb.ToGray(p=0.2)
			])

		# Normalization transforms.
		# Torchvision's built-in models already have normalization stage,
		# so it's not needed here.
		# transforms.extend([
		# 	alb.Resize(height=target_height, width=target_width, p=1),
		# 	alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		# ])

		# By using ParseCOCOSample class we've transformed bboxes from
		# [x, y, w, h] to [x1, y1, x2, y2] format.
		# That's why 'pascal_voc' was set as bbox format type.
		self._transforms = alb.Compose(
			transforms,
			bbox_params=alb.BboxParams(format="pascal_voc", label_fields=["bbox_classes"])
		) if augs_enabled else None

		self._to_tensor = T.ToTensor()
		self._pil_to_tensor = T.PILToTensor()
		self._parse_sample = ParseCOCOSample()

	def __getitem__(self, idx):
		img, target = super().__getitem__(idx)
		image_id = self.ids[idx]
		target = dict(image_id=image_id, annotations=target)

		img, target = self._parse_sample(img, target)

		if self._transforms is not None:
			transformed = self._transforms(
				image=np.asarray(img, dtype=np.float32),
				bboxes=np.asarray(target['boxes'], dtype=np.float32),
				bbox_classes=np.asarray(target['labels'], dtype=np.int64),
			)

			img = self._to_tensor(transformed["image"])

			target['boxes'] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
			target['labels'] = torch.as_tensor(transformed["bbox_classes"], dtype=torch.int64)

		else:
			img = self._pil_to_tensor(img)
			img = img.type(torch.float32)
		return img, target


class ParseCOCOSample:
	def __call__(self, image, target):

		w, h = image.size

		image_id = target["image_id"]
		image_id = torch.tensor([image_id])

		anno = target["annotations"]

		anno = [obj for obj in anno if obj["iscrowd"] == 0]

		boxes = [obj["bbox"] for obj in anno]

		# guard against no boxes via resizing
		boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
		boxes[:, 2:] += boxes[:, :2]
		boxes[:, 0::2].clamp_(min=0, max=w)
		boxes[:, 1::2].clamp_(min=0, max=h)

		classes = [obj["category_id"] for obj in anno]
		classes = torch.tensor(classes, dtype=torch.int64)

		keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
		boxes = boxes[keep]
		classes = classes[keep]

		target = {
			"boxes": boxes,
			"labels": classes,
			"image_id": image_id,
			"area": torch.as_tensor([obj["area"] for obj in anno], dtype=torch.float32),
			"iscrowd": torch.as_tensor([obj["iscrowd"] for obj in anno], dtype=torch.int64),
		}

		return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
	def _has_only_empty_bbox(anno):
		return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

	def _has_valid_annotation(anno):
		# if it's empty, there is no annotation
		if len(anno) == 0:
			return False
		# if all boxes have close to zero area, there is no annotation
		if _has_only_empty_bbox(anno):
			return False

		if "keypoints" not in anno[0]:
			return True

		return False

	if not isinstance(dataset, torchvision.datasets.CocoDetection):
		raise TypeError(
			f"This function expects dataset of type "
			f"torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
		)
	ids = []
	for ds_idx, img_id in enumerate(dataset.ids):
		ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
		anno = dataset.coco.loadAnns(ann_ids)
		if cat_list:
			anno = [obj for obj in anno if obj["category_id"] in cat_list]
		if _has_valid_annotation(anno):
			ids.append(ds_idx)

	dataset = torch.utils.data.Subset(dataset, ids)
	return dataset


def collate_fn(batch):
	return tuple(zip(*batch))


def main():
	from cfg import cfg

	# split = 'train'
	split = 'val'

	dataset = load_dataset(cfg, split=split, batch_size=1, augs_enabled=False, verbose=True)

	for batch_id, (images, samples_data) in enumerate(dataset):
		print('Images in batch:', len(images))
		print('Samples data in batch:', len(samples_data))

		for image, sample_data in zip(images, samples_data):
			print(image)
			print(sample_data)
			print('\nImage shape:', image.shape)
			print('Boxes shape:', sample_data['boxes'].size())
			print('Objects in the image:', sample_data['labels'].shape[0])
			print('Label ids:', sample_data['labels'])
			print('Sample data keys:', list(sample_data.keys()))
			visualize_sample(image, sample_data)

			assert len(sample_data['boxes'].size()) == 2


if __name__ == '__main__':
	main()
