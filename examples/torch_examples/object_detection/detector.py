import torch
import matplotlib.pyplot as plt
from create_model import get_model
from dataloader import load_dataset
from torchvision.utils import draw_bounding_boxes


def main():

	from cfg import cfg

	checkpoint_path = 'models/model_name.pth'
	checkpoint = torch.load(checkpoint_path)

	device = 'cuda:0'

	ds = load_dataset(cfg, split='val', batch_size=1, augs_enabled=False, verbose=False)

	model = get_model(cfg)
	model.load_state_dict(checkpoint['model'])
	model.to(device)
	model.eval()

	for images, targets in ds:
		for image, target in zip(images, targets):

			detections = model([image.to(device)])[0]

			print()
			print(detections)
			print('Detected boxes shape:', detections['boxes'].size())
			visualize_detections(image, detections)


def visualize_detections(image, detections):

	image = image.type(torch.uint8)

	visualized_boxes = None
	for detection in detections:
		visualized_boxes = draw_bounding_boxes(
			image,
			boxes=detection['boxes'],
			width=3
		)

	fig, ax = plt.subplots()
	ax.imshow(visualized_boxes.permute(1, 2, 0).numpy())
	ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
	fig.tight_layout()

	plt.show()


if __name__ == '__main__':
	main()
