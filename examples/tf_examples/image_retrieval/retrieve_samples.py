"""
Script searches k nearest neighbors in gallery for each query sample
and draws results of search - query image merges with retrieved images
with classification result:
	* green rectangle - retrieved sample is True Positive
	* red rectangle - retrieved sample is not True Positive

It uses only preprocessed tf dataset splits as inputs.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow_similarity.distances import EuclideanDistance

import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

from dataset.dataloader import load_dataset
from dataset.metrics import build_embeddings_from_dataset, ValIndexer


def main():
	####################
	# Params:

	from cfg import cfg

	# Path to trained model
	model_path = 'models/model_name.h5'

	# Path to source images of the dataset
	source_images = 'Polyvore/images'

	# Show each query image merged with retrieved one
	show_images = True

	# Save merged images to dir
	save_images = False
	output_images_path = 'retrieved_samples/retrieving_1'

	# How many nearest neighbors retrieve for each query sample
	k_nearest_neighbors = 5

	# Number of results per output image (1...N)
	results_per_img = 1

	# Number of query batches to process.
	# Batch size defined in cfg.dataset.batch_size
	query_batches_to_process = int(
		np.ceil(cfg.dataset.query_samples / cfg.dataset.batch_size)
	)

	# Number of gallery samples
	gallery_samples_num = cfg.dataset.gallery_samples

	# It's also possible to search 'gallery' queries in 'train' gallery
	query_split = 'query'
	gallery_split = 'gallery'

	####################

	# Load model from saved h5 file
	model = tf.keras.models.load_model(model_path)

	# Load query and gallery datasets
	query_dataset = load_dataset(
		tfrecords=cfg.dataset.tfrecords,
		split=query_split,
		batch_size=cfg.dataset.batch_size,
		samples_per_class=cfg.dataset.samples_per_class,
		target_image_shape=cfg.dataset.input_shape,
	)
	gallery_dataset = load_dataset(
		tfrecords=cfg.dataset.tfrecords,
		split=gallery_split,
		batch_size=cfg.dataset.batch_size,
		samples_per_class=cfg.dataset.samples_per_class,
		target_image_shape=cfg.dataset.input_shape,
	)

	# Create gallery indexer
	indexer = ValIndexer(cfg, distance_function=EuclideanDistance())
	indexer.load_dataset_to_gallery(
		model, gallery_dataset, gallery_samples_num, verbose=True
	)

	# Build query embeddings
	query_embeddings, query_labels, query_image_ids = build_embeddings_from_dataset(
		model, query_dataset, query_batches_to_process,
		return_img_ids=True,
	)
	print('Query samples shape. Embeddings:', query_embeddings.shape, 'labels:', query_labels.shape)

	# Exit script if we don't want to show or save kNN images
	if not save_images and not show_images:
		sys.exit()

	# Retrieve from gallery k_nearest_neighbors samples for each query sample
	retrieved_samples = indexer.get_kNN(query_embeddings, k=k_nearest_neighbors)

	print(
		f'\nColumns legend:'
		f'\nquery_id, gt label, the closest labels (rank {k_nearest_neighbors}), '
		f'query image id, top 1 result is TP, distance to top 1'
	)
	if save_images:
		if not os.path.exists(output_images_path):
			os.makedirs(output_images_path)

	query_labels_np = query_labels.numpy()
	query_image_ids_np = query_image_ids.numpy()
	stacked_images = {}
	for q_id, retrieved_data in enumerate(retrieved_samples):

		gt_label = query_labels_np[q_id]
		query_image_id = query_image_ids_np[q_id]
		query_img_path = os.path.join(source_images, f'{query_image_id}.jpg')
		query_img = cv2.imread(query_img_path)

		retrieved_labels = ''
		output_img = query_img.copy()
		top1_is_tp = False
		for k_nearest_neighbor in range(k_nearest_neighbors):

			retrieved_label = retrieved_data[k_nearest_neighbor].label  # this is class_id
			retrieved_img_id = retrieved_data[k_nearest_neighbor].data.numpy()
			retrieved_img_path = os.path.join(source_images, f'{retrieved_img_id}.jpg')

			if not query_img_path or not retrieved_img_path:
				continue

			true_positive = True if gt_label == retrieved_label else False
			retrieved_img = cv2.imread(retrieved_img_path)

			# Draw rect based on retrieval result (red if wrong, green if correct)
			# All images must have the same resolution
			bbox_color = (0, 226, 30) if true_positive else (0, 0, 219)
			(img_h, img_w) = retrieved_img.shape[:2]
			x1 = img_w * (k_nearest_neighbor + 1) + 1
			x2 = img_w * (k_nearest_neighbor + 2) - 2
			y1 = 1
			y2 = img_h - 1

			# Merge previous image with retrieved one
			merged_img = np.concatenate((output_img, retrieved_img), axis=1)
			output_img = cv2.rectangle(merged_img.copy(), (x1, y1), (x2, y2), bbox_color, 4)

			retrieved_labels += f'_{retrieved_label}'

			if k_nearest_neighbor == 0:
				top1_is_tp = true_positive

		print(q_id, gt_label, retrieved_labels, query_image_id, top1_is_tp, retrieved_data[0].distance)

		if show_images:
			plt.figure()
			plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
			plt.show()

		if save_images:

			# Save result of each query to a single image
			if results_per_img == 1:

				# Filename format: gtLabel_retrievedLabels_isTruePositive_queryImgID.jpg
				filename = f'{gt_label}{retrieved_labels}_{int(top1_is_tp)}_{query_image_id}.jpg'
				output_image_path = os.path.join(output_images_path, filename)
				cv2.imwrite(output_image_path, output_img)

			# Save multiple results to a single image
			elif results_per_img > 1:

				if q_id % (results_per_img + 1) == 0 and q_id != 0:

					# queryImgID1_queryImgID2_..._queryImgIDN.jpg
					filename = f'{"_".join(str(query_id) for query_id in stacked_images.keys())}.jpg'
					output_image_path = os.path.join(output_images_path, filename)
					cv2.imwrite(
						output_image_path,
						np.concatenate(list(stacked_images.values()), axis=0)
					)
					stacked_images = {}
				else:
					stacked_images[query_image_id] = output_img.copy()


if __name__ == '__main__':
	main()
