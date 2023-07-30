"""
Simple Tensorflow pipeline with training
classification model on the Fashion-MNIST dataset.

Reference:
https://www.tensorflow.org/tutorials/keras/classification
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import Model
from typing import Tuple, Dict
from time import time
import json
import os

from evolly import compute_fitness
from evolly import get_flops_tf

from create_model import my_model

import tensorflow as tf


def main() -> None:

	from cfg import cfg
	cfg.model.name = '0000_00001'

	# Assign which accelerator to use during training
	accelerator_type = 'GPU'
	# (utilize first GPU even if you have multiple)
	accelerators = [tf.config.list_logical_devices('GPU')[0]]

	cfg.train.accelerators = accelerators
	cfg.train.accelerator_type = accelerator_type

	train_wrapper(cfg)


def train_wrapper(cfg) -> None:

	# Here weights of the last epoch will be returned,
	# but it's better to return weights of the "best" epoch
	model, meta_data = train(cfg)

	# Compute fitness value
	meta_data['fitness'] = compute_fitness(
		val_metrics=meta_data['val_metric'],
		target_params=cfg.search.target,
		model_params=meta_data['parameters'],
		w=cfg.search.w,
		metric_op=cfg.val.metric_op
	)

	# Save trained model to file.
	# NOTE: weights of the last epoch will be saved.
	cfg.model.name += f'_{meta_data["fitness"]:.5f}'
	model_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}.h5')
	model.save(model_path, save_format='h5')

	# Save metadata json
	metadata_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}_meta.json')
	save_json(metadata_path, meta_data)


def train(cfg) -> Tuple[Model, Dict]:

	meta_data = {'train_loss': [], 'val_metric': [], 'config': cfg}

	start_time = time()

	fashion_mnist = tf.keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# Build model from genotype
	model = my_model(cfg)

	meta_data['parameters'] = model.count_params()
	meta_data['flops'] = get_flops_tf(model)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	history = model.fit(
		train_images, train_labels,
		batch_size=cfg.train.batch_size,
		epochs=cfg.train.epochs,
		validation_data=(test_images, test_labels)
	)

	for epoch_id, epoch_train_loss in enumerate(history.history['loss']):
		meta_data['train_loss'].append(float(epoch_train_loss))
		meta_data['val_metric'].append(float(
			history.history['val_accuracy'][epoch_id]
		))

	meta_data['training_time'] = float(time() - start_time)

	return model, meta_data


def save_json(path, output_dict):
	with open(path, "w") as j:
		json.dump(output_dict, j, indent=2)


if __name__ == '__main__':
	main()
