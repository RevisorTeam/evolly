"""
Simple Tensorflow pipeline with training
classification model on the Fashion-MNIST dataset.

Reference:
https://www.tensorflow.org/tutorials/keras/classification
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.framework.convert_to_constants import (
	convert_variables_to_constants_v2_as_graph
)
from tensorflow.keras import Model

from typing import Tuple, Dict

from time import time
import numpy as np
import tempfile
import json
import os

from evolly import compute_fitness

from create_model import my_model


def main() -> None:

	from cfg import cfg

	cfg.model.name = '0000_00001'

	config_path = None          # path to config file
	if config_path:
		cfg.merge_from_file(config_path)

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

	# Save trained model to file
	cfg.model.name += f'_{meta_data["fitness"]:.5f}'
	model_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}.h5')
	model.save(model_path, save_format='h5')

	# Save metadata json
	metadata_path = os.path.join(cfg.train.save_dir, f'{cfg.model.name}_meta.json')
	save_json(metadata_path, meta_data)


def train(cfg) -> Tuple[Model, Dict]:
	learning_rate = 0.1
	batch_size = 64

	meta_data = {'train_loss': [], 'val_metric': [], 'config': cfg}

	start_time = time()

	fashion_mnist = tf.keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# Build model from genotype
	model = my_model(cfg)

	meta_data['parameters'] = model.count_params()
	meta_data['flops'] = get_flops(model)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	history = model.fit(
		train_images, train_labels,
		batch_size=batch_size,
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


def get_flops(model, write_path=tempfile.NamedTemporaryFile().name) -> int:
	concrete = tf.function(lambda inputs: model(inputs))
	concrete_func = concrete.get_concrete_function(
		[tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
	frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
		concrete_func,
		# lower_control_flow=False		# uncomment it if flops of LSTM / GRU layers compute wrong
	)
	with tf.Graph().as_default() as graph:
		tf.graph_util.import_graph_def(graph_def, name='')
		run_meta = tf.compat.v1.RunMetadata()
		opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
		if write_path:
			opts['output'] = 'file:outfile={}'.format(write_path)  # suppress output
		flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

		# The //2 is necessary since `profile` counts multiply and accumulate
		# as two flops, here we report the total number of multiply accumulate ops
		flops_total = flops.total_float_ops // 2
		return flops_total


def save_json(path, output_dict):
	with open(path, "w") as j:
		json.dump(output_dict, j, indent=2)


if __name__ == '__main__':
	main()
