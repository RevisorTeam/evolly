"""
Script parses Polyvore dataset, creates train and test (query, gallery)
splits and makes tfrecords files.

These tfrecords are used while loading splits to Tensorflow Dataset object.
It's much faster to load tfrecords than manually load data to tf dataset.
"""

import tensorflow as tf
import json
import os as os
import numpy as np
import statistics


def main():

	# Paths to dataset images and json with labelled data.
	# Dataset can be downloaded from official GitHub repo:
	# https://github.com/xthan/polyvore-dataset
	images_path = 'Polyvore/images'
	json_path = 'Polyvore/polyvore_item_metadata.json'

	# Path to directory with output splits
	tfrecords_path = '../tfrecords'

	# How many classes ids will be in one tfrecord.
	# Each class_id has 1 ... N samples.
	classes_per_record = 4

	# Names of the output splits.
	# Gallery and query splits are splits that are used during test stage.
	splits = ['train', 'gallery', 'query']

	# Which approach to use while splitting samples:
	# 'uniform' - pick all classes to train and test.
	# 'categorical' - pick X% of classes to train and other 100%-X% classes to test.
	# 	For example, train set may contain 70% of classes while test set
	# 	contains 30% of classes.
	splitting_type = 'categorical'

	seed = 12345
	np.random.seed(seed)

	# Parse raw data and create train and test (query and gallery) splits.
	classes, samples, split_samples = parse_data(
		json_path, images_path, splits,
		splitting_type=splitting_type,

		# Do not take too much samples to gallery and query
		# in order to speed up validation stage
		max_gallery_samples_per_class=500,
		max_query_samples_per_class=100,
	)

	print('\nCreating tfrecords...')
	for split in splits:
		tfrecords_split_path = os.path.join(tfrecords_path, split)

		if not os.path.exists(tfrecords_split_path):
			os.makedirs(tfrecords_split_path)

		create_tfrecords(
			samples,
			split_samples[split],
			split_name=split,
			tfrecords_path=tfrecords_path,
			classes_per_record=classes_per_record
		)


def parse_data(
		json_path,
		images_path,
		splits,
		train_size=0.7,
		gallery_size=0.2,
		query_size=0.1,
		samples_thresh=10,
		max_gallery_samples_per_class=None,
		max_query_samples_per_class=None,
		splitting_type='uniform',
):

	assert splitting_type in ['uniform', 'categorical']

	if max_gallery_samples_per_class is None:
		max_gallery_samples_per_class = int(1e10)
	if max_query_samples_per_class is None:
		max_query_samples_per_class = int(1e10)

	metadata = open_json(json_path)

	classes = {}
	samples = {}

	print('Parsing samples...')
	for sample_id, sample_data in metadata.items():

		sample_id = int(sample_id)
		class_id = int(sample_data['category_id'])
		class_name = sample_data['semantic_category']
		sample_description = sample_data['description']
		image_path = os.path.join(images_path, f'{sample_id}.jpg')

		if not os.path.exists(image_path):
			continue

		samples[sample_id] = {
			'sample_id': sample_id,
			'class_id': class_id,
			'class_name': class_name,
			'sample_description': sample_description,
			'image_path': image_path,
		}

		if class_id not in classes.keys():
			classes[class_id] = []

		classes[class_id].append(sample_id)

	# Make sure each class has at least samples_thresh samples.
	classes_to_del = []
	for class_id, class_samples in classes.items():
		if len(class_samples) < samples_thresh:
			classes_to_del.append(class_id)
	for class_id_to_del in classes_to_del:
		classes.pop(class_id_to_del)
	print(
		f'{len(classes_to_del)} classes will not be written '
		f'in tfrecords due to lack of samples.'
	)

	# Distribute samples to splits and count stats
	dataset_stats = {split: {'classes': 0, 'samples': 0} for split in splits}
	split_samples = {split: {} for split in splits}
	samples_per_class = {split: [] for split in splits}

	if splitting_type == 'uniform':
		for class_id, class_samples in classes.items():

			samples_per_class_total = len(classes[class_id])
			query_samples = int(np.ceil(query_size * samples_per_class_total))
			gallery_samples = int(np.ceil(gallery_size * samples_per_class_total))
			train_samples = samples_per_class_total - gallery_samples - query_samples

			for split in splits:
				if class_id not in split_samples[split].keys():
					split_samples[split][class_id] = []
				dataset_stats[split]['classes'] += 1

				if split == 'train':
					split_samples[split][class_id] = class_samples[:train_samples]

				elif split == 'gallery':
					split_samples[split][class_id] = \
						class_samples[train_samples:train_samples + gallery_samples]
					if len(split_samples[split][class_id]) > max_gallery_samples_per_class:
						split_samples[split][class_id] = \
							split_samples[split][class_id][:max_gallery_samples_per_class]

				elif split == 'query':
					split_samples[split][class_id] = \
						class_samples[train_samples + gallery_samples:samples_per_class_total]
					if len(split_samples[split][class_id]) > max_query_samples_per_class:
						split_samples[split][class_id] = \
							split_samples[split][class_id][:max_query_samples_per_class]

				dataset_stats[split]['samples'] += len(split_samples[split][class_id])
				samples_per_class[split].append(len(split_samples[split][class_id]))

	else:
		# Distribute classes ids to train and test (query, gallery) splits
		classes_ids = set(classes.keys())
		train_classes_ids = set(np.random.choice(
			list(classes_ids),
			size=int(train_size * len(classes_ids)),
			replace=False
		))
		test_classes_ids = classes_ids - train_classes_ids

		# Distribute samples to splits and count stats
		for raw_split in ['train', 'test']:
			raw_split_class_ids = train_classes_ids if raw_split == 'train' else test_classes_ids
			for class_id in raw_split_class_ids:

				if raw_split == 'train':
					if class_id not in split_samples['train'].keys():
						split_samples['train'][class_id] = set()
					dataset_stats['train']['classes'] += 1
				else:
					if class_id not in split_samples['query'].keys():
						split_samples['query'][class_id] = set()
					if class_id not in split_samples['gallery'].keys():
						split_samples['gallery'][class_id] = set()
					dataset_stats['query']['classes'] += 1
					dataset_stats['gallery']['classes'] += 1

				# Number of samples that should be placed in query the split
				query_samples = int(query_size * len(classes[class_id]))

				for sample_idx, sample_id in enumerate(classes[class_id]):

					if raw_split == 'train':
						split_samples['train'][class_id].add(sample_id)
						dataset_stats['train']['samples'] += 1

					if raw_split == 'test':
						# Put first query_samples to query set
						if sample_idx < query_samples:
							if sample_idx > max_query_samples_per_class:
								continue
							split_samples['query'][class_id].add(sample_id)
							dataset_stats['query']['samples'] += 1
						else:
							if sample_idx > max_gallery_samples_per_class:
								continue
							split_samples['gallery'][class_id].add(sample_id)
							dataset_stats['gallery']['samples'] += 1

		for split, class_ids in split_samples.items():
			for class_id, sample_ids in class_ids.items():
				samples_per_class[split].append(len(split_samples[split][class_id]))

	print()
	print('Parsed dataset stats:')
	print(dataset_stats)
	print('Samples per class:')
	print(samples_per_class)
	print('Average samples per class:')
	for split in splits:
		print(split, statistics.mean(samples_per_class[split]))

	return classes, samples, split_samples


def create_tfrecords(
		samples,
		split_samples,
		split_name,
		tfrecords_path='tfrecords',
		classes_per_record=50
):
	"""
	Writes samples of split classes ids to tfrecords.
	Each tfrecords contain classes_per_record unique_ids (labels).
	Last tfrecord has (1 <= N <= classes_per_record) classes_ids

	For instance:
		There are 14 samples of 3 classes_ids (333, 444, 555). classes_per_record=2.
		Their corresponding classes_ids (labels):
		[333, 333, 333, 444, 444, 444, 444, 444, 444, 555, 555, 555, 555, 666]

		Output tfrecords will contain samples with classes_ids:
		00001.tfrecord : [333, 333, 333, 444, 444, 444, 444, 444, 444]
		00002.tfrecord : [555, 555, 555, 555, 666]
	"""
	records_classes = distribute_classes(
		classes_ids=split_samples.keys(),
		classes_per_record=classes_per_record
	)

	shards_written = 0
	total_samples = 0
	for tfrecord_id, record_classes in enumerate(records_classes):

		tfrecord_path = os.path.join(tfrecords_path, split_name, '{:05d}.tfrecord'.format(tfrecord_id))
		samples_in_shard = 0
		with tf.io.TFRecordWriter(tfrecord_path) as writer:
			for class_id in record_classes:
				for sample_id in split_samples[class_id]:
					sample = samples[sample_id]
					tf_example = make_tf_example(sample)
					writer.write(tf_example.SerializeToString())
					samples_in_shard += 1

		total_samples += samples_in_shard
		shards_written += 1

		print('tfrecord', tfrecord_path, 'ready. Samples in shard:', samples_in_shard)

	print(f'Shards written in {split_name}: {shards_written}. Samples: {total_samples}')


def distribute_classes(classes_ids, classes_per_record=100):
	"""
	Distributes classes_ids to chunks of classes_per_record size
	"""
	return chunk_to_n_size(list(classes_ids), classes_per_record)


def chunk_to_n_size(seq, num):
	return [seq[i * num:(i + 1) * num] for i in range((len(seq) + num - 1) // num)]


def make_tf_example(sample_dict):

	image_bytes = tf.io.decode_jpeg(tf.io.read_file(sample_dict['image_path']))

	features = {
		'sample_id': _int64_feature(sample_dict['sample_id']),
		'class_id': _int64_feature(sample_dict['class_id']),
		'class_name': _bytes_feature(sample_dict['class_name']),
		'sample_description': _bytes_feature(sample_dict['sample_description']),
		'image_path': _bytes_feature(sample_dict['image_path']),
		'image_raw': _image_feature(image_bytes),
	}

	return tf.train.Example(features=tf.train.Features(feature=features))


def _image_feature(value):
	"""Returns a bytes_list from an image bytes string."""
	return tf.train.Feature(
		bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
	)


def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
	"""Returns an int64_list from a bool / enum / int / uint (list)."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature_list(value):
	"""Returns a list of float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def open_json(path_to_json):
	with open(path_to_json) as f:
		json_file = json.load(f)
	return json_file


if __name__ == '__main__':
	main()
