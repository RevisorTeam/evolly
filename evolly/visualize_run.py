from evolly import unpack_genotype
from evolly.utils import get_models
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import shutil
import os

import cv2
import numpy as np

from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.analytics import Databricks
from diagrams.custom import Custom

from time import sleep


plt.style.use('seaborn-paper')
plt.rc('figure', figsize=(8, 6))


_graph_attr = {
	"pad": "0.0",
	'splines': 'spline',
	"nodesep": "0.20",
	"ranksep": "0.35",
	"fontsize": "20",
	"labeljust": "l",
	# "bgcolor": "transparent"
}


def visualize_run(
		run_path: str,
		block_icons: list = None,
		metric_name: str = '',
		output_path: str = 'diagrams',
		oneline_placing: bool = True,
		metric_op: str = 'max',
		skip_init_model: bool = True,
		pad: int = 5,
		target_diagram_h: int = 90,
		frame_diagrams: int = 4,
		plot_pad_w: int = 1000,
		vid_width: int = 1920,
		vid_height: int = 600,
		vid_fps: int = 15,
		**kwargs,
):
	"""
	Make a video visualizing evolution progress of Evolly run. Video will contain:
	* Accuracy progress plot.
	* Diagram with evolution of backbone. Marked block with yellow
		background is a mutated one.
	Source plot and diagram frames will be saved as well.

	Each frame is a visualized model with its backbone architecture (diagram)
	and accuracy (plot).

	:param run_path: path to directory with run data.

	:param block_icons: dictionary with custom paths to block images. Mapping:
		{ block_type1: path1, block_type2: path2, ... }

	:param metric_name: name of the Y-axis on val_metric plot.

	:param output_path: path to the output directory.

	:param oneline_placing: whether to place plot and diagram
	in one line on the output video:
		* True - plot and diagram will be placed in one line
		* False - in two lines

	:param metric_op: whether to maximize or minimize val metrics
	to find "the best" model.

	:param skip_init_model: do not use first model while building plots and diagrams.
	If you manually pretrained initial model, some of the keys
	in a metadata json might be missed. In that case set this arg to True.

	:param pad: pad between diagrams in pixels.

	:param target_diagram_h: height of the output diagram.

	:param frame_diagrams: max number of diagrams per frame.

	:param plot_pad_w: horizontal pad of the plot

	:param vid_width: width of the output video.

	:param vid_height: height of the output video.

	:param vid_fps: output video FPS.
	"""

	diagrams_path = os.path.join(output_path, 'diagrams')
	if not os.path.exists(diagrams_path):
		os.makedirs(diagrams_path)

	plots_path = os.path.join(output_path, 'plots')
	if not os.path.exists(plots_path):
		os.makedirs(plots_path)

	trained_models, max_generation_id, max_model_id = get_models(run_path)

	# TODO check meta_info['config'] keys

	df = pd.DataFrame(columns=['model_id', 'val_metric'])

	op = max if metric_op == 'max' else min
	model_metrics = {}
	warning_raised = False
	models = {}
	for model_id, meta_info in trained_models.items():

		print((model_id / max_model_id) * 100, '%')

		if skip_init_model and model_id == 1:
			continue

		###########################
		# Unpack info needed for visualization and put it to models dict
		parent_id = meta_info['config']['model'].get('parent_id', '')
		branches = meta_info['config']['genotype'].get('branches')
		branch_names = meta_info['config']['genotype'].get('branch_names')
		mutated_depth_ids = meta_info['config']['genotype'].get('mutated_depth_ids', [])
		mutations_string = meta_info['config']['genotype'].get('mutations_string', '')
		parameters = meta_info.get('parameters') / 1e6			# millions
		flops = meta_info.get('parameters') / 1e9				# billions
		val_metric = op(meta_info.get('val_metric')) 			# best val metric
		fitness = meta_info.get('fitness')
		
		branch_name = branch_names[0]
		if not warning_raised:
			if len(branches) > 1:
				warnings.warn(
					f'Warning! Multiple branches is not supported in visualization yet. '
					f'First branch {branch_name} will be used to visualize evolly.\n'
					f'({len(branches)} branches have been passed)'
				)
			warning_raised = True

		unpacked_blocks, blocks_order = unpack_genotype(branches, branch_names)
		blocks_order = blocks_order[branch_name]

		models[model_id] = {
			'model_id': model_id,
			'parent_id': parent_id, 
			'blocks': unpacked_blocks, 
			'blocks_order': blocks_order,
			'mutated_depth_ids': mutated_depth_ids,
			'mutations_string': mutations_string,
			'parameters': parameters,
			'flops': flops,
			'val_metric': val_metric,
			'fitness': fitness,
		}

		if model_id not in model_metrics.keys():
			model_metrics[model_id] = val_metric

		best_model_id = op(model_metrics, key=model_metrics.get)
		best_val_metric = model_metrics[best_model_id]

		###########################
		# Create diagram with mutated backbone and save it to output_path
		filename = f'diagram_{model_id}.png'
		diagram_path = os.path.join(diagrams_path, filename)
		create_diagram(
			diagram_path=diagram_path,
			model_metrics=model_metrics,
			block_icons=block_icons,
			metric_name=metric_name,
			metric_op=metric_op,
			**models[model_id]
		)

		###########################
		# Create plot with model's val metric progress
		df = df.append({
			'model_id': model_id, 'val_metric': val_metric,
		}, ignore_index=True)

		ax = df.set_index('model_id').plot(
			use_index=True,
			y='val_metric',
			color='tab:orange',
			label='',
			grid=True,
			legend=False,
			linestyle='solid',
			linewidth=1.0,
			alpha=1.0,
		).set(xlabel='Model ID', ylabel=metric_name)

		plt.plot(
			[best_model_id], [best_val_metric],
			marker='o', markersize=6, color="green",
			label='The best model ID'
		)
		plt.plot(
			[model_id], [val_metric],
			marker='o', markersize=6, color="orange",
			label='Current model ID'
		)

		plt.legend().get_frame().set_alpha(None)
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.075), fancybox=True, ncol=2)

		plot_filename = f'plot_{model_id}.png'
		plot_path = os.path.join(plots_path, plot_filename)
		# plt.show()
		plt.savefig(
			plot_path,
			# transparent=True
		)
		plt.close()

	###########################
	# Compose diagrams and plots into a single image
	print('Saving output video...')
	init_diagram_pad = 100 if oneline_placing else 10
	plot_pad_h = (target_diagram_h + pad) * frame_diagrams + init_diagram_pad + 20

	if not oneline_placing:
		vid_width = int(vid_width / 1.5)
		vid_height = vid_height + plot_pad_h

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	output_video_path = os.path.join(output_path, 'visualized_run.mp4')
	vid_writer = cv2.VideoWriter(output_video_path, fourcc, vid_fps, (vid_width, vid_height))

	model_ids_range = list(range(
		2 if skip_init_model else 1, max_model_id)
	)

	diagram_filenames = os.listdir(diagrams_path)
	plots_filenames = os.listdir(plots_path)

	assert len(diagram_filenames) == len(plots_filenames)

	model_ids_chunks = chunk_to_n_size(model_ids_range, num=frame_diagrams)
	for model_ids in model_ids_chunks:

		out_image = np.full((vid_height, vid_width, 3), fill_value=255, dtype=np.uint8)

		diagram_pad = init_diagram_pad
		for model_id in model_ids:

			diagram_filename = get_filename_by_id(diagram_filenames, target_id=model_id)
			plot_filename = get_filename_by_id(plots_filenames, target_id=model_id)

			if not diagram_filename or not plot_filename:
				continue

			diagram_path = os.path.join(diagrams_path, diagram_filename)
			plot_path = os.path.join(plots_path, plot_filename)

			diagram = cv2.imread(diagram_path)
			plot = cv2.imread(plot_path)

			(diagram_h, diagram_w) = diagram.shape[:-1]
			(plot_h, plot_w) = plot.shape[:-1]

			diagram_ratio = target_diagram_h / diagram_h
			target_diagram_w = int(diagram_ratio * diagram_w)

			resized_diagram = cv2.resize(diagram, (target_diagram_w, target_diagram_h))

			out_image[diagram_pad:diagram_pad + target_diagram_h, 20:20 + target_diagram_w] = resized_diagram
			if oneline_placing:
				out_image[0:0 + plot_h, plot_pad_w:plot_pad_w + plot_w] = plot
			else:
				out_image[plot_pad_h:plot_pad_h + plot_h, 0:0 + plot_w] = plot

			diagram_pad += target_diagram_h + pad

		vid_writer.write(out_image)

	vid_writer.release()


def get_filename_by_id(filenames, target_id):
	for filename in filenames:
		model_id = int(filename.split('.')[0].split('_')[-1])
		if model_id == target_id:
			return filename
	return None


def chunk_to_n_size(seq, num):
	return [seq[i * num:(i + 1) * num] for i in range((len(seq) + num - 1) // num)]


def create_diagram(
		blocks=None,
		blocks_order=None,
		mutated_depth_ids=None,
		model_metrics=None,
		model_id=1,
		parent_id=1,
		parameters=0.0,
		val_metric=0.0,
		mutations_string='',
		metric_op='max',
		block_icons=None,
		diagram_path='',
		metric_name='val metric',
		**kwargs,
):
	better_than_parent = False
	if not parent_id or parent_id == 1:
		better_than_parent = True
	else:
		if metric_op == 'max' and val_metric > model_metrics[parent_id]:
			better_than_parent = True
		if metric_op == 'min' and val_metric < model_metrics[parent_id]:
			better_than_parent = True

	# Remove mutations branch from str
	mutations_string = ', '.join(mutations_string.split(',')[1:])
	metric_name = 'val metric' if not metric_name else metric_name
	diagram_name = \
		f"Model ID: {model_id} | " \
		f"Parent ID: {parent_id} | " \
		f"{metric_name}: {val_metric * 100:.2f} | " \
		f"Params: {parameters:.2f} mil |" \
		f" Mutations{mutations_string}"

	filename = diagram_path.split('/')[-1].split('.')[0]
	with Diagram(
		name=diagram_name,
		direction='LR',
		graph_attr=_graph_attr,
		filename=filename,
		outformat='png',
		show=False
	) as diagram:

		# E5F5FD blue, EBF3E7 green, ECE8F6 purple, FDF7E3 yellow, F6E8E8 red
		with Cluster(label='') as block_cluster:
			block_cluster.dot.graph_attr['bgcolor'] = '#EBF3E7' if better_than_parent else '#F6E8E8'

			unchanged_blocks = {}
			for depth_id, block_ids in blocks_order.items():
				if depth_id not in mutated_depth_ids:
					block_id = block_ids[0]
					block_data = blocks[block_id]
					block_type = block_data['block_type']
					block_name = get_block_name(depth_id, block_type)
					unchanged_blocks[depth_id] = Custom(
						block_name, block_icons[block_type]
					) if block_icons is not None else Databricks(block_name)

			# Mutated blocks must be defined inside the mutated_cluster
			# in order to display correctly
			mutated_blocks = {}
			for depth_id, block_ids in blocks_order.items():
				if depth_id in mutated_depth_ids:
					block_id = block_ids[0]
					block_data = blocks[block_id]
					block_type = block_data['block_type']
					block_name = get_block_name(depth_id, block_type)
					with Cluster(label=str(depth_id), graph_attr={'labeljust': 'l'}) as mutated_cluster:
						mutated_cluster.dot.graph_attr['bgcolor'] = '#FDF7E3'
						mutated_cluster.dot.graph_attr["label"] = ''
						mutated_blocks[depth_id] = Custom(
							block_name, block_icons[block_type]
						) if block_icons is not None else Databricks(block_name)

			node_start = None
			for depth_id, block_ids in blocks_order.items():

				if depth_id not in mutated_depth_ids:
					node = unchanged_blocks[depth_id]
				else:
					node = mutated_blocks[depth_id]

				# First depth_id
				if node_start is None:
					node_start = node
					continue

				node_end = node

				edge = Edge(color='black', style='bold', forward=True)
				diagram.connect(node=node_start, node2=node_end, edge=edge)

				node_start = node_end

	# To make sure that image has been saved
	sleep(0.01)

	shutil.move(src=f'{filename}.png', dst=diagram_path)


def get_block_name(depth_id, block_type):
	# TODO add output shapes to the end of block name?
	return f'#{depth_id} {block_type}'

