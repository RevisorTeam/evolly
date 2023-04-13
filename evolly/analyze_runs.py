import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from evolly.utils import get_models

plt.style.use('seaborn-paper')
plt.rc('figure', figsize=(8, 6))		# set default plot size

supported_scopes = ['models', 'generations']
supported_metric_ops = ['max', 'min']


def analyze_runs(
		runs: dict,
		scope: str = 'generations',			# models / generations
		metric_op: str = 'max',
		val_metric_name: str = 'val_metric',
		print_table: bool = False,
		skip_init_model: bool = True,
		draw_fitness: bool = True,
		moving_avg_step: int = 16,
		show_plots: bool = True,
		save_plots: bool = False,
		plots_path: str = 'plots',
		**kwargs		# for compatibility
) -> pd.DataFrame:
	"""
	Analyze Evolly's runs: parse metadata files, put them in
	pandas dataframe and build following plots:
				Y axis	   					|   	X axis
		1. Val_metrics 						| model_id or generation_id
		2. Val_metrics 						| parameters
		3. Flops (bil) 						| model_id or generation_id
		4. Parameters (mil) 				| model_id or generation_id
		5. Val_metric divided by parameters | model_id or generation_id


	:param runs: Dictionary with paths to the run directories. Mapping:
		{run1_name: run1_path, run2_name: run2_path, ...}
		You can pass as much runs as you want (>= 1).

	:param scope: atomic unit on a X-axis. Accepted values:
		* 'models' - each model's metadata will be plotted
		* 'generations' - only "the best" model of the generation
		will be picked up for plotting

	:param metric_op: whether to maximize or minimize val metrics
	to find "the best" model:
		* 'max' or 'min'

	:param val_metric_name: name of the Y-axis on val_metric plot

	:param print_table: whether to print resulting dataframe with parsed metadata

	:param skip_init_model: do not use first model while building plots.
	If you manually pretrained initial model, some of the keys
	in a metadata json might be missed. In that case set this arg to True.

	:param draw_fitness: whether to draw fitness curve on the first plot.
		* if set to True: both val_metric and fitness curves will be drawn
		* if set to false: only val_metric curve will be drawn

	:param moving_avg_step: step size of the moving average
	(used only in models scope).

	:param show_plots: whether to show plots in a window.

	:param save_plots: whether to save plots as an image to plots_path.

	:param plots_path: path where output images will be saved.

	:return: pandas dataframe with parsed run metadata.
	"""
	assert runs.keys() >= 1, \
		'Runs dictionary is empty! It must have at least one element.'

	assert scope in supported_scopes, \
		f'Scope {scope} is not in supported scopes: {supported_scopes}'

	assert metric_op in supported_metric_ops, \
		f'Metric operation {metric_op} is not in supported metric ' \
		f'operations: {supported_metric_ops}'

	if save_plots:
		if not os.path.exists(plots_path):
			os.makedirs(plots_path)
	
	legend = True if len(runs.keys()) > 1 else False

	colors = [
		'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
		'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
		'tab:olive', 'tab:cyan'
	]

	df = pd.DataFrame(columns=[
		'run_name', 'model_id', 'generation_id', 'parent_id', 'fitness', 'val_metric',
		'parameters', 'flops',
		'mutated_branch', 'mutation_type', 'mutated_depth_ids',
		'mutated_block_ids', 'mutations_string'
	])

	# Load results of all runs into a single dataframe (df)
	for run_name, run_path in runs.items():

		trained_models, max_generation_id, max_model_id = get_models(run_path)

		op = max if metric_op == 'max' else min
		for model_id, model_info in trained_models.items():

			if skip_init_model and model_id == 1:
				continue

			df = df.append({
				'run_name': run_name,
				'model_id': model_id,
				'generation_id': model_info['config']['model']['generation_id'],
				'parent_id': model_info['config']['model']['parent_id']
					if 'parent_id' in model_info['config']['model'].keys() else 0,
				'fitness': model_info['fitness'],
				'val_metric': op(model_info['val_metric']),
				'train_loss': op(model_info['train_loss']),
				'parameters': model_info['parameters'] / 1e6,
				'flops': model_info['flops'] / 1e9,
				'mutated_branch': model_info['config']['genotype']['mutated_branch']
					if 'mutated_branch' in model_info['config']['genotype'].keys() else '',
				'mutation_type': model_info['config']['genotype']['mutation_type']
					if 'mutation_type' in model_info['config']['genotype'].keys() else '',
				'mutated_depth_ids': model_info['config']['genotype']['mutated_depth_ids']
					if 'mutated_depth_ids' in model_info['config']['genotype'].keys() else '',
				'mutated_block_ids': model_info['config']['genotype']['mutated_block_ids']
					if 'mutated_block_ids' in model_info['config']['genotype'].keys() else '',
				'mutations_string': model_info['config']['genotype']['mutations_string']
					if 'mutations_string' in model_info['config']['genotype'].keys() else '',
			}, ignore_index=True)

		print(
			f"Stats of {run_name}:"
			f"\n\tgenerations total: {max_generation_id}"
			f"\n\tmodels total: {max_model_id}\n"
		)

	if scope == 'generations':
		df = df.groupby(['run_name', 'generation_id'], as_index=False).max()

	best_model = df.loc[df['fitness'].idxmax()]
	print(
		f"Best model: run {best_model['run_name']}, model_id {best_model['model_id']}, "
		f"generation_id {best_model['generation_id']}\n"
	)

	if print_table:
		print(f'models:')
		print(tabulate(df, headers='keys', tablefmt='psql'))

	if not show_plots and not save_plots:
		return df

	plot_index = ['model_id'] if scope == 'models' else ['generation_id']
	scope_x_title = 'Model ID' if scope == 'models' else 'Generation ID'

	#######################################
	# 1. Val_metrics / model_id graph:
	cols_to_plot = ['val_metric', 'fitness'] if draw_fitness else ['val_metric']
	val_metrics_df = df.pivot_table(
		index=plot_index,
		columns=['run_name'],
		values=cols_to_plot,
	).interpolate(method='nearest')

	# Smooth values by applying moving average window of size: moving_avg_step.
	# For models scope only (because of strong data dispersion)
	if scope == 'models':
		for run_name in runs.keys():
			if draw_fitness:
				val_metrics_df[('fitness', run_name)] = val_metrics_df[
					('fitness', run_name)].rolling(moving_avg_step).sum() / moving_avg_step
			val_metrics_df[('val_metric', run_name)] = val_metrics_df[
				('val_metric', run_name)].rolling(moving_avg_step).sum() / moving_avg_step

	axes = []
	run_names = list(runs.keys())
	ax_id = 0
	for run_id, run_name in enumerate(run_names):
		color_id = run_id % len(colors)
		for col_to_plot in cols_to_plot:
			ax = val_metrics_df.plot(
				use_index=True,		# x
				y=(col_to_plot, run_name),
				color=colors[color_id],
				label=f'{run_name}, {"fitness" if col_to_plot == "fitness" else val_metric_name}',
				grid=True,
				legend=legend,
				linestyle=':' if col_to_plot == 'fitness' else 'solid',
				linewidth=0.9 if col_to_plot == 'fitness' else 1.0,
				alpha=0.9 if col_to_plot == 'fitness' else 1.0,
				ax=None if ax_id == 0 else axes[ax_id - 1],
			)
			ax_id += 1

			if legend:
				ax.get_legend().set_title('')
				ax.get_legend().get_frame().set_alpha(None)

			axes.append(ax)
	axes[-1].set(xlabel=scope_x_title)

	if show_plots:
		plt.show()
	if save_plots:
		image_filename = f'search_progress-{scope}_scope.jpg'
		image_path = os.path.join(plots_path, image_filename)
		plt.savefig(image_path)

	#######################################
	# 2. Val_metrics / parameters graph:
	metric_params_df = df.pivot_table(
		index=plot_index,
		columns=['run_name'],
		values=['val_metric', 'parameters']
	).interpolate(method='nearest')

	axes = []
	run_names = list(runs.keys())
	for run_id, run_name in enumerate(run_names):
		color_id = run_id % len(colors)
		ax = metric_params_df.plot.scatter(
			x=('parameters', run_name),
			y=('val_metric', run_name),
			color=colors[color_id],
			label=run_name,
			alpha=0.5,
			edgecolors='none',
			grid=True,
			ax=None if run_id == 0 else axes[run_id - 1],
			legend=legend
		)
		if legend:
			ax.get_legend().set_title('')
			# ax.legend(loc='lower center', ncol=2)
			ax.get_legend().get_frame().set_alpha(None)

		axes.append(ax)
	axes[-1].set(xlabel='Parameters, mil', ylabel=val_metric_name)

	if show_plots:
		plt.show()
	if save_plots:
		image_filename = f'metric_w_params-{scope}_scope.jpg'
		image_path = os.path.join(plots_path, image_filename)
		plt.savefig(image_path)

	#######################################
	# 3. Flops (bil) / model_id graph:
	flops_df = df.pivot_table(
		index=plot_index,
		columns=['run_name'],
		values='flops',
	).interpolate(method='nearest')

	if scope == 'models':
		for run_name in runs.keys():
			flops_df[run_name] = flops_df[run_name].rolling(
				moving_avg_step).sum() / moving_avg_step

	flops_plot = flops_df.plot(
		use_index=True,
		y=list(runs.keys()),
		grid=True,
		legend=legend
	)
	if legend:
		plt.legend(list(runs.keys())).get_frame().set_alpha(None)

	flops_plot.set(xlabel=scope_x_title, ylabel='Flops, bil')

	if show_plots:
		plt.show()
	if save_plots:
		image_filename = f'flops-{scope}_scope.jpg'
		image_path = os.path.join(plots_path, image_filename)
		plt.savefig(image_path)

	#######################################
	# 4. Parameters (mil) / model_id graph:
	params_df = df.pivot_table(
		index=plot_index,
		columns=['run_name'],
		values='parameters',
	).interpolate(method='nearest')

	if scope == 'models':
		for run_name in runs.keys():
			params_df[run_name] = params_df[run_name].rolling(
				moving_avg_step).sum() / moving_avg_step

	params_plot = params_df.plot(
		use_index=True,
		y=list(runs.keys()),
		grid=True,
		legend=legend
	)
	if legend:
		plt.legend(list(runs.keys())).get_frame().set_alpha(None)

	params_plot.set(xlabel=scope_x_title, ylabel='Parameters, mil')

	if show_plots:
		plt.show()
	if save_plots:
		image_filename = f'parameters-{scope}_scope.jpg'
		image_path = os.path.join(plots_path, image_filename)
		plt.savefig(image_path)

	#######################################
	# 5. Val_metric : parameters / model_id graph:

	metric_div_params_df = df.pivot_table(
		index=plot_index,
		columns=['run_name'],
		values=['val_metric', 'parameters'],
	).interpolate(method='nearest')

	for run_name in runs.keys():
		metric_div_params_df[('metric_div_params', run_name)] =\
			metric_div_params_df[('val_metric', run_name)] / metric_div_params_df[('parameters', run_name)]

	# Change column name of metric_div_params from tuple to string
	metric_div_params_df.columns = [
		run_name if col_name == 'metric_div_params' else (col_name, run_name)
		for (col_name, run_name) in metric_div_params_df.columns
	]

	# print(tabulate(metric_div_params_df, headers='keys', tablefmt='psql'))

	if scope == 'models':
		for run_name in runs.keys():
			metric_div_params_df[run_name] = metric_div_params_df[run_name].rolling(
				moving_avg_step).sum() / moving_avg_step

	metric_div_params_plot = metric_div_params_df.plot(
		use_index=True,
		y=list(runs.keys()),
		grid=True,
		legend=legend
	)
	if legend:
		plt.legend(list(runs.keys())).get_frame().set_alpha(None)

	metric_div_params_plot.set(xlabel=scope_x_title, ylabel='Val metric divided by parameters')

	if show_plots:
		plt.show()
	if save_plots:
		image_filename = f'metric_div_params-{scope}_scope.jpg'
		image_path = os.path.join(plots_path, image_filename)
		plt.savefig(image_path)

	return df
