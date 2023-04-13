"""
Example of analyzing Evolly's runs
"""

from evolly import analyze_runs


def main():

	# You can pass single or multiple runs at one time
	runs = {
		'test_tuning': 'examples/tf_examples/searches/test_tuning',
		'test_searching': 'examples/torch_examples/searches/test_searching',
	}

	# Path to directory with output plots
	plots_path = 'searches_plots/test'

	params = {
		'scope': 'models',  				# 'models' / 'generations'
		'metric_op': 'max',					# whether to maximize or minimize val metric
		'val_metric_name': 'accuracy',		# Y-axis title
		'print_table': True,
		'draw_fitness': False,
		'show_plots': False,
		'save_plots': False,
		'plots_path': plots_path,
	}

	# results_df is a pandas dataframe with runs stats
	results_df = analyze_runs(runs, **params)


if __name__ == '__main__':
	main()
