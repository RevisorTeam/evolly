"""
Example of visualizing Evolly's run
(making video with evolution progress)
"""

from evolly import visualize_run


def main():

	# Path to run directory
	# run = 'tf_examples/searches/test_tuning'
	run = 'torch_examples/searches/test_searching'

	# Path to output directory with a video and diagram, plot images
	output_path = 'visualized_runs/test_searching'

	# Path to custom block icons.
	# For each block type you can set unique image
	# which will be used in a diagram.
	# If none will be passed, default image is used for all block types.
	block_icons = {
		'resnet': 'images/layers1.png',
		'mobilenet': 'images/layers2.png',
		'inception_a': 'images/layers3.png',
		'inception_b': 'images/layers4.png',
	}

	params = {
		'output_path': output_path,
		'oneline_placing': True,
		'block_icons': block_icons,
		'metric_name': 'accuracy',
		'metric_op': 'max',
		'show_diagrams': True,
		'save_diagrams': False,
		'target_diagram_h': 130,
		'vid_width': 2400,
		'vid_height': 1000,
	}

	visualize_run(run, **params)


if __name__ == '__main__':
	main()
