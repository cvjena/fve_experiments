import cvargparse as cvargs

from chainer_addons.models import PrepareType
from chainer_addons.training import OptimizerType


def _visualization_args(factory):
	parser = factory.add_mode("show")

	parser.add_args([
		cvargs.Arg("--subset", choices=["train", "test"], default="train"),
		cvargs.Arg("--n_samples", type=int, default=25),
	])

def _training_args(factory):
	parser = factory.add_mode("train")

	parser.add_args([
		cvargs.Arg("--fve_type", choices=["no", "grad", "em"],
			default="no",
			help="Type of parameter update."
			"\"no\": FVE-Layer is disabled, "
			"\"grad\": FVE-Layer parameters are learned with a gradient descent, "
			"\"em\": FVE-Layer parameters are learned with an iterative EM Algorithm."),

		cvargs.Arg("--n_components", default=1, type=int),
		cvargs.Arg("--comp_size", default=256, type=int),
		cvargs.Arg("--init_mu", default=0.0, type=float),
		cvargs.Arg("--init_sig", default=1.0, type=float),

		cvargs.Arg("--ema_alpha", default=0.99, type=float),
		cvargs.Arg("--aux_lambda", default=0.9, type=float),
		cvargs.Arg("--aux_lambda_rate", default=0.5, type=float),

		cvargs.Arg("--mask_features", action="store_true"),
		cvargs.Arg("--no_gmm_update", action="store_true"),
		cvargs.Arg("--only_mu_part", action="store_true"),

		cvargs.Arg("--augment_features", action="store_true"),
	], group_name="FVE arguments")

	parser.add_args(cvargs.ArgFactory([
			cvargs.Arg("--update_size", type=int, default=-1,
				help="if != -1 and > batch_size, then perform gradient acummulation until the update size is reached"),

			OptimizerType.as_arg("optimizer", "opt",
				help_text="type of the optimizer"),

			# cvargs.Arg("--from_scratch", action="store_true",
			# 	help="Do not load any weights. Train the model from scratch"),

			# cvargs.Arg("--label_smoothing", type=float, default=0,
			# 	help="Factor for label smoothing"),

			# cvargs.Arg("--test_fold_id",type=int, default=0,
			# 	help="ID of the test split"),

			# cvargs.Arg("--only_clf", action="store_true",
			# 	help="Train only the classification layer"),
		])\
		.batch_size()\
		.epochs()\
		.learning_rate(lr=1e-4, lrd=0.1, lrs=20, lrt=1e-6)\
		.weight_decay(default=5e-4),
		group_name="Training arguments")

	# parser.add_args([
	# 	cvargs.Arg("--augmentations",
	# 		choices=[
	# 			"random_crop",
	# 			"random_flip",
	# 			"random_rotation",
	# 			"center_crop",
	# 			"color_jitter"
	# 		],
	# 		default=["random_crop", "random_flip", "color_jitter"],
	# 		nargs="*"),

	# 	cvargs.Arg("--center_crop_on_val", action="store_true"),
	# 	cvargs.Arg("--brightness_jitter", type=int, default=0.3),
	# 	cvargs.Arg("--contrast_jitter", type=int, default=0.3),
	# 	cvargs.Arg("--saturation_jitter", type=int, default=0.3),

	# ], group_name="Augmentation options")

	parser.add_args([
		cvargs.Arg("--no_progress", action="store_true", help="dont show progress bar"),
		cvargs.Arg("--no_snapshot", action="store_true", help="do not save trained model"),
		cvargs.Arg("--output", "-o", type=str, default=".out", help="output folder"),
	], group_name="Output arguments")

def _dataset_args(factory):
	factory.subp_parent.add_args([
		# cvargs.Arg("data"),
		cvargs.Arg("dataset"),
		# cvargs.Arg("parts"),

		cvargs.Arg("--n_classes", type=int, default=10,
			help="number of classes"),

		cvargs.Arg("--extend_size", type=int, default=64,
			help="size of resulting images after transformation"),

		cvargs.Arg("--n_patches", type=int, default=10,
			help="number of patches for the clutter"),

		cvargs.Arg("--patch_size", type=int, default=10,
			help="size of clutter patches"),

		cvargs.Arg("--scale_down", action="store_true",
			help="scale down extended images to original size"),

		# cvargs.Arg("--label_shift", type=int, default=1,
		# 	help="label shift"),

		# cvargs.Arg("--swap_channels", action="store_true",
		# 	help="preprocessing option: swap channels from RGB to BGR"),

		# cvargs.Arg("--cache_images", action="store_true",
		# 	help="chaches resized (but not augmented yet) images. "
		# 	"reduces image processing times after the 1st epoch"),

		cvargs.Arg("--n_jobs", "-j", type=int, default=0,
			help="number of loading processes. If 0, then images are loaded in the same process"),

	], group_name="Dataset arguments")

def _model_args(factory):
	factory.subp_parent.add_args([
		cvargs.Arg("--model_type", "-mt"),

		PrepareType.as_arg("prepare_type",
			default="model",
			help_text="type of image preprocessing"),

		# cvargs.Arg("--separate_model", action="store_true",
		# 	help="if set, create a separate models for parts and the global image"),

		cvargs.Arg("--input_size", type=int, nargs="+"),

		cvargs.Arg("--load",
			help="loads trained weights (last classification layer is NOT changed)"),

		cvargs.Arg("--weights",
			help="loads pre-trained weights (last classification layer is changed"),

		cvargs.Arg("--load_path",
			help="loads a subtree of the given weights"),

		cvargs.Arg("--headless", action="store_true",
			help="ignore classifier weights during loading"),

		cvargs.Arg("--load_strict", action="store_true",
			help="load weights in a strict mode (raises errors if names do not match, etc.)"),

	], group_name="Model arguments")


def parse_args():

	factory = cvargs.ModeParserFactory(parser_cls=cvargs.GPUParser)
	factory.subp_parent.add_args(cvargs.ArgFactory()\
		.debug()\
		.seed())
	_dataset_args(factory)
	_model_args(factory)
	_training_args(factory)
	_visualization_args(factory)

	return factory.parse_args()

