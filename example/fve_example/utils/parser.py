from cvargparse import Arg
from cvargparse import ArgFactory
from cvargparse import BaseParser
from cvargparse import GPUParser
from cvargparse import ModeParserFactory

from chainer_addons.models import PrepareType
from chainer_addons.training import OptimizerType

def _training_args(factory):

	parser = factory.add_mode("train")

	parser.add_args([
		Arg("--fve_type", choices=["no", "grad", "em"],
			default="no",
			help="Type of parameter update."
			"\"no\": FVE-Layer is disabled, "
			"\"grad\": FVE-Layer parameters are learned with a gradient descent, "
			"\"em\": FVE-Layer parameters are learned with an iterative EM Algorithm."),

		Arg("--n_components", default=1, type=int),
		Arg("--comp_size", default=256, type=int),
		Arg("--init_mu", default=0.0, type=float),
		Arg("--init_sig", default=1.0, type=float),

		Arg("--post_fve_size", default=0, type=int),

		Arg("--ema_alpha", default=0.99, type=float),
		Arg("--aux_lambda", default=0.9, type=float),
		Arg("--aux_lambda_rate", default=0.5, type=float),

		Arg("--mask_features", action="store_true"),
		Arg("--no_gmm_update", action="store_true"),
		Arg("--only_mu_part", action="store_true"),
		Arg("--normalize", action="store_true"),

		Arg("--augment_features", action="store_true"),
	], group_name="FVE arguments")

	parser.add_args(ArgFactory([
			Arg("--update_size", type=int, default=-1,
				help="if != -1 and > batch_size, then perform gradient acummulation until the update size is reached"),

			OptimizerType.as_arg("optimizer", "opt",
				help_text="type of the optimizer"),

			Arg("--from_scratch", action="store_true",
				help="Do not load any weights. Train the model from scratch"),

			Arg("--label_smoothing", type=float, default=0,
				help="Factor for label smoothing"),

			Arg("--test_fold_id",type=int, default=0,
				help="ID of the test split"),

			Arg("--only_clf", action="store_true",
				help="Train only the classification layer"),

			Arg("--analyze_features", action="store_true",
				help="Add a feature analyzis extension to report estimated and actual feature statistics"),

			Arg("--only_analyze", action="store_true",
				help="Do not train, but analyze features once and save resulting conv maps"),
		])\
		.debug()\
		.batch_size()\
		.seed()\
		.epochs()\
		.learning_rate(lr=1e-4, lrd=0.1, lrs=20, lrt=1e-6)\
		.weight_decay(default=5e-4),
		group_name="Training arguments")

	parser.add_args([
		Arg("--augmentations",
			choices=[
				"random_crop",
				"random_flip",
				"random_rotation",
				"center_crop",
				"color_jitter"
			],
			default=["random_crop", "random_flip", "color_jitter"],
			nargs="*"),

		Arg("--center_crop_on_val", action="store_true"),
		Arg("--brightness_jitter", type=int, default=0.3),
		Arg("--contrast_jitter", type=int, default=0.3),
		Arg("--saturation_jitter", type=int, default=0.3),

	], group_name="Augmentation options")

	parser.add_args([
		Arg("--no_progress", action="store_true", help="dont show progress bar"),
		Arg("--no_snapshot", action="store_true", help="do not save trained model"),
		Arg("--output", "-o", type=str, default=".out", help="output folder"),
	], group_name="Output arguments")

def _visualize_args(factory):
	parser = factory.add_mode("visualize")

	parser.add_args(ArgFactory([

			Arg("--subset", choices=["train", "val"], default="val"),
			Arg("--classes", nargs="*", type=int),
			Arg("--class_names"),
		])\
		.batch_size(),
		group_name="Visualization options")

class ModelChoices(object):
	choices = ["inception_imagenet", "inception", "resnet"]

	def __contains__(self, value):
		return value.startswith("cv2_") or value in self.choices

	def __iter__(self):
		return iter(self.choices + ["cv2_<any other model>"])

def parse_args():

	factory = ModeParserFactory(parser_cls=GPUParser)

	base_parser = factory.subp_parent

	base_parser.add_args([
		Arg("data"),
		Arg("dataset"),
		Arg("parts"),

		Arg("--label_shift", type=int, default=1,
			help="label shift"),

		Arg("--swap_channels", action="store_true",
			help="preprocessing option: swap channels from RGB to BGR"),

		# Arg("--cache_images", action="store_true",
		# 	help="chaches resized (but not augmented yet) images. "
		# 	"reduces image processing times after the 1st epoch"),

		Arg("--n_jobs", "-j", type=int, default=0,
			help="number of loading processes. If 0, then images are loaded in the same process"),

	], group_name="Dataset arguments")

	base_parser.add_args([
		Arg("--model_type", "-mt", choices=ModelChoices()),

		PrepareType.as_arg("prepare_type",
			default="model",
			help_text="type of image preprocessing"),

		Arg("--separate_model", action="store_true",
			help="if set, create a separate models for parts and the global image"),

		Arg("--input_size", type=int, nargs="+"),
		Arg("--parts_input_size", type=int, nargs="+"),

		Arg("--load",
			help="loads trained weights (last classification layer is NOT changed)"),

		Arg("--weights",
			help="loads pre-trained weights (last classification layer is changed"),

		Arg("--load_path",
			help="loads a subtree of the given weights"),

		Arg("--headless", action="store_true",
			help="ignore classifier weights during loading"),

		Arg("--load_strict", action="store_true",
			help="load weights in a strict mode (raises errors if names do not match, etc.)"),

	], group_name="Model arguments")

	_training_args(factory)
	_visualize_args(factory)

	return factory.parse_args()
