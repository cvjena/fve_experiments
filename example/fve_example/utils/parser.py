import logging
import platform
import chainermn as mn

from cvargparse import Arg
from cvargparse import ArgFactory
from cvargparse import GPUParser
from cvargparse import ModeParserFactory
from cvargparse import utils

from chainer_addons.models import PrepareType
from chainer_addons.training import OptimizerType
from cvfinetune import parser as parser_module

class Parser(GPUParser):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._file_handler = None

	def _logging_config(self, simple=False):
		if not self.has_logging: return
		fmt = '{levelname:s} - [{asctime:s}] {filename:s}:{lineno:d} [{funcName:s}]: {message:s}'

		handlers = []
		comm = mn.create_communicator("pure_nccl")
		if comm.rank == 0:
			handler0 = logging.StreamHandler()
			handler0.addFilter(parser_module.base.HostnameFilter())
			fmt0 = "<{hostname:^10s}>: " + fmt
			handlers.append((handler0, fmt0, logging.INFO))

		if self._args.logfile in [ None, "" ] :
			filename = f"{platform.node()}.log"
		else:
			filename = self._args.logfile

		self._file_handler = handler1 = logging.FileHandler(filename=filename, mode="w")
		handlers.append((handler1, fmt, logging.INFO))

		utils.logger_config.init_logging_handlers(handlers)

	def __del__(self):
		try:
			if getattr(self, "_file_handler", None) is not None:
				self._file_handler.flush()
		except Exception as e:
			warnings.warn("Could not flush logs to file: {}".format(e))

def add_model_args(factory: ModeParserFactory):
	base_parser = factory.subp_parent
	parser_module.add_model_args(base_parser)
	base_parser.add_args([
		Arg("--feature_aggregation", choices=["mean", "concat"], default="mean",
			help="Part feature aggregation after GAP. Ignored in case of FVE"),

		Arg("--pred_comb", choices=["no", "sum", "linear"], default="no",
			help="Combination strategy of the global and part features"),

		Arg("--copy_mode", choices=["copy", "share", "init"], default="copy",
			help="Copy mode for the separate model. See chainer.Link.copy for more information"),

	], group_name="Model arguments")


def add_dataset_args(factory: ModeParserFactory):
	base_parser = factory.subp_parent
	parser_module.add_dataset_args(base_parser)

	base_parser.add_args([
		Arg("--shuffle_parts", action="store_true"),
	], group_name="Dataset arguments")

def add_training_args(factory: ModeParserFactory):

	parser = factory.add_mode("train")
	parser.add_args([
		Arg("--profile", action="store_true"),
		Arg("--only_klass", type=int),
	])

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
		Arg("--aux_lambda_step", default=20, type=int),

		Arg("--mask_features", action="store_true"),
		Arg("--no_gmm_update", action="store_true"),
		Arg("--only_mu_part", action="store_true"),
		Arg("--normalize", action="store_true"),

		Arg("--no_sacred", action="store_true"),

		Arg("--augment_features", action="store_true"),
	], group_name="FVE arguments")

	parser_module.add_training_args(parser)

	parser.add_args([
			Arg("--update_size", type=int, default=-1,
				help="if != -1 and > batch_size, then perform gradient acummulation until the update size is reached"),

			Arg("--test_fold_id", type=int, default=0,
				help="ID of the test split"),

			Arg("--analyze_features", action="store_true",
				help="Add a feature analyzis extension to report estimated and actual feature statistics"),

			Arg("--mpi", action="store_true",
				help="use multi-GPU training with OpenMPI"),

			Arg("--only_analyze", action="store_true",
				help="Do not train, but analyze features once and save resulting conv maps"),

			Arg("--loss_scaling", type=float, default=65000.0),
			Arg("--opt_epsilon", type=float, default=1e-1,
				help="epsilon for Adam and RMSProp optimizers"),
		], group_name="Training arguments")

def add_visualize_args(factory: ModeParserFactory):
	parser = factory.add_mode("visualize")

	parser.add_args(ArgFactory([

		Arg("--subset", choices=["train", "val"], default="val"),
		Arg("--classes", nargs="*", type=int),
		Arg("--class_names"),
	])\
	.batch_size(),
	group_name="Visualization options")

def add_evaluation_args(factory: ModeParserFactory):

	parser = factory.add_mode("evaluate")
	parser.add_args([
		Arg("--mpi", action="store_true",
			help="use multi-GPU evaluation with OpenMPI"),

		Arg("--batch_size", type=int, default=32,
			help="use multi-GPU evaluation with OpenMPI"),

	], group_name="Evaluation arguments")


def parse_args():

	factory = ModeParserFactory(parser_cls=Parser)


	add_model_args(factory)
	add_dataset_args(factory)
	add_training_args(factory)
	add_evaluation_args(factory)
	add_visualize_args(factory)


	return factory.parse_args()
