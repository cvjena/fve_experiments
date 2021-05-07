
from cvargparse import Arg
from cvargparse import BaseParser

def parse_args(*args, **kwargs):

	parser = BaseParser([

	    Arg("--analyze", nargs="*",
	    	choices=[
	    		"data", "data_change",
	    		"baseline",
	    		"classifier",
	    		"gradient"],

	    	default=["data", "baseline", "classifier"]
    	),
	    Arg("--output", type=str, default=None),
	])

	parser.add_args([
	    Arg("--n_dims", type=int, default=2),
	    Arg("--embed", action="store_true"),
	    Arg("--n_components", "-nc", type=int, default=1),

	    Arg("--n_classes", type=int, default=4),
	    Arg("--n_samples", type=int, default=128),

	    Arg("--sample_std", type=float, default=1),
	    Arg("--data_scale", type=float, default=10),
	    Arg("--data_shift", type=float, nargs="*"),
	], group_name="Data options")

	parser.add_args([
	    Arg("--init_mu", type=float, default=10),
	    Arg("--init_sig", type=float, default=1),

	    Arg("--fve_linear", action="store_true"),
	    Arg("--fve_only_mu", action="store_true"),
	    Arg("--fve_only_sig", action="store_true"),
	    Arg("--fve_normalize", action="store_true"),
	    Arg("--fve_no_update", action="store_true"),
	], group_name="FVE options")

	parser.add_args([
	    Arg("--epochs", type=int, default=100),
	    Arg("--batch_size", type=int, default=32),
	    Arg("--learning_rate", type=float, default=1e-2),

	    Arg("--seed", type=int),
	    Arg("--device", type=int, default=-1),
	], group_name="Training options")

	parser.add_args([
	    Arg("--no_plot", "-np", action="store_true"),
	], group_name="Plotting options")

	return parser.parse_args(*args, **kwargs)
