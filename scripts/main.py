import sys

sys.path.append("..")

from EAGLE.config import args
from EAGLE.utils.mutils import *
from EAGLE.utils.data_util import *
from EAGLE.utils.util import init_logger

import warnings
import networkx as nx

warnings.simplefilter("ignore")

# load data
args, data = load_data(args)

# pre-logs
log_dir = args.log_dir
init_logger(prepare_dir(log_dir) + "log_" + args.dataset + ".txt")
info_dict = get_arg_dict(args)

# Runner
from EAGLE.runner import Runner
from EAGLE.model import EADGNN
from EAGLE.model import ECVAE

model = EADGNN(args=args).to(args.device)
cvae = ECVAE(args=args).to(args.device)
runner = Runner(args, model, cvae, data)

results = []

if args.mode == "train":
    results = runner.run()
elif args.mode == "eval":
    results = runner.re_run()

# post-logs
measure_dict = results
info_dict.update(measure_dict)
filename = "info_" + args.dataset + ".json"
json.dump(info_dict, open(osp.join(log_dir, filename), "w"))
