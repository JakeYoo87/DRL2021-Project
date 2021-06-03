import os
import argparse

from methods.bohb import run_bohb
from methods.random_search import random_search
from methods.regularized_evolutionary import regularized_evolution
from methods.reinforce import run_reinforce
from methods.rl import run_rl
from nasbench101_cifar10 import NASCifar10A, NASCifar10B

import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--run_method', default="rl", type=str)
parser.add_argument('--runtime', default=4e6, type=float)
parser.add_argument('--window_len', default=100, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--benchmark', default="nas_cifar10a", type=str, help='nas_cifar10a, nas_cifar10b')
parser.add_argument('--output_path', default="result", type=str)
parser.add_argument('--data_dir', default="checkpoint/", type=str)

args = parser.parse_args()

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)

os.makedirs(os.path.join(args.output_path), exist_ok=True)
cs = b.get_configuration_space()

if args.run_method == 'rs':
    random_search(args.runtime, b, cs)
elif args.run_method == 're':
    _ = regularized_evolution(b, cs, args.runtime,
                              population_size=100,
                              sample_size=10)
elif args.run_method == 'reinforce':
    run_reinforce(args.runtime, b, cs)
elif args.run_method == 'rl':
    run_rl(args.runtime, b, cs)
elif args.run_method == 'bohb':
    run_bohb(args.runtime, b, cs)

res, res_len = b.get_results(ignore_invalid_configs=False, window_len=args.window_len)
save_file = '%s_%.1E_seed%d.txt' % (args.run_method, args.runtime, args.seed)
with open(os.path.join(args.output_path, save_file), "w") as f:
    f.write('runtime val_min val_avg test_avg\n')
    record = 0
    record_offset = 1e4
    for i in range(res_len):
        if res['runtime'][i] >= record + record_offset:
            record = record + record_offset
            f.write('%d %.4f %.4f %.4f\n' % (record,
                                             res['regret_validations_min'][i],
                                             res['regret_validations_moving_avg'][i],
                                             res['regret_tests_moving_avg'][i])
                    )

# best_valid = 0.9505
# best_test = 0.9431
