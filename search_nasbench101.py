import os
import json
import argparse

from nasbench101_cifar10 import NASCifar10A, NASCifar10B, NASCifar10C
from nasbench101_method import random_search, regularized_evolution, run_rl, run_bohb

parser = argparse.ArgumentParser()
parser.add_argument('--run_iter', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--run_method', default="bohb", type=str)
parser.add_argument('--runtime', default=1e7, type=float)
parser.add_argument('--benchmark', default="nas_cifar10a", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--output_path', default="result", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="checkpoint/", type=str, nargs='?', help='specifies the path to the tabular data')

args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)

# elif args.benchmark == "nas_cifar10c":
#     b = NASCifar10C(data_dir=args.data_dir)

os.makedirs(os.path.join(args.output_path), exist_ok=True)
cs = b.get_configuration_space()

if args.run_method == 'rs':
    random_search(args.runtime, b, cs)
elif args.run_method == 're':
    _ = regularized_evolution(b, cs, args.runtime,
                              population_size=100,
                              sample_size=10)
elif args.run_method == 'rl':
    run_rl(args.runtime, b, cs)
elif args.run_method == 'bohb':
    run_bohb(args.runtime, b, cs)

res, res_len = b.get_results(ignore_invalid_configs=True)
save_file = '%s_%d.txt' % (args.run_method, args.run_iter)
with open(os.path.join(args.output_path, save_file), "w") as f:
    for i in range(res_len):
        f.write('%.4f %.4f %.4f \n' % (res['runtime'][i],
                                       res['regret_validation'][i],
                                       res['regret_test'][i]))

# best_valid = 0.9505
# best_test = 0.9431