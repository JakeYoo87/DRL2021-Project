import os
import json
import argparse

from nasbench101_cifar10 import NASCifar10A, NASCifar10B, NASCifar10C

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--run_method', default="rs", type=str)
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

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir)

output_path = os.path.join(args.output_path, "random_search")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()

# runtime = []
# regret = []
# curr_incumbent = None
# curr_inc_value = None

last_time = 0
while b.get_runtime() < args.runtime:
    if b.get_runtime() - last_time > 1e5:
        last_time = b.get_runtime()
        print('runtime: %.4f best_valid: %.4f best_test: %.4f' % (b.get_runtime(), b.get_best_valid(), b.get_best_test()))
    config = cs.sample_configuration()
    b.objective_function(config)


res, res_len = b.get_results(ignore_invalid_configs=True)
with open(os.path.join(output_path, '%s_%d.txt' % (args.run_method, args.run_id)), "w") as f:
    for i in range(res_len):
        f.write('%.4f %.4f %.4f \n' % (res['runtime'][i], res['validation'][i], res['test'][i]))

# best_valid = 0.9505
# best_test = 0.9431