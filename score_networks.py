import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
from tqdm import tqdm
import os
from scores import get_score_func
from scipy import stats
from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='datasets/cifar10/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='checkpoint/NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--nasspace', default='nds_resnet', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.batch_size)

# os.makedirs(args.save_loc, exist_ok=True)
# filename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}'
accs = np.zeros(len(searchspace))

for i, (uid, network) in tqdm(enumerate(searchspace)):
    try:
        network = network.to(device)
        # data_iterator = iter(train_loader)
        # x, target = next(data_iterator)
        # x, target = x.to(device), target.to(device)
        accs[i] = searchspace.get_final_accuracy(uid)
        if i% 1000 == 0:
            print(i)
    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid)
# np.save(filename, scores)
