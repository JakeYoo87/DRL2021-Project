import os

import ConfigSpace
import numpy as np
from nasbench import api
from nasbench.lib import graph_util
import collections

MAX_EDGES = 9
VERTICES = 7


class NASCifar10(object):

    def __init__(self, data_dir, multi_fidelity=False):

        self.multi_fidelity = multi_fidelity
        if self.multi_fidelity:
            self.dataset = api.NASBench(os.path.join(data_dir, 'nasbench_full.tfrecord'))
        else:
            self.dataset = api.NASBench(os.path.join(data_dir, 'nasbench_only108.tfrecord'))
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.num_edges = []

        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []

    @staticmethod
    def objective_function_from_config(self, config):
        pass

    def get_runtime(self):
        return sum(self.costs)

    def get_regret(self):
        min_index = self.y_valid.index(min(self.y_valid))
        regret_validation = self.y_valid[min_index] - self.y_star_valid
        regret_test = self.y_test[min_index] - self.y_star_test
        return regret_validation, regret_test

    def record_invalid(self, config, valid, test, costs):
        self.X.append(config)
        self.y_valid.append(valid)
        self.y_test.append(test)
        self.costs.append(costs)

    def record_valid(self, config, data, model_spec, num_edges):

        self.num_edges.append(num_edges)
        self.X.append(config)

        # compute mean test error for the final budget
        _, metrics = self.dataset.get_metrics_from_spec(model_spec)
        mean_test_error = 1 - np.mean([metrics[108][i]["final_test_accuracy"] for i in range(3)])
        self.y_test.append(mean_test_error)

        # compute validation error for the chosen budget
        valid_error = 1 - data["validation_accuracy"]
        self.y_valid.append(valid_error)

        runtime = data["training_time"]
        self.costs.append(runtime)

    @staticmethod
    def get_configuration_space():
        pass

    def get_results(self, ignore_invalid_configs=False, window_len=10):

        regret_validations = []
        regret_validations_min = []
        regret_validations_window = collections.deque(maxlen=window_len)
        regret_validations_moving_avg = []
        #
        regret_tests = []
        regret_tests_min = []
        regret_tests_window = collections.deque(maxlen=window_len)
        regret_tests_moving_avg = []
        #
        runtime = []
        rt = 0

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            regret_validation = float(self.y_valid[i] - self.y_star_valid)
            regret_test = float(self.y_test[i] - self.y_star_test)
            #
            regret_validations.append(regret_validation)
            regret_validations_min.append(np.min(regret_validations))
            regret_validations_window.append(regret_validation)
            regret_validations_moving_avg.append(np.mean(regret_validations_window))

            regret_tests.append(regret_test)
            regret_tests_min.append(np.min(regret_tests))
            regret_tests_window.append(regret_validation)
            regret_tests_moving_avg.append(np.mean(regret_tests_window))
            #
            rt += self.costs[i]
            runtime.append(float(rt))

        res = dict()
        res['regret_validations'] = regret_validations
        res['regret_tests'] = regret_tests
        res['regret_validations_min'] = regret_validations_min
        res['regret_tests_min'] = regret_tests_min
        res['regret_validations_moving_avg'] = regret_validations_moving_avg
        res['regret_tests_moving_avg'] = regret_tests_moving_avg
        res['runtime'] = runtime
        res['num_edges_avg'] = np.mean(self.num_edges)

        return res, len(runtime)


class NASCifar10A(NASCifar10):
    def objective_function_from_config(self, config, budget=108, cond_record=0):
        if self.multi_fidelity is False:
            assert budget == 108

        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = config["edge_%d" % i]

        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        num_edges = graph_util.num_edges(matrix)
        if num_edges > MAX_EDGES:
            # self.record_invalid(config, 1, 1, 0)
            return 1, 0

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            # self.record_invalid(config, 1, 1, 0)
            return 1, 0

        if cond_record > 0:
            if num_edges >= cond_record:
                self.record_valid(config, data, model_spec, num_edges)
        else:
            self.record_valid(config, data, model_spec, num_edges)
        return 1 - data["validation_accuracy"], data["training_time"]

    def objective_function_from_matrix(self, matrix, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            # self.record_invalid(config, 1, 1, 0)
            return 1, 0

        labeling = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu',
                    'conv3x3-bn-relu', 'maxpool3x3', 'output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            # self.record_invalid(config, 1, 1, 0)
            return 1, 0

        self.record_valid(matrix, data, model_spec)
        return 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs


class NASCifar10B(NASCifar10):
    def objective_function_from_config(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        bitlist = [0] * (VERTICES * (VERTICES - 1) // 2)
        for i in range(MAX_EDGES):
            bitlist[config["edge_%d" % i]] = 1
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit

        matrix = np.fromfunction(graph_util.gen_is_edge_fn(out),
                                 (VERTICES, VERTICES),
                                 dtype=np.int8)
        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        self.record_valid(config, data, model_spec)

        return 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        cat = [i for i in range((VERTICES * (VERTICES - 1)) // 2)]
        for i in range(MAX_EDGES):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, cat))
        return cs
