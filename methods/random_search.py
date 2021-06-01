import tensorflow as tf

from methods.common import print_stats, random_architecture


def random_search(runtime, b, cs):
    last_time = 0
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        b.objective_function(random_architecture(cs))
    tf.enable_eager_execution()
    tf.enable_resource_variables()
