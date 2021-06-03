from methods.common import print_stats, random_architecture


def random_search(runtime, b, cs):
    last_time = 0
    while b.get_runtime() < runtime:
        last_time = print_stats(b, last_time)
        b.objective_function_from_config(random_architecture(cs))
