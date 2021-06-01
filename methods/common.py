# common functions
def random_architecture(cs):
    config = cs.sample_configuration()
    return config


def print_stats(b, last_time, offset=1e5):
    if b.get_runtime() - last_time > offset:
        last_time = b.get_runtime()
        regret_validation, regret_test = b.get_regret()
        print('runtime: %.4f regret_validation: %.4f regret_test: %.4f'
              % (b.get_runtime(), regret_validation, regret_test))

    return last_time