# BOHB
from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
import logging


# logging.basicConfig(level=logging.ERROR)


class MyWorker(Worker):

    def __init__(self, b, run_id, id, nameserver, nameserver_port):
        super(MyWorker, self).__init__(run_id=run_id, id=id,
                                       nameserver=nameserver, nameserver_port=nameserver_port)
        self.b = b

    def compute(self, config, budget, **kwargs):
        y, cost = self.b.objective_function_from_config(config, budget=108)
        return ({
            'loss': float(y),
            'info': float(cost)})


def run_bohb(runtime, b, cs):
    min_budget = 4
    max_budget = 108
    hb_run_id = '0'
    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1
    workers = []

    for i in range(num_workers):
        w = MyWorker(b=b, run_id=hb_run_id, id=i,
                     nameserver=ns_host, nameserver_port=ns_port)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=cs, run_id=hb_run_id,
                min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host, nameserver_port=ns_port,
                ping_interval=10, min_bandwidth=0.3)

    n_iters = 300
    results = bohb.run(n_iters, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()