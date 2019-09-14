import concurrent.futures
import os
import subprocess
from functools import partial

import cloudpickle
import fire

def launch(name, f, *, namespace='safety', mode='local', mpi=1) -> None:
    if mode == 'local':
        with open('/tmp/pickle_fn', 'wb') as file:
            cloudpickle.dump(f, file)

        subprocess.check_call(['mpiexec', '-n', str(mpi), 'python', '-c', 'import sys; import pickle; pickle.loads(open("/tmp/pickle_fn", "rb").read())()'])
        return
    raise Exception('Other modes unimplemented!')

def parallel(jobs, mode):
    if mode == 'local':
        assert len(jobs) == 1, "Cannot run jobs in parallel locally"
        for job in jobs:
            job()
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(job) for job in jobs]
            for f in futures:
                f.result()

def launch_trials(name, fn, trials, hparam_class, extra_hparams=None, dry_run=False, mpi=1, mode='local', save_dir=None):
    jobs = []
    for trial in trials:
        descriptors = []
        kwargs = {}
        for k, v, s in trial:
            if k is not None:
                if k in kwargs:
                    print(f'WARNING: overriding key {k} from {kwargs[k]} to {v}')
                kwargs[k] = v
            if s.get('descriptor'):
                descriptors.append(str(s['descriptor']))
        hparams = hparam_class()
        hparams.override_from_dict(kwargs)
        if extra_hparams:
            hparams.override_from_str_dict(extra_hparams)
        job_name = (name + '/' + '-'.join(descriptors)).rstrip('/')
        hparams.validate()
        if dry_run:
            print(f"{job_name}: {kwargs}")
        else:
            if save_dir:
                hparams.run.save_dir = os.path.join(save_dir, job_name)
            trial_fn = partial(fn, hparams)
            jobs.append(partial(launch, job_name, trial_fn, mpi=mpi, mode=mode))

    parallel(jobs, mode=mode)

def main(commands_dict):
    """Similar to fire.Fire, but with support for multiple commands without having a class."""
    class _Commands:
        def __init__(self):
            for name, cmd in commands_dict.items():
                setattr(self, name, cmd)
    fire.Fire(_Commands)
