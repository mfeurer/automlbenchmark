from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir



def setup(*args, **kwargs):
    #call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


#def run(*args, **kwargs):
def run(dataset: Dataset, config: TaskConfig):
    #from .exec import run
    from frameworks.shared.caller import run_in_venv
    #return run(*args, **kwargs)
    data = dict(
        train=dict(
            X_enc=dataset.train.X_enc,
            y_enc=dataset.train.y_enc
        ),
        test=dict(
            X_enc=dataset.test.X_enc,
            y_enc=dataset.test.y_enc
        ),
        train_path=dataset.train.path,
        test_path=dataset.test.path,
        target_index=dataset.target.index,
    )

    return run_in_venv(__file__, "exec.py",
                        input_data=data, dataset=dataset, config=config)


def docker_commands(*args, setup_cmd=None):
    return """
{cmd}
EXPOSE 54321
EXPOSE 54322
""".format(
        cmd="RUN {}".format(setup_cmd) if setup_cmd is not None else ""
    )


#There is no network isolation in Singularity,
#so there is no need to map any port.
#If the process inside the container binds to an IP:port,
#it will be immediately reachable on the host.
def singularity_commands(*args, setup_cmd=None):
    return """
{cmd}
""".format(
        cmd="{}".format(setup_cmd) if setup_cmd is not None else ""
    )
