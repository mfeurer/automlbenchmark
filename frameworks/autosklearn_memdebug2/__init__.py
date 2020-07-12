from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    try:
        print(f"dataset.train.X_enc={dataset.train.X_enc}")
        print(f"dataset.train.y_enc={dataset.train.y_enc}")
        print(f"dataset.test.X_enc={dataset.test.X_enc}")
        print(f"dataset.test.y_enc={dataset.test.y_enc}")
        print(f"dataset.predictors={dataset.predictors}")
        predictors_type=['Categorical' if p.is_categorical() else 'Numerical' for p in dataset.predictors]
        print(f"predictors_type={predictors_type}")
        for p in dataset.predictors:
            print(f"p={p} type={type(p)}")
            print(f"p.is_categorical()={p.is_categorical()}")

        data = dict(
            train=dict(
                X_enc=dataset.train.X_enc,
                y_enc=dataset.train.y_enc
            ),
            test=dict(
                X_enc=dataset.test.X_enc,
                y_enc=dataset.test.y_enc
            ),
            predictors_type=['Categorical' if p.is_categorical() else 'Numerical' for p in dataset.predictors]
        )
        print(f)
    except Exception as e:
        print(f"Exception={e}")
        raise e

    print(f"Input data is data={data}")
    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

