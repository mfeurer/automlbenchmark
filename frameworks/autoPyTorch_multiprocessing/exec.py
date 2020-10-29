from multiprocessing import Process
import logging
import typing
import math
import os
import tempfile as tmp
import warnings
import time

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import ConfigSpace as cs
import autoPyTorch.utils.create_trajectory as traj
from autoPyTorch import AutoNetClassification, AutoNetRegression, AutoNetEnsemble
from autoPyTorch import HyperparameterSearchSpaceUpdates
from autoPyTorch.pipeline.nodes import LogFunctionsSelector, BaselineTrainer
from autoPyTorch.components.metrics.additional_logs import *
from autoPyTorch.utils.ensemble import test_predictions_for_ensemble

from frameworks.shared.callee import call_run, result, utils, output_subdir

log = logging.getLogger(__name__)


def get_hyperparameter_search_space_updates_lcbench():
    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="InitializationSelector",
                                hyperparameter="initializer:initialize_bias",
                                value_range=["Yes"])
    search_space_updates.append(node_name="CreateDataLoader",
                                hyperparameter="batch_size",
                                value_range=[16, 512],
                                log=True)
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[50, 50])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:activation",
                                value_range=["relu"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:max_units",
                                value_range=[64, 1024],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:max_units",
                                value_range=[32,512],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:num_groups",
                                value_range=[1,5])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:blocks_per_group",
                                value_range=[1,3])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:resnet_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:activation",
                                value_range=["relu"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:mlp_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[1, 6])
    return search_space_updates


def get_autonet_config_lcbench(min_budget, max_budget, max_runtime, run_id, task_id, num_workers, logdir, seed, memory_limit_mb):
    autonet_config = {
            'additional_logs': [],
            'additional_metrics': ["balanced_accuracy"],
            'algorithm': 'bohb',
            'batch_loss_computation_techniques': ['standard', 'mixup'],
            'best_over_epochs': False,
            'budget_type': 'epochs',
            'categorical_features': None,
            #'cross_validator': 'stratified_k_fold',
            #'cross_validator_args': dict({"n_splits":5}),
            'cross_validator': 'none',
            'cuda': False,
            'dataset_name': None,
            'early_stopping_patience': 10,
            'early_stopping_reset_parameters': False,
            'embeddings': ['none', 'learned'],
            'eta': 2,
            'final_activation': 'softmax',
            'full_eval_each_epoch': True,
            'hyperparameter_search_space_updates': get_hyperparameter_search_space_updates_lcbench(),
            'imputation_strategies': ['mean'],
            'initialization_methods': ['default'],
            'initializer': 'simple_initializer',
            'log_level': 'debug',
            'loss_modules': ['cross_entropy_weighted'],
            'lr_scheduler': ['cosine_annealing'],
            'max_budget': max_budget,
            'max_runtime': max_runtime,
            'memory_limit_mb': memory_limit_mb,
            'min_budget': min_budget,
            'min_budget_for_cv': 0,
            'min_workers': num_workers,
            #'network_interface_name': 'eth0',
            'network_interface_name': 'lo',
            'networks': ['shapedmlpnet', 'shapedresnet'],
            'normalization_strategies': ['standardize'],
            'num_iterations': 300,
            'optimize_metric': 'accuracy',
            'optimizer': ['sgd', 'adam'],
            'over_sampling_methods': ['none'],
            'preprocessors': ['none', 'truncated_svd'],
            'random_seed': seed,
            'refit_validation_split': 0.33,
            'result_logger_dir': logdir,
            'run_id': run_id,
            'run_worker_on_master_node': True,
            'shuffle': True,
            'target_size_strategies': ['none'],
            'task_id': task_id,
            'torch_num_threads': 2,
            'under_sampling_methods': ['none'],
            'use_pynisher': True,
            'use_tensorboard_logger': False,
            'validation_split': 0.33,
            'working_dir': tmp.gettempdir(),
            }
    return autonet_config


def get_ensemble_config():
    ensemble_config = {
            "ensemble_size":40,
            "ensemble_only_consider_n_best":10,
            "ensemble_sorted_initialization_n_best":0
            }
    return ensemble_config


def get_autonet_instance_for_id(
    run_id: str,
    task_id: int,
    logdir: str,
    config: typing.Any,  # Automlbenchmark Namespace Object
    cat_feats: typing.List[str],
    ensemble_setting: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perf_metric: str,
) -> typing.Union[AutoNetClassification, AutoNetRegression, AutoNetEnsemble]:
    """
    Given an automl task setting, this function returns an autonet object that is able to fit a X_train/Y_train pair.

    Notice that for multiprocessing jobs this function is likely gonna be called with the same run_id among autopytorch objects,
    and task_id from 1 to n_jobs, where 1 is going to be the master

    Args:
        run_id (str): A common identifier among all autonet objects
        task_id (int): The id of the multiprocessing object, where 0 is master
        logdir (str): where the results are gonna be populated
        config (typing.Any): A configuration that dictates information about the task
        cat_feats (typing.List[str]): The list of categorical features
        ensemble_setting (str): ensemble/normal
        X_test (np.ndarray): testing features
        y_test (np.ndarray): testing labels
        perf_metric (str): The performance metric to optimizer

    Returns:
        Autonet object that is able to fit a model
    """
    # Extract runtime settings from the configuration
    max_search_runtime = config.max_runtime_seconds - 5*60 if config.max_runtime_seconds > 6*60 else config.max_runtime_seconds
    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    ml_memory_limit = config.framework_params.get('_ml_memory_limit', 'auto')

    # Use for now same auto memory setting from autosklearn
    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = utils.system_memory_mb().total
    if ml_memory_limit == 'auto':
        ml_memory_limit = max(min(config.max_mem_size_mb,
                                  math.ceil(total_memory_mb / n_jobs)),
                              3072)  # 3072 is autosklearn defaults

    autonet_config = get_autonet_config_lcbench(
            min_budget=10,
            max_budget=50,
            max_runtime=max_search_runtime,
            run_id=run_id,
            task_id=task_id,
            num_workers=n_jobs,
            logdir=logdir,
            seed=1,
            memory_limit_mb=ml_memory_limit,
    )

    autonet = AutoNetRegression
    if config.type == 'classification':
        autonet_config["algorithm"] = "portfolio_bohb"
        autonet_config["portfolio_type"] = "greedy"
        autonet = AutoNetClassification

    # Categoricals
    if any(cat_feats):
        autonet_config["categorical_features"] = cat_feats
    autonet_config["embeddings"] = ['none', 'learned']

    # Test logging
    autonet_config["additional_logs"] = [
        test_predictions_for_ensemble.__name__,
        test_result_ens.__name__
    ]

    # Set up ensemble
    ensemble_config = get_ensemble_config()
    autonet_config = {**autonet_config, **ensemble_config}
    autonet_config["optimize_metric"] = perf_metric

    if ensemble_setting == "ensemble":
        auto_pytorch = AutoNetEnsemble(autonet, config_preset="full_cs", **autonet_config)
    else:
        auto_pytorch = autonet(config_preset="full_cs", **autonet_config)

    # Test logging cont.
    auto_pytorch.pipeline[LogFunctionsSelector.get_name()].add_log_function(
        name=test_predictions_for_ensemble.__name__,
        log_function=test_predictions_for_ensemble(auto_pytorch, X_test, y_test),
        loss_transform=False
    )
    auto_pytorch.pipeline[LogFunctionsSelector.get_name()].add_log_function(
        name=test_result_ens.__name__,
        log_function=test_result_ens(auto_pytorch, X_test, y_test)
    )

    auto_pytorch.pipeline[BaselineTrainer.get_name()].add_test_data(X_test)

    return auto_pytorch


def delayed_auto_pytorch_fit(
    auto_pytorch: typing.Union[AutoNetClassification, AutoNetRegression, AutoNetEnsemble],
    arguments: typing.Dict[str, typing.Any]
) -> None:
    """
    We need the master to be ready when we setup this function. This handy
    function just fits a model AFTER waiting for the master to be ready

    Args:
        auto_pytorch ()
        arguments ()
    """

    time.sleep(10)
    print(f"started autopytorch child with {arguments} PID={os.getpid()}")
    try:
        auto_pytorch.fit(**arguments)
    except Exception as e:
        # Print as in multiprocessing, the stdout is dis-associated
        print(f"Failed with {e}")
    return


def launch_complementary_autopytorch_jobs(
    run_id: str,
    logdir: str,
    config: typing.Any,  # Automlbenchmark Namespace Object
    cat_feats: typing.List[str],
    ensemble_setting: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perf_metric: str,
) -> typing.List[Process]:
    """
    HPBanster requires that we launch the workers manually, so this function creates n_job-1 threads,
    accounting for the fact that the master process will also be a job.

    Args:
        run_id (str): A common identifier among all autonet objects
        task_id (int): The id of the multiprocessing object, where 0 is master
        logdir (str): where the results are gonna be populated
        config (typing.Any): A configuration that dictates information about the task
        cat_feats (typing.List[str]): The list of categorical features
        ensemble_setting (str): ensemble/normal
        X_test (np.ndarray): testing features
        y_test (np.ndarray): testing labels
        perf_metric (str): The performance metric to optimizer

    Returns:
        List of process launched for final join()
    """
    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    launched = []
    for task_id in range(2, n_jobs+1):
        auto_pytorch_child = get_autonet_instance_for_id(
            run_id=run_id,
            task_id=task_id,
            logdir=logdir,
            config=config,
            cat_feats=cat_feats,
            ensemble_setting=ensemble_setting,
            X_test=X_test,
            y_test=y_test,
            perf_metric=perf_metric
        )
        arguments = {
            'X_train': X_train,
            'Y_train': y_train,
        }
        arguments.update(auto_pytorch_child.get_current_autonet_config())
        arguments['refit'] = False
        p = Process(target=delayed_auto_pytorch_fit, args=(auto_pytorch_child, arguments))
        p.start()
        launched.append(p)

    return launched


def run(dataset, config):
    """
    This Method builds a Pytorch network that best fits a given dataset
    setting
    Args:
            dataset: object containing the data to be fitted as well as the expected values
            config: Configuration of additional details, like whether is a regression or
                    classification task
    Returns:
            A dict with the number of elements that conform the PyTorch network that best
            fits the data (models_count) and the duration of the task (training_duration)
    """

    log.info("\n**** AutoPyTorch ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    # Mapping of benchmark metrics to AutoPyTorch metrics
    # TODO Some metrics are not yet implemented in framework
    metrics_mapping = dict(
        acc='accuracy',
        bac='balanced_accuracy',
        auc='auc_metric',
        f1='auc_metric',
        logloss='cross_entropy',
        mae='mean_distance',
        mse='mean_distance',
        r2='auc_metric'
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        log.exception("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("Running AutoPyTorch with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric)
    log.info("Environment: %s", os.environ)

    # Data Processing
    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    X_test= dataset.test.X_enc
    y_test = dataset.test.y_enc
    cat_feats = [type(f)==str for f in X_train[0]]

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    # Find the model that best fit the data
    # TODO add seed support when implemented on the framework
    is_classification = config.type == 'classification'

    run_id = int(time.time())
    ensemble_setting = "ensemble"
    logdir = os.path.join(
        tmp.gettempdir(),
        "logs/run_" + str(run_id),
    )

    # Master Job
    auto_pytorch = get_autonet_instance_for_id(run_id, 1, logdir, config, cat_feats, ensemble_setting, X_test, y_test, perf_metric)

    # Child jobs if any
    complementary_autopytorch_jobs = launch_complementary_autopytorch_jobs(
        run_id,
        logdir,
        config,
        cat_feats,
        ensemble_setting,
        X_train,
        y_train,
        X_test,
        y_test,
        perf_metric
    )

    with utils.Timer() as training:
        auto_pytorch.fit(X_train, y_train, **auto_pytorch.get_current_autonet_config(), refit=False)

    # Build ensemble
    log.info("Building ensemble and predicting on the test set.")
    ensemble_config = traj.get_ensemble_config()

    simulator = traj.EnsembleTrajectorySimulator(ensemble_pred_dir=logdir, ensemble_config=ensemble_config, seed=1)
    simulator.simulate_trajectory()
    simulator.save_trajectory(save_file=os.path.join(logdir, "ensemble_trajectory.json"))
    simulator.save_trajectory(save_file=os.path.join(logdir, "ensemble_trajectory_test.json"), test=True)

    incumbent_score_val, incumbent_ind_val = simulator.get_incumbent_at_timestep(timestep=3600, use_val=True)

    predictions = simulator.test_preds[incumbent_ind_val]

    # Convert output to strings for classification
    if is_classification:
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`
    else:
        probabilities = None

    save_artifacts(auto_pytorch, config)
    [p.join() for p in complementary_autopytorch_jobs]
    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=None,
                  training_duration=training.duration)


def save_artifacts(autonet, config):
    try:
        models_repr = autonet.get_pytorch_model()
        log.debug("Trained Model:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
    except:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
