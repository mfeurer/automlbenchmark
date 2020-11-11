import logging
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
    print(f"In function get_hyperparameter_search_space_updates_lcbench")
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
    nics = os.listdir('/sys/class/net/') if os.path.exists('/sys/class/net/') else []
    nic = 'eth0' if 'eth0' in nics else 'enp2s0' if 'enp2s0' in nics else nics[1]
    print(f"Using nic={nic} from {nics}")
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
            'log_level': 'info',
            'loss_modules': ['cross_entropy_weighted'],
            'lr_scheduler': ['cosine_annealing'],
            'max_budget': max_budget,
            'max_runtime': max_runtime,
            'memory_limit_mb': memory_limit_mb,
            'min_budget': min_budget,
            'min_budget_for_cv': 0,
            'min_workers': num_workers,
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
    print(f"Just before returning the autonet config")
    return autonet_config

def get_ensemble_config():
    ensemble_config = {
            "ensemble_size":40,
            "ensemble_only_consider_n_best":10,
            "ensemble_sorted_initialization_n_best":0
            }
    return ensemble_config

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

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

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


    # Find the model that best fit the data
    # TODO add seed support when implemented on the framework
    is_classification = config.type == 'classification'
    autonet = AutoNetClassification if is_classification else AutoNetRegression

    max_search_runtime = config.max_runtime_seconds-5*60 if config.max_runtime_seconds>6*60 else config.max_runtime_seconds

    run_id = int(time.time())
    logdir = os.path.join(
        tmp.gettempdir(),
        "logs/run_" + str(run_id),
    )

    autonet_config = get_autonet_config_lcbench(
            min_budget=10,
            max_budget=50,
            max_runtime=max_search_runtime,
            run_id=run_id,
            task_id=1,
            num_workers=1,  #n_jobs
            logdir=logdir,
            seed=1,
            memory_limit_mb=ml_memory_limit,
    )

    if is_classification:
        autonet_config["algorithm"] = "portfolio_bohb"
        autonet_config["portfolio_type"] = "greedy"

    print(f"after the is classification")

    # Categoricals
    cat_feats = [type(f)==str for f in X_train[0]]
    if any(cat_feats):
        autonet_config["categorical_features"] = cat_feats
    autonet_config["embeddings"] = ['none', 'learned']

    # Test logging
    autonet_config["additional_logs"] = [test_predictions_for_ensemble.__name__, test_result_ens.__name__]

    # Set up ensemble
    print(f"Get the ensmeble config")
    ensemble_config = get_ensemble_config()
    autonet_config = {**autonet_config, **ensemble_config}
    autonet_config["optimize_metric"] = perf_metric
    print(f"Before creating the object autonet")
    auto_pytorch = AutoNetEnsemble(autonet, config_preset="full_cs", **autonet_config)

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

    with utils.Timer() as training:
        print(f"Just before calling print")
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
    print(f"probabilities={probabilities}")
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
