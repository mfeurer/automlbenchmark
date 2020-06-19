import logging
import json
import math
import multiprocessing
import os
import pickle
import tempfile as tmp
import warnings

import numpy as np

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import autosklearn
from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
import autosklearn.metrics as metrics
from ConfigSpace import Configuration

from frameworks.shared.callee import call_run, result, Timer, touch
from utils import system_memory_mb

log = logging.getLogger(__name__)


CALLBACK_COUNTER = multiprocessing.Value('i', 0)


def get_smac_object_callback(portfolio, lock):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)

        lock.acquire()
        try:
            global CALLBACK_COUNTER
            print(CALLBACK_COUNTER.value, flush=True)
            if CALLBACK_COUNTER.value == 0:
                initial_configurations = [
                    Configuration(configuration_space=scenario.cs, values=member)
                    for member in portfolio.values()]
            else:
                initial_configurations = [scenario.cs.sample_configuration(size=1)]
            CALLBACK_COUNTER.value += 1
        finally:
            lock.release()

        rh2EPM = RunHistory2EPM4LogCost
        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
        )
    return get_smac_object


def get_sh_or_hb_object_callback(budget_type, bandit_strategy, eta, initial_budget, portfolio,
                                 lock):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.intensification.hyperband import Hyperband
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)

        lock.acquire()
        try:
            global CALLBACK_COUNTER
            if CALLBACK_COUNTER.value == 0:
                initial_configurations = [
                    Configuration(configuration_space=scenario.cs, values=member)
                    for member in portfolio.values()]
            else:
                initial_configurations = [scenario.cs.sample_configuration(size=1)]
            CALLBACK_COUNTER.value += 1
        finally:
            lock.release()

        rh2EPM = RunHistory2EPM4LogCost

        ta_kwargs['budget_type'] = budget_type

        if bandit_strategy == 'sh':
            bandit = SuccessiveHalving
        elif bandit_strategy == 'hb':
            bandit = Hyperband
        else:
            raise ValueError(bandit_strategy)

        smac4ac = SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
            intensifier=bandit,
            intensifier_kwargs={
                'initial_budget': initial_budget,
                'max_budget': 100,
                'eta': eta,
                'min_chall': 1},
            )
        smac4ac.solver.epm_chooser.min_samples_model = int(len(scenario.cs.get_hyperparameters()) / 2)
        return smac4ac
    return get_smac_object


def run(dataset, config):
    log.info(f"\n**** AutoSklearn2={autosklearn.__version__} ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    # Mapping of benchmark metrics to autosklearn metrics
    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        bac=metrics.balanced_accuracy,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        rmse=metrics.mean_squared_error,  # autosklearn can optimize on mse, and we compute rmse independently on predictions
        r2=metrics.r2
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("Running auto-sklearn with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric)
    log.info("Environment: %s", os.environ)

    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    predictors_type = dataset.predictors_type
    log.debug("predictors_type=%s", predictors_type)
    # log.info("finite=%s", np.isfinite(X_train))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    ml_memory_limit = config.framework_params.get('_ml_memory_limit', 'auto')
    ensemble_memory_limit = config.framework_params.get('_ensemble_memory_limit', 'auto')

    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = system_memory_mb().total
    if ml_memory_limit == 'auto':
        ml_memory_limit = max(min(config.max_mem_size_mb / n_jobs,
                                  math.ceil(total_memory_mb / n_jobs)),
                              3072)  # 3072 is autosklearn defaults
    if ensemble_memory_limit == 'auto':
        ensemble_memory_limit = max(math.ceil(ml_memory_limit - (total_memory_mb - config.max_mem_size_mb)),
                                    math.ceil(ml_memory_limit / 3),  # default proportions
                                    1024)  # 1024 is autosklearn defaults
    log.info("Using %sMB memory per ML job and %sMB for ensemble job on a total of %s jobs.", ml_memory_limit, ensemble_memory_limit, n_jobs)

    log.warning("Using meta-learned initialization, which might be bad (leakage).")
    # TODO: do we need to set per_run_time_limit too?

    if is_classification:
        this_directory = os.path.dirname(__file__)
        selector_file = os.path.join(this_directory, 'selector_0.pkl')
        with open(selector_file, 'rb') as fh:
            selector = pickle.load(fh)
        metafeatures = np.array([len(np.unique(y_train)), X_train.shape[1], X_train.shape[0]])
        selection = np.argmax(selector['selector'].predict(metafeatures))
        automl_policy = selector['methods_to_choose_from'][selection]
        print('Selected policy', automl_policy)

        """
        Get portfolios from github with
        
        for policy in RF_None_holdout_iterative_es_if RF_None_3CV_iterative_es_if RF_None_5CV_iterative_es_if RF_None_10CV_iterative_es_if ;
        do 
            scp feurerm@aadlogin.informatik.uni-freiburg.de:/mhome/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/60MIN/ASKL_create_portfolio/${policy}/32_None_None_0.json /home/feurerm/sync_dir/projects/openml/automlbenchmark/frameworks/autosklearn2/portfolios/${policy}.json
        done
        
        for policy in RF_SH-eta4-i_holdout_iterative_es_if RF_SH-eta4-i_3CV_iterative_es_if RF_SH-eta4-i_5CV_iterative_es_if RF_SH-eta4-i_10CV_iterative_es_if;
        do 
            scp feurerm@aadlogin.informatik.uni-freiburg.de:/mhome/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/60MIN/ASKL_create_portfolio/${policy}/32_SH_None_0.json /home/feurerm/sync_dir/projects/openml/automlbenchmark/frameworks/autosklearn2/portfolios/${policy}.json
        done
        """

        setting = {
            'RF_None_holdout_iterative_es_if': {
                'resampling_strategy': 'holdout-iterative-fit',
                'fidelity': None,
            },
            'RF_None_3CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 3,
                'fidelity': None,
            },
            'RF_None_5CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 5,
                'fidelity': None,
            },
            'RF_None_10CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 10,
                'fidelity': None,
            },
            'RF_SH-eta4-i_holdout_iterative_es_if': {
                'resampling_strategy': 'holdout-iterative-fit',
                'fidelity': 'SH',
            },
            'RF_SH-eta4-i_3CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 3,
                'fidelity': 'SH',
            },
            'RF_SH-eta4-i_5CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 5,
                'fidelity': 'SH',
            },
            'RF_SH-eta4-i_10CV_iterative_es_if': {
                'resampling_strategy': 'cv-iterative-fit',
                'folds': 10,
                'fidelity': 'SH',
            }
        }[automl_policy]

        resampling_strategy = setting['resampling_strategy']
        if resampling_strategy == 'cv-iterative-fit':
            resampling_strategy_kwargs = {'folds': setting['folds']}
        else:
            resampling_strategy_kwargs = None

        portfolio_file = os.path.join(this_directory, 'portfolios', '%s.json' % automl_policy)
        with open(portfolio_file) as fh:
            portfolio_json = json.load(fh)
        portfolio = portfolio_json['portfolio']

        lock = multiprocessing.Lock()
        if setting['fidelity'] == 'SH':
            smac_callback = get_sh_or_hb_object_callback('iterations', 'sh', 4, 5.0, portfolio,
                                                         lock)
        else:
            smac_callback = get_smac_object_callback(portfolio, lock)

        auto_sklearn = AutoSklearnClassifier(
            time_left_for_this_task=config.max_runtime_seconds,
            n_jobs=n_jobs,
            ml_memory_limit=ml_memory_limit,
            ensemble_memory_limit=ensemble_memory_limit,
            seed=config.seed,
            resampling_strategy=resampling_strategy,
            resampling_strategy_arguments=resampling_strategy_kwargs,
            initial_configurations_via_metalearning=0,
            get_smac_object_callback=smac_callback,
            include_estimators=[
                'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting',
            ],
            include_preprocessors=["no_preprocessing"],
            **training_params
        )

    else:

        auto_sklearn = AutoSklearnRegressor(time_left_for_this_task=config.max_runtime_seconds,
                                            n_jobs=n_jobs,
                                            ml_memory_limit=ml_memory_limit,
                                            ensemble_memory_limit=ensemble_memory_limit,
                                            seed=config.seed,
                                            **training_params)
    with Timer() as training:
        auto_sklearn.fit(X_train, y_train, metric=perf_metric, feat_type=predictors_type)

    # Convert output to strings for classification
    log.info("Predicting on the test set.")
    X_test = dataset.test.X_enc
    y_test = dataset.test.y_enc
    predictions = auto_sklearn.predict(X_test)
    probabilities = auto_sklearn.predict_proba(X_test) if is_classification else None

    save_artifacts(auto_sklearn, config)

    print(auto_sklearn.sprint_statistics())

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(auto_sklearn.get_models_with_weights()),
                  training_duration=training.duration)


def make_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


def save_artifacts(estimator, config):
    try:
        models_repr = estimator.show_models()
        log.debug("Trained Ensemble:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            models_file = os.path.join(make_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
    except:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
