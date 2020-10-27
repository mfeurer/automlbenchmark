import pandas as pd
import contextlib
import logging
import os
import psutil
import shutil

import h2o
from h2o.automl import H2OAutoML

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import to_data_frame, write_csv
#from amlb.results import NoResultError, save_predictions_to_file
#from amlb.utils import Monitoring, walk_apply, zip_path
from amlb.resources import config as rconfig
#from frameworks.shared.callee import output_subdir
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, balanced_accuracy_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, roc_auc_score
import math
from operator import itemgetter
from frameworks.shared.callee import call_run, result, output_subdir, utils

log = logging.getLogger(__name__)


#class BackendMemoryMonitoring(Monitoring):
#
#    def __init__(self, name=None, frequency_seconds=300, check_on_exit=False,
#                 verbosity=0, log_level=logging.INFO):
#        super().__init__(name, frequency_seconds, check_on_exit, "backend_monitoring_")
#        self._verbosity = verbosity
#        self._log_level = log_level
#
#    def _check_state(self):
#        sd = h2o.cluster().get_status_details()
#        log.log(self._log_level, "System memory (bytes): %s", psutil.virtual_memory())
#        log.log(self._log_level, "DKV: %s MB; Other: %s MB", sd['mem_value_size'][0] >> 20, sd['pojo_mem'][0] >> 20)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** H2O AutoML ****\n")
    # Mapping of benchmark metrics to H2O metrics
    metrics_mapping = dict(
        acc='mean_per_class_error',
        balacc='mean_per_class_error',
        auc='AUC',
        logloss='logloss',
        mae='mae',
        mse='mse',
        r2='r2',
        rmse='rmse',
        rmsle='rmsle'
    )
    sort_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if sort_metric is None:
        # TODO: Figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported, defaulting to AUTO.", config.metric)

    try:
        training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
        nthreads = config.framework_params.get('_nthreads', config.cores)
        jvm_memory = str(round(config.max_mem_size_mb * 2/3))+"M"   # leaving 1/3rd of available memory for XGBoost

        log.info("Starting H2O cluster with %s cores, %s memory.", nthreads, jvm_memory)
        max_port_range = 49151
        min_port_range = 1024
        rnd_port = os.getpid() % (max_port_range-min_port_range) + min_port_range
        port = config.framework_params.get('_port', rnd_port)

        h2o.init(nthreads=nthreads,
                 port=port,
                 min_mem_size=jvm_memory,
                 max_mem_size=jvm_memory,
                 strict_version_check=config.framework_params.get('_strict_version_check', True)
                 # log_dir=os.path.join(config.output_dir, 'logs', config.name, str(config.fold))
                 )

        # Load train as an H2O Frame, but test as a Pandas DataFrame
        log.debug("Loading train data from %s.", dataset.train_path)
        train = h2o.import_file(dataset.train_path, destination_frame=frame_name('train', config))
        # train.impute(method='mean')
        log.debug("Loading test data from %s.", dataset.test_path)
        test = h2o.import_file(dataset.test_path, destination_frame=frame_name('test', config))
        # test.impute(method='mean')

        log.info("Running model on task %s, fold %s.", config.name, config.fold)
        log.debug("Running H2O AutoML with a maximum time of %ss on %s core(s), optimizing %s.",
                  config.max_runtime_seconds, config.cores, sort_metric)

        aml = H2OAutoML(max_runtime_secs=config.max_runtime_seconds,
                        max_runtime_secs_per_model=round(config.max_runtime_seconds/2),  # to prevent timeout on ensembles
                        sort_metric=sort_metric,
                        seed=config.seed,
                        **training_params)

        #monitor = (BackendMemoryMonitoring(frequency_seconds=rconfig().monitoring.frequency_seconds,
        #                                  check_on_exit=True,
        #                                  verbosity=rconfig().monitoring.verbosity) if config.framework_params.get('_monitor_backend', False)
        #           # else contextlib.nullcontext  # Py 3.7+ only
        #           else contextlib.contextmanager(iter)([0])
        #           )
        with utils.Timer() as training:
            aml.train(y=dataset.target_index, training_frame=train)
            #with monitor:
            #    aml.train(y=dataset.target.index, training_frame=train)

        if not aml.leader:
            raise ValueError("H2O could not produce any model in the requested time.")

        # save_predictions(aml, test, dataset=dataset, config=config)
        overfit_frame = generate_overfit_artifacts(aml, config, test, dataset)
        save_artifacts(aml, dataset=dataset, config=config, overfit_frame=overfit_frame)

        #return dict(
        #    models_count=len(aml.leaderboard),
        #    training_duration=training.duration
        #)
        h2o_preds = aml.predict(test).as_data_frame(use_pandas=False)
        preds = to_data_frame(h2o_preds[1:], columns=h2o_preds[0])
        y_pred = preds.iloc[:, 0]

        h2o_truth = test[:, dataset.target_index].as_data_frame(use_pandas=False, header=False)
        y_truth = to_data_frame(h2o_truth)

        predictions = y_pred.values
        probabilities = preds.iloc[:, 1:].values
        truth = y_truth.values

        return result(output_file=config.output_predictions_file,
                      predictions=predictions,
                      truth=truth,
                      probabilities=probabilities,
                      # target_is_encoded=is_classification,
                      models_count=len(aml.leaderboard),
                      training_duration=training.duration)

    finally:
        if h2o.connection():
            # h2o.remove_all()
            h2o.connection().close()
        if h2o.connection().local_server:
            h2o.connection().local_server.shutdown()
        # if h2o.cluster():
        #     h2o.cluster().shutdown()


def frame_name(fr_type, config):
    return '_'.join([fr_type, config.name, str(config.fold)])


def generate_overfit_artifacts(estimator, config, test, dataset):

    models = [l for l in estimator.leaderboard.as_data_frame()['model_id'].to_list()]
    train_scores = []
    val_scores = []
    test_scores = []

    for model in models:
        model = h2o.get_model(model)
        test_metric = model.model_performance(test)

        if config.metric == 'acc':
            # empirically found :D, None means threshold and True are to get the train,..
            data = model.accuracy(None, True, True, True)
            train_scores.append(data['train'][0][1])
            val_scores.append(data['xval'][0][1])
            test_score = test_metric.accuracy()[0][1]
        elif config.metric == 'balacc':
            data = model.mean_per_class_error(None, True, True, True)
            train_scores.append(1-data['train'][0][1])
            val_scores.append(1-data['xval'][0][1])
            test_score = 1 - test_metric.mean_per_class_error()[0][1]
        else:
            if config.metric == 'auc':
                data = model.auc(True, True, True)
                test_score = test_metric.auc()
            if config.metric == 'logloss':
                data = model.logloss(True, True, True)
                test_score = test_metric.logloss()
            if config.metric == 'mae':
                data = model.mae(True, True, True)
                test_score = test_metric.mae()
            if config.metric == 'mse':
                data = model.mse(True, True, True)
                test_score = test_metric.mse()
            if config.metric == 'r2':
                data = model.r2(True, True, True)
                test_score = test_metric.r2()
            if config.metric == 'rmse':
                data = model.rmse(True, True, True)
                test_score = test_metric.rmse()
            if config.metric == 'rmsle':
                data = model.rmsle(True, True, True)
                test_score = test_metric.rmsle()
            train_scores.append(data['train'])
            val_scores.append(data['xval'])
        test_scores.append(test_score)

    dataframe = []
    individual = [(m, tr, v, te) for m, tr, v, te in zip(models, train_scores, val_scores, test_scores) if 'Stacked' not in m]
    m, tr, v, te = sorted(individual, key=itemgetter(2))[-1]
    dataframe.append({
        'model': 'best_individual_model' + m,
        'test': te,
        'val': v,
        'train': tr,
    })

    best_ensemble_index = models.index(str(estimator.leader.model_id))
    dataframe.append({
        'model': 'best_ensemble_model' + models[best_ensemble_index],
        'test': test_scores[best_ensemble_index],
        'val': val_scores[best_ensemble_index],
        'train': train_scores[best_ensemble_index],
    })

    return pd.DataFrame(dataframe)


def save_artifacts(automl, dataset, config, overfit_frame):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        lb = automl.leaderboard.as_data_frame()
        log.debug("Leaderboard:\n%s", lb.to_string())
        if 'overfit' in artifacts:
            overfit_file = os.path.join(output_subdir('overfit', config), 'overfit.csv')
            overfit_frame.to_csv(overfit_file)
        if 'leaderboard' in artifacts:
            models_dir = output_subdir("models", config)
            write_csv(lb, os.path.join(models_dir, "leaderboard.csv"))
        if 'models' in artifacts:
            models_dir = output_subdir("models", config)
            all_models_se = next((mid for mid in lb['model_id'] if mid.startswith("StackedEnsemble_AllModels")),
                                 None)
            mformat = 'mojo' if 'mojos' in artifacts else 'json'
            if all_models_se and mformat == 'mojo':
                save_model(all_models_se, dest_dir=models_dir, mformat=mformat)
            else:
                for mid in lb['model_id']:
                    save_model(mid, dest_dir=models_dir, mformat=mformat)
                models_archive = os.path.join(models_dir, "models.zip")
                #zip_path(models_dir, models_archive)

                def delete(path, isdir):
                    if path != models_archive and os.path.splitext(path)[1] in ['.json', '.zip']:
                        os.remove(path)
                #walk_apply(models_dir, delete, max_depth=0)

        if 'models_predictions' in artifacts:
            predictions_dir = output_subdir("predictions", config)
            test = h2o.get_frame(frame_name('test', config))
            for mid in lb['model_id']:
                model = h2o.get_model(mid)
                save_predictions(model, test,
                                 dataset=dataset,
                                 config=config,
                                 predictions_file=os.path.join(predictions_dir, mid, 'predictions.csv'),
                                 preview=False
                                 )
            #zip_path(predictions_dir,
            #         os.path.join(predictions_dir, "models_predictions.zip"))

            def delete(path, isdir):
                if isdir:
                    shutil.rmtree(path, ignore_errors=True)
            walk_apply(predictions_dir, delete, max_depth=0)

        if 'logs' in artifacts:
            logs_dir = output_subdir("logs", config)
            h2o.download_all_logs(dirname=logs_dir)
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


def save_model(model_id, dest_dir='.', mformat='mojo'):
    model = h2o.get_model(model_id)
    if mformat == 'mojo':
        model.save_mojo(path=dest_dir)
        # model.download_mojo(path=dest_dir, get_genmodel_jar=True)
    else:
        model.save_model_details(path=dest_dir)


if __name__ == '__main__':
    call_run(run)
