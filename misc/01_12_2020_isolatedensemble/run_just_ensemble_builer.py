############################################################
# Imports
############################################################
import logging.handlers
import os
import autosklearn.classification
import autosklearn.metrics as metrics
from shutil import copyfile
from sklearn.metrics import balanced_accuracy_score


############################################################
# Variables from the environment
############################################################
#INPUT_DIR = os.environ['INPUT_AUTOSKLEARNDIR']
#SEED = os.environ['SEED']
INPUT_DIR = '/home/chico/master_thesis/automlbenchmark/results/autosklearncopyall_test_test_docker_20201201T180904/debug/kc2/1'
SEED = 73421283

if __name__ == '__main__':
    strategies = [
        'autosklearnBBCEnsembleSelectionNoPreSelect',
        'autosklearnBBCEnsembleSelection',
        'autosklearnBBCScoreEnsemble',
        'autosklearnBBCEnsembleSelection',
        'autosklearnBBCEnsembleSelectionNoPreSelect',
        'autosklearnBBCSMBOAndEnsembleSelection',
	'autosklearnBBCEnsembleSelectionPreSelectInES',
        'bagging',
    ]
    # Prepare the automl object
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=3600,
        n_jobs=1,
        memory_limit=4096,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        seed=SEED,
        metric=metrics.balanced_accuracy,
        per_run_time_limit=30,
    )
    cls.automl_ = cls.build_automl(
        seed=SEED,
        ensemble_size=cls.ensemble_size,
        initial_configurations_via_metalearning=0,
        tmp_folder=cls.tmp_folder,
        output_folder=cls.output_folder,
    )

    # Copy the data from existing run
    files_to_copy = []
    for r, d, f in os.walk(os.path.join(
        INPUT_DIR,
        '.auto-sklearn',
    )):
        for file_name in f:
            # Ignore the ensemble memory
            if 'ensemble_' in file_name: continue
            files_to_copy.append(os.path.join(r, file_name))
    for filename in files_to_copy:
        dst = filename.replace(INPUT_DIR, cls.automl_._backend.temporary_directory + '/')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            copyfile(filename, dst)

    # Restore original run specific settings
    cls.automl_._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
    datamanager = cls.automl_._backend.load_datamanager()
    cls.automl_._task = datamanager.info['task']
    cls.automl_._label_num = datamanager.info['label_num']
    X_test = datamanager.data['X_test']
    Y_test = datamanager.data['Y_test']
    dataset_name = datamanager.name
    y_train = cls.automl_._backend.load_targets_ensemble()

    # Then finally fit the ensemble
    cls.fit_ensemble(y_train, ensemble_size=50, dataset_name=dataset_name)

    # Score
    predictions = cls.predict(X_test)
    print("Accuracy score:", balanced_accuracy_score(Y_test, predictions))

    # Cleanup this mess
    cls.automl_._clean_logger()
    cls.automl_._close_dask_client()
