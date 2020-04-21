import argparse
import collections
import os

import pandas as pd


def generate_run_file(framework, benchmark, constrain, task, fold, force=False):
    """Generates a bash script for the sbatch command"""

    run_file = f"results/{framework}_{benchmark}_{constrain}_{task}_{fold}.sh"
    if os.path.exists(run_file) and not force:
        return

    with open(run_file, 'w') as f:
        f.write(f"#!/bin/bash
#Setup the run
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export SINGULARITY_TMPDIR=/home/riverav/tmp
source /home/riverav/work/venv/bin/activate
cd /home/riverav/work/automlbenchmark

# Config the run
export framework={framework}
export benchmark={benchmark}
export constrain={constrain}
export task={task}
export fold={fold}
echo 'python runbenchmark.py {framework} {benchmark} {constrain} --task {task} --fold {fold} -m singularity'
python runbenchmark.py {framework} {benchmark} {constrain} --task {task} --fold {fold} -m singularity
echo 'Finished the run'
"
    )
    return run_file

def get_task_from_benchmark(benchmarks)
    """Returns a dict with benchmark to task mapping"""
    task_from_benchmark = {}

    for benchmark in benchmarks:
        task_from_benchmark[benchmark] = []
        filename= os.path.join('resources','benchmarks', f"{benchmark}.yaml")
        if os.path.exists(filename):
            raise Exception(f"File {filename} not found!")
        with open(filename ) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        for task in data:
            if task.name in ['__dummy-task', '__defaults__']:
                continue
            task_from_benchmark[benchmark].append(task.name)

def score(row, res_col='result'):
    """
    Get the results as the automlbenchmark team through https://github.com/openml/automlbenchmark/blob/master/reports/report/results.py
    return row[res_col] if row[res_col] in [row.auc, row.acc] else -row[res_col]
    """
    result = df['result'].iloc[0]
    if result in [ df['auc'].iloc[0], df['acc'].iloc[0] ]
        return result
    else:
        return -result

def get_results(framework, benchmark, constrain, task, fold, force=False):
    result_file = 'results/results.csv'
    df = pd.read_csv(result_file)
    df = df[(df['framework']==framework) & (df['task']==task) & (df['fold']==fold)]

    # If no run return
    if df.empty:
        return None

    if df.shape[0] != 1:
        print(f"-W-: More than 1 column ({df.shape[0]}) matched the criteria {framework} {benchmark} {constrain} {task} {fold}. Picking the first one: {df} ")

    result = df['result'].iloc[0]
    if result is None or pd.isnull(result):
        return df['info'].iloc[0]

    return score(df)


def query_yes_no(question, default='no'):
    if default is None:
        prompt = " [y/n] "
    elif default == 'yes':
        prompt = " [Y/n] "
    elif default == 'no':
        prompt = " [y/N] "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return distutils.util.strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def launch_run(run_file):
    """Sends a job to sbatcht"""
    name, ext = os.path.splitext(run_file)
    returned_value = os.system(
        "sbatch -p ml_cpu-ivy -c 5 --job-name {} -o {} {}".format(
            name,
            os.path.join('logs', name + '.out'),
            run_file,
        )
    )
    return returned_value

def str2bool(v):
    """
    Process any bool
    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'framework',
        choices=['tpot', 'H2OAutoML'],
        help='What framework to manage'
    )
    parser.add_argument(
        '--benchmark',
        default='test',
        choices=['test', 'small', 'medium', 'large', 'validation'],
        help='What benchmark to run'
    )
    parser.add_argument(
        '--task',
        help='What specific task to run'
    )
    parser.add_argument(
        '--constrain',
        default='1h4c',
        help='What framework to manage'
    )
    parser.add_argument(
        '--force',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Force everything-- Maybe use for a bad run'
    )

    args = parser.parse_args()
    frameworks = [args.framework]
    if args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = ['test', 'small', 'medium', 'large', 'validation']

    if args.task:
        tasks = {}
        count = 0
        tasks_from_benchmark = get_task_from_benchmark(benchmarks)
        for benchmark in benchmarks:
            if args.task in tasks_from_benchmark[benchmark]:
                tasks[benchmark] = args.task
                count +=1
        if count != 1:
            raise Exception(f"For task={args.task} it was not uniquely defined for benchmark={benchmarks}({count})")
    else:
        tasks = get_task_from_benchmark(benchmarks)

    # Get the task
    jobs = collections.defaultdict(dict)
    for framework in frameworks:
        for benchmark in benchmarks:
            for task in tasks[benchmark]:
                for fold in folds:
                    # Check if the run files for this task exist
                    jobs[framework][benchmark][task]['run_file'] = generate_run_file(
                        framework=framework,
                        benchmark=benchmark,
                        constrain=args.constrain,
                        task=task,
                        fold=fold,
                        force=args.force
                    )

                    # Check if there are results already
                    jobs[framework][benchmark][task]['results'] = get_results(
                        framework=framework,
                        benchmark=benchmark,
                        constrain=args.constrain,
                        task=task,
                        fold=fold,
                        force=args.force
                    )

                    print(f"Returned = {jobs[framework][benchmark][task]['results']}")
                    continue

                    # Launch the run if it was not yet launched
                    if jobs[framework][benchmark][task]['results'] is None or args.force:
                        launch_run(jobs[framework][benchmark][task]['run_file'])
                    else:
                        # If the run failed, then ask the user to relaunch
                        if not  jobs[framework][benchmark][task]['results'].replace('.','',1).isdigit():
                            if query_yes_no(f"For framework={framework} benchmark={benchmark} constrain={args.constrain} task={task} fold={fold} obtained: {jobs[framework][benchmark][task]['results']}. Do you want to relaunch this run?"):
                                launch_run(jobs[framework][benchmark][task]['run_file'])

