import argparse
import collections
import mmap
import os
import subprocess

import numpy as np
import pandas as pd

import yaml


def generate_run_file(framework, benchmark, constraint, task, fold, force=False):
    """Generates a bash script for the sbatch command"""

    run_file = f"results/{framework}_{benchmark}_{constraint}_{task}_{fold}.sh"
    if os.path.exists(run_file) and not force:
        return run_file

    command = f"""#!/bin/bash
#Setup the run
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export SINGULARITY_TMPDIR=/home/riverav/tmp
source /home/riverav/work/venv/bin/activate
cd /home/riverav/work/automlbenchmark

# Config the run
export framework={framework}
export benchmark={benchmark}
export constraint={constraint}
export task={task}
export fold={fold}
echo 'python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity'
python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity
echo 'Finished the run'
"""

    with open(run_file, 'w') as f:
        f.write(command)
    return run_file

def get_task_from_benchmark(benchmarks):
    """Returns a dict with benchmark to task mapping"""
    task_from_benchmark = {}

    for benchmark in benchmarks:
        task_from_benchmark[benchmark] = []
        filename= os.path.join('resources','benchmarks', f"{benchmark}.yaml")
        if not  os.path.exists(filename):
            raise Exception(f"File {filename} not found!")
        with open(filename ) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        for task in data:
            if task['name'] in ['__dummy-task', '__defaults__']:
                continue
            task_from_benchmark[benchmark].append(task['name'])

    return task_from_benchmark


def score(df, res_col='result'):
    """
    Get the results as the automlbenchmark team through https://github.com/openml/automlbenchmark/blob/master/reports/report/results.py
    return row[res_col] if row[res_col] in [row.auc, row.acc] else -row[res_col]
    """
    result = df['result'].iloc[0]
    auc = df['auc'].iloc[0] if 'auc' in df else None
    acc = df['acc'].iloc[0] if 'acc' in df else None
    if result in [ auc, acc]:
        return result
    else:
        return -result

def norm_score(framework, benchmark, constraint, task, fold, score):
    zero = get_results('constantpredictor', benchmark, constraint, task, fold)
    one = get_results('TunedRandomForest', benchmark, constraint, task, fold)
    if zero is None or one is None or not is_number(score):
        return score
    return (score - zero) / (one - zero)


def get_results(framework, benchmark, constraint, task, fold):
    result_file = 'results/results.csv'

    if not os.path.exists(result_file):
        return None

    df = pd.read_csv(result_file)
    df["fold"] = pd.to_numeric(df["fold"])
    df = df[(df['framework']==framework) & (df['task']==task) & (df['fold']==fold)]

    # If no run return
    if df.empty:
        return None

    if df.shape[0] != 1:
        print(f"-W-: More than 1 column ({df.shape[0]}) matched the criteria {framework} {benchmark} {constraint} {task} {fold}. Picking the first one: {df} ")

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
                return str2bool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def check_if_crashed(run_file):
    """Checks if a run crashed from the batch log file"""
    name, ext = os.path.splitext(os.path.basename(run_file))
    logfile = os.path.join('logs', name + '.out')
    if not os.path.exists(logfile):
        return False

    causes = [
        'error: Exceeded job memory limit'
    ]
    with open(logfile, 'rb', 0) as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        for cause in causes:
            if s.find(cause.encode()) != -1:
                return cause
    return 'UNDEFINED CAUSE'


def check_if_running(run_file):
    name, ext = os.path.splitext(os.path.basename(run_file))
    result = subprocess.run([
        'squeue',
        f"--format=\"%.50j\" --noheader -u {os.environ['USER']}"
    ], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    for i, line in enumerate(result.splitlines()):
        if name in line:
            return True
    return False


def launch_run(run_file, partition):
    """Sends a job to sbatcht"""
    name, ext = os.path.splitext(os.path.basename(run_file))
    if check_if_running(run_file):
        return
    command = "sbatch -p {} -c 5 --job-name {} -o {} {}".format(
            partition,
            name,
            os.path.join('logs', name + '.out'),
            run_file,
        )
    print(f"-I-: Running command={command}")
    returned_value = os.system(
        command
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

def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        'framework',
        choices=['TPOT', 'H2OAutoML', 'constantpredictor', 'TunedRandomForest'],
        help='What framework to manage'
    )
    parser.add_argument(
        '--benchmark',
        choices=['test', 'small', 'medium', 'large', 'validation'],
        help='What benchmark to run'
    )
    parser.add_argument(
        '--task',
        help='What specific task to run'
    )
    parser.add_argument(
        '--constraint',
        default='1h4c',
        help='What framework to manage'
    )
    parser.add_argument(
        '--partition',
        default='ml_cpu-ivy',
        help='What framework to manage'
    )
    parser.add_argument(
        '--fold',
        type=int,
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
    parser.add_argument(
        '--run',
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
        #benchmarks = ['test', 'small', 'medium', 'large']
        benchmarks = ['test', 'small']

    if args.task:
        tasks = {}
        count = 0
        tasks_from_benchmark = get_task_from_benchmark(benchmarks)
        for benchmark in benchmarks:
            if args.task in tasks_from_benchmark[benchmark]:
                tasks[benchmark] = [args.task]
                count +=1
            else:
                tasks[benchmark] = []
        if count != 1:
            raise Exception(f"For task={args.task} it was not uniquely defined for benchmark={benchmarks}({count})")
    else:
        tasks = get_task_from_benchmark(benchmarks)

    if args.fold or args.fold == 0:
        folds = [args.fold]
    else :
        folds = np.arange(10)

    # Get the task
    jobs = collections.defaultdict(dict)
    total = 0
    for framework in frameworks:
        jobs[framework] = dict()
        print('_'*40)
        print(f"\t\t{framework}")
        print('_'*40)
        for benchmark in benchmarks:
            jobs[framework][benchmark] = dict()
            print('_'*40)
            print(f"\tbenchmark={benchmark}")
            for task in tasks[benchmark]:
                jobs[framework][benchmark][task] = dict()
                print('_'*40)
                print(f"\t\t task={task}")
                for fold in folds:

                    # Check if the run files for this task exist
                    jobs[framework][benchmark][task]['run_file'] = generate_run_file(
                        framework=framework,
                        benchmark=benchmark,
                        constraint=args.constraint,
                        task=task,
                        fold=fold,
                        force= False if not args.force else True,
                    )

                    # Check if there are results already
                    jobs[framework][benchmark][task]['results'] = get_results(
                        framework=framework,
                        benchmark=benchmark,
                        constraint=args.constraint,
                        task=task,
                        fold=fold,
                    )

                    # Normalize the score as in the paper
                    jobs[framework][benchmark][task]['norm_score'] = norm_score(
                        framework=framework,
                        benchmark=benchmark,
                        constraint=args.constraint,
                        task=task,
                        fold=fold,
                        score=jobs[framework][benchmark][task]['results'],
                    )

                    # Show status to see what is going on
                    valid_result = is_number(jobs[framework][benchmark][task]['results'])
                    if valid_result:
                        status = 'Completed'
                    elif check_if_running(jobs[framework][benchmark][task]['run_file']):
                        status = 'Running'
                    else:
                        status = 'N/A'
                        crashed = check_if_crashed(jobs[framework][benchmark][task]['run_file'])
                        if crashed:
                            jobs[framework][benchmark][task]['results'] = crashed
                            status = 'Chrashed'

                    if args.run:
                        # Launch the run if it was not yet launched
                        if jobs[framework][benchmark][task]['results'] is None or args.force:
                            if check_if_running(jobs[framework][benchmark][task]['run_file']):
                                status = 'Running'
                            else:
                                launch_run(
                                    jobs[framework][benchmark][task]['run_file'],
                                    partition=args.partition,
                                )
                                status = 'Launched'
                        else:
                            # If the run failed, then ask the user to relaunch
                            if not valid_result:
                                if check_if_running(jobs[framework][benchmark][task]['run_file']):
                                    status = 'Running'
                                else:
                                    status = 'Failed'
                                    if query_yes_no(f"For framework={framework} benchmark={benchmark} constraint={args.constraint} task={task} fold={fold} obtained: {jobs[framework][benchmark][task]['results']}. Do you want to relaunch this run?"):
                                        launch_run(
                                            jobs[framework][benchmark][task]['run_file'],
                                            partition=args.partition,
                                        )
                                        status = 'Relaunched'

                    print(f"\t\t\tFold:{fold} Status = {status} ({jobs[framework][benchmark][task]['results']})")
                    total = total + 1

print('_'*40)
print(f" A total of {total} runs checked")
print('_'*40)
