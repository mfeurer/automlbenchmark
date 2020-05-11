import argparse
import collections
import mmap
import os
import re
import time
import subprocess
import logging
import glob

import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.options.display.max_colwidth = 200

import yaml


# Logger Setup
logger = logging.getLogger('manager')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def generate_run_file(framework, benchmark, constraint, task, fold, rundir):
    """Generates a bash script for the sbatch command"""

    run_file = f"results/{framework}_{benchmark}_{constraint}_{task}_{fold}.sh"
    if os.path.exists(run_file):
        return run_file

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export SINGULARITY_TMPDIR=/home/riverav/tmp
source /home/riverav/work/venv/bin/activate
cd {rundir}
export TMPDIR=/tmp/$SLURM_JOB_NAME
mkdir -p $TMPDIR

# Config the run
export framework={framework}
export benchmark={benchmark}
export constraint={constraint}
export task={task}
export fold={fold}
echo 'python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity'
python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity
echo "Deleting temporal folder $TMPDIR"
rm -rf $TMPDIR
echo 'Finished the run'
"""

    with open(run_file, 'w') as f:
        f.write(command)
    return run_file


def generate_cleanup_file(run_file, rundir):
    """Generates a bash script for the sbatch command"""

    name, ext = os.path.splitext(os.path.basename(run_file))
    clean_file = f"results/{name}_cleanup.sh"
    if os.path.exists(clean_file):
        return clean_file

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export SINGULARITY_TMPDIR=/home/riverav/tmp
source /home/riverav/work/venv/bin/activate
export TMPDIR=/tmp/{name}
export host=`grep HOSTNAME= {rundir}/logs/{name}.out | sed 's/^.*HOSTNAME=\([_[:alnum:]]*\).*/\\1/'`
ssh -o 'StrictHostKeyChecking no' $host ls -l /tmp/{name} &&  rm -rf /tmp/{name}
echo 'Finished the run'
"""

    with open(clean_file, 'w') as f:
        f.write(command)
    return clean_file

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
    bac = df['bac'].iloc[0] if 'bac' in df else None
    acc = df['acc'].iloc[0] if 'acc' in df else None
    if result in [ auc, acc, bac]:
        return result
    else:
        return -result


def norm_score(framework, benchmark, constraint, task, fold, score):
    zero = get_results('constantpredictor', benchmark, constraint, task, fold)
    one = get_results('RandomForest', benchmark, constraint, task, fold)
    if zero is None or one is None or not is_number(score):
        return score
    print(f"score={score}({type(score)})")
    print(f"one={one}({type(one)})")
    print(f"zero={zero}({type(zero)})")
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
        #logger.warn(f"More than 1 column ({df.shape[0]}) matched the criteria {framework} {benchmark} {constraint} {task} {fold}. Picking the first one: {df} ")
        if df[df.applymap(np.isreal)['result']].shape[0] >= 1:
            df = df[df.applymap(np.isreal)['result']]

    result = df['result'].iloc[-1]
    if result is None or pd.isnull(result):
        return df['info'].iloc[-1]

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
            logger.critical("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def check_if_crashed(run_file):
    """Checks if a run crashed from the batch log file"""
    name, ext = os.path.splitext(os.path.basename(run_file))
    logfile = os.path.join('logs', name + '.out')
    if not os.path.exists(logfile):
        return False

    causes = [
        'error: Exceeded job memory limit',
        'DUE TO TIME LIMIT',
        'MemoryError',
        'OutOfMemoryError',
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


def get_node(partition):
    """
    Gets a free node to be able to run the run without HTTP errors
    PARTITION                AVAIL  TIMELIMIT  NODES  STATE NODELIST
meta_gpu-black              up   infinite      4  down* metagpu[2-4,8]
meta_gpu-black              up   infinite      1   drng metagpu9
meta_gpu-black              up   infinite      1  drain metagpu1
meta_gpu-black              up   infinite      1  alloc metagpu5
test_cpu-ivy                up    1:05:00      1    mix metaex01
ml_cpu-ivy                  up 4-00:00:00      1    mix metaex16
ml_cpu-ivy                  up 4-00:00:00      2   idle metaex[15,17]
cpu_ivy                     up    2:05:00      1    mix metaex16
cpu_ivy                     up    2:05:00     10  alloc metaex[18-27]
cpu_ivy                     up    2:05:00      7   idle metaex[15,17,28-32]
bosch_cpu-cascadelake       up   infinite     36  alloc kisexe[09-44]
allbosch_cpu-cascadelake    up    2:05:00      3    mix kisexe[03,06,08]
allbosch_cpu-cascadelake    up    2:05:00     41  alloc kisexe[01-02,04-05,07,09-44]
    """

    if partition == 'test_cpu-ivy':
        return 'metaex01'
    elif partition == 'ml_cpu-ivy':
        # Sleep for 5 seconds to make sure slurm updated
        time.sleep(5)

        result = subprocess.run(
            f"squeue --format=\"%.50R\" --noheader -u {os.environ['USER']}",
            shell=True,
            stdout=subprocess.PIPE
        )
        result = result.stdout.decode('utf-8')
        machines = {
            'metaex15':0,
            'metaex16':0,
            'metaex17':0,
        }
        for i, line in enumerate(result.splitlines()):
            line = line.lstrip().strip()
            if line in machines:
                machines[line] = machines[line]+1
        return min(machines, key=machines.get)
    else:
        raise Exception(f"Unsupported partition={partition} provided")


def launch_run(run_file, partition, constraint, rundir):
    """Sends a job to sbatcht"""
    name, ext = os.path.splitext(os.path.basename(run_file))

    if check_if_running(run_file):
        return

    # Run options
    if partition == 'bosch_cpu-cascadelake':
        extra = '--bosch --nodelist=kisexe[33-40] '
        extra = '--bosch --nodelist=' + ','.join([f"kisexe{f}" for f in range(20,30)])
        extra = '--bosch'
    else:
        extra = ''

    if constraint == '1h8c':
        memory = '32G'
        cores = 8
    elif constraint == '1h4c':
        memory = '16G'
        cores = 4
    elif constraint == 'test':
        memory = '8G'
        cores = 2
    else:
        raise Exception(f"Unsupported constrain provided {constraint}")

    command = "sbatch -p {} --mem {} -c {} --job-name {} -o {} {} {}".format(
        partition,
        memory,
        cores,
        name,
        os.path.join('logs', name + '.out'),
        run_file,
        extra
    )
    logger.debug(f"-I-: Running command={command}")
    returned_value = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE
    )
    returned_value = returned_value.stdout.decode('utf-8')

    success = re.compile('Submitted batch job (\d+)').match(returned_value)
    if success:
        jobid = int(success[1])
        print(f"jobid={jobid}")
        logfile = os.path.join('logs', name + '.out')

        # Wait 5 seconds and launch the dependent cleanup job in case of failure
        command = "sbatch --dependency=afternotok:{} --kill-on-invalid-dep=yes -p {} -c 1 --job-name {} -o {} {} {}".format(
            jobid,
            partition,
            name+'_cleanup',
            os.path.join('logs', name + '_cleanup' + '.out'),
            generate_cleanup_file(run_file, rundir),
            extra
        )
        logger.debug(f"-I-: Running command={command}")
        returned_value = os.system(command)

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

def get_job_status(frameworks, benchmarks, tasks, folds, partition, constraint, run, rundir):

    # Get the task
    jobs = collections.defaultdict(dict)
    total = 0
    for framework in frameworks:
        jobs[framework] = dict()
        logger.debug('_'*40)
        logger.debug(f"\t\t{framework}")
        logger.debug('_'*40)

        for benchmark in benchmarks:
            jobs[framework][benchmark] = dict()
            logger.debug('_'*40)
            logger.debug(f"\tbenchmark={benchmark}")
            for task in tasks[benchmark]:
                jobs[framework][benchmark][task] = dict()
                logger.debug('_'*40)
                logger.debug(f"\t\t task={task}")
                for fold in folds:
                    jobs[framework][benchmark][task][fold] = dict()

                    # Check if the run files for this task exist
                    jobs[framework][benchmark][task][fold]['run_file'] = generate_run_file(
                        framework=framework,
                        benchmark=benchmark,
                        constraint=constraint,
                        task=task,
                        fold=fold,
                        rundir=rundir,
                    )

                    # Check if there are results already
                    jobs[framework][benchmark][task][fold]['results'] = get_results(
                        framework=framework,
                        benchmark=benchmark,
                        constraint=constraint,
                        task=task,
                        fold=fold,
                    )

                    # Normalize the score as in the paper
                    jobs[framework][benchmark][task][fold]['norm_score'] = norm_score(
                        framework=framework,
                        benchmark=benchmark,
                        constraint=constraint,
                        task=task,
                        fold=fold,
                        score=jobs[framework][benchmark][task][fold]['results'],
                    )

                    # Show status to see what is going on
                    valid_result = is_number(jobs[framework][benchmark][task][fold]['results'])
                    if valid_result:
                        status = 'Completed'
                    elif check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                        status = 'Running'
                    else:
                        status = 'N/A'
                        crashed = check_if_crashed(jobs[framework][benchmark][task][fold]['run_file'])
                        if crashed:
                            jobs[framework][benchmark][task][fold]['results'] = crashed
                            status = 'Chrashed'

                    if run:
                        # Launch the run if it was not yet launched
                        if jobs[framework][benchmark][task][fold]['results'] is None:
                            if check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                                status = 'Running'
                            else:
                                launch_run(
                                    jobs[framework][benchmark][task][fold]['run_file'],
                                    partition=partition,
                                    constraint=constraint,
                                    rundir=rundir,
                                )
                                status = 'Launched'
                        else:
                            # If the run failed, then ask the user to relaunch
                            if not valid_result:
                                if check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                                    status = 'Running'
                                else:
                                    status = 'Failed'
                                    if query_yes_no(f"For framework={framework} benchmark={benchmark} constraint={constraint} task={task} fold={fold} obtained: {jobs[framework][benchmark][task][fold]['results']}. Do you want to relaunch this run?"):
                                        launch_run(
                                            jobs[framework][benchmark][task][fold]['run_file'],
                                            partition=partition,
                                            constraint=constraint,
                                            rundir=rundir,
                                        )
                                        status = 'Relaunched'
                    jobs[framework][benchmark][task][fold]['status'] = status

                    logger.debug(f"\t\t\tFold:{fold} Status = {status} ({jobs[framework][benchmark][task][fold]['results']})")
                    total = total + 1


    logger.debug('_'*40)
    logger.debug(f" A total of {total} runs checked")
    logger.debug('_'*40)

    return jobs


def get_normalized_score(frameworks, benchmarks, tasks, folds):
    # Create a Dataframe
    dataframe = []
    for benchmark in benchmarks:
        for task in tasks[benchmark]:
            row = {
                'benchmark': benchmark,
                'Task': task,
            }
            for framework in frameworks:
                average = []
                for fold in folds:
                    score = jobs[framework][benchmark][task][fold]['norm_score']
                    if is_number(score):
                        average.append(score)
                if len(average) < 1:
                    average = 'N/A'
                else:
                    average = np.nanmean(average) if np.any(average) else 0
                    framework: average
                row[framework] = average
            dataframe.append(row)
    dataframe = pd.DataFrame(dataframe)
    return dataframe


def get_problems(frameworks, benchmarks, tasks, folds):
    # Create a Dataframe
    dataframe = []
    for benchmark in benchmarks:
        for task in tasks[benchmark]:
            row = {
                'benchmark': benchmark,
                'Task': task,
                'Issues': [],
            }
            for framework in frameworks:
                row[framework] = 0
                for fold in folds:
                    # If this run crashed,  register why
                    if 'Chrashed' in jobs[framework][benchmark][task][fold]['status']:
                        if jobs[framework][benchmark][task][fold]['results'] not in row['Issues']:
                            row['Issues'].append(jobs[framework][benchmark][task][fold]['results'])
                        row[framework] +=1
            dataframe.append(row)
    dataframe = pd.DataFrame(dataframe)
    return dataframe

def complete_missing_csv():
    results_file = 'results.csv'
    results = pd.read_csv(os.path.join('results', results_file))

    for local_path in glob.glob('results/*/scores/results.csv'):
        local_result = pd.read_csv(local_path)
        if not local_result.empty:
            #Just take rsults that make sense
            local_result = local_result[local_result.applymap(np.isreal)['result']]
            local_result = local_result[~local_result['fold'].isnull()]
            if not local_result.empty:
                results = pd.concat([results, local_result], sort=False).drop_duplicates()

    results['fold'] = results['fold'].astype('int32')
    results.to_csv("final.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        'framework',
        #choices=['TPOT', 'H2OAutoML', 'constantpredictor', 'RandomForest'],
        help='What framework to manage'
    )
    parser.add_argument(
        '--benchmark',
        #choices=['test', 'small', 'medium', 'large', 'validation'],
        help='What benchmark to run'
    )
    parser.add_argument(
        '--task',
        help='What specific task to run'
    )
    parser.add_argument(
        '--constraint',
        default='1h8c',
        choices=['test', '1h4c', '1h8c'],
        help='What framework to manage'
    )
    parser.add_argument(
        '--partition',
        default='ml_cpu-ivy',
        choices=['ml_cpu-ivy', 'test_cpu-ivy', 'bosch_cpu-cascadelake'],
        help='What framework to manage'
    )
    parser.add_argument(
        '--fold',
        type=int,
        help='What framework to manage'
    )
    parser.add_argument(
        '--run',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Launches the run to sbatch'
    )
    parser.add_argument(
        '--verbose',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Prints more debug info'
    )
    parser.add_argument(
        '--problems',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='generates a problems dataframe'
    )
    parser.add_argument(
        '--rundir',
        default='/home/riverav/work/automlbenchmark_fork',
        help='The area from where to run'
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    frameworks = args.framework.split() if ' ' in args.framework else [args.framework]
    if args.benchmark:
        benchmarks = args.benchmark.split() if ' ' in args.benchmark else [args.benchmark]
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

    # Get the job status
    jobs = get_job_status(frameworks, benchmarks, tasks, folds, args.partition, args.constraint, args.run, args.rundir)

    # Print the reports if needed
    if args.problems:
        problems = get_problems(frameworks, benchmarks, tasks, folds)
        logger.warn("This Run problems are:\n")
        logger.warn(problems)
        logger.warn("\n")
        problems.to_csv('problems.csv', index=False)

    # Get and print the normalized score
    logger.warn("\nCheck the file normalized_score.csv!")
    normalized_score_dataframe = get_normalized_score(frameworks, benchmarks, tasks, folds)
    logger.warn(normalized_score_dataframe)
    normalized_score_dataframe.to_csv('normalized_score.csv', index=False)
