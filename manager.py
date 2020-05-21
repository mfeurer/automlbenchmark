import argparse
from random import randrange
import collections
import mmap
import os
import re
import time
import subprocess
import logging
import glob
import json

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

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export SINGULARITY_TMPDIR=/home/riverav/tmp
source /home/riverav/work/venv/bin/activate
cd {rundir}
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then export TMPDIR=/tmp/{framework}_{benchmark}_{constraint}_{task}_{fold}_$SLURM_JOB_ID; else export TMPDIR=/tmp/{framework}_{benchmark}_{constraint}_{task}_{fold}$SLURM_ARRAY_JOB_ID'_'$SLURM_ARRAY_TASK_ID; fi
echo TMPDIR=$TMPDIR
mkdir -p $TMPDIR
export SINGULARITY_BINDPATH="$TMPDIR:/tmp"

# Config the run
export framework={framework}
export benchmark={benchmark}
export constraint={constraint}
export task={task}
export fold={fold}
echo 'python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity --session {framework}_{benchmark}_{constraint}_{task}_{fold}'
python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity --session {framework}_{benchmark}_{constraint}_{task}_{fold}
echo "Deleting temporal folder $TMPDIR"
rm -rf $TMPDIR
echo 'Finished the run'
"""

    with open(run_file, 'w') as f:
        f.write(command)
    return run_file


def generate_cleanup_file(run_file, jobid, rundir):
    """Generates a bash script for the sbatch command"""

    name, ext = os.path.splitext(os.path.basename(run_file))
    clean_file = f"results/{name}_cleanup.sh"

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
export SINGULARITY_TMPDIR=/home/riverav/tmp
source /home/riverav/work/venv/bin/activate
export host=`sacct -j {jobid} --format=NodeList --noheader | sort -u`
if [ -z "$host" ]
then
    echo "No host could be found for cleanup"
else
    ssh -o 'StrictHostKeyChecking no' $host ls -l /tmp/{jobid} &&  rm -rf /tmp/{jobid}
fi
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
    if score is None or not is_number(score):
        return score
    zero = get_results('constantpredictor', benchmark, constraint, task, fold)
    if zero is None or not is_number(zero):
        return score
    one = get_results('RandomForest', benchmark, constraint, task, fold)
    if one is None or not is_number(one):
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
        #logger.warn(f"More than 1 column ({df.shape[0]}) matched the criteria {framework} {benchmark} {constraint} {task} {fold}. Picking the first one: {df} ")
        if not df[df.result.notnull()].empty:
            df = df[df.result.notnull()]
            df = df.iloc[[-1]]

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
        'timeout',
    ]
    with open(logfile, 'rb', 0) as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        for cause in causes:
            if s.find(cause.encode()) != -1:
                return cause
    return 'UNDEFINED CAUSE'


def check_if_running(run_file):
    name, ext = os.path.splitext(os.path.basename(run_file))

    # First check if there is a job with this name
    result = subprocess.run([
        'squeue',
        f"--format=\"%.50j\" --noheader -u {os.environ['USER']}"
    ], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    for i, line in enumerate(result.splitlines()):
        if name in line:
            return True

    # The check in the user job arrays
    result = subprocess.run([
        f"squeue --format=\"%.50i\" --noheader -u {os.environ['USER']}"
    ], shell=True, stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    jobids = [jobid.split('_')[0] for jobid in result.splitlines() if "_" in jobid]
    if jobids:
        jobids = list(set(jobids))
    for i, array_job in enumerate(jobids):
        command = subprocess.run([
            f"scontrol show jobid -dd {array_job} | grep Command |  sort --unique",
        ], shell=True, stdout=subprocess.PIPE)
        command, filename = command.stdout.decode('utf-8').split('=')
        filename = filename.lstrip().rstrip()
        if not os.path.exists(filename):
            raise Exception(f"Could not find file {filename} from command={command} array_job={array_job }")
        with open(filename, 'r') as data_file:
            if name in data_file.read():
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

def to_array_run(run_file, memory, cores, rundir):

    name = f"{rundir}/results/arrayjob_{len(run_file)}_{os.getpid()}_{randrange(100)}.sh"

    command = f"""#!/bin/bash
#SBATCH -o {rundir}/logs/%x.%A.%a.out
#SBATCH -c {cores}
#SBATCH --mem {memory}

echo "Here's what we know from the SLURM environment"
echo SHELL=$SHELL
echo HOME=$HOME
echo CWD=$(pwd)
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
echo "Job started at: `date`"

"""
    for i, task in enumerate(run_file):
        command += f"\nif [ $SLURM_ARRAY_TASK_ID -eq {i} ]"
        command += f"\n\tthen"
        command += f"\n\t\techo \"Chrashed {rundir}/logs/$SLURM_JOB_NAME.$SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID.out\" > {rundir}/logs/{os.path.splitext(os.path.basename(task))[0]}.out"
        command += f"\n\t\tbash -x {task}"
        command += f"\n\t\t cp {rundir}/logs/$SLURM_JOB_NAME.$SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID.out {rundir}/logs/{os.path.splitext(os.path.basename(task))[0]}.out"
        command +="\n\tfi\n"
    command += "\necho DONE at: `date`"

    with open(name, 'w') as f:
        f.write(command)
    return name


def launch_run(run_file, partition, constraint, rundir, run_mode):
    """Sends a job to sbatcht"""

    # Check if run mode is not understandle
    if 'array' not in run_mode and 'single' not in run_mode:
        raise Exception(f"Unsupported run_mode {run_mode} provided")

    # make sure we work with lists
    not_launched_runs = []
    for task in run_file:
        if not  check_if_running(task):
            not_launched_runs.append(task)
    if not not_launched_runs:
        return
    run_file = not_launched_runs

    # Run options
    extra = ''
    if partition == 'bosch_cpu-cascadelake':
        extra += ' --bosch'

    if constraint == '1h1c':
        memory = '12G'
        # In the file .config/automlbenchmark/constraints.yaml 1 core is assigned
        # but reallty in the cluster we provide 2
        cores = 2
    elif constraint == '8h1c':
        memory = '12G'
        # In the file .config/automlbenchmark/constraints.yaml 1 core is assigned
        # but reallty in the cluster we provide 2
        cores = 2
    elif constraint == '1h8c':
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

    # For array
    if 'array' in run_mode:
        job_list_file = to_array_run(run_file, memory, cores, rundir)
        name, ext = os.path.splitext(os.path.basename(job_list_file))
        _, max_active_runs = run_mode.split('_')
        max_run = min(int(max_active_runs), len(run_file))
        extra += f" -p {partition} --array=0-{len(run_file)-1}%{max_run} --job-name {name}"
        jobid = _launch_sbatch_run(extra, job_list_file)

    elif run_mode == 'single':
        for task in run_file:
            name, ext = os.path.splitext(os.path.basename(task))
            this_extra = extra + f" -p {partition} --mem {memory} -c {cores} --job-name {name} -o {os.path.join('logs', name + '.out')}"
            jobid = _launch_sbatch_run(this_extra, task)

    # Wait 5 seconds and launch the dependent cleanup job in case of failure
    for i, task in enumerate(run_file):
        name, ext = os.path.splitext(os.path.basename(task))
        local_jobid = jobid if 'array' not in run_mode else f"{jobid}_{i}"
        options = "--dependency=afternotok:{} --kill-on-invalid-dep=yes -p {} {} -c 1 --job-name {} -o {}".format(
            local_jobid,
            partition,
            '--bosch' if partition == 'bosch_cpu-cascadelake' else '',
            name+'_cleanup',
            os.path.join('logs', name + '_cleanup' + '.out'),
        )
        #_launch_sbatch_run(options, generate_cleanup_file(task, local_jobid, rundir))


def _launch_sbatch_run(options, script):
    command = "sbatch {} {}".format(
        options,
        script
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
        return int(success[1])
    else:
        raise Exception("Could not launch job for script={script}")


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


def get_job_status(frameworks, benchmarks, tasks, folds, partition, constraint, run_mode, rundir):

    # Get the task
    jobs = collections.defaultdict(dict)
    total = 0

    # get a list of jobs to run
    run_files = []

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

                    if run_mode:
                        # Launch the run if it was not yet launched
                        if jobs[framework][benchmark][task][fold]['results'] is None:
                            if check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                                status = 'Running'
                            else:
                                run_files.append(jobs[framework][benchmark][task][fold]['run_file'])
                                status = 'Launched'
                        else:
                            # If the run failed, then ask the user to relaunch
                            if not valid_result:
                                if check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                                    status = 'Running'
                                else:
                                    status = 'Failed'
                                    if query_yes_no(f"For framework={framework} benchmark={benchmark} constraint={constraint} task={task} fold={fold} obtained: {jobs[framework][benchmark][task][fold]['results']}. Do you want to relaunch this run?"):
                                        run_files.append(jobs[framework][benchmark][task][fold]['run_file'])
                                        status = 'Relaunched'
                    jobs[framework][benchmark][task][fold]['status'] = status

                    logger.debug(f"\t\t\tFold:{fold} Status = {status} ({jobs[framework][benchmark][task][fold]['results']})")
                    total = total + 1

    # Launch the runs in sbatch
    if run_files:
        launch_run(
            run_files,
            partition=partition,
            constraint=constraint,
            rundir=rundir,
            run_mode=run_mode,
        )

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
                    average_result = 'N/A'
                    row[framework + '_mean'] = average_result
                    row[framework + '_std'] = average_result
                else:
                    row[framework + '_mean'] = np.nanmean(average) if np.any(average) else 0
                    row[framework + '_std'] = np.nanstd(average) if np.any(average) else 0
                row[framework + '_num_folds'] = len(average)
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


def get_autosklearn_model_list(model_file):

    if not os.path.exists(model_file):
        logger.error(f"Could not find model file: {model_file}")
        return []

    pattern = re.compile(".*'classifier:__choice__': '(\w+)'.*")
    model_list = []
    for i, line in enumerate(open(model_file)):
        match = re.match(pattern, line)
        if match:
            model_list.append(match.group(1))
    return model_list


def get_h2o_model_list(model_file):

    # Read the json file
    model_file = model_file.replace('full_models', 'models').replace('.zip', '.json')
    if not os.path.exists(model_file):
        logger.error(f"Could not find model file: {model_file}")
        return []

    if 'StackedEnsemble' not in model_file:
        # Sometimes a model is better than the stack
        model_file = os.path.basename(model_file).split('_')[0]
        return [model_file]

    model_list = []
    parsed = json.load(open(model_file))
    for i, params in enumerate(parsed['parameters']):
        if not parsed['parameters'][i]['name'] == 'base_models':
            continue
        for model in parsed['parameters'][i]['actual_value']:
            model_list.append(model['name'].split('_')[0])
        break
    return model_list


def get_model_list(run_file):
    "Tries to extract the list of models from a runfile"
    name, ext = os.path.splitext(os.path.basename(run_file))
    logfile = os.path.join('logs', name + '.out')
    match = re.match("([\w-]+)_(small|medium|large|test)_([\dA-Za-z]+)_([.\w-]+)_(\d)", name)
    if match is None:
        logger.error(f"Could not match name={name}")
        return []
    framework, benchmark, constraint, task, fold = match.groups()

    # If we cant file the model file, bail out
    if not os.path.exists(logfile):
        logger.error(f"Could not find log file: {logfile}")
        return []

    # Get the path where models where saved
    pattern = re.compile(".*Scores\s*saved\s*to\s*`\/output\/(.*)\/scores\/results.csv.*")
    result_file = None
    for i, line in enumerate(open(logfile)):
        match = re.match(pattern, line)
        if match:
            result_file = match.group(1)
            break
    if result_file is None:
        logger.error(f"Could not extract result file from: {logfile}")
        return []

    # Extract per framework models
    if 'autosklearn' in framework:
        model_file = os.path.join('results', result_file, 'models', task, fold, 'models.txt')
        return get_autosklearn_model_list(model_file)
    elif 'H2O' in framework:
        model_file = glob.glob(os.path.join('results', result_file, 'full_models', task, fold, '*.zip'))
        if len(model_file) < 1:
            print(f"Error while parsing {run_file}.. No model file found!")
            return []
        return get_h2o_model_list(model_file[-1])
    else:
        raise ValueError(f"No support for framework={framework}")

def get_used_models_per_framework(jobs):
    """
    Returns a per framework dict with a dataframe of models used.
    If more than 1 fold is provided, the average is taken
    """
    framework_models = {}
    for framework, framework_dict in jobs.items():
        rows = []
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                model_usage = {'benchmark':benchmark, 'task':task}
                for fold, data in task_dict.items():
                    run_file =  jobs[framework][benchmark][task][fold]['run_file']
                    for model in get_model_list(run_file):
                        if model not in model_usage:
                            model_usage[model] = 1
                        else:
                            model_usage[model] += 1

                # Calculate average
                for key, value in model_usage.items():
                    if key == 'task':continue
                    if key == 'benchmark':continue
                    model_usage[key] = value/len(task_dict.keys())

                rows.append(model_usage)
        framework_models[framework] = pd.DataFrame(rows)
        print(framework_models[framework])
        framework_models[framework].to_csv(f"{framework}_models.csv", index=False)
    return framework_models


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
        choices=['test', '1h4c', '1h8c', '1h1c', '8h1c'],
        help='What number o fcores and runtime is allowed'
    )
    parser.add_argument(
        '--partition',
        default='ml_cpu-ivy',
        choices=['ml_cpu-ivy', 'test_cpu-ivy', 'bosch_cpu-cascadelake'],
        help='In what partition to launch'
    )
    parser.add_argument(
        '--fold',
        type=int,
        choices=list(range(10)),
        help='What fold out of 10 to launch'
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
        '--array_run',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Launches the run to sbatch on an array fashion'
    )
    parser.add_argument(
        '--max_active_runs',
        type=int,
        default=5,
        help='maximum number of active runs for a batch array'
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
        default=False,
        help='generates a problems dataframe'
    )
    parser.add_argument(
        '--models',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a models used dataframe'
    )
    parser.add_argument(
        '--rundir',
        default='/home/riverav/work/automlbenchmark_new',
        help='The area from where to run'
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    # Check framewors
    frameworks = args.framework.split() if ' ' in args.framework else [args.framework]
    #valid_frameworks = [os.path.basename(path) for path in glob.glob('frameworks/*')]
    #for framework in frameworks:
    #    if framework not in valid_frameworks:
    #        raise Exception(f"Unsupported framework={framework}...")

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

    # Can only run on array or normal mode
    if args.run and args.array_run:
        raise Exception('Only one can be specified --run or --array_run')
    elif args.run:
        run_mode = 'single'
    elif args.array_run:
        run_mode = 'array_' + str(args.max_active_runs)
    else:
        run_mode = None

    # Get the job status
    jobs = get_job_status(frameworks, benchmarks, tasks, folds, args.partition, args.constraint, run_mode, args.rundir)

    # Print the reports if needed
    if args.problems:
        problems = get_problems(frameworks, benchmarks, tasks, folds)
        logger.warn("This Run problems are:\n")
        logger.warn(problems)
        logger.warn("\n")
        problems.to_csv('problems.csv', index=False)

    # Print models used
    if args.models:
        get_used_models_per_framework(jobs)

    # Get and print the normalized score
    logger.warn("\nCheck the file normalized_score.csv!")
    normalized_score_dataframe = get_normalized_score(frameworks, benchmarks, tasks, folds)
    logger.warn(normalized_score_dataframe)
    normalized_score_dataframe.to_csv('normalized_score.csv', index=False)
