import argparse
import glob
import typing
import os

import pandas as pd

import networkx as nx

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# ====================================================================
#                               Functions
# ====================================================================
def parse_data(csv_location: str) -> pd.DataFrame:
    """
    Collects the data of a given experiment, annotated in a csv.
    we expect the csv to look something like:
    (index),tool,task,model,fold,train,val,test,overfit
    """
    data = []
    for data_file in glob.glob(os.path.join(csv_location, '*overfit.csv')):
        data.append(
            pd.read_csv(
                data_file,
                index_col=0,
            )
        )

    if len(data) == 0:
        print(f"ERROR: No overfit data to parse on {csv_location}")
        return

    data = pd.concat(data).reindex()

    # Only plot ensemble for now
    model = 'best_ensemble_model'
    data = data[data['model'] == model]

    # Make sure our desired columns are numeric
    data['test'] = pd.to_numeric(data['test'])
    data['overfit'] = pd.to_numeric(data['overfit'])

    # then we want to fill in the missing values
    all_tools = [t for t in data['tool'].unique().tolist()]
    num_rows = [len(data[data['tool'] == t].index) for t in all_tools]
    tool_with_more_rows = all_tools[np.argmax(num_rows)]
    required_columns = ['task', 'model', 'fold']

    # There is a function called isin pandas, but it gives
    # wrong results -- do this fill in manually
    # base df has all the task/fold/models in case one is missing, like for a crash
    #base_df = data[data['tool'] == tool_with_more_rows][required_columns].reset_index(drop=True)
    base_df = data.loc[data['tool'] == tool_with_more_rows, required_columns].reset_index(drop=True)
    for tool in list(set(all_tools) - {tool_with_more_rows}):
        fix_df = data.loc[data['tool'] == tool, required_columns].reset_index(drop=True)

        # IsIn from pandas does it base on the index. We need to unstack/stack values
        # for real comparisson
        #missing_rows = base_df.iloc[base_df[~base_df.stack(
        #).isin(fix_df.stack().values).unstack()].dropna(how='all').index]
        mask = base_df.stack().isin(fix_df.stack().values).unstack()
        missing_rows = base_df.iloc[base_df[~mask].dropna(how='all').index].copy()
        missing_rows['tool'] = tool
        data = pd.concat([data, missing_rows], sort=True).reindex()

    # A final sort
    data = data.sort_values(by=['tool']+required_columns).reset_index(drop=True)

    return data


def plot_relative_performance(df: pd.DataFrame, tools: typing.List[str],
                              metric: str = 'test', output_dir: typing.Optional[str] = None,
                              ) -> None:
    """
    Generates a relative performance plot, always compared to
    autosklearn.
    """

    if 'autosklearn' not in df['tool'].tolist():
        raise ValueError('We need autosklearn in the dataframe to compare')

    if any([tool not in df['tool'].tolist() for tool in tools]):
        raise ValueError(f"Experiment {tools} was not found in the dataframe {df['tool']}")

    # Get the desired frames
    autosklearn_df = df[df['tool'] == 'autosklearn'].reset_index(drop=True)

    for tool in tools:
        desired_df = df[df['tool'] == tool].reset_index(drop=True)
        desired_df[metric] = desired_df[metric].subtract(autosklearn_df[metric])

        # make sure autosklearn is in the data
        sns.set_style("whitegrid")
        sns.lineplot(
            'task',
            metric,
            data=desired_df,
            ci='sd',
            palette=sns.color_palette("Set2"),
            err_style='band',
            label=tool,
        ).set_title(f"Relative {metric} metric against Autosklearn")

    plt.legend()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, '_'.join(tools) + '.pdf'))

    plt.show()


def plot_ranks(df: pd.DataFrame,
               metric: str = 'test', output_dir: typing.Optional[str] = None,
               ) -> None:
    """
    Generates a relative performance plot, always compared to
    autosklearn.
    """

    if 'autosklearn' not in df['tool'].tolist():
        raise ValueError('We need autosklearn in the dataframe to compare')

    # Step 1: Calculate the mean and std of each fold,
    # That is, collapse the folds
    df = df.groupby(['tool', 'model', 'task']).mean().add_suffix('_mean').reset_index()
    df.to_csv('debug.csv')

    # Sadly the rank method of group by gives weird result, so rank manually
    df['rank'] = 0
    df['seed'] = 0
    for task in df['task'].unique():
        df.loc[df['task'] == task, 'rank'] = df.loc[df['task'] == task, metric + '_mean'].rank()

    # make sure autosklearn is in the data
    sns.set_style("whitegrid")
    sns.pointplot(
        'seed',
        'rank',
        data=df,
        ci='sd',
        platte=sns.diverging_palette(250, 30, 65, center="dark", as_cmap=True),
        dodge=True,
        join=False,
        hue='tool',
    ).set_title(f"Ranking Plot")

    plt.legend(ncol=4, loc='upper center')
    plt.xlim(-1, 1)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'ranking_plot.pdf'))

    plt.show()


def generate_pairwise_comparisson_matrix(df: pd.DataFrame,
                                         metric: str = 'test',
                                         tools: typing.List = [],
                                         ) -> pd.DataFrame:
    """
    Generates pairwise matrices as detailed in
    https://en.wikipedia.org/wiki/Condorcet_method

    "For each possible pair of candidates, one pairwise count indicates how many voters
    prefer one of the paired candidates over the other candidate, and another pairwise count
    indicates how many voters have the opposite preference.
    The counts for all possible pairs of candidates summarize
    all the pairwise preferences of all the voters."

    """
    # Step 1: Calculate the mean and std of each fold,
    # That is, collapse the folds
    df = df.groupby(['tool', 'model', 'task']).mean().add_suffix('_mean').reset_index()

    # Just care about the provided tools
    if len(tools) < 1:
        tools = sorted(df['tools'].unique())
    df = df[df['tool'].isin(tools)]

    pairwise_comparisson_matrix = pd.DataFrame(data=np.zeros((len(tools), len(tools))),
                                               columns=tools, index=tools)

    for task in df['task'].unique():
        # we use datasets as our voters
        # Then we add each ballot
        df_task = df[df['task'] == task]

        # So we create a ballot for this dataset
        this_pairwise_comparisson_matrix = pd.DataFrame(
            data=np.zeros((len(tools), len(tools))),
            columns=tools, index=tools
        )

        for candidate in tools:
            # Every row in the matrix dictates over who the current tool
            # wins in performance. Equal performance (likely diagonal)
            # should get zero
            candidate_performance = df_task.loc[df_task['tool'] == candidate,
                                                metric + '_mean'].to_numpy().item(0)

            for opponent in tools:

                # Do not update same performance
                if opponent == candidate:
                    continue

                opponent_performance = df_task.loc[df_task['tool'] == opponent,
                                                   metric + '_mean'].to_numpy().item(0)

                # We update the this_pairwise_comparissin_matrix cells of the current tool
                # that have a better performance than the current tool
                if not np.isnan(candidate_performance) and np.isnan(opponent_performance):
                    # If we have a NaN as opponent, we always win
                    this_pairwise_comparisson_matrix.loc[candidate, opponent] = 1
                elif metric == 'test' and candidate_performance > opponent_performance:
                    this_pairwise_comparisson_matrix.loc[candidate, opponent] = 1
                elif metric == 'overfit' and candidate_performance < opponent_performance:
                    this_pairwise_comparisson_matrix.loc[candidate, opponent] = 1
                elif metric not in ['test', 'overfit']:
                    raise NotImplementedError(metric)

        pairwise_comparisson_matrix = pairwise_comparisson_matrix.add(
            this_pairwise_comparisson_matrix,
            fill_value=0
        )
    return pairwise_comparisson_matrix


def get_sorted_locked_winner_losser_tuples(pairwise_comparisson_matrix: pd.DataFrame
                                           ) -> typing.List[typing.Tuple[str, str]]:
    """
    Implements the sort and lock step of
    https://en.wikipedia.org/wiki/Ranked_pairs

    Sort:
    The pairs of winners, called the "majorities", are then sorted from the largest majority to the smallest majority. A majority for x over y precedes a majority for z over w if and only if one of the following conditions holds:

Vxy > Vzw. In other words, the majority having more support for its alternative is ranked first.
Vxy = Vzw and Vwz > Vyx. Where the majorities are equal, the majority with the smaller minority opposition is ranked first.[vs 1]

    Lock
    The next step is to examine each pair in turn to determine the pairs to "lock in".

    Lock in the first sorted pair with the greatest majority.
    Evaluate the next pair on whether a Condorcet cycle occurs when this pair is added to the locked pairs.
    If a cycle is detected, the evaluated pair is skipped.
    If a cycle is not detected, the evaluated pair is locked in with the other locked pairs.
    Loop back to Step #2 until all pairs have been exhausted.
    """

    pairs = []
    for candidate, row in pairwise_comparisson_matrix.iterrows():
        for opponent in pairwise_comparisson_matrix.columns:
            if opponent == candidate:
                next
            pair = (candidate, opponent)
            # we count positive votes first
            number_of_votes = row[opponent]
            number_of_oposition = pairwise_comparisson_matrix.loc[opponent, candidate]

            # We ask for number_of_vote > 0 so that we do not double count pairs
            # that is, it is the same candidate, opponent than opponent, candidate
            winner = pairwise_comparisson_matrix.loc[
                candidate, opponent] > pairwise_comparisson_matrix.loc[
                    opponent, candidate]
            if number_of_votes > 0 and winner:
                pairs.append((pair, number_of_votes, number_of_oposition))

    # Sort the list
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    print(f"sorted=>{sorted_pairs}")
    return [pair for pair, v, o, in sorted_pairs]


def plot_ranked_pairs_winner(df: pd.DataFrame,
                             metric: str = 'test',
                             tools: typing.List = [],
                             output_dir: typing.Optional[str] = None) -> None:
    """
    Implements the voting mechanism from
    https://en.wikipedia.org/wiki/Ranked_pairs.
    """

    # Tally: To tally the votes, consider each voter's preferences.
    # For example, if a voter states "A > B > C" (A is better than B, and B is better than C),
    # the tally should add one for A in A vs. B, one for A in A vs. C, and one for B
    # in B vs. C. Voters may also express indifference (e.g., A = B), and unstated candidates
    # are assumed to be equal to the stated candidates.
    pairwise_comparisson_matrix = generate_pairwise_comparisson_matrix(
        df, metric, tools
    )
    print(f"pairwise_comparisson_matrix={pairwise_comparisson_matrix}")
    pairwise_comparisson_matrix.to_csv('pairwise_comparisson_matrix.csv')

    # Sort and lock
    # a sorted list of locked winners (winner, losser)
    pair_of_wrinners = get_sorted_locked_winner_losser_tuples(pairwise_comparisson_matrix)

    G = nx.DiGraph()
    G.add_edges_from(pair_of_wrinners)
    #pos = nx.spring_layout(G)
    pos = nx.spring_layout(G,k=0.95,iterations=50)
    # k controls the distance between the nodes and varies between 0 and 1
    # iterations is the number of times simulated annealing is run
    # default k =0.1 and iterations=50
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_size = 500, node_color='none')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=pair_of_wrinners, edge_color='b', arrows=True, arrowsize=20)
    plt.show()


def plot_testsubtrain_history(csv_location: str, tools: typing.List[str],
                              output_dir: typing.Optional[str] = None) -> None:
    """
    Parses a list of ensemble history files and plot the difference between
    train and test
    """
    dfh = []
    for data_file in glob.glob(os.path.join(csv_location, '*ensemble_history.csv')):
        dfh.append(
            pd.read_csv(
                data_file,
                index_col=0,
            )
        )

    if len(dfh) == 0:
        print(f"ERROR: No ensemble history data to parse on {csv_location}")
        return

    dfh = pd.concat(dfh).reindex()

    # Data needs to be sorted so that we can subtract, add, etc as everything is ordered
    dfh = dfh.sort_values(by=['tool', 'task', 'fold', 'Timestamp']).reset_index(drop=True)

    if any([tool not in dfh['tool'].tolist() for tool in tools]):
        raise ValueError(f"Experiment {tools} was not found in the dataframe {dfh['tool']}")

    # amount of data and for that we re-build the dataframe because it is easier
    recompleted_desired_df = []
    for tool in tools:
        desired_df = dfh[dfh['tool'] == tool].reset_index(drop=True)
        desired_df['TestMinusTrain'] = desired_df['ensemble_test_score'].subtract(
            desired_df['ensemble_optimization_score'])

        # Make the timestamp the same, as the longest stamp so that
        # sns does everything for us
        # That is, we want to make sure every fold of every task has the same
        for task in desired_df['task'].unique():
            mask = desired_df['task'] == task
            all_folds = desired_df[mask]['fold'].unique().tolist()
            count_folds = [desired_df[mask][desired_df[mask]['fold'] == a].shape[
                0] for a in all_folds]
            argmax = np.argmax(count_folds)
            biggest_fold = all_folds[np.argmax(count_folds)]

            # Make timestamp a range
            time_mask = (desired_df['task'] == task) & (desired_df['fold'] == biggest_fold)
            desired_df.loc[time_mask, 'Timestamp'] = pd.Series(range(count_folds[argmax]), index = desired_df.loc[time_mask, 'Timestamp'].index)

            # So the strategy here is to copy over the biggest fold,
            # and re-place values of other folds into it. So the expectation
            # is that we have the same timestamt and biggest fold will have 1000
            # elements, it is easier to make a copy of this biggest fold data and
            # replace a the rows with that of other fold. This will collapse uncertainty
            # in the rows that only the biggest fold has

            # No need to to do anything for the biggest fold. For the remaining
            # folds, we copy the biggest fold data as a base and overwrite with desired data
            recompleted_desired_df.append(
                desired_df[mask][desired_df[mask]['fold'] == biggest_fold]
            )

            for fold in set(all_folds) - {biggest_fold}:
                base_frame = desired_df[mask][desired_df[mask]['fold'] == biggest_fold
                                              ].reset_index(drop=True)
                base_frame['fold'] = fold
                this_frame = desired_df[mask][desired_df[mask]['fold'] == fold
                                              ].reset_index(drop=True)
                # Copy values from original frame into the base frame that is gonna be
                # used to create a new frame with same number of timestamps
                base_frame.loc[:this_frame['TestMinusTrain'].shape[0], 'TestMinusTrain'
                               ] = this_frame['TestMinusTrain']
                recompleted_desired_df.append(base_frame)

    desired_df = pd.concat(recompleted_desired_df).reset_index(drop=True)

    # make sure autosklearn is in the data
    sns.set_style("whitegrid")
    ordered_tasks = desired_df.task.value_counts().index
    g = sns.FacetGrid(desired_df, col="task", col_wrap=2, sharex=False, sharey=False, hue="tool")
    # height=1.7, aspect=4,)
    g.map(sns.lineplot, 'Timestamp', 'TestMinusTrain', ci='sd',

        palette=sns.color_palette("Set2"),
        err_style='band',
        label=tool,
    )
    g.set(xticks=[])

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Test-Train History')

    plt.legend()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, '_'.join(tools) + '.pdf'))

    plt.show()


# def printsomethind():
#     experiment_results = {}
#     for tool_task, test_value in data.groupby(['tool', 'task']).mean()['test'].to_dict().items():
#         tool, task = tool_task
#         if tool not in experiment_results:
#             experiment_results[tool] = {}
#         if task not in experiment_results[tool]:
#             experiment_results[tool][task] = test_value
#
#     summary = []
#     for tool in experiment_results:
#         row = experiment_results[tool]
#         row['tool'] = tool
#         summary.append(row)
#
#     summary = pd.DataFrame(summary)
#     print(summary)
#
#     # The best per task:
#     for task in [c for c in summary.columns if c != 'tool']:
#         best = summary[task].argmax()
#         print(f"{task}(best) = {summary['tool'].iloc[best]}")
#
#     # How many times better than autosklearn
#     summary_no_tool_column = summary.loc[:, summary.columns != 'tool']
#     baseline_results = summary[summary['tool']=='autosklearn'].loc[:, summary[summary['tool']=='autosklearn'].columns != 'tool']
#     for index, row in summary.iterrows():
#         tool = row['tool']
#         if tool == 'autosklearn': continue
#         print(f"{tool} (better_than_baseline): {np.count_nonzero(summary_no_tool_column.iloc[index] > baseline_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to plot CSV results')
    parser.add_argument(
        '--csv_location',
        help='Where to look for csv files',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--experiment',
        action='append',
        help='Generates results/plots just for this experiment',
        default=None,
        type=str,
        required=False,
    )
    args = parser.parse_args()

    # First get the data
    df = parse_data(args.csv_location)

    tools = df['tool'].unique()
    #for category in ['Stacking', 'BBCSMBOAnd', 'BBCEnsembleSelection', 'ScoreEnsemb']:
    #    this_tools = [t for t in tools if category in t]
    #    # Relative performance difference
    #    plot_relative_performance(df.copy(),
    #                              tools = this_tools,
    #                              metric='test', output_dir=None)
    #    # Plot the training/test history
    #    plot_testsubtrain_history(args.csv_location,
    #                              tools = ['autosklearn'] + this_tools, output_dir=None)
    #plot_relative_performance(df.copy(), tools = ['autosklearnStacking', 'autosklearnBBCEnsembleSelection_B_50_Nb_25'], metric='overfit', output_dir=None)

    # Plot Ranking Loss
    #plot_ranks(df.copy(), metric='overfit', output_dir=None)

    # Make pairwise matrix
    plot_ranked_pairs_winner(metric='test', df=df, tools=['autosklearn'] + [t for t in df['tool'].unique() if 'BBCScore' in t])

