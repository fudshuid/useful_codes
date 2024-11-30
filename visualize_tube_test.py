############ INFO ############
# Author : Gyu-Hwan Lee (Korea Institute of Science and Technology, Seoul National University)
# Contact: gh.lee@kist.re.kr
# Written : 2022-04-11
# Last edit : 2023-07-14 (Gyu-Hwan Lee)

'''
- Usage ->
$ python visualize_tube_test.py --file winner_loser.txt

- Description: Usual way of determining rank in tube test (cf. Wang et al. Science, 2011) is simply to count the number of wins for each mouse.
               In such case, many ties (mice with same number of wins) occur. Inspired by Elo rating, this code tries to get a better representation of 
               hierarchy obtained by tube test by considering how easy or hard for a mouse to win the opponent.

- Input text (tsv format) format:

Day  winner  loser
1    B0      R0
1    G3	     B1
1    G2      B2

'''
##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str,
                help="Data file name (must be in a tsv format)")
args = ap.parse_args()
data_file = args.file

# load tube test data
data_df = pd.read_csv(data_file, sep='\t')
days = np.sort(np.unique(data_df.Day.to_numpy()))

best_rankings = []
tier_dicts = []
for day_idx, day in enumerate(days):
    print("#"*20 + f" Day{day} " + "#"*20)
    day_df = data_df[data_df['Day'] == day]
    mice = np.sort(np.unique(list(day_df.winner.to_numpy()) + list(day_df.loser.to_numpy())))
    
    # count wins of each mouse
    winners = day_df.winner.to_numpy()
    wins = []
    win_dict = {}

    for mouse in mice:
        n_win = np.sum(winners == mouse)
        win_dict[mouse] = n_win
        # print(f"Mouse: {mouse}, Win count: {n_win}")
        wins.append(n_win)
        
    unique_wins = np.unique(wins)[::-1]

    tiers = {}
    mouse2tier = {}
    for idx, win_count in enumerate(unique_wins):
        tier = idx+1
        mice_in_tier = [mouse for mouse in mice if win_dict[mouse] == win_count]

        tiers[win_count] = mice_in_tier
        for mouse in mice_in_tier:
            mouse2tier[mouse] = tier

    print(f"Tiers: {tiers}")
    tiers['nwins'] = unique_wins
    tier_dicts.append(tiers)

    # calculate score for each individual (per tier)
    ordered_ranking = []
    for n_win in unique_wins:
        mice_in_tier = tiers[n_win]

        # when there are more than two mice in a tier, try to order them:
        #   1) mouse with lower score is ranked higher
        #   2) if more than two mice have the same score, rank them according to the minumum tier number that a mouse won
        if len(mice_in_tier) > 1:
            print(f"Ordering mice with {n_win} wins: {mice_in_tier}")
            scores = []
            loser_tiers_list = []
            for winner in mice_in_tier:
                lost_mice = day_df[day_df['winner'] == winner].loser.to_numpy()
                loser_tiers = [mouse2tier[loser] for loser in lost_mice]
                score = np.sum(loser_tiers)

                scores.append(score)
                loser_tiers_list.append(loser_tiers)

            for mouse, loser_tiers in zip(mice_in_tier, loser_tiers_list):
                print(f"  {mouse} won mice with tier {loser_tiers} (min={min(loser_tiers)}, sum={np.sum(loser_tiers)})")

            unique_scores = np.unique(scores)
            tier_ranking = []
            for score in unique_scores:
                idxs_w_score = np.where(scores == score)[0]
                mice_w_score = [mice_in_tier[i] for i in idxs_w_score]
                loser_tiers_collected = [loser_tiers_list[i] for i in idxs_w_score]

                if len(idxs_w_score) == 1:
                    tier_ranking.append(mice_in_tier[idxs_w_score[0]])
                else:
                    min_loser_tiers = [min(loser_tiers) for loser_tiers in loser_tiers_collected]
                    order = np.argsort(min_loser_tiers)
                    mice_ordered = np.array(mice_w_score)[order]
                    
                    tier_ranking += mice_ordered.tolist()

            ordered_ranking += tier_ranking
            print(f"  --> Ordering result: {tier_ranking}")

        # when there is only one mice in a tier, just add it to the ordered ranking         
        else:
            ordered_ranking += mice_in_tier

    best_rankings.append(ordered_ranking)

# generate trajectory of each mouse (trajectory: a specific mouse's ranks over trial dates)(eg. [3, 5, 10])
trajectories = {m: [] for m in mice}
for day, ranking in zip(days, best_rankings):
    for i, mouse in enumerate(ranking):
        trajectories[mouse].append(i+1)
        
## Start visualization
fig, ax = plt.subplots(figsize=(len(days)*2,4))

# change this color list to get a more customized color visualization
colors = sns.color_palette('pastel', n_colors=len(mice))
# draw trajectory for each mouse
for i, m in enumerate(mice):
    ax.plot(np.arange(1,len(days)+1), trajectories[m], label=m, 
            color='gray', alpha=0.8)
    
for day, ranking, tiers in zip(days, best_rankings, tier_dicts):
    for i, mouse in enumerate(ranking):
        # add rectangle for each mouse in each day
        ax.add_patch(patches.Rectangle(
                        (day-0.2, i+0.5), 0.4, 1,
                        edgecolor = 'black',
                        facecolor = colors[np.where(mice == mouse)[0][0]],
                        fill=True
                     ))
        # add text to annotate the identity of the mouse in that position
        ax.text(day, i+1, mouse, fontsize = 12,
               horizontalalignment='center', verticalalignment='center',
               fontfamily='sans-serif')

    # delimit tier boundaries
    nwins = tiers['nwins']
    for nwin in nwins:
        tier_mice = tiers[nwin]
        
        # get index of mice in the same tier
        idxs = [i for i, m in enumerate(ranking) if m in tier_mice]
        start_idx = min(idxs)
        end_idx = max(idxs)
        
        # draw thick rectangle
        ax.add_patch(patches.Rectangle(
                        (day-0.2, start_idx+0.5), 0.4, (end_idx-start_idx+1),
                        edgecolor = 'black',
                        linewidth = 3,
                        fill=False
                     ))

ax.invert_yaxis()
ax.set_xlabel("Day")
ax.set_xticks(days)
ax.set_xticklabels(days)

ax.set_ylabel("Rank")
ax.set_yticks(np.arange(1,len(mice)+1))
ax.set_yticklabels(np.arange(1,len(mice)+1))

# save figure
fig.savefig(f"Tube_test_rank_visualized_Days_{days}.pdf", bbox_inches='tight', pad_inches=0.1)
