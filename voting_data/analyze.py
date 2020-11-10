import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# column names
# votes,eevp,eevp_source,timestamp,state,expected_votes,trump2016,votes2012,votes2016,vote_share_rep,vote_share_dem,vote_share_trd
df = pd.read_csv("voting_data.csv")

states = [
 'alaska', 'alabama', 'arkansas', 'arizona', 'california', 'colorado',
 'connecticut', 'delaware', 'florida', 'georgia',
 'hawaii', 'iowa', 'idaho', 'illinois', 'indiana', 'kansas', 'kentucky',
 'louisiana', 'massachusetts', 'maryland', 'maine', 'michigan',
 'minnesota', 'missouri', 'mississippi', 'montana', 'north-carolina',
 'north-dakota', 'nebraska', 'new-hampshire', 'new-jersey', 'new-mexico',
 'nevada', 'new-york', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
 'rhode-island', 'south-carolina', 'south-dakota', 'tennessee', 'texas',
 'utah', 'virginia', 'vermont', 'washington', 'wisconsin',
 'west virginia', 'wyoming',
]

for state in states:
    print(state)
    current_state = state
    # Grab this state's data 
    mask = (df['state'] == current_state)
    subset = df[mask]
    
    # Get ratio (dem_vots / total_votes) / (rep_votes / total_votes)
    trimmed_subset = subset[['vote_share_dem']].values[1:] / subset[['vote_share_rep']].values[1:]
    vote_ratio = pd.DataFrame(trimmed_subset, columns=['Democratic/Republican Vote Share'])
    vote_ratio = vote_ratio.reset_index()

    vote_subset = subset[['votes']].values[1:]
    vote_subset = pd.DataFrame(vote_subset, columns=['votes'])
    vote_subset = vote_subset.reset_index()

    # Plot
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_ylim([0.0, 2.0])
    vote_ratio.plot(kind='scatter', x='index', y='Democratic/Republican Vote Share', ax=ax, label='vote_share_ratio', color='DarkGreen')
    vote_subset.plot(x='index', y='votes', ax=ax2)
    plt.title("{}".format(current_state))
    plt.savefig("state_plots/{}_plot.png".format(current_state))
    plt.close()
    plt.clf()
    

