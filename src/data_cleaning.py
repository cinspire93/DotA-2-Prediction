import pandas as pd
import numpy as np
import random
import requests
import time
from bs4 import BeautifulSoup
from itertools import combinations

hero_attribute_df = pd.read_csv('dota-2-matches/hero_attributes.csv')


def assign_player_team(players_df):
    '''
    Input: player info dataframe
    Output: one extra column (radiant_player) indicating the player team, Radiant=True and Dire=False
    '''
    players_df['radiant_player'] = players_df['player_slot'] < 50
    return players_df

def construct_hero_selection_df(players_df, heros_chart):
    '''
    Input: player info dataframe and heros info dataframe
    Output: hero_selection_df that gives 1 for every selected hero, and 0 otherwise for every game(row)
    '''
    # Quick function to help me get nice snake form hero names from the name column in heros_chart
    hero_names = heros_chart.localized_name.apply(lambda x: '_'.join(x.split()))
    # The number of heros is very important for separating radiant selections from dire ones
    num_heros = len(hero_names)
    # Get both Radiant and Dire heros (basically a symmetric copy of hero names), later use the hero_two_teams as column
    # for the hero_selection_df
    radiant_heros = hero_names.apply(lambda x: x + '_radiant').tolist()
    dire_heros = hero_names.apply(lambda x: x + '_dire').tolist()
    hero_two_teams = radiant_heros + dire_heros

    hero_selection_raw = players_df.groupby('match_id').apply(get_hero_selection, num_heros=num_heros).apply(pd.Series).values
    return pd.DataFrame(data=hero_selection_raw, index=range(50000), columns=hero_two_teams)

def get_hero_selection(single_game_player_info, num_heros):
    '''
    Input: player info from a single game
    Output: one row of NumPy array representing the hero selection by both teams
    '''
    hero_selection_row = np.zeros(2 * num_heros)
    # Must subtract hero_id by 1 because they start from 1 to 112, index start 1 lower
    radiant_heros = single_game_player_info[single_game_player_info.radiant_player].hero_id.values - 1
    dire_heros = single_game_player_info[~single_game_player_info.radiant_player].hero_id.values + num_heros - 1

    hero_selections = np.append(radiant_heros, dire_heros)
    hero_selection_row[hero_selections] = 1
    return hero_selection_row

def construct_x_seconds_df(players_time_df, threshold=600):
    '''
    Input: player info dataframe
    Output: all player info before the desired time threshold, defaulted to be 10 minutes/600 seconds
    '''
    x_seconds_df = players_time_df.groupby('match_id').apply(lambda game: game[game.times <= threshold])
    x_seconds_df.reset_index(drop=True, inplace=True)
    return x_seconds_df

def construct_x_seconds_max_wealth(x_seconds_df):
    '''
    Input: player info before desired time threshold
    Output: the max wealth, experience, number of last hits for every player before 10 minutes
    '''
    x_seconds_max_wealth = x_seconds_df.groupby('match_id').max().ix[:, 1:]
    return x_seconds_max_wealth

def construct_x_seconds_gold_growth_benchmark(x_seconds_df):
    '''
    Input: player info before desired time threshold
    Output: the mean, std of gold growth before 10 minutes
    '''
    # Thanks to a well-structured player info dataframe, I can refer to all gold columns quickly
    gold_growth_df = x_seconds_df.ix[:, 2::3].diff().join(x_seconds_df.match_id)
    # Subtracting 100 from each minute-to-minute gold growth to normalize for actual gold growth
    gold_growth_df = gold_growth_df[(gold_growth_df.ix[:, 0]>=0)] - 100
    gold_growth_df['match_id'] += 100

    gold_growth_mean = gold_growth_df.groupby('match_id').mean()
    gold_growth_std = gold_growth_df.groupby('match_id').std()
    return gold_growth_mean.join(gold_growth_std, lsuffix='_mean', rsuffix='_std')

def construct_num_team_fights(team_fights_df, threshold=600):
    '''
    Input: team fight info before desired time threshold
    Output: the number of team fights for every match in a dataframe
    '''
    num_team_fights_df = pd.DataFrame(team_fights_df[team_fights_df.end < threshold].groupby('match_id')['start'].count())
    num_team_fights_df = num_team_fights_df.rename(columns={'start': 'count'})
    return pd.DataFrame(index=range(50000)).join(num_team_fights_df).fillna(0).astype(int)

def construct_net_death_count_from_teamfights(teamfight_players_df, num_team_fights_df):
    '''
    Input: teamfight info for players by match, number of teamfights before 10 mins by match
    Output: net death count for radiant and dire players from all teamfights before 10 mins in each match
    '''
    team_fights_before_ten_min_raw = teamfight_players_df.groupby('match_id')\
                                    .apply(get_net_death_count, num_team_fights_df=num_team_fights_df)
    # The missing_index is very important here, as it captures matches where no team fights happened(abandoned).
    # In this case, groupby('match_id') will not give us those matches, so we have to fill them in manually with missing_index
    missing_index = np.array(list(set(range(50000)) - set(team_fights_before_ten_min_raw.index)))
    team_fights_before_ten_min_raw = team_fights_before_ten_min_raw.reindex(index=range(50000), fill_value=[0, 0])
    return pd.DataFrame(team_fights_before_ten_min_raw.apply(pd.Series).values, index=range(50000),
                        columns=['radiant_net_death_count', 'dire_net_death_count'])

def get_net_death_count(single_game_teamfight_info, num_team_fights_df):
    '''
    Input: teamfight info for a single match, number of teamfights before 10 mins by match
    Output: net death count for radiant and dire players from all teamfights before 10 mins in that single match
    '''
    match_id = int(single_game_teamfight_info['match_id'].mean())
    num_teamfights = num_team_fights_df.ix[match_id, 0]
    teamfights_before_ten_min = single_game_teamfight_info.reset_index(drop=True).ix[:num_teamfights*10 - 1]
    if teamfights_before_ten_min.shape[0] == 0:
        return [0, 0]
    radiant_net_death_count = teamfights_before_ten_min[teamfights_before_ten_min.player_slot<50]['deaths'].sum()
    dire_net_death_count = teamfights_before_ten_min[teamfights_before_ten_min.player_slot >50]['deaths'].sum()
    return [radiant_net_death_count, dire_net_death_count]

def get_hero_index_mapping(heros_chart):
    '''
    Input: hero info dataframe
    Output: a dictionary where the key is the hero name, and the value its index
    '''
    index_to_hero = heros_chart['localized_name'].apply(lambda name: '_'.join(name.split()).replace("'", "")).to_dict()
    hero_index_mapping = {hero: index for index, hero in index_to_hero.iteritems()}
    return hero_index_mapping

def get_hero_roles(link):
    '''
    Input: URL of a hero's DotA 2 webpage
    Output: A list of all roles fulfilled by the selected hero
    '''
    hero_page = requests.get(link).content

    # Making sure to save all html files for future reference, no one likes a scraper.
    f = open('heros_html/{}.txt'.format(link.split('/')[-2]), 'w')
    f.write(hero_page)
    f.close()

    hero_info = BeautifulSoup(hero_page, 'html.parser')
    return hero_info.find('p', id='heroBioRoles').text.split(' - ')

def construct_hero_roles(hero_index_mapping=None):
    '''
    Input: Optional hero_index_mapping
    Output: if no hero_index_mapping is provided, then return dictionary with hero names as keys and\
            list of roles as values. Otherwise, the key will be mapped indexes.
    '''
    hero_facebook = requests.get('http://www.dota2.com/heroes/')
    soup = BeautifulSoup(hero_facebook.content, 'html.parser')
    hero_links = [link['href'] for link in soup.find_all('a', class_='heroPickerIconLink')]
    hero_roles = {}
    # Let the scraping begin
    while len(hero_roles) != 113:
        for link in hero_links:
            # Make sure that I do not scrape the same thing twice
            if link not in hero_roles.keys():
                try:
                    hero_roles[link] = get_hero_roles(link)
                except AttributeError:
                    continue
            time.sleep(random.random())
    for link in hero_roles.keys():
        hero_roles[link.split('/')[-2]] = hero_roles.pop(link)
    if hero_index_mapping:
        for hero in hero_roles.keys():
            # The get method here is specifically used to prevent conflict with MonkeyKing, who is not present for the current data
            hero_roles[hero_index_mapping.get(hero, 112)] = hero_roles.pop(hero)
    return hero_roles

# Remember to integrate construct_hero_composition_df with get_game_hero_composition, better in a class without GLOBALs
def construct_hero_composition_df(players_df, hero_attribute_df):
    '''
    Input: player info dataframe, heros info dataframe and a roles for each hero
    Output: hero composition for both teams for all games
    '''
    total_roles = hero_attribute_df.columns.tolist()
    roles_both_teams = [role+'_radiant' for role in total_roles] + [role+'_dire' for role in total_roles]
    hero_compositions = players_df.groupby('match_id').apply(get_game_hero_composition,
                        hero_attribute_df=hero_attribute_df).apply(pd.Series).values
    return pd.DataFrame(data=hero_compositions, index=range(50000), columns=roles_both_teams)

def get_game_hero_composition(single_game_player_info, hero_attribute_df):
    '''
    Input: player info from a single game
    Output: hero compositions for both radiants and dires from the given game
    '''
    radiant_heros = single_game_player_info[single_game_player_info.radiant_player]['hero_id']
    dire_heros = single_game_player_info[~single_game_player_info.radiant_player]['hero_id']
    return np.array(hero_attribute_df.ix[radiant_heros].sum().tolist() + hero_attribute_df.ix[dire_heros].sum().tolist())

def construct_hero_attribute_df(hero_roles):
    '''
    Input: a dictionary with hero indexes as keys and lists of roles as values
    Output: an attribute DataFrame that gives 1 if a hero takes on a role, and 0 otherwise
    '''
    total_roles = []
    for roles in hero_roles.itervalues():
        total_roles += roles
    total_roles = sorted(list(set(total_roles) - {'Pusher', 'Ranged', 'Melee', 'Durable', 'Escape', 'Jungler', 'Nuker'}))
    num_heros = len(hero_roles)-1
    hero_attribute_df = pd.DataFrame(index=range(num_heros), columns=total_roles)
    for hero_index in xrange(num_heros):
        hero_attribute_df.ix[hero_index] = [1 if role in hero_roles[hero_index] else 0 for role in total_roles]
    return hero_attribute_df

def construct_hard_coded_hero_attribute_df(filepath='dota-2-matches/hero_attributes.csv'):
    '''
    Input: path to the hardcoded hero attribute csv file
    Output: hardcoded hero attribute DataFrame
    '''
    hero_attribute_df = pd.read_csv(filepath)
    return hero_attribute_df

def get_interaction_terms(hero_composition_df):
    '''
    Input: hero composition DataFrame
    Output: expanded hero composition DataFrame with all interaction terms included
    '''
    for interaction_term in combinations(hero_composition_df.columns.tolist(), 2):
        hero_composition_df[interaction_term[0]+'-x-'+interaction_term[1]] = hero_composition_df[interaction_term[0]] * hero_composition_df[interaction_term[1]]
    return hero_composition_df

def construct_ten_min_firstblood_df(objectives_df):
    '''
    Input: objectives DataFrame
    Output: every first blood that happened within the first 10 minutes of the game, 1 for radiant, 0 for dire.
            The output will get joined with the feature DataFrame later, any NaNs that results from the join will be filtered
            out when I use pd.get_dummies().
    '''
    ten_min_firstblood = objectives_df[(objectives_df.time<=600) & (objectives_df.subtype=='CHAT_MESSAGE_FIRSTBLOOD')]
    ten_min_firstblood = ten_min_firstblood.drop(['player1', 'player2', 'subtype', 'team', 'time', 'value'], axis=1)
    ten_min_firstblood['radiant_firstblood'] = (ten_min_firstblood['slot'] < 5).astype(int)
    ten_min_firstblood = ten_min_firstblood.drop('slot', axis=1)
    return ten_min_firstblood

def get_game_hero_info(players_df, hero_attribute_df):
    '''
    Input: players info DataFrame and hero attribute DataFrame
    Output: reduced hero info for each game, includes match number, player position,
            whether the player is on radiant or not and the main role of the hero selected
    '''
    major_roles = hero_attribute_df.apply(np.argmax, axis=1)
    game_hero_info_df = players_df[['match_id', 'hero_id', 'player_slot', 'radiant_player']].copy()
    game_hero_info_df['role'] = major_roles.ix[game_hero_info_df.hero_id].reset_index(drop=True)
    return game_hero_info_df

def construct_role_check_df(game_hero_info_df):
    '''
    Input: reduced hero info for each game
    Output: check whether each team in every game has too many carries, 1 for yes, 0 otherwise
    '''
    checked_roles = ['Radiant_Carries', 'Dire_Carries', 'Radiant_Supports', 'Dire_Supports']
    role_check_raw = game_hero_info_df.groupby('match_id').apply(get_carry_support_count).apply(pd.Series).values
    role_check_df = pd.DataFrame(role_check_raw, index=range(50000), columns=checked_roles)
    role_check_df['too_many_carries_radiant'] = (role_check_df['Radiant_Carries'] >= 3).astype(int)
    role_check_df['too_many_carries_dire'] = (role_check_df['Dire_Carries'] >= 3).astype(int)
    return role_check_df[['too_many_carries_radiant', 'too_many_carries_dire']]

def get_carry_support_count(single_game_hero_info):
    '''
    Input: reduced hero info for one single game
    Output: number of players in each role=> [Carry, Support(merged from Hard and Farm Support), Offlane, Mid]
    '''
    radiant_count = single_game_hero_info[single_game_hero_info.radiant_player].role.value_counts().to_dict()
    dire_count = single_game_hero_info[~single_game_hero_info.radiant_player].role.value_counts().to_dict()
    radiant_support_count = radiant_count.get('Farm_Support', 0) + radiant_count.get('Hard_Support', 0)
    dire_support_count = dire_count.get('Farm_Support', 0) + dire_count.get('Hard_Support', 0)
    return [radiant_count.get('Carry', 0), dire_count.get('Carry', 0), radiant_support_count, dire_support_count]

def construct_long_player_gold_df(ten_min_max_wealth):
    '''
    Input: the max amount of wealth (gold, last hits, experience) for each player within the first 10 minutes
    Output: long formatted DataFrame that unpivots the gold columns (for every player) onto the match_id column.
            This will be merged with the game_hero_info_df
    '''
    long_player_gold_df = pd.melt(ten_min_max_wealth.ix[:, 0::3].reset_index(), id_vars='match_id', var_name='player_slot',
                                  value_name='max_gold')
    long_player_gold_df['player_slot'] = long_player_gold_df['player_slot'].apply(lambda string: int(string.split('_')[-1]))
    return long_player_gold_df.sort_values(by=['match_id', 'player_slot']).reset_index(drop=True)

if __name__ == '__main__':
    players_df = pd.read_csv('dota-2-matches/players.csv')
    heros_chart = pd.read_csv('dota-2-matches/hero_names.csv')
    players_time_df = pd.read_csv('dota-2-matches/player_time.csv')

    players_df = assign_player_team(players_df)

    hero_selection_df = construct_hero_selection_df(players_df, heros_chart)
    # These two should check that columns are correctly specified
    # and that each game(row) has 10 selections
    print '----------Hero Selection DataFrame----------'
    print hero_selection_df.head()
    print hero_selection_df.sum(axis=1)[:5]
    print '\n'

    x_seconds_df = construct_x_seconds_df(players_time_df, x=600)
    print '----------Ten Min Bench Mark----------'
    print ten_min_bench_mark.head()
