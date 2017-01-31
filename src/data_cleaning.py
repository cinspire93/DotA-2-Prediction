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

# Remember to integrate construct_hero_selection_df with get_hero_selection, better in a class
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

    hero_selection_raw = players_df.groupby('match_id').apply(get_hero_selection)
    return pd.DataFrame(data=hero_selection_raw.apply(pd.Series).values, index=range(50000), columns=hero_two_teams)

def get_hero_selection(single_game_player_info):
    '''
    Input: player info from a single game
    Output: one row of NumPy array representing the hero selection by both teams
    '''
    hero_selection_row = np.zeros(224)
    # Must subtract hero_id by 1 because they start from 1 to 112, index start 1 lower
    radiant_heros = single_game_player_info[single_game_player_info.radiant_player].hero_id.values - 1
    dire_heros = single_game_player_info[~single_game_player_info.radiant_player].hero_id.values + 112 - 1

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
    return pd.DataFrame(index=range(50000)).join(num_team_fights_df).fillna(0)

def construct_net_death_count_from_teamfights(teamfight_players_df, num_team_fights_df):
    '''
    Input: teamfight info for players by match, number of teamfights before 10 min by match
    Output: net gold change for radiant and dire by players in each match before 10 mins
    '''
    net_death_count = pd.DataFrame(index=range(50000), columns=['net_death_count_radiant', 'net_death_count_dire'])
    for match_number in xrange(50000):
        num_team_fights = num_team_fights_df.ix[match_number, 0]
        match_teamfights_before_ten_min = teamfight_players_df[teamfight_players_df.match_id==match_number].reset_index(drop=True).ix[:num_team_fights*10-1]
        radiant_net_death_count = match_teamfights_before_ten_min[match_teamfights_before_ten_min.player_slot<50]['deaths'].sum()
        dire_net_death_count = match_teamfights_before_ten_min[match_teamfights_before_ten_min.player_slot>50]['deaths'].sum()
        net_death_count.ix[match_number] = radiant_net_death_count, dire_net_death_count
    return net_death_count

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
    Output: if no hero_index_mapping is provided, then return dictionary with hero names as keys and list of roles as values. Otherwise, the key will be mapped indexes.
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
            # time.sleep(random.random())
    for link in hero_roles.keys():
        hero_roles[link.split('/')[-2]] = hero_roles.pop(link)
    if hero_index_mapping:
        for hero in hero_roles.keys():
            # The get method here is specifically used to prevent conflict with MonkeyKing, who is not present for the current data
            hero_roles[hero_index_mapping.get(hero, 113)] = hero_roles.pop(hero)
    return hero_roles

# Remember to integrate construct_hero_composition_df with get_game_hero_composition, better in a class without GLOBALs
def construct_hero_composition_df(players_df, hero_attribute_df):
    '''
    Input: player info dataframe, heros info dataframe and a roles for each hero
    Output: hero composition for both teams for all games
    '''
    total_roles = hero_attribute_df.columns.tolist()
    roles_both_teams = [role+'_radiant' for role in total_roles] + [role+'_dire' for role in total_roles]
    hero_compositions = players_df.groupby('match_id').apply(get_game_hero_composition)
    return pd.DataFrame(data=hero_compositions.apply(pd.Series).values, index=range(50000), columns=roles_both_teams)

def get_game_hero_composition(single_game_player_info):
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
    global hero_attribute_df
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
    global hero_attribute_df
    hero_attribute_df = pd.read_csv(filepath)
    return hero_attribute_df

def get_interaction_terms(hero_composition_df):
    '''
    Input: hero composition dataframe
    Output: expanded hero composition dataframe with all interaction terms included
    '''
    for interaction_term in combinations(hero_composition_df.columns.tolist(), 2):
        hero_composition_df[interaction_term[0]+'-x-'+interaction_term[1]] = hero_composition_df[interaction_term[0]] * hero_composition_df[interaction_term[1]]
    return hero_composition_df

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
