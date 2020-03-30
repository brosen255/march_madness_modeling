import pandas as pd
from opencage.geocoder import OpenCageGeocode
import geopy.distance
import re
import numpy as np
from collections import defaultdict
from statistics import mean, median

def convert_string_to_tuple(x):
    return x.replace(')','').replace('(','').split(',')
def kenpom_dict(merger):    
    merger['teamid'] = merger.zipper.apply(lambda x: x[0])
    merger = merger[~merger.teamid.isnull()]
    merger['teamid'] = merger.teamid.apply(int)
    merger['year'] = merger.zipper.apply(lambda x: x[1])
    merger['zipper2'] = list(zip(merger.teamid,merger.year))
    dict_values = [i for i in merger[cols_not_zipper].values]
    team_kenpom_dict = dict(zip(merger['zipper2'],dict_values))
    kp_dict = {k:list(v) for k,v in team_kenpom_dict.items()}
    return kp_dict
def seed_dict(merger,seed_fn):
    seeds = pd.read_csv(filepath+seed_fn)
    seeds = seeds[seeds.Season >= 2002]
    seeds['Seed'] = seeds.Seed.apply(lambda x: re.findall("\d+", x)[0])
    seeds['zipper'] = list(zip(seeds.TeamID,seeds.Season))
    seed_dict = dict(zip(seeds.zipper,seeds.Seed))
    return dict

def team_location_dict(team_coords_fn):
    coors_df = pd.read_csv(team_coords_fn,sep='|')
    coors_dict = dict(zip(coors_df.TeamID,coors_df.coords))
    uniq_teams = list(set([k for k,v in coors_dict.items()]))
    new_coors_dict = {}
    for team in uniq_teams:
        for year in range(2002,2020):
            new_coors_dict[ (team,year) ] = coors_dict[team]
    return coors_dict

def game_loc_dict(game_loc_fn):
    game_loc = pd.read_csv(game_loc_fn)
    game_loc_dict = dict(zip(game_loc.game_zip,game_loc.game_coors))
    gl = pd.DataFrame(list(game_loc_dict.items()),columns = ['gz','game_coors'])
    gl['tpair'] = gl.gz.apply(convert_string_to_tuple_to_tuple)
    gl['zip'] = gl.tpair.apply(lambda x: [x[0]] +sorted([x[1],x[2]]))
    gl.drop(['gz','tpair'],axis=1,inplace=True)
    gl['zip'] = gl.zip.apply(lambda x: str([int(i.strip()) for i in x]))
    final_game_loc_dict = dict(zip(gl.zip,gl.game_coors))
    return final_game_loc_dict

def num_chmps_dict(total_dict_chmps):
    ch = pd.DataFrame(list(total_dict_chmps.items()),columns=['zipper','chmp_value'])
    ch['zipper'] = ch.zipper.apply(convert_string_to_tuple)
    ch['zipper'] = ch.zipper.apply(lambda x: tuple([int(i.strip()) for i in x]))
    ch['zipper'] = ch.zipper.apply(lambda x: (x[1],x[0]))
    final_chmp_dict = dict(zip(ch.zipper,ch.chmp_value))
    return final_chmp_dict
def num_wins_dict(total_dict_wins):
    w = pd.DataFrame(list(total_dict_wins.items()),columns=['zipper','win_value'])
    w['zipper'] = w.zipper.apply(convert_string_to_tuple)
    w['zipper'] = w.zipper.apply(lambda x: tuple([int(i.strip()) for i in x]))
    w['zipper'] = w.zipper.apply(lambda x: (x[1],x[0]))
    final_win_dict = dict(zip(w.zipper,w.win_value))
    return final_win_dict
def conf_dict(conf_fn):
    conf = pd.read_csv(conf_fn)
    conf = conf[conf.Season >= 2002][conf.Season < 2020]
    conf['zipper'] = list(zip(conf.TeamID,conf.Season))
    conf['conf_z'] = list(zip(conf.Season,conf.ConfAbbrev))
    conf['conf_z'] = conf.conf_z.apply(str)
    conf['conf_value'] = conf.conf_z.map(total_conf_dict).fillna(0)
    final_conf_dict = dict(zip(conf.zipper,conf.conf_value))
    return final_conf_dict

def season_avg_dict(kp_dict):
    cols = ['Adjusted_eff_margin', 'AdjustedO', 'AdjustD',
           'Adjusted_tempo', 'Luck', 'Ajusted_sos', 'SOS_opp_off', 'SOS_opp_def',
           'NCSOS', 'win_pct','seed','conf_value','chmp_value','win_value','school_coors']
    df = pd.DataFrame(list(kp_dict.items()),columns = ['zipper','value_list'])
    df['seed'] = df.zipper.map(seed_dict).apply(lambda x: [x])
    df['conf_value'] = df.zipper.map(final_conf_dict).apply(lambda x: [x])
    df['chmp_value'] = df.zipper.map(final_chmp_dict).fillna(0).apply(lambda x: [x])
    df['win_value'] = df.zipper.map(final_win_dict).fillna(0).apply(lambda x: [x])
    df['tot'] = df.value_list+df.seed+df.conf_value+df.chmp_value+df.win_value
    df.drop(['value_list','seed','conf_value','chmp_value','win_value',],axis=1,inplace=True)
    df['tot'] = df.tot.apply(lambda x: [float(i) for i in x])
    df['tot'] = df.tot.apply(lambda x: np.asarray(x, dtype=np.float32))
    overall_dict = dict(zip(df.zipper,df.tot))
    return overall_dict

def season_avg_submission(subm_fn,diffs_dict):
    sub = pd.read_csv(filepath + subm_fn)
    sub['year'] = sub.ID.apply(lambda x: x.split('_')[0])
    sub['team1'] = sub.ID.apply(lambda x: x.split('_')[1])
    sub['team2'] = sub.ID.apply(lambda x: x.split('_')[2])
    sub['id1'] = list(zip(sub.team1,sub.year))
    sub['id2'] = list(zip(sub.team2,sub.year))
    sub.drop(['ID','Pred','team1','team2','year'],axis=1,inplace=True)
    sub['id1'] = sub.id1.apply(lambda x: tuple([int(i) for i in list(x)]))
    sub['id2'] = sub.id2.apply(lambda x: tuple([int(i) for i in list(x)]))

    cols = ['Adjusted_eff_margin', 'AdjustedO', 'AdjustD',
           'Adjusted_tempo', 'Luck', 'Ajusted_sos', 'SOS_opp_off', 'SOS_opp_def',
           'NCSOS', 'win_pct','seed','conf_value','chmp_value','win_value']

    l1 = sub.id1.tolist()
    l2 = sub.id2.tolist()

    vector_diff = [overall_dict[l1[i]] - overall_dict[l2[i]] for i in range(len(l1))]
    diffs_df = pd.DataFrame(vector_diff,columns = cols)
    fram = pd.concat([sub,diffs_df],axis=1)
    return fram

def add_coors_conf(fram):
    fram['coors1'] = fram.id1.map(new_coors_dict)
    fram['coors2'] = fram.id2.map(new_coors_dict)
    fram['temp_tzip'] = list(zip(fram.id1.apply(lambda x:x[0]),fram.id2.apply(lambda x:x[0])))
    fram['temp_tzip'] = fram.temp_tzip.apply(lambda x: sorted(list(x)))
    fram['ozip'] = list(zip(fram.id1.apply(lambda x:x[1]),fram.temp_tzip))
    fram['ozip'] = fram.ozip.apply(lambda x: [x[0],x[1][0],x[1][1]]).apply(str)
    fram.drop(['coors1','coors2','temp_tzip','ozip'],axis=1,inplace=True)

    fram['conf_win1'] = fram.id1.map(conf_games_dict).fillna(0)
    fram['conf_win2'] = fram.id2.map(conf_games_dict).fillna(0)
    fram['win_conf_diff'] = fram.conf_win1 - fram.conf_win2
    fram.drop(['conf_win1','conf_win2'],axis=1,inplace=True)
    return fram
def add_rank_info(fram):
    fram['id11'] = fram.id1.apply(str)
    fram['id22'] = fram.id2.apply(str)
    fram['rank_info1'] = fram.id11.map(rank_dict).apply(convert_string_to_tuple)
    fram['last_rank1'] = fram.rank_info1.apply(lambda x: x[0])
    fram['change_rank1'] = fram.rank_info1.apply(lambda x: x[1])

    fram['rank_info2'] = fram.id22.map(rank_dict).apply(convert_string_to_tuple)
    fram['last_rank2'] = fram.rank_info2.apply(lambda x: x[0])
    fram['change_rank2'] = fram.rank_info2.apply(lambda x: x[1])
    fram['last_rank_diff'] = fram.last_rank1.apply(float) - fram.last_rank2.apply(float)
    fram['change_rank_diff'] = fram.change_rank1.apply(float) - fram.change_rank2.apply(float)
    cols_to_drop = ['rank_info1','rank_info2','last_rank1','change_rank1','last_rank2','change_rank2']
    fram.drop(cols_to_drop,axis=1,inplace=True)
    return fram
def add_season_avg(fram):
    fram['reg_vector1'] = fram.id11.map(reg_dict)
    fram['reg_vector2'] = fram.id22.map(reg_dict)
    fram['reg_vector_diff'] = fram.reg_vector1 - fram.reg_vector2

    cols_ = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']
    vector_df = pd.DataFrame(fram.reg_vector_diff.tolist(),columns=cols_)
    fram.drop(['reg_vector1','reg_vector2','reg_vector_diff'],axis=1,inplace=True)
    Xtest = pd.concat([fram,vector_df],axis=1)

    return Xtest