import pandas as pd
from opencage.geocoder import OpenCageGeocode
import geopy.distance
import re
import numpy as np
from collections import defaultdict
from statistics import mean, median

def name_merger(filepath,df_fn,mapper_fn):    
    kenpom = pd.read_csv(filepath+df_fn,sep='|')
    kenpom['Team'] = kenpom['Team'].str.replace('\d+', '').apply(lambda x: x.lower().strip())
    team_mapper = pd.read_csv(filepath+mapper_fn,encoding='iso-8859-1')
    team_mapper['TeamNameSpelling'] = team_mapper.TeamNameSpelling.str.replace('-',' ')
    merger = pd.merge(kenpom,team_mapper,how='left',left_on='Team',right_on='TeamNameSpelling')
    merger = merger[merger.Year < 2020]
    merger['zipper'] = list(zip(merger.TeamID,merger.Year))
    
    merger['win_pct'] = merger['W-L'].apply(win_pct)
    cols_to_drop = ['Rank','Team','Year','TeamID','TeamNameSpelling','Conference','W-L']
    merger = merger[[i for i in merger.columns if 'rank' not in i]].drop(cols_to_drop,axis = 1)
    cols_not_zipper = [i for i in merger.columns if 'zipper' != i]
    order_cols = ['zipper'] + cols_not_zipper
    return merger[order_cols]


def win_pct(row):
    lis = row.split('-')
    n_wins = int(lis[0])
    n_losses = int(lis[1])
    winn_pct = n_wins/(n_wins + n_losses)
    return winn_pct
def name_dict(merger,cols):
    dict_values = [i for i in merger[cols].values]
    team_dict = dict(zip(merger['zipper'],dict_values))
    return team_dict

def prep_results(filename):
    results = pd.read_csv(filepath+filename)
    results.drop(['WLoc','NumOT'],axis=1,inplace= True)
    results = results[results.Season >= 2002]
    results['Wzip'] = list(zip(results.WTeamID,results.Season))
    results['Lzip'] = list(zip(results.LTeamID,results.Season))
    results['score_diff'] = results.WScore - results.LScore
    return results

def merge_results_conf(results,conf_fn):
    conf = pd.read_csv(filepath+conf_fn)
    conf['zipper'] = list(zip(conf.TeamID,conf.Season))
    conf.drop(['TeamID','Season'],axis=1,inplace=True)
    r = pd.merge(results,conf,how='left',left_on = 'Wzip',right_on='zipper')
    r = r.rename(columns={'ConfAbbrev':'Wconf'}).drop('zipper',axis=1,inplace=True)
    r = pd.merge(r,conf,how='left',left_on = 'Lzip',right_on='zipper')
    r = r.rename(columns={'ConfAbbrev':'Lconf'})
    r.drop('zipper',axis=1,inplace=True)
    return r

def clean_seeding(seed_fn):
    seeds = pd.read_csv(filepath+seed_fn)
    seeds['zipper'] = list(zip(seeds.TeamID,seeds.Season))
    seeds.drop(['TeamID','Season'],axis=1,inplace=True)
    r = pd.merge(r,seeds,how='left',left_on='Wzip',right_on='zipper').drop('zipper',axis=1)
    r.rename(columns={'Seed':'Wseed'},inplace=True)
    r = pd.merge(r,seeds,how='left',left_on='Lzip',right_on='zipper').drop('zipper',axis=1)
    r.rename(columns={'Seed':'Lseed'},inplace=True)
    r['Wseed'] = r.Wseed.apply(lambda q: ''.join(list(filter(type(q).isdigit,q)))).apply(int)
    r['Lseed'] = r.Lseed.apply(lambda q: ''.join(list(filter(type(q).isdigit,q)))).apply(int)
    return r
def seed_relative_conversion(r):

    ## LOGIC GAMES
    r['better_team'] = ''
    r['worse_team'] = ''
    r.loc[r.Wseed<r.Lseed,'better_team'] = r.Wzip
    r.loc[r.Wseed>r.Lseed,'better_team'] = r.Lzip
    r.loc[r.Wseed==r.Lseed,'better_team'] = r.Wzip
    r.loc[r.better_team == r.Wzip,'worse_team'] = r.Lzip
    r.loc[r.better_team == r.Lzip,'worse_team'] = r.Wzip
    r.loc[r.Wzip != r.better_team,'score_diff'] = r.score_diff.apply(lambda x: (x)*-1)

    r['Hseed_conf'] = ''
    r['Lseed_conf'] = ''
    r.loc[r.better_team == r.Wzip,'Hseed_conf'] = r.Wconf
    r.loc[r.better_team == r.Lzip,'Hseed_conf'] = r.Lconf
    r.loc[r.Hseed_conf == r.Wconf,'Lseed_conf'] = r.Lconf
    r.loc[r.Hseed_conf == r.Lconf,'Lseed_conf'] = r.Wconf

    r['higher_seed'] = ''
    r['worse_seed'] = ''
    r.loc[r.better_team == r.Wzip,'higher_seed'] = r.Wseed
    r.loc[r.better_team == r.Lzip,'higher_seed'] = r.Lseed
    r.loc[r.higher_seed == r.Wseed,'worse_seed'] = r.Lseed
    r.loc[r.higher_seed == r.Lseed,'worse_seed'] = r.Wseed
    return r

def prep_seeding_df(r):
    cols_to_drop = ['WScore','LScore','Wzip','Lzip','Wconf','Lconf','Wseed','Lseed']
    col_order = ['Season','WTeamID','LTeamID','DayNum','better_team','worse_team','Hseed_conf','Lseed_conf',
                 'higher_seed','worse_seed','score_diff']
    r = r.drop(cols_to_drop,axis=1)[col_order]
    r.columns = ['year','WTeamID','LTeamID','daynum','Hseed_zipper','Lseed_zipper','Hseed_conf','Lseed_conf',
                'Hseed_value','Lseed_value','score_diff']
    r['seed_diff'] = r.Hseed_value - r.Lseed_value
    r['Hseed_won'] = ''
    r.loc[r.score_diff > 0,'Hseed_won'] = 1
    r.loc[r.score_diff < 0,'Hseed_won'] = 0
    r.drop(['Hseed_value','Lseed_value'],axis=1,inplace=True)
    return r

def subtract_team_vects(r):
    l1 = r.Hseed_zipper.tolist()
    l2 = r.Lseed_zipper.tolist()
    vector_diff = [team_dict[l1[i]] - team_dict[l2[i]] for i in range(len(l1))]
    df_vects = pd.DataFrame(vector_diff,columns = [i for i in merger.columns if 'zipper' != i])
    full_df = pd.concat([r,df_vects],axis=1)
    return full_df

def map_team_names(mapper_fn):
    team_mapper = pd.read_csv(filepath+mapper_fn,encoding='iso-8859-1')
    team_mapper['TeamNameSpelling'] = team_mapper.TeamNameSpelling.str.replace('-',' ')
    team_dict = dict(zip(team_mapper.TeamID,team_mapper.TeamNameSpelling))
    td = dict(zip(team_mapper.TeamNameSpelling,team_mapper.TeamID))
    f['Hteam_name'] = f.Hseed_zipper.apply(lambda x: x[0]).map(team_dict)
    f['Lteam_name'] = f.Lseed_zipper.apply(lambda x: x[0]).map(team_dict)
    f = f[[i for i in f.columns if 'team_' in i] + [i for i in f.columns if 'team_' not in i]]
    return f

def matchup_city_dict(city_fn):
    gc = pd.read_csv(filepath+city_fn)
    gc['team1'] = gc.team1.apply(lambda x: x.lower().strip())
    gc['team2'] = gc.team2.apply(lambda x: x.lower().strip())
    gc['TeamID1'] = gc['team1'].map(td)
    gc['TeamID2'] = gc['team2'].map(td)
    gc['mini_z'] = list(zip(gc.TeamID1,gc.TeamID2))
    gc['mini_z'] = gc.mini_z.apply(lambda x: sorted(x))
    gc['zipper'] = list(zip(gc.year,gc.mini_z))
    gc['zipper'] = gc.zipper.apply(str)
    return gc

def consol_game_city(result_fn):
    game_log = pd.read_csv(filepath+result_fn)
    game_log = game_log[(game_log.Season >= 2002)&(game_log.Season < 2010)]
    game_log['mini_z'] = list(zip(game_log.WTeamID,game_log.LTeamID))
    game_log['mini_z'] = game_log.mini_z.apply(lambda x: sorted(x))
    game_log['zipper'] = list(zip(game_log.Season,game_log.mini_z))
    game_log['zipper'] = game_log.zipper.apply(str)
    Lmerger = pd.merge(game_log,gc,how = 'left',on='zipper')
    drop_cols = ['mini_z_x','zipper','year','team1','team2','TeamID1','TeamID2','mini_z_y','WLoc',
                'WScore','LScore','NumOT']
    Lmerger.drop(drop_cols,axis=1,inplace=True)
    return Lmerger

def formatting_city_info(city_fn,game_fn):
    city = pd.read_csv(filepath+city_fn)
    city['Loc'] = city.City+', '+city.State
    city_dict = dict(zip(city.CityID,city.Loc))
    ncdf = pd.read_csv(filepath+game_fn)
    ncdf = ncdf[ncdf.CRType == 'NCAA'].reset_index(drop=True)
    ncdf['location'] = ncdf.CityID.map(city_dict)
    ncdf.drop(['CRType','CityID'],axis=1,inplace=True)
    locations = pd.concat([Lmerger,ncdf])
    locations['zipper'] = list(zip(locations.Season,locations.DayNum,locations.WTeamID,locations.LTeamID))
    locations.drop(['Season','DayNum','WTeamID','LTeamID'],axis=1,inplace=True)
    f['HSeed_teamID'] = f.Hseed_zipper.apply(lambda x: x[0])
    f['LSeed_teamID'] = f.Lseed_zipper.apply(lambda x: x[0])
    f.drop(['Hseed_zipper','Lseed_zipper'],axis=1,inplace=True)
    f['zipper'] = list(zip(f.year,f.daynum,f.WTeamID,f.LTeamID))
    return f,locations

def school_location_dict(school_city_fn):
    schoolL = pd.read_csv(filepath+school_city_fn,encoding='iso-8859-1')
    locU_dict = dict(zip(schoolL.TeamID,schoolL.SchoolLocation))
    return locU_dict

def isolate_gps_info(result_fn):
    result = pd.merge(f,**formatting_city_info(city_fn,game_fn),how='left',on='zipper')
    result.drop(['Hteam_name','Lteam_name','zipper','WTeamID','LTeamID'],axis=1,inplace=True)
    result.rename(columns={'location':'game_location'},inplace=True)
    result['Hseed_location'] = result.HSeed_teamID.map(locU_dict).apply(lambda x: x.lower())
    result['Lseed_location'] = result.LSeed_teamID.map(locU_dict).apply(lambda x: x.lower())
    gps = result[['game_location','Hseed_location','Lseed_location']]
    result.drop([i for i in result.columns if 'location' in i],axis=1,inplace=True)
    gps['game_location'] = gps.game_location.apply(lambda x: x.lower().strip())
    gps['Hseed_location'] = gps.Hseed_location.apply(lambda x: x.lower().strip())
    gps['Lseed_location'] = gps.Lseed_location.apply(lambda x: x.lower().strip())
    return gps

def find_coordinates(city_name):    
    key = '85f15821952a4e2d9aa204ba3336b3d2'
    geocoder = OpenCageGeocode(key)
    results = geocoder.geocode(city_name)
    coordinates = (results[0]['geometry']['lat'], results[0]['geometry']['lng'])
    return coordinates

def fetch_lat_long(gps):
    print('getting lat/long coordinates for each game venue...')
    gl = list(set(gps.game_location.tolist()))
    gl_dict = {i:find_coordinates(i) for i in gl}
    gps['game_coords'] = gps.game_location.map(gl_dict)

    print('getting lat/long coordinates for each school....')
    HsL = list(set(gps.Hseed_location.tolist()))
    Ls_dict = {i:find_coordinates(i) for i in HsL}
    gps['Hs_coords'] = gps.Hseed_location.map(Ls_dict)

    LsL = list(set(gps.Lseed_location.tolist()))
    gl_dict = {i:find_coordinates(i) for i in LsL}
    gps['Ls_coords'] = gps.Lseed_location.map(gl_dict)
    return gps

def format_gps(gps):    
    gps = pd.read_csv(filepath+'loc_coords_df.psv',sep='|')

    game_coords = gps.game_coords.tolist()
    Hs_coords = gps.Hs_coords.tolist()
    Ls_coords = gps.Ls_coords.tolist()

    dist_for_Hs=[]
    for i in range(len(game_coords)):
        h = tuple([float(i.strip()) for i in Hs_coords[i].replace('(','').replace(')','').split(',')])
        g = tuple([float(i.strip()) for i in game_coords[i].replace('(','').replace(')','').split(',')])
        dist_for_Hs.append(geopy.distance.distance(h,g).miles)

    dist_for_Ls=[]
    for i in range(len(game_coords)):
        h = tuple([float(i.strip()) for i in Ls_coords[i].replace('(','').replace(')','').split(',')])
        g = tuple([float(i.strip()) for i in game_coords[i].replace('(','').replace(')','').split(',')])
        dist_for_Ls.append(geopy.distance.distance(h,g).miles)
    
    gps['dist_for_Hs'] = dist_for_Hs
    gps['dist_for_Ls'] = dist_for_Ls
    return gps

def finalize_location(gps,result):
    cols_to_drop = ['game_location','Hseed_location','Lseed_location','game_coords','Hs_coords','Ls_coords']
    framer = pd.concat([result,gps],axis=1).drop(cols_to_drop,axis= 1)
    framer['dist_diff'] = framer.dist_for_Hs - framer.dist_for_Ls
    framer.drop(['dist_for_Hs','dist_for_Ls'],axis=1,inplace=True)

    ordr =['year']+[i for i in framer.columns if 'teamID' in i]+[i for i in framer.columns if 'year' != i if 'teamID' not in i]
    framer = framer[ordr]
    return framer

def team_championships(champ_fn):
    champ = pd.read_csv(filepath+champ_fn)
    champ.columns = ['year','champion','runnerup']
    champ['runnerup'] = champ.runnerup.apply(lambda x: x.lower().strip())
    champ['champion'] = champ.champion.str.split('(').apply(lambda x: x[0].strip().lower())

    years = champ.year.tolist() * 2
    teams = champ.champion.tolist() + champ.runnerup.tolist()
    c = pd.DataFrame()
    c['year'] = years
    c['team'] = teams
    c['value'] = [1]*81 + [.5]*81
    c['teamID'] = c.team.map(td)
    c = c[~c.teamID.isnull()]
    c['teamID'] = c.teamID.astype('int')
    return c

def finalize_championships(framer,c):
    total_dict_chmps = {}
    for i in range(2002,2020):
        m = c[(c.year < i)&(c.year > i-10)].groupby(['teamID'])['value'].sum().reset_index()
        keys_ = [str((i, teamer)) for teamer in m.teamID.tolist()]
        values_ = m.value.tolist()
        total_dict_chmps = dict(total_dict_chmps, **dict(zip(keys_,values_)))

    framer['Hzip'] = list(zip(framer.year,framer.HSeed_teamID))
    framer['Lzip'] = list(zip(framer.year,framer.LSeed_teamID))
    framer['Hzip'] = framer.Hzip.apply(str)
    framer['Lzip'] = framer.Lzip.apply(str)
    framer['HS_championships'] = framer.Hzip.map(total_dict_chmps)
    framer.loc[framer.HS_championships.isnull(),'HS_championships'] = 0
    framer['LS_championships'] = framer.Lzip.map(total_dict_chmps)
    framer.loc[framer.LS_championships.isnull(),'LS_championships'] = 0
    framer['diff_chmps_L10years'] = framer.HS_championships-framer.LS_championships
    framer.drop(['HS_championships','LS_championships'],axis=1,inplace=True)
    return framer

def past_tournament_wins(framer,wins_fn):
    tresults = pd.read_csv(filepath+wins_fn)
    teams = tresults[['Season','WTeamID']]

    total_dict_wins = {}
    for i in range(2002,2020):
        m = teams[(teams.Season < i)&(teams.Season > i-6)].groupby(['WTeamID']).size().reset_index()
        m.columns = ['WTeamID','value']
        keys_ = [str((i, teamer)) for teamer in m.WTeamID.tolist()]
        values_ = m['value'].tolist()
        total_dict_wins = dict(total_dict_wins, **dict(zip(keys_,values_)))

    framer['HS_wins_L10'] = framer.Hzip.map(total_dict_wins)
    framer['LS_wins_L10'] = framer.Lzip.map(total_dict_wins)
    framer.loc[framer['HS_wins_L10'].isnull(),'HS_wins_L10'] = 0
    framer.loc[framer['LS_wins_L10'].isnull(),'LS_wins_L10'] = 0
    framer['diff_wins_L10years'] = framer.HS_wins_L10 - framer.LS_wins_L10
    framer.drop(['HS_wins_L10','LS_wins_L10'],axis=1,inplace=True)
    framer['daynum'] = framer.daynum.apply(lambda x: x-framer.daynum.min())
    return framer

def rank_conferences(framer,conf_fn):
    conf = pd.read_csv(filepath+'MTeamConferences.csv')
    conf = conf[['TeamID','ConfAbbrev']]
    conf_dict = dict(zip(conf.TeamID,conf.ConfAbbrev))

    tt = tresults[['Season','WTeamID','WScore']]
    tt['conf'] = tt.WTeamID.map(conf_dict)

    total_conf_dict = {}
    for i in range(2002,2020):
        m = tt[(tt.Season < i)&(tt.Season > i-4)].groupby(['conf','WTeamID']).size().reset_index()
        m = m.groupby('conf')[0].mean().reset_index()
        m.columns = ['Conf','value']
        m = m.sort_values('value',ascending = False)
        keys_ = [str((i, confer)) for confer in m.Conf.tolist()]
        values_ = m['value'].tolist()
        total_conf_dict = dict(total_conf_dict, **dict(zip(keys_,values_)))

    framer['HSconf_zipper'] = list(zip(framer.year,framer.Hseed_conf))
    framer['LSconf_zipper'] = list(zip(framer.year,framer.Lseed_conf))
    framer['HSconf_zipper'] = framer.HSconf_zipper.apply(str)
    framer['LSconf_zipper'] = framer.LSconf_zipper.apply(str)
    framer['HSconf_value'] = framer.HSconf_zipper.map(total_conf_dict).fillna(0)
    framer['LSconf_value'] = framer.LSconf_zipper.map(total_conf_dict).fillna(0)
    framer.drop(['Hseed_conf','Lseed_conf','HSconf_zipper','LSconf_zipper'],axis=1,inplace=True)
    framer.drop(['Hzip','Lzip'],axis=1,inplace=True)
    framer['diff_conf_value'] = framer.HSconf_value - framer.LSconf_value
    framer.drop(['HSconf_value','LSconf_value'],axis=1,inplace=True)
    return framer

def school_location_dict(framer):
    hs_dist_dict = dict(zip(framer.HSeed_teamID,framer.Hs_coords))
    ls_dist_dict = dict(zip(framer.LSeed_teamID.apply(str),framer.Ls_coords.apply(str)))
    coords_dict = dict(hs_dist_dict,**ls_dist_dict)
    coors_df = pd.DataFrame(list(coords_dict.items()),columns=['TeamID','coords'])
    coors_df.to_csv('team_coords_dict.psv',index=False,sep='|')

    framer['zipper'] = list(zip(framer.year,framer.HSeed_teamID,framer.LSeed_teamID))
    game_loc_dict = dict(zip(framer.zipper,framer.game_coords))
    game_loc = pd.DataFrame(list(game_loc_dict.items()),columns=['game_zip','game_coors'])
    return game_loc

def conf_winner(framer,conf_winn_fn):
    conf_games = pd.read_csv(filepath+conf_winn_fn)
    conf_games.drop(['DayNum','LTeamID'],axis=1,inplace=True)
    conf_games.drop_duplicates(subset=['Season','ConfAbbrev'],keep='last',inplace=True)
    conf_games['z'] = list(zip(conf_games.WTeamID,conf_games.Season))
    conf_games_dict = dict(zip(conf_games.z,[1]*conf_games.shape[0]))
    framer['hs_zip'] = list(zip(framer.HSeed_teamID,framer.year))
    framer['ls_zip'] = list(zip(framer.LSeed_teamID,framer.year))
    framer['HSconf_win'] = framer.hs_zip.map(conf_games_dict).fillna(0)
    framer['LSconf_win'] = framer.ls_zip.map(conf_games_dict).fillna(0)
    framer['conf_win_diff'] = framer.HSconf_win - framer.LSconf_win
    framer.drop(['HSconf_win','LSconf_win'],axis=1,inplace=True)
    framer = framer[framer.year > 2002].reset_index(drop=True)
    return conf_winner

def historial_ranking(hrank_fn):
    
    massey = pd.read_csv(filepath+hrank_fn)
    m = massey.groupby(['TeamID','Season','RankingDayNum'])['OrdinalRank'].mean().reset_index()
    m['zipper'] = list(zip(m.TeamID,m.Season))
    m.drop(['TeamID','Season'],axis=1,inplace = True)

    rank_dict={}
    uniq_zip = list(m.zipper.unique())
    for team in uniq_zip:
        subber= m[m.zipper == team]
        last_rank = subber['OrdinalRank'].values[-1]
        median_day = subber.RankingDayNum.tolist()[int(subber.shape[0]/2)]
        mid_point=subber[subber.RankingDayNum == median_day]['OrdinalRank'].tolist()[0]
        rank_dict[str(team)] = (last_rank,last_rank-mid_point)

    ranker = pd.DataFrame(rank_dict.items(),columns = ['zipper','lastr_changer'])
    return ranker

def incorporate_rankings(framer,ranker):
    rank_dict = dict(zip(ranker.zipper,ranker.lastr_changer))
    framer['hs_zip2'] = framer.hs_zip.apply(str)
    framer['ls_zip2'] = framer.ls_zip.apply(str)
    framer['hs_rank_info'] = framer.hs_zip2.map(rank_dict)
    framer['ls_rank_info'] = framer.ls_zip2.map(rank_dict)
    framer['hs_rank_info'] = framer.hs_rank_info.apply(lambda x:tuple(str(x).replace(')','').replace('(','').split(',')))
    framer['ls_rank_info'] = framer.ls_rank_info.apply(lambda x:tuple(str(x).replace(')','').replace('(','').split(',')))

    framer['hs_last_rank'] = framer.hs_rank_info.apply(lambda x: x[0])
    framer['ls_last_rank'] = framer.ls_rank_info.apply(lambda x: x[0])
    framer['hs_change_rank'] = framer.hs_rank_info.apply(lambda x: x[1])
    framer['ls_change_rank'] = framer.ls_rank_info.apply(lambda x: x[1])

    framer['last_rank_diff'] = framer.hs_last_rank.apply(lambda x: float(x)) - framer.ls_last_rank.apply(lambda x: float(x))
    framer['change_rank_diff'] = framer.hs_change_rank.apply(lambda x: float(x)) - framer.ls_change_rank.apply(lambda x: float(x))
    framer.drop(['hs_last_rank','ls_last_rank','hs_change_rank','ls_change_rank',
                          'hs_rank_info','ls_rank_info'],axis=1,inplace=True)
    return framer

def incorporate_season_averages(framer,avg_fn):
    cols_ = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']
    Wcols_ = ['W'+i for i in cols_]
    Lcols_ = ['L'+i for i in cols_]

    reg = pd.read_csv(filepath+avg_fn)
    reg['wzip'] = list(zip(reg.WTeamID,reg.Season))
    reg['wzip'] = reg.wzip.apply(str)
    reg['lzip'] = list(zip(reg.LTeamID,reg.Season))
    reg['lzip'] = reg.lzip.apply(str)

    wdf = reg.groupby('wzip')[Wcols_].mean().reset_index()
    Wdict = dict(zip(wdf.wzip,wdf[Wcols_].values))
    ldf = reg.groupby('lzip')[Lcols_].mean().reset_index()
    Ldict = dict(zip(ldf.lzip,ldf[Lcols_].values))

    reg_dict = dict(Wdict,**Ldict)

    framer['Wreg_vector'] = framer.hs_zip2.map(reg_dict)
    framer['Lreg_vector'] = framer.ls_zip2.map(reg_dict)
    framer['reg_vector_diff'] = framer.Wreg_vector - framer.Lreg_vector
    vect_df = pd.DataFrame(framer.reg_vector_diff.tolist(),columns=cols_)
    framer.drop(['Wreg_vector','Lreg_vector','reg_vector_diff'],axis=1,inplace=True)
    Xtrain = pd.concat([framer,vect_df],axis=1)
    return Xtrain