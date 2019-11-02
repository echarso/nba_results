import pandas as pd
import numpy as np
import datetime
import pandas as pd
import numpy as np
import datetime
mydate = datetime.datetime.now()
mydate.strftime("%B")

'''
 create a salarycap  csv
 create date objects in python done
 create a column games in a row : games played with 1 or 2 days in between ?
    a game comes with a new date we should compare the current day with the last day played
    if the same then push up the counter
    if the team is away we should step up the away_games_counter for the next game
    if the team is in we should stet up the in_games_counter


 create a column how many days away from home :
 create a column how many days playing  home :
 create a column relax sequence ?
 create a column relax sequence ?


'''


columns = [
            'date','h_team','h_wins','h_gm_pnt','h_def_avrg','h_off_avrg','h_slr','h_shape',
           'v_team','v_wins','v_gm_pnt','v_def_avrg','v_off_avrg','v_slr','v_shape',
           'difference','total_points']
df = pd.DataFrame(columns=columns )
h_wins={}
h_offence_avrg={}
h_defence_avrg={}
v_wins={}
v_offence_avrg={}
v_defence_avrg={}


TOTAL_THREASHOLD = 240
differenceBetweenTwoTeams=pd.DataFrame()


def getDateGravity(x ):
    switcher = {
        "Jan":2,
        "Feb":3,
        "Mar":3,
        "Apr":4,
        "May":4,
        "Jun":4,
        "Jul":4,
        "Aug":0,
        "Sep":0,
        "Oct":1,
        "Nov":1,
        "Dec":2
    }
    x = x.lstrip()
    return switcher[x]



def getValue(name ,array):
    if name in array:
        return array[name]
    else:
        return 0

def numbersForTeamWhenHosting(team,offence,defence):
    if team in h_offence_avrg:
        if ( offence > defence):
            h_wins[team]=h_wins[team]+1
        h_offence_avrg[team]=(int(h_offence_avrg[team])/2)+(offence/2)
        h_defence_avrg[team]=(int(h_defence_avrg[team])/2)+(defence/2)
    else:
        if ( offence > defence):
            h_wins[team]=1
        else:
            h_wins[team]=0
        h_offence_avrg[team]=offence
        h_defence_avrg[team]=defence



def numbersForTeamWhenVisiting(team,offence,defence):
    if team in v_wins:
        if ( offence > defence):
             v_wins[team]=v_wins[team]+1
        v_offence_avrg[team]=(int(v_offence_avrg[team])/2)+(offence/2)
        v_defence_avrg[team]=(int(v_defence_avrg[team])/2)+(defence/2)
    else:
        if ( offence > defence):
            v_wins[team]=1
        else:
            v_wins[team]=0
        v_offence_avrg[team]=offence
        v_defence_avrg[team]=defence


def my_function(array,team,points):
    if team in array:
      array[team]=int(array[team])+int(points)
    else:
      array[team]=int(points)
    return array


def visiting_flag(name):
    teams_df.at[name,'lst_gm_out']=1
    teams_df.at[name,'lst_gm_in']=0


def hosting_flag(name):
    teams_df.at[name,'lst_gm_in']=1
    teams_df.at[name,'lst_gm_out']=0

def team_time_calc(name,current_day):
    days_bfr_lst_gm = teams_df.at[name,'prv_gm_date']
    if days_bfr_lst_gm != 0:
        prev_day = teams_df.at[name,'prv_gm_date']
        delta =  current_day - prev_day
        delta = delta.days
        #print '------> delta ', delta
        teams_df.at[name,'days_bfr_lst_gm'] =  delta
        if delta < 3 :
            teams_df.at[name,'gms_in_row'] = teams_df.at[name,'gms_in_row']+1
        else:
            ## be carefull with that
            ## at this point you are missing the logic
            teams_df.at[name,'gms_in_row'] =0;

# fixing the team dataframe
teams_df = pd.read_csv("salaries.csv")
teams_df.index=teams_df['Team'].astype(str).str[:3]
teams_df['counter_hosting_games']=0
teams_df['counter_visiting_games']=0
teams_df['days_bfr_lst_gm']=0
teams_df['prv_gm_date']=0
teams_df['gms_in_row']=0
teams_df['lst_gm_in']=0
teams_df['lst_gm_out']=0


with open("nba_games20.csv", "r") as ins:
    array = {}
    i=0
    h_before_game_wins = 0
    h_before_game_offence_avrg =0
    h_before_game_defence_avrg =0
    v_before_game_wins= 0

    for line in ins:

        #splitted = line.split('pm')[1]
        print line
        splitted = line.split(',')

        #array_visitors = my_function(array,splitted[1],splitted[2])
        #array_hosts = my_function(array,splitted[3],splitted[4])

        host = splitted[3];
        h_points = splitted[4]

        visitor = splitted[1];
        v_points = splitted[2];

        df.at[i,'date']=splitted[0][4:15]
        if len(df.at[i,'date']) == 10:
           df.at[i,'date'] =df.at[i,'date'][:4]+'0'+df.at[i,'date'][4:]
        df.at[i,'datetime']= datetime.datetime.strptime(df.at[i,'date'].strip(), "%b %d %Y").date()
        current_day = df.at[i,'datetime']

        df.at[i,'h_team']=host
        df.at[i,'v_team']=visitor
        host = host[:3]
        visitor=visitor[:3]
        h_before_game_wins_in=0
        v_before_game_wins_in=0
        h_before_game_wins_out=0
        v_before_game_wins_out=0
        if host in h_wins:
            h_before_game_wins_in = h_wins[host]
        if visitor in h_wins:
            v_before_game_wins_in = h_wins[visitor]
        if host in v_wins:
            h_before_game_wins_out = v_wins[host]
        if visitor in v_wins:
            v_before_games_wins_out= v_wins[visitor]

        df.at[i,'h_win_when_in'] = h_before_game_wins_in
        df.at[i,'h_win_when_out'] = h_before_game_wins_out

        df.at[i,'h_off_avrg'] = getValue(host, h_offence_avrg)
        df.at[i,'h_def_avrg'] = getValue(host,h_defence_avrg)

        df.at[i,'v_win_when_in'] = v_before_game_wins_in
        df.at[i,'v_win_when_out'] = v_before_game_wins_out
        df.at[i,'v_off_avrg'] = getValue(visitor, v_offence_avrg)
        df.at[i,'v_def_avrg'] = getValue(visitor,v_defence_avrg)


        df.at[i,'in_win_diff']=h_before_game_wins_in - v_before_game_wins_in
        df.at[i,'out_win_diff']=h_before_game_wins_out - v_before_game_wins_out

        host = host[:3]
        visitor=visitor[:3]

        team_time_calc(host,current_day)
        team_time_calc(visitor,current_day)


        teams_df.at[host,'prv_gm_date'] = current_day
        teams_df.at[visitor,'prv_gm_date'] = current_day


        df.at[i,'h_sal2018']        =teams_df.at[host,'sal2018']
        df.at[i,'h_sal2019']        =teams_df.at[host,'sal2019']
        df.at[i,'h_sal2020']        =teams_df.at[host,'sal2020']
        df.at[i,'h_sal2021']        =teams_df.at[host,'sal2021']
        df.at[i,'h_sal2022']        =teams_df.at[host,'sal2022']
        df.at[i,'days_bfr_lst_gm']    =teams_df.at[host,'days_bfr_lst_gm']
        df.at[i,'h_gms_in_row'] =teams_df.at[host,'gms_in_row']
        df.at[i,'h_lst_gm_out']        =teams_df.at[host,'lst_gm_out']
        df.at[i,'h_lst_gm_in']        =teams_df.at[host,'lst_gm_in']
        df.at[i,'h_d_lst_gm']      =teams_df.at[visitor,'days_bfr_lst_gm']



        df.at[i,'v_sal2018']        =teams_df.at[visitor,'sal2018']
        df.at[i,'v_sal2019']        =teams_df.at[visitor,'sal2019']
        df.at[i,'v_sal2020']        =teams_df.at[visitor,'sal2020']
        df.at[i,'v_sal2021']        =teams_df.at[visitor,'sal2021']
        df.at[i,'v_sal2022']        =teams_df.at[visitor,'sal2022']
        df.at[i,'v_sal2022']        =teams_df.at[visitor,'sal2022']
        df.at[i,'v_gms_in_row']     =teams_df.at[visitor,'gms_in_row']
        df.at[i,'v_lst_gm_out']     =teams_df.at[visitor,'lst_gm_out']
        df.at[i,'v_lst_gm_in']      =teams_df.at[visitor,'lst_gm_in']
        df.at[i,'v_d_lst_gm']      =teams_df.at[visitor,'days_bfr_lst_gm']

        df.at[i,'sal_diff'] = df.at[i,'h_sal2018']-df.at[i,'v_sal2018']
        df.at[i,'points_scored_average']=df.at[i,'h_off_avrg']+df.at[i,'v_off_avrg']
        df.at[i,'defence_scored_average']=df.at[i,'h_def_avrg']+df.at[i,'v_def_avrg']

        df.at[i,'h_t_score'] =h_points
        df.at[i,'v_t_score'] =v_points
        df.at[i,'difference']=int(splitted[4])-int(splitted[2])
        df.at[i,'total_points']=int(splitted[4])+int(splitted[2])
        if df.at[i,'total_points'] > TOTAL_THREASHOLD:
            df.at[i,'target_flag']=1
        else:
            df.at[i,'target_flag']=0

        numbersForTeamWhenHosting(host,int(splitted[4]),int(splitted[2]))
        numbersForTeamWhenVisiting(visitor,int(splitted[2]),int(splitted[4]))
        hosting_flag(host)
        visiting_flag(visitor)
        i=i+1
#print df[['h_team','v_team','h_win','v_win']]
teamsDataFrame = pd.DataFrame({ "h_wins":h_wins,
                                "h_offence_avrg":h_offence_avrg,
                                "h_defence_avrg":h_defence_avrg,
                                "v_wins":v_wins,
                                "v_offence_avrg":v_offence_avrg})

df['h_team']=df['h_team'].astype(str).str[:3]
df['v_team']=df['v_team'].astype(str).str[:3]
df['d_name'] = df['date'].astype(str).str[:3]
df['d_gravity'] =df['d_name'].apply(getDateGravity);

df =  df[[ 'defence_scored_average',
            'points_scored_average',
            'sal_diff',
          'in_win_diff',
          'out_win_diff',
          'difference',
          'total_points',
          'target_flag']]
#df =df.drop(['h_team', 'v_team','date'], axis=1)

df.to_csv('my_game_records.csv')
