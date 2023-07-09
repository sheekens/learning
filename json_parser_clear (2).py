import json
from datetime import datetime

path_to_json = 'C:/Users/Admin/Downloads/Telegram Desktop/Stats_version_1_seas_20_21.json'
# path_to_json = '/Users/dkruglov/Downloads/Stats_version_1_seas_20_21.json'
path_to_json = 'D:/testing/Stats_version_1_seas_20_21.json'
with open(path_to_json) as file:
    json_data = json.load(file)
print()

####### 1. Написать функцию, которая достает из jsona все встречающиеся команды в виде словаря (k - id команды, v - название)
def team_list_dict_from_json(json_data):
    '''huinya'''
    team_dict = {}
    for match in json_data['matches']: 
        for key, value in match['homeTeam'].items():
            if value not in team_dict.values():
                team_dict[match['homeTeam']['id']] = match['homeTeam']['name']
    return team_dict

team_dict = team_list_dict_from_json(json_data)
# print(team_dict)


####### 2. Написать функцию, которая принимает дату и возвращает
# пары команд, которые играют в этот день

####### Форматировать вводимую дату и дату из json!!!!!!
# @gerzog:: сделай, чтобы функция умела принимать не дату_время, а только дату (match_date = '2020-09-12')
def teams_on_matchday_from_json(match_date2search, json_data):
    match_date_datetime = datetime.strptime(match_date, '%Y-%m-%d')
    # print(match_date_datetime)
    teams_on_one_matchday = {}
    for match in json_data['matches']:
        # if match['utcDate'][0:10] == match_date[0:10]:
        # match_datetime = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ').date()
        match_datetime_str = match['utcDate']
        match_datetime = datetime.strptime(match_datetime_str, '%Y-%m-%dT%H:%M:%SZ')
        print(match_datetime)
        if match_date_datetime.date() == datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ').date():
            teams_on_one_matchday[match['homeTeam']['name']] = match['awayTeam']['name']
            # teams_on_one_matchday = [match_date[0:10], 'igraet', match['homeTeam']['name'], 'protiv', match['awayTeam']['name']]
            # print(match_date[0:10], 'igraet', match['homeTeam']['name'], 'protiv', match['awayTeam']['name'])
    return teams_on_one_matchday

match_date = '2020-09-12'
teams_on_one_matchday = teams_on_matchday_from_json(match_date)
print(match_date, 'igrayut', teams_on_one_matchday)
exit()

###### 3. Написать функцию, которая принимает название команды и возвращает,
# в каком количестве матчей участвовала команда
# (сколько из них побед/поражений/ничьих)

def score_by_team_name(team_name):
    score = {'total': 0, 'win' : 0, 'lose' : 0, 'draw' : 0}
    for match in json_data['matches']:
        if match['awayTeam']['name'] == team_name:
            score['total'] += 1
            if match['score']['winner'] == 'AWAY_TEAM':
                score['win'] += 1
            elif match['score']['winner'] == 'HOME_TEAM':
                score['lose'] += 1
            else:
                score['draw'] += 1
        if match['homeTeam']['name'] == team_name:
            score['total'] += 1
            if match['score']['winner'] == 'AWAY_TEAM':
                score['lose'] += 1
            elif match['score']['winner'] == 'HOME_TEAM':
                score['win'] += 1
            else:
                score['draw'] += 1
    return score

team_name = 'Arsenal FC'
# team_name = 'Fulham FC'
score = score_by_team_name(team_name)
# print(team_name, score)


####### 4. Написать функцию, которая принимает название команды и возвращает,
# сколько голов команда забила за все матчи, 
# сколько в среднем команда забивает за один матч,
# за один матч в первом тайме/во втором/в доп. время/пентальти

def detailed_score_by_team_name(team_name):
    team_score = {
        'totalGoals' : 0,
        'avgFulltime' : 0,
        'avg1stTime' : 0,
        'avg2ndTime' : 0,
        'avgExtraTime' : 0,
        'avgPenalties' : 0
    }
    for match in json_data['matches']:
        if match['homeTeam']['name'] == team_name:
            team_score['totalGoals'] += match['score']['fullTime']['homeTeam']
            team_score['avg1stTime'] += match['score']['halfTime']['homeTeam']
        if match['awayTeam']['name'] == team_name:
            team_score['totalGoals'] += match['score']['fullTime']['awayTeam']
            team_score['avg1stTime'] += match['score']['halfTime']['awayTeam']
        if match['score']['extraTime']['homeTeam'] is not None:
            team_score['avgExtraTime'] += match['score']['extraTime']['homeTeam']
        if match['score']['extraTime']['awayTeam'] is not None:
            team_score['avgExtraTime'] += match['score']['extraTime']['awayTeam']
        if match['score']['penalties']['homeTeam'] is not None:
            team_score['avgPenalties'] += match['score']['penalties']['homeTeam']
        if match['score']['penalties']['awayTeam'] is not None:
            team_score['avgPenalties'] += match['score']['penalties']['awayTeam']
    team_score['avg2ndTime'] = (team_score['totalGoals'] - team_score['avg1stTime'] - team_score['avgExtraTime'] - team_score['avgPenalties']) / score['total']
    avgFulltime = team_score['totalGoals'] / score['total']
    avg1stTime = team_score['avg1stTime'] / score['total']
    avgExtraTime = team_score['avgExtraTime'] / score['total']
    avgPenalties = team_score['avgPenalties'] / score['total']
    team_score['avgFulltime'] = avgFulltime
    team_score['avg1stTime'] = avg1stTime
    team_score['avgExtraTime'] = avgExtraTime
    team_score['avgPenalties'] = avgPenalties
    return team_score

# for team_name in team_dict.values():
#     team_score = detailed_score_by_team_name(team_name)
#     print(team_name, 'detailed score:', team_score)
#     print()


###### Добавить в функции 3, 4 дополнительную переменную, с помощью которой можно будет
# считать необходимые метрики во всех матчах/в тех, в которых команда играла в гостях/дома.

# @gerzog: тут я имел ввиду в уже имеюющуюся функцию добавить дополнительный атрибут (типо режим работы home/away/total) и в зависимости от 
# значения этого дополнительного атрибута, будет возвращаться соответствующий словарик.

# разбить на несколько словарей??????????

# 3
def score_by_team_name_home_away(team_name):
    score_home_away = {
        'home' : {
            'total': 0,
            'win' : 0,
            'lose' : 0,
            'draw' : 0
            },
        'away' : {
            'total': 0,
            'win' : 0,
            'lose' : 0,
            'draw' : 0
            }
        }
    for match in json_data['matches']:
        if match['awayTeam']['name'] == team_name:
            score_home_away['away']['total'] += 1
            if match['score']['winner'] == 'AWAY_TEAM':
                score_home_away['away']['win'] += 1
            elif match['score']['winner'] == 'HOME_TEAM':
                score_home_away['away']['lose'] += 1
            else:
                score_home_away['away']['draw'] += 1
        if match['homeTeam']['name'] == team_name:
            score_home_away['home']['total'] += 1
            if match['score']['winner'] == 'AWAY_TEAM':
                score_home_away['home']['lose'] += 1
            elif match['score']['winner'] == 'HOME_TEAM':
                score_home_away['home']['win'] += 1
            else:
                score_home_away['home']['draw'] += 1
    return score_home_away

team_name = 'Arsenal FC'
# team_name = 'Fulham FC'
score_home_away = score_by_team_name_home_away(team_name)
# print(team_name, score_home_away)


# №4
def super_detailed_score_by_team_name(team_name):
    # @gerzog: следующая строка очень длинная -> запиши ее в таком формате:
    # dict_name = {
    #     'key1':{
    #         'k1': v1,
    #         'k2': v2
    #     }
    #     'key2':{
    #        'k1': v1,
    #        'k2': v2
    #     }
    # }

    # в других местах тоже строчки длинные, погугли, как можно в разных случаях переносить

    score_home_away = {
        'home' : {
            'totalGoals' : 0,
            'avgFulltime' : 0,
            'avg1stTime' : 0,
            'avg2ndTime' : 0,
            'avgExtraTime' : 0,
            'avgPenalties' : 0
            },
        'away' : {
            'totalGoals' : 0,
            'avgFulltime' : 0,
            'avg1stTime' : 0,
            'avg2ndTime' : 0,
            'avgExtraTime' : 0,
            'avgPenalties' : 0
        }
    }
    for match in json_data['matches']:
        if match['homeTeam']['name'] == team_name:
            score_home_away['home']['totalGoals'] += match['score']['fullTime']['homeTeam']
            score_home_away['home']['avg1stTime'] += match['score']['halfTime']['homeTeam']
        if match['awayTeam']['name'] == team_name:
            score_home_away['away']['totalGoals'] += match['score']['fullTime']['awayTeam']
            score_home_away['away']['avg1stTime'] += match['score']['halfTime']['awayTeam']
        if match['score']['extraTime']['homeTeam'] is not None:
            score_home_away['home']['avgExtraTime'] += match['score']['extraTime']['homeTeam']
        if match['score']['extraTime']['awayTeam'] is not None:
            score_home_away['away']['avgExtraTime'] += match['score']['extraTime']['awayTeam']
        if match['score']['penalties']['homeTeam'] is not None:
            score_home_away['home']['avgPenalties'] += match['score']['penalties']['homeTeam']
        if match['score']['penalties']['awayTeam'] is not None:
            score_home_away['away']['avgPenalties'] += match['score']['penalties']['awayTeam']

    score_home_away['home']['avg2ndTime'] = (score_home_away['home']['totalGoals'] - score_home_away['home']['avg1stTime'] - score_home_away['home']['avgExtraTime'] - score_home_away['home']['avgPenalties']) / score['total']
    score_home_away['away']['avg2ndTime'] = (score_home_away['away']['totalGoals'] - score_home_away['away']['avg1stTime'] - score_home_away['away']['avgExtraTime'] - score_home_away['away']['avgPenalties']) / score['total']
    avgFulltime = score_home_away['home']['totalGoals'] / score['total']
    score_home_away['home']['avgFulltime'] = avgFulltime
    avg1stTime = score_home_away['home']['avg1stTime'] / score['total']
    score_home_away['home']['avg1stTime'] = avg1stTime
    avgExtraTime = score_home_away['home']['avgExtraTime'] / score['total']
    score_home_away['home']['avgExtraTime'] = avgExtraTime
    avgPenalties = score_home_away['home']['avgPenalties'] / score['total']
    score_home_away['home']['avgPenalties'] = avgPenalties
    avgFulltime = score_home_away['away']['totalGoals'] / score['total']
    score_home_away['away']['avgFulltime'] = avgFulltime
    avg1stTime = score_home_away['away']['avg1stTime'] / score['total']
    score_home_away['away']['avg1stTime'] = avg1stTime
    avgExtraTime = score_home_away['away']['avgExtraTime'] / score['total']
    score_home_away['away']['avgExtraTime'] = avgExtraTime
    avgPenalties = score_home_away['away']['avgPenalties'] / score['total']
    score_home_away['away']['avgPenalties'] = avgPenalties
    return score_home_away

score_home_away = super_detailed_score_by_team_name(team_name)
# print(score_home_away)


###### 5. Написать функцию, которая принимает названия двух команд, проверяет
# играли ли они друг против друга и если играли - у какой команды больше побед
# (вывести что-то вроде 'za sezon FK Torpedo deret Arsenal s summarnim scetom 20-4') .

def derby_season_result(team_1, team_2):
    derby_result = {
        team_1 : 0,
        team_2 : 0
    }
    for match in json_data['matches']:
        # @gerzog: код в 198-205 и 206-213 подозрительно одинаковый, надо бы это исправить
        if (team_1 in match['homeTeam']['name'] and team_2 in match['awayTeam']['name']) or (team_2 in match['homeTeam']['name'] and team_1 in match['awayTeam']['name']):
            if match['score']['winner'] == 'AWAY_TEAM':
                derby_result[match['awayTeam']['name']] += 1
            elif match['score']['winner'] == 'HOME_TEAM':
                derby_result[match['homeTeam']['name']] += 1
    return derby_result

team_1 = 'Arsenal FC'
team_2 = 'Fulham FC'

derby_result = derby_season_result(team_1, team_2)
print(derby_result)


###### 6. Нам очень важно проверить, присутствовали ли на матчах подлые ирландские
# ублюдки в качестве рефери. Написать функцию, которая это делает.

def referees_nationality_check(country):
    referees_nationality_checkdict = []
    index = 0
    for match in json_data['matches']:
        for referee in match['referees']:
            if json_data['matches'][i]['referees'][j]['nationality'] != country:
                referees_nationality_checkdict.insert(index, ({'matchid' : json_data['matches'][i]['id'], 'referee_name' : json_data['matches'][i]['referees'][j]['name'], 'country' : json_data['matches'][i]['referees'][j]['nationality']}))
                index += 1
    return referees_nationality_checkdict

country = 'England'
referees_nationality_checkdict = referees_nationality_check(country)
# print(referees_nationality_checkdict)