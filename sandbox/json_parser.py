import json

path_to_json = 'D:/testing/Stats_version_1_seas_20_21.json'
with open(path_to_json) as file:
    json_data = json.load(file)
# print(json_data)
# print(type(json_data))
# print('keys', json_data.keys())
print()

# Что за конструкция? json_data/словарь/ ['matches']/ключ json_data/ [0]/??????/ .keys()/ключи ключа matches словаря json_data/
# Вывод ключей ключа matches словаря json_data
# print(json_data['matches'][1].keys())
# print(json_data['matches'][0]['score']['fullTime'])
# print(json_data['filters']['status'])
# print(type(json_data['matches'][0].keys()))

# team_list = []
# index = 0
# for i in range(311):
#     home_team = json_data['matches'][i]['homeTeam']['name']
#     away_team = json_data['matches'][i]['awayTeam']['name']
#     if home_team not in team_list:
#         team_list.insert(index, home_team)
#         index += 1
#     if away_team not in team_list:
#         team_list.insert(index, away_team)
#         index += 1

# team_list = []
# index = 0
# for i in range(311):
#     home_team = json_data['matches'][i]['homeTeam']['name']
#     away_team = json_data['matches'][i]['awayTeam']['name']
#     if home_team not in team_list:
#         print('popka')
#         team_list.insert(index, home_team)
#         index += 1
#     for j in range(311):
#         # index = 0
#         if away_team not in team_list:
#             print('piska')
#             team_list.insert(index, away_team)
#             index += 1


# print(team_list)
# print(len(team_list))

team_dict = {}
for i in range(311):
    for key, value in json_data['matches'][i]['homeTeam'].items():
        if value not in team_dict.values():
            team_dict[json_data['matches'][i]['homeTeam']['id']] = json_data['matches'][i]['homeTeam']['name']
print(team_dict)
# for i in range(311):
#     id = json_data['matches'][i]['homeTeam']['id']
    
#     name = json_data['matches'][i]['homeTeam']['name']
#     for j in 'homeTeam':

#     if home_team not in team_list:
#         team_list.insert(index, home_team)
#         index += 1
#     if away_team not in team_list:
#         team_list.insert(index, away_team)
#         index += 1