# # test_list = []
# test_list = ['banan', 'kokos', 'noski', 'avtobus']
# # test_list.insert(0, 'banan')
# print(test_list)

# test_dict = {}
# test_dict['banan'] = 1
# print(test_dict)

# big_dict = {}
# big_dict['pokupki'] = test_list
# print(big_dict)

import json

path_to_json = 'D:/testing/Stats_version_1_seas_20_21.json'
with open(path_to_json) as file:
    json_data = json.load(file)

# print(json_data['filters']['status'])
# print(json_data['matches'][0]['season']['id'])

# for i in range(json_data['count']):
# # for i in range(311):
#     id_matcha = json_data['matches'][i]['id']
#     print(i, id_matcha)

# for match in json_data['matches']:
#     id_matcha2 = match['id']
#     print(type(match))

# Написать функцию, которая достает из jsona все
#  встречающиеся команды в виде словаря (k - id команды, v - название)

team_names_dict = {}
for match in json_data['matches']:
    hometeam_name = match['homeTeam']['name']
    hometeam_id = match['homeTeam']['id']
    team_names_dict[hometeam_id] = hometeam_name
    # print(hometeam_name, hometeam_id)
sorted_dict = dict(sorted(team_names_dict.items(), key = lambda x: x[0], reverse = False))
print(sorted_dict)