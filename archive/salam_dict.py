import string
import collections
# words_dict = {}
# words_dict = {'welcome': 1, 'to': 999 }
# print(type(words_dict))
# print(words_dict)
# words_dict['to'] = 0
# words_dict['from'] = 10
# print(words_dict)
# print(words_dict.keys())
# exit()

words_dict = {}
with open('C:/Users/Admin/Downloads/Telegram Desktop/dz_3.txt', 'r') as text:
    text = str(text.readlines())

# print(text)
# print(type(text))
# exit()
words = text.translate(str.maketrans('', '',string.punctuation ),).split()
# print(words)
for word in words:
    if word not in words_dict.keys():
    # if (word or word.capitalize) not in words_dict.keys():
        words_dict[word] = 1
        # print(words_dict)
    else:
        words_dict[word] += 1
# print(words_dict)    

# for key, value in words_dict.items():
#     print(key, value)

# print()
# print(text)
print()
# print(len(words_dict))
# print(words_dict.items())

# rabotaet, no hz chto za lambda i item
# print(dict(sorted(words_dict.items(), key=lambda item: item[1])))
sorted_dict = dict(sorted(words_dict.items(), key=lambda x: x[1], reverse = True))
print(sorted_dict)
# sorted_dict = sorted(words_dict)
# print(sorted_dict)
# words_dict(dict(sorted()))
print()
print('samoe 4astoe slovo - "', max(sorted_dict, key=sorted_dict.get), '", kol-vo povtorenii -', sorted_dict['{}'.format(max(sorted_dict, key=sorted_dict.get))])
print()
# word_check = input('kogo proverit? ')
word_check = 'the'
if word_check in words_dict:
    print('vstre4alos', words_dict.get(word_check), 'raz')
else:
    print('ne bylo')
# for word in words_dict:
#     current = words_dict[word]
#     print(current)
#     if 'current' in words_dict:
#         print('vstre4alos', words_dict['current'], 'raz')
#         break
#     else:
#         print('ne bylo')