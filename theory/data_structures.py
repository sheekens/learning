from pprint import pprint

# 4

l = [1, 2, 3, 4]

# for i in l:
#     i = 0
# print(l)
# exit()

# for i in range(len(l)):
#     l[i] = l[i-1] 
#     print(i, l)

# print(l)

# 7

# lst = [1, 2, 3, 4, 5]
# # Временная сложность поиска по списку = O(n)-где n длинна списка

# lst = [2, 4, 3, 1, 5]
# srt_lst = sorted(lst)
# print(lst)
# print(srt_lst)
# Временная сложность сортировки списка = O(n log n)-где n длинна списка

# поикс в словаре python осуществляется по ключам, временная сложность = О(1)
dct = {
    'Abram': 500,
    'Batman': 10000,
    'Cunt': 100,
    'Debik': 5000
}
dct_rev = {v: k for k, v in dct.items()}

srt_dct = sorted(dct.keys())
srt_dct1 = sorted(dct.values())

dct2 = {}

# for i in srt_dct1:
#     for k, v in dct.items():
#         if i == v:
#             dct2[k] = i

for i in srt_dct1:
    dct2[dct_rev[i]] = i
    
print(dct2)  
    
# д\з залить файлик на гит, добавить в описании карточек ссылку на него по каждому вопросу







# dct1 = {srt_dct[x]:srt_dct1[x] for x in range(len(srt_dct))}


# pprint(dct1)
# print(srt_dct)
