from pprint import pprint

# 4
""" 
    В Python итерация по списку с помощью for x in list не позволяет напрямую изменять элементы списка, 
    потому что x является копией текущего элемента списка, а не ссылкой на него. 
    Изменения, внесенные в x, не отразятся на самом списке.
"""

my_list = [1, 2, 3, 4]
# Итерация с копированием элемента
for i in my_list: 
    i = 0    # Это не изменит исходный список
print(my_list)

""" С другой стороны, итерация с использованием for i in range(len(list)) позволяет изменять элементы списка, 
    так как вы работаете с индексами элементов, что дает прямой доступ к элементам списка для их изменения.
"""
# Итерация с доступом по индексу
for i in range(len(my_list)):
    my_list[i] = my_list[i-1] # Это изменит исходный список
    print(i, my_list)

print(my_list)

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
