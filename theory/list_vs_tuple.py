import sys 

def print_object_info(b):
    print(b)
    print(type(b))
    print(id(b))
    print(sys.getsizeof(b)) 
    

# a = (10, 20)
# print(a[0])

# a[1] = 30 # нельзя делать, тип не изменяем
# print(a)

# b = ('aaaaa', 12.3, [1, 2, 3])
# print(b)
# print(type(b))
# print(id(b))
# print(sys.getsizeof(b))
# b = list(b)
# print(b)
# print(type(b))
# print(id(b))
# print(sys.getsizeof(b))

list2fill = []
print_object_info(list2fill)

size = 10
for i in range(size):
    # list2fill.append((i, i))
    list2fill.append(i)
    print_object_info(list2fill[-1])
    print('*'*50)

    