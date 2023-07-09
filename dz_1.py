print()
input_1 = '124124.eaew13212aue'
input_2 = 'da7.9sdfg,86de'
print('input_1=', input_1)
print('input_2=', input_2)
# print(len(input_1))
# print(len(input_2))
#print(input_1[:-2])
numbers = [0,1,2,3,4,5,6,7,8,9,'.']
for i in range(len(numbers)):
    numbers[i] = str(numbers[i])

is_input_1_correct = True
is_input_2_correct = True
input_1_float = '0'
input_2_float = '0'
input_1_str = '0'
input_2_str = '0'

for i in range(len(input_1)):
    print()
    print('{} of {}'.format(i, len(input_1)))
    print('current:', input_1[i])
    current = input_1[i]
    if current not in numbers:
        is_input_1_correct = False
        input_1_str = input_1_str + str(input_1[i])
        print('naydeny stringi')
    if current in numbers:
        input_1_float = input_1_float + str(input_1[i])

for i in range(len(input_2)):
    print()
    print('{} of {}'.format(i, len(input_2)))
    print('current:', input_2[i])
    current = input_2[i]
    if current not in numbers:
        is_input_2_correct = False
        input_2_str = input_2_str + str(input_2[i])
        print('naydeny stringi')
    if current in numbers:
        input_2_float = input_2_float + str(input_2[i])
input_1_float = float(input_1_float[1:])
input_2_float = float(input_2_float[1:])
input_1_str = input_1_str[1:]
input_2_str = input_2_str[1:]
print()
print ('input_1_float=', input_1_float, is_input_1_correct)
print ('input_1_str=', input_1_str)
print ('input_1_float type', type(input_1_float))
print()
print ('input_2_float=', input_2_float, is_input_2_correct)
print ('input_2_str=', input_2_str)
print()
print('str sum', input_1_str + input_2_str)
print('float sum', (input_1_float + input_2_float))


print()
print('program finish')