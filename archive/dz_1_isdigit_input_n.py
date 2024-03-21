print()
input_1 = '345durcvy87,i7t6bnm,9097'
input_2 = '8888'
# print('input_1 =', input_1)
# print('input_2 =', input_2)

input_number = [1,2]
for i in input_number:
    print('input_{} = {}'.format(i, input_{}.format(i)))
    print(i)
exit()
is_input_1_correct = True
input_1_dot_found = False
input_1_second_dot = False
input_1_float = ''
input_1_str = ''



if input_1 is '':
    input_1_float = 0.0
else:
    for i in range(len(input_1)):
        print()
        print('{} of {}'.format(i, len(input_1)))
        print('input_1 current:', input_1[i])
        current = input_1[i]
        if current.isdigit():
            input_1_float += str(input_1[i])
        elif current is '.' and input_1_dot_found is False:
            input_1_dot_found = True
            input_1_float += str(input_1[i])
        elif current is '.' and input_1_dot_found is True:
            input_1_second_dot = True
        else:
            is_input_1_correct = False
            input_1_str += str(input_1[i])
            print('naydeny stringi')
if input_1_float is '':
    input_1_float = 0.0

input_1_float = float(input_1_float)

print()
print ('input_1_float =', input_1_float, is_input_1_correct)
print ('input_1_str =', input_1_str)
if input_1_second_dot is True:
    print ('input_1 second dot found')
print ('input_1_float type', type(input_1_float))


print()
print('float sum =', (input_1_float + input_2_float))
print('str sum =', input_1_str + input_2_str)


print()
print('program finish')