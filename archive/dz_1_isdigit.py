print()
input_1 = '0000765.7wexc6898.09j8'
input_2 = '8xtc+=8.vu- -,8.8'

# input_1 = ''
# input_2 = 'sdgfhgjhk'

# input_1 = '    '
# input_2 = ' .987g .,jh7'

# input_1 = '------'
# input_2 = '8xtc+=8.vu- -,8.8'

print('input_1 =', input_1)
print('input_2 =', input_2)

is_input_1_correct = True
is_input_2_correct = True
input_1_dot_found = False
input_2_dot_found = False
input_1_second_dot = False
input_2_second_dot = False
input_1_float = ''
input_2_float = ''
input_1_str = ''
input_2_str = ''

if input_1 is '':
    input_1_float = 0.0
else:
    for i in range(len(input_1)):
        print()
        print('{} of {}'.format(i, len(input_1)))
        print('input_1 current:', input_1[i])
        current = input_1[i]
        if current.isdigit():
            input_1_float += input_1[i]
        elif current is '.' and input_1_dot_found is False:
            input_1_dot_found = True
            input_1_float += input_1[i]
        elif current is '.' and input_1_dot_found is True:
            input_1_second_dot = True
            is_input_1_correct = False
            input_1_str += input_1[i]
        else:
            is_input_1_correct = False
            input_1_str += input_1[i]
            print('naydeny stringi')
if input_1_float is '':
    input_1_float = 0.0

if input_2 is '':
    input_2_float = 0.0
else:
    for i in range(len(input_2)):
        print()
        print('{} of {}'.format(i, len(input_2)))
        print('input_2 current:', input_2[i])
        current = input_2[i]
        if current.isdigit():
            input_2_float += input_2[i]
        elif current is '.' and input_2_dot_found is False:
            input_2_dot_found = True
            input_2_float += input_2[i]
        elif current is '.' and input_2_dot_found is True:
            input_2_second_dot = True
            is_input_2_correct = False
            input_2_str += input_2[i]
        else:
            is_input_2_correct = False
            input_2_str += input_2[i]
            print('naydeny stringi')
if input_2_float is '':
    input_2_float = 0.0

input_1_float = float(input_1_float)
input_2_float = float(input_2_float)

print()
print ('input_1_float =', input_1_float, is_input_1_correct)
print ('input_1_str =', input_1_str)
if input_1_second_dot is True:
    print ('input_1 second dot found')
print ('input_1_float type', type(input_1_float))

print()
print ('input_2_float =', input_2_float, is_input_2_correct)
print ('input_2_str =', input_2_str)
if input_2_second_dot is True:
    print ('input_2 second dot found')
print ('input_2_float type', type(input_2_float))

print()
print('float sum =', (input_1_float + input_2_float))
print('str sum =', input_1_str + input_2_str)

print()
print('program finish')