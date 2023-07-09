def palindrom_check(input):
    # input = '123456765432'
    lenght = len(input)
    half_lenght = int(lenght/2)
    palindrom_flag = True
    if input == '':
        print('no input')
    else:
        for i in range(half_lenght):
            input_lr = input[i]
            rl_index = lenght - i - 1
            input_rl = input[-i-1]
            #input_rl = input[rl_index]
            print('input_lr =', input_lr)
            print('input_rl =', input_rl)
            if input_lr == input_rl:
                print('сходится')
            else:
                print('не сходится')
                palindrom_flag = False
                break
        print()
        print(input_lr, input_rl)
        if palindrom_flag:
            print('Palindrom!!!')
        else:
            print('ne palindrom')

# text = input('vvodi suka!! ')
# palindrom = palindrom_check(text)

#exit()
############str_to_char_num

def lenght_count(input):
    # input = 'жаль, что ты лох'
    char_lenght = []
    char_count = 0
    char_index = 0
    if input == '':
        print('no input')
    else:
        for i in range(len(input)):
            current = input[i]
            is_word_finished = False
            if (current.isalpha() or current.isdigit()) or current == "'":
                char_count += 1
                if i == len(input) - 1:
                    is_word_finished = True
                print('char', current, 'char_count', char_count)
            elif char_count > 0:
                is_word_finished = True
            if is_word_finished:
                char_lenght.insert(char_index, char_count)
                char_index += 1
                char_count = 0
        print()
        print(char_lenght)
        print('ok')

input = " жаль, что ты ло4х's"
palindrom = lenght_count(input)

############anagramm_check

# str1 = input('str1= ')
# str2 = input('str2= ')
# chars = []
# char_index = 0
# anagramma_flag = True
# for i in range(len(str1)):
#     current = str1[i]
#     if current not in chars:
#         chars.insert(char_index, current)
#         char_index += 1
# print(chars)
# for i in range(len(str2)):
#     current = str2[i]
#     if current not in chars:
#         anagramma_flag = False
#         break
# if anagramma_flag:
#     print('anagramma')
# else:
#     print('dermo, a ne anagramma')