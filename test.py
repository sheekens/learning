a = 72.78888888888889%
b = 2
print('{0:04d}'.format(a))
# print('{}/simple2DConv.{0:04d}.pt'.format(b, a))
# print('{b}/simple2DConv.{a:04d}.pt'.format(b, a))
print(f'{b}/simple2DConv.{a:04f}.pt')


exit()
import os
import cv2
from varname.helpers import debug

img1 = cv2.imread('datasets/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset/img1/000001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('datasets/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset/img1/000002.jpg', cv2.IMREAD_GRAYSCALE)

result = img2 - img1
result = cv2.absdiff(img2, img1)
cv2.imshow('jeezus', result)
cv2.waitKey(-1)



exit()
print()
input_1 = input('gfug')
input_2 = 'da7.9sdfg,86de'
print('input_1=', input_1)
print('input_2=', input_2)
# print(len(input_1))
# print(len(input_2))
#print(input_1[:-2])
numbers = [0,1,2,3,4,5,6,7,8,9]
for i in range(len(numbers)):
    #numbers[i] = str(numbers[i])
    print(numbers[i])
    print(i)
    print()
is_input_1_correct = True
is_input_2_correct = True
input_1_float = ''
input_2_float = ''
input_1_str = ''
input_2_str = ''

for i in range(len(input_1)):
    print()
    print('{} of {}'.format(i, len(input_1)))
    print('current:', input_1[i])
    current = input_1[i]
    # if current not in numbers:
    #     is_input_1_correct = False
    #     input_1_str = input_1_str + str(input_1[i])
    #     print('naydeny stringi')
    if current.isdigit:
        input_1_float = True
    else:
        input_1_float = False
        
for i in range(len(input_2)):
    print()
    print('{} of {}'.format(i, len(input_2)))
    print('current:', input_2[i])
    current = input_2[i]
    if current not in numbers:
        is_input_2_correct = False
        input_2_str = input_2_str + str(input_2[i])
        print('naydeny stringi')
    if current.isdigit:
        input_2_float = input_2_float + str(input_2[i])
print (input_1_float)
exit()
input_1_float = float(input_1_float)
input_2_float = float(input_2_float)
input_1_str = input_1_str
input_2_str = input_2_str
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