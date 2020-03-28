import re

def getLabelFromTxt(add):
    f = open(add,'r',encoding='utf8')
    char_dict = {}

    for line in f.readlines():

        '''
        label获取方式
        '''
        label = line.split()[1]

        for char in label:
            if char not in char_dict.keys():
                char_dict[char] = 1
            else:
                char_dict[char] += 1
    char_dict = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    return char_dict

def getLabelFrom800wTxt(add):
    f = open(add, 'r', encoding='utf8')
    char_dict = {}

    for line in f.readlines():

        '''
        label获取方式
        '''
        label = re.findall(r'_(.*?)_',line)[0]
        # print(label)

        for char in label:
            if char not in char_dict.keys():
                char_dict[char] = 1
            else:
                char_dict[char] += 1
    char_dict = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    return char_dict


if __name__ == '__main__':
    # dic_mjq = getLabelFromTxt('/home/cjy/mjq/result2.txt')
    dic_800w = getLabelFrom800wTxt('/home/cjy/800w_train.txt')

    f = open('./temp.txt','w',encoding='utf-8')
    for pair in dic_800w:
        f.write(str(pair)+'\n')
        print(pair)

    f.close()

# x1 = [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25, 9.25, 10.25, 11.25, 12.25, 13.25, 14.25]
# x2 = [0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75, 10.75, 11.75, 12.75, 13.75, 14.75]

# plt.figure(figsize=(10, 5))
# plt.bar(x1, y1, width=0.5, label='A')
# plt.bar(x2, y2, width=0.5, label='B')
# plt.title('Weight change in 15 months')
# plt.xlabel('Month')
# plt.ylabel('kg')
# plt.legend()
# plt.show()
