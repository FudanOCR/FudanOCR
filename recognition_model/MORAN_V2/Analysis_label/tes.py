import re

def cal_len(str):
    length = len(str)
    for ch in str:
        if '\u4e00' <= ch <= '\u9fff':
            length += 1
    return length




f = open('log.txt',encoding='utf-8')

record = []
for line in f.readlines():
    print(line)
    record.append(re.findall(' (\S+)\$',line))

right = open('right.txt','w',encoding='utf-8')
wrong = open('wrong.txt','w',encoding='utf-8')

right.write("预测" + " "*35 + "目标\n")
wrong.write("预测" + " "*35 + "目标\n")

for pair in record:
    # print(pair)
    if len(pair) != 2:
        continue
    if pair[0].lower() == pair[1].lower():
        right.write(pair[0] + " "*(35-cal_len(pair[0]))+"|" + pair[1] + "\n")
    else:
        wrong.write(pair[0] + " "*(35-cal_len(pair[0]))+"|" + pair[1] + "\n")

right.close()
wrong.close()