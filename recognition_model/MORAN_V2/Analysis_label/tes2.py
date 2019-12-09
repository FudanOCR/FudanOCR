f = open('ArTtrainhp.txt','r',encoding='utf-8')

record = []

for line in f.readlines():
    record.append(line.split()[1])

print(record)

str = ''.join(record)
print(str)
print("训练集Label所有字符长度：",len(str))

sum = 0
for ch in str:
    if '\u4e00' <= ch <= '\u9fff':
        sum += 1
print("训练集Label中文字符长度：",sum)