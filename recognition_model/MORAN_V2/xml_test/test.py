import re

f = open('./word.xml','r',encoding='iso-8859-1')

string = ""
for line in f.readlines():
    print(line)
    string += line

print(string)

# 记录文件路径
result1 = re.findall(r'file=\"(.*?)\"',string)
print(result1)

result2= re.findall(r'tag=\"(.*?)\"',string)
print(result2)





