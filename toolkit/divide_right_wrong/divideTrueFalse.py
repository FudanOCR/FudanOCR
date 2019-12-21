# -*- coding: utf-8 -*-

txt_file = "../MORAN_DEMO/result.txt"
true_file = open("./right.txt",'w')
false_file = open('./wrong.txt','w')


f = open(txt_file)
for line in f.readlines():
    print(line)

    address , target , gt = line.split()
    if target.lower() == gt.lower():
        true_file.write(line)
    else:
        false_file.write(line)

true_file.close()
false_file.close()