all_lines = open("/home/miaosiyu/code_dataset/dataset/OCR_dataset/train.txt",'r').readlines()
# all_lines = open("/home/cqy/chuqianyun1/script/conbine1/syn_dataset.txt",'r').readlines()
all_chars=[]
for each_line in all_lines:
    all_chars.extend(list(each_line.strip()))

print(len(all_chars))
print(set(all_chars))
# with open("./conbine1/syn_keys.txt",'w') as f:
with open("/home/miaosiyu/code_dataset/dataset/OCR_dataset/train_keys.txt",'w') as f:
    for each in set(all_chars):
        f.write(each)
