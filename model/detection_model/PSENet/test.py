test_ex = []
ex = {
    "K": 0,
    "N": 0,
    "list": []
}


def mov(lis, index, cur_iter):
    max = lis.pop(index)
    lis.insert(cur_iter, max)
    return lis


T = int(input())
for i in range(T):
    temp_k = int(input())
    temp_n = int(input())
    temp_list = input()
    temp_list = temp_list.split()
    temp_list = list(map(int, temp_list))
    ex["K"] = temp_k
    ex["N"] = temp_n
    ex["list"] = temp_list
    test_ex.append(ex.copy())

for i in range(T):
    cur_iter = 0
    left_exchange = test_ex[i]["K"]
    cur_list = test_ex[i]["list"]
    cur_len = test_ex[i]["N"]
    print(left_exchange)
    while (left_exchange > 0):
        # if (left_exchange > (cur_len - cur_iter-1)):
        #     # print(cur_list[cur_iter:].index(max(cur_list[cur_iter:])))
        #
        #     index = cur_list[cur_iter:].index(max(cur_list[cur_iter:])) + cur_iter
        #     cur_list = mov(cur_list, index, cur_iter)
        #     left_exchange = left_exchange - (index - cur_iter)
        #     cur_iter += 1
        # else:
           # print(cur_list[cur_iter:cur_iter + left_exchange])
        index = cur_list[cur_iter:cur_iter + left_exchange+1].index(max(cur_list[cur_iter:cur_iter + left_exchange+1])) + cur_iter
        cur_list = mov(cur_list, index, cur_iter)
        left_exchange = left_exchange - (index - cur_iter)
        cur_iter += 1
    print(cur_list)














