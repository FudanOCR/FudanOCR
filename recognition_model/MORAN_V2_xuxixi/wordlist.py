
words = "0123456789abcdefghijklmnopqrstuvwxyz"
result = ""
for word in words:
    result += word+":"

# result[-1] = "$"
result = result[:-1]
result += '$'
print(result)

