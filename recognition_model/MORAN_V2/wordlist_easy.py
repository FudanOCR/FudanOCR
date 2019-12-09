# coding=utf-8

words = "s:e:l:f:a:d:h:i:v:r:b:3:6:8:9:m:c:2:5:0:o:n:N:t:w:G:R:E:K:A:S:T:O:M:Y:C:W:I:L:U:B:P:F:V:D:7:H:y:g:p:u:J:-:k:4:1:j:Z:.:X:x:':!:(:&:q:;:):?:£:é:::Q:Ñ:É"
result = ""
for word in words:
    result += word+":"

# result[-1] = "$"
result = result[:-1]
result += '$'
print(result)
