import json

# 指定您想要读取的JSON文件名
filename = 'Operation.json'
target = 'toOperation.json'

# 使用 `json.load()` 函数从JSON文件中读取数据
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

dic = {}
idx = 0
for i, j in data.items():
    dic[idx] = i
    idx += 1

with open(target, 'w', encoding='utf-8') as f:
    json.dump(dic, f, indent=4, ensure_ascii=False)

