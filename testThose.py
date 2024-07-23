import torch

# 假设这是你原始的tensor
original_tensor = torch.randn(1, 64)  # 8行64列

# 创建一个与original_tensor行数相同，列数为1，且填充为你想添加的整数的张量
integer_to_add = torch.full((original_tensor.size(0), 1), 1, dtype=original_tensor.dtype)

# 使用torch.cat来合并这两个tensor，拼接维度为1（列方向）
new_tensor = torch.cat((original_tensor, integer_to_add), dim=1)

print(new_tensor)  # 输出应该是torch.Size([8, 65])