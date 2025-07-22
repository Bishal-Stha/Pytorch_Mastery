import torch

my_list = [[1,2,3], [4,5,6]]
tensor = torch.tensor(my_list)
print("Tensor:\n",tensor)

print("Tensor Attributes:\n")
print("Tensor Shape: ",tensor.shape)
print("Tensor DataType: ",tensor.dtype)

a = torch.tensor([
    [1,2],
    [3,4]
])

b = torch.tensor([
    [2,1],
    [4,3]
])

print("\nElement wise Tensor Operations:\n")
print("Tensor add:\n",a+b)
print("Tensor subtract:\n",a-b)
print("Tensor multiplication:\n",a*b)
print("Tensor Division", a/b)
print("Matrix Multiplication:\n",a@b)
"""

[[1,2],    [[1,2],
[3,4]]      [3,4]]
])

A@A = 
1*1+2*3  1*2+2*4
3*1+3*3  3*2+4*4

7   10
15  22

"""
# print("Tensor add:\n",)