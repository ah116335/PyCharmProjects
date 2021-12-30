from typing import Optional, Union, Tuple

import numpy as geek
import pandas as pd

# aa=([1,2,3], [4,5,6])
# print("aa = ",aa)
# bb=[1,2,3],[4,5,6]
# print("bb = ",bb)
# print("######")
# print(bb[:1])
# print(bb[0])
# print(bb[::-1])
# print("######")

#
# print(aa==bb)
#
# aaa=np.array(aa)
# print("aaa is ", aaa)
#
# a=np.array([(1,3,2),(4,0,1)])
# # b=np.matrix(np.random.random((3,4)))
# # c=np.ones((4,3) )
# # d=np.eye(3)
# c=np.array([(1,3),(0,1), (5,2)])
#
# print("a is ", a)
#
# print("shape of a is ", a.shape)
# # print("b")
# # print(b)
# # print(b.shape)
#
# print("c is \n", c)
# print("shape f c is \n", c.shape)
# print(np.dot(a,c))
# # print("d")
# # print(d)
# # print(d.shape)
from numpy.core._multiarray_umath import ndarray

# g=np.linspace(1,10,4)
# print(g)
# print(g.sum())
# print(g.min())


# e=np.full((2,4),7)
# print(e)

array = geek.arange(8)
print("Original array : \n", array)

# shape array with 2 rows and 4 columns
array = geek.arange(8).reshape(2, 4)
print("\narray reshaped with 2 rows and 4 columns : \n", array)

# shape array with 2 rows and 4 columns
array = geek.arange(8).reshape(4, 2)
print("\narray reshaped with 2 rows and 4 columns : \n", array)

# Constructs 3D array
array = geek.arange(8).reshape(2, 2, 2)
print("\nOriginal array reshaped to 3D : \n", array)
