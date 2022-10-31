import numpy as np
from boltons import iterutils
# dict1 = [1,2,3]
#
# lst_val = []
#
# def func():
#         for value in dict1:
#             global lst_val
#             lst_val.append(value)
#         return lst_val
#
#
#
# dict1.insert(1,5)
# dict1.insert(3,5)
# del dict1[3]
# a=4
# b=6
# c=np.arange(a,b+1,1)
#
# FF = np.arange(0,10,1)
# FF=np.delete(FF,2)
# print(FF)
# f= np.zeros((3,3))
# f[:,2]= c
#
# print(f)



def grouponpairs(l, f):
    groups = []
    g = []
    pairs = iterutils.pairwise(l + [None])
    for a, b in pairs:
        g.append(a)
        if b is None:
            continue
        if not f(a, b):
            groups.append(g)
            g = []
    groups.append(g)
    return groups



n = 3
y = np.array([1,1,1,1,1,9,9,9,1,1,5,5,5,5,1,1,1,1,9,9,9,1,1,5,5,5,5,1,1,1,1,9,9,9,1,1,5,5,5,5,1,1,1,1,9,9,9,1,1,5,5,5,5])

b = [abs(i - j) > n for i, j in zip(y[:-1], y[1:])]
m = [i + 1 for i, j in enumerate(b) if j is True]
m = [0] + m + [len(y)]
result = [y[i: j] for i, j in zip(m[:-1], m[1:])]

print(result)