import numpy as np
import numpy.matlib
import math
from scipy.spatial import distance

a=[[1,2],[3,4],[5,6]]
b=[[7,8],[9,10],[11,12]]

p = np.vstack((a,b))

print(p)


# a=[[1,2],[3,4]]
# a = np.array(a)
# theta = [5, 5]
# print(np.argwhere(np.sum(a>1, 1) == 1))

# V1 = np.linspace(0,1,10)
# V2 = 1-V1
# V = np.vstack((V1, V2)).T
# cosine = []
# for i in range(len(V)):
#     tmp = []
#     for j in range(len(V)):
#         # print(i , j, V[i],V[j], 1-distance.pdist((V[i],V[j]), 'cosine'))
#         dis = 1-distance.pdist((V[i],V[j]), 'cosine')
#         tmp.append(dis.tolist()[0])
#     cosine.append(tmp)

# cosine = cosine - np.diag(np.diag(cosine))

# print(cosine)

# for i in range(len(cosine)):
# data = numpy.arccos(cosine)
# print(data)
# dd = numpy.min(, axis=1)
# print(numpy.argmin(data, axis=1))


# x = [0, 0]
# y = [0, 1]
# dis = 1-distance.pdist((x,y), 'cosine')
# if np.isnan(dis):
#     print('nan')
#     dis = 1
# print(numpy.arccos(dis))


