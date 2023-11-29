import numpy as np
import random
import numpy.matlib
import math
from scipy.spatial import distance

limit = 2
a=np.array([[1,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1]])
index_array = np.argwhere(a[0]==1)
print(index_array)

# if len(index_array) > limit:
#     need_reduce = len(index_array) - limit
#     res = random.sample(range(0,len(index_array)), need_reduce)
#     print(res)





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


