import numpy as np

a = np.array([1,2,3,4])
b = a
c = np.multiply(a,b)
d = np.dot(a,b)
e = np.transpose(a)
f = np.dot(a,e)
print(c)
print(d)
print(f)

tt = np.random.rand(1,2)
print(tt[0][0])

gg = np.dot(e, a)
print(gg)

m = np.array([2,2,2,2])
m = a
print(m)