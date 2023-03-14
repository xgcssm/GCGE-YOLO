import torch as t
import math
import numpy as np

alist = t.randn(2, 6, 7)

inputsz = np.array(alist.shape[1:])
outputsz = np.array([2, 3])

stridesz = np.floor(inputsz / outputsz).astype(np.int32)

kernelsz = inputsz - (outputsz - 1) * stridesz

adp = t.nn.AdaptiveAvgPool2d(list(outputsz))
avg = t.nn.AvgPool2d(kernel_size=list(kernelsz), stride=list(stridesz))
adplist = adp(alist)
avglist = avg(alist)

print(alist)
print(adplist)
print(avglist)