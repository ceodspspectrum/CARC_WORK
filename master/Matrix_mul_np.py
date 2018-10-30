import numpy as np
import timeit
import pandas as pd
N=20
i=1
s=2
num_repeats=10
res=pd.DataFrame(columns=('N','Time'))
for j in range(0,N):

    A = np.random.rand(i, i).astype(np.float32)
    B = np.random.rand(i, i).astype(np.float32)

    timer = timeit.Timer("np.dot(A, B)", "import numpy as np; from __main__ import A, B")
    numpy_times_list = timer.repeat(num_repeats, 1)
    times=np.mean(numpy_times_list)
    res.loc[j]=[i,times]
    res.to_csv('Result_mkl_mul_CV_base'+str(s)+'.csv',index=False)
    i=i*s
