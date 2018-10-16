# go build -buildmode c-shared -o goNN2ST.so

import numpy as np
import ctypes
import numpy.ctypeslib as npct
import subprocess

subprocess.check_call(
    ["go", "build",
     "-buildmode", "c-shared",
     "-o", "goNN2ST.so"]
)

goNN2ST = npct.load_library("goNN2ST", ".")

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

goNN2ST.TS_for_numpy.restype = None
goNN2ST.TS_for_numpy.argtypes = [
    array_1d_double, ctypes.c_int, ctypes.c_int,
    array_1d_double, ctypes.c_int, ctypes.c_int]


def py2goNN2ST(x_benchmark, x_test):
    cols, b_rows = x_benchmark.shape
    cols, t_rows = x_test.shape
    goNN2ST.TS_for_numpy(
        np.reshape(x_benchmark, cols*b_rows, "F"), b_rows, cols,
        np.reshape(x_test, cols*t_rows, "F"), t_rows, cols
    )
