import __init__
import numpy as np
import time
from NN2ST import GaussianData
from NN2ST import HypothesisTest

mean_b = np.array([1.0, 1.0])
cov_b = np.array([[1, 0.], [0., 1]])
# Trial sample
mean_t = np.array([1.15, 1.15])
cov_t = np.array([[1.0, 0.0], [0.0, 1.0]])
gdata = GaussianData(mean_b, cov_b, mean_t, cov_t, n_points=20000)
gdata.generate_data()

start_python = time.time()
HT = HypothesisTest(gdata.x_benchmark, gdata.x_trial, K=5, n_perm=100)
print(" python TS {}".format(
    HT.TestStatistic(gdata.x_benchmark, gdata.x_trial)))
end_python = time.time()

__init__.py2goNN2ST(gdata.x_benchmark, gdata.x_trial)

start_go = time.time()
__init__.py2goNN2ST(gdata.x_benchmark, gdata.x_trial)
end_go = time.time()

print("python time   {}".format(end_python - start_python))
print("go time       {}".format(end_go - start_go))
