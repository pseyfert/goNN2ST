package main

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func randomMatrix(n int, points int) *mat.Dense {
	tmp := make([]float64, n*points)
	for i := 0; i < n*points; i++ {
		tmp[i] = rand.NormFloat64()
	}

	A := mat.NewDense(n, points, tmp)
	return A
}

func newSample(mb mat.Vector, cb mat.Symmetric, npoints int) (*mat.Dense, error) {

	rand_b := randomMatrix(mb.Len(), npoints)

	var chol_b mat.Cholesky
	if ok := chol_b.Factorize(cb); !ok {
		return nil, fmt.Errorf("benchmark covariance matrix not positive semi-definite")
	}

	L_b := chol_b.LTo(nil)
	rand_b.Product(L_b, rand_b)

	// benchmark:
	// slightly faster like this, than flipping i and j
	for i := 0; i < mb.Len(); i++ {
		for j := 0; j < npoints; j++ {
			rand_b.Set(i, j, rand_b.At(i, j)+mb.AtVec(i))
		}
	}

	retval := new(testdata)
	retval.B = rand_b
	return rand_b, nil
}

func newTestdata(mb mat.Vector, cb mat.Symmetric, mt mat.Vector, ct mat.Symmetric, npoints int) (*testdata, error) {
	// original python code:
	// L_b = np.linalg.cholesky(self.cov_b)
	// L_t = np.linalg.cholesky(self.cov_t)
	//
	// self.x_benchmark = np.dot(L_b,(np.random.randn(self.n_points,2) + self.mean_b).T).T
	// self.x_trial = np.dot(L_t,(np.random.randn(self.n_points,2) + self.mean_t).T).T

	if mb.Len() != mt.Len() {
		return nil, fmt.Errorf("median for benchmark and test don't match in size")
	}
	{
		cbr, cbc := cb.Dims()
		ctr, ctc := ct.Dims()
		if cbr != ctr || cbc != ctc {
			return nil, fmt.Errorf("covariance for benchmark and test don't match in dimensions")
		}
	}
	if _, cols := cb.Dims(); cols != mb.Len() {
		return nil, fmt.Errorf("mean and covariance dimensions don't match")
	}

	rand_b, err := newSample(mb, cb, npoints)
	if nil != err {
		return nil, fmt.Errorf("benchmark covariance matrix not positive semi-definite")
	}
	rand_t, err := newSample(mt, ct, npoints)
	if nil != err {
		return nil, fmt.Errorf("test covariance matrix not positive semi-definite")
	}

	retval := new(testdata)
	retval.B = rand_b
	retval.T = rand_t
	return retval, nil
}
