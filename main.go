package main

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type testdata struct {
	B mat.Matrix
	T mat.Matrix
}

func randomMatrix(n int, points int) *mat.Dense {
	tmp := make([]float64, n*points)
	for i := 0; i < n*points; i++ {
		tmp[i] = rand.NormFloat64()
	}

	A := mat.NewDense(n, points, tmp)
	return A
}

func newTestdata(mb mat.Vector, cb mat.Symmetric, mt mat.Vector, ct mat.Symmetric, npoints int) *testdata {
	// original python code:
	// L_b = np.linalg.cholesky(self.cov_b)
	// L_t = np.linalg.cholesky(self.cov_t)
	//
	// self.x_benchmark = np.dot(L_b,(np.random.randn(self.n_points,2) + self.mean_b).T).T
	// self.x_trial = np.dot(L_t,(np.random.randn(self.n_points,2) + self.mean_t).T).T

	if mb.Len() != mt.Len() {
		fmt.Println("median for benchmark and test don't match in size")
	}
	{
		cbr, cbc := cb.Dims()
		ctr, ctc := ct.Dims()
		if cbr != ctr || cbc != ctc {
			fmt.Println("covariance for benchmark and test don't match in dimensions")
		}
	}
	if _, cols := cb.Dims(); cols != mb.Len() {
		fmt.Println("mean and covariance dimensions")
	}

	var chol_b mat.Cholesky
	if ok := chol_b.Factorize(cb); !ok {
		fmt.Println("benchmark covariance matrix not positive semi-definite")
	}
	var chol_t mat.Cholesky
	if ok := chol_t.Factorize(ct); !ok {
		fmt.Println("test covariance matrix not positive semi-definite")
	}

	rand_b := randomMatrix(mb.Len(), npoints)
	rand_t := randomMatrix(mt.Len(), npoints)

	L_b := chol_b.LTo(nil)
	L_t := chol_t.LTo(nil)
	rand_b.Product(L_b, rand_b)
	rand_t.Product(L_t, rand_t)

	// TODO: benchmark me
	for i := 0; i < mb.Len(); i++ {
		for j := 0; j < npoints; j++ {
			rand_b.Set(i, j, rand_b.At(i, j)+mb.AtVec(i))
		}
	}
	for i := 0; i < mt.Len(); i++ {
		for j := 0; j < npoints; j++ {
			rand_t.Set(i, j, rand_t.At(i, j)+mt.AtVec(i))
		}
	}

	retval := new(testdata)
	retval.B = rand_b
	retval.T = rand_t
	return retval
}

func main() {
	// rand.Seed(int64(0))

	mean_b := mat.NewVecDense(2, []float64{0.0, 0.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 0.2})

	mean_t := mat.NewVecDense(2, []float64{1.00, 1.0})
	cov_t := mat.NewSymDense(2, []float64{0.0001, 0.0, 0.0, 0.0001})

	toy := newTestdata(mean_b, cov_b, mean_t, cov_t, 2000)

	plotToy(toy)
}

func plotToy(toy *testdata) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	plotpoints_b := toyToPlot(toy.B)
	plotpoints_t := toyToPlot(toy.T)

	p.Title.Text = "generated random data"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err = plotutil.AddLinePoints(p,
		"B", plotpoints_b,
		"T", plotpoints_t,
		"frame", frame())
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		panic(err)
	}
}

func toyToPlot(in mat.Matrix) plotter.XYs {
	_, npoints := in.Dims()
	pts := make(plotter.XYs, npoints)
	for i := range pts {
		pts[i].X = in.At(0, i)
		pts[i].Y = in.At(1, i)
	}
	return pts
}

func frame() plotter.XYs {
	pts := make(plotter.XYs, 4)
	pts[0].X, pts[0].Y = -4.0, 4.0
	pts[1].X, pts[1].Y = -4.0, -4.0
	pts[2].X, pts[2].Y = 4.0, -4.0
	pts[3].X, pts[3].Y = 4.0, 4.0
	return pts
}
