package main

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func BenchmarkDirect(b *testing.B) {
	mean_b := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	mean_t := mat.NewVecDense(2, []float64{1.15, 1.15})
	cov_t := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	rand.Seed(int64(0))
	for n := 0; n < b.N; n++ {
		toy, _ := newTestdata(mean_b, cov_b, mean_t, cov_t, 200000)
		_ = toy.T.At(0, 0)
	}
}
