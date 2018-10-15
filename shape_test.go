package main

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestWrongShapes(t *testing.T) {
	mean_b := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	mean_t := mat.NewVecDense(3, []float64{1.15, 1.15, 1.15})
	cov_t := mat.NewSymDense(3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0})

	_, err := newTestdata(mean_b, cov_b, mean_t, cov_t, 1)
	if err == nil || err.Error() != "median for benchmark and test don't match in size" {
		t.Errorf("Benchmark and Test medians with different dimensions were not rejected")
	}

	mean_t = mat.NewVecDense(2, []float64{1.15, 1.15})
	_, err = newTestdata(mean_b, cov_b, mean_t, cov_t, 1)
	if err == nil || err.Error() != "covariance for benchmark and test don't match in dimensions" {
		t.Errorf("Benchmark and Test covariances with different dimensions were not rejected")
	}

	cov_b = mat.NewSymDense(3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0})
	_, err = newTestdata(mean_b, cov_b, mean_t, cov_t, 1)
	if err == nil || err.Error() != "mean and covariance dimensions don't match" {
		t.Errorf("covariances and means with mismatching dimensions were not rejected")
	}

	mean_b = mat.NewVecDense(2, []float64{1.15, 1.15})
	mean_t = mat.NewVecDense(2, []float64{1.15, 1.15})
	cov_b = mat.NewSymDense(2, []float64{
		1.0, 2.0,
		2.0, 1.0})
	cov_t = mat.NewSymDense(2, []float64{
		5.0, 0.0,
		0.0, 5.0})
	_, err = newTestdata(mean_b, cov_b, mean_t, cov_t, 1)
	if err == nil || err.Error() != "benchmark covariance matrix not positive semi-definite" {
		t.Errorf("not positive semi-definite matrix accepted for benchmark")
	}
	cov_b = mat.NewSymDense(2, []float64{
		5.0, 0.0,
		0.0, 5.0})
	cov_t = mat.NewSymDense(2, []float64{
		1.0, 2.0,
		2.0, 1.0})
	_, err = newTestdata(mean_b, cov_b, mean_t, cov_t, 1)
	if err == nil || err.Error() != "test covariance matrix not positive semi-definite" {
		t.Errorf("not positive semi-definite matrix accepted for test")
	}
}
