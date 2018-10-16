package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func get_benchmark_ts(N int) *testdata {
	mean_b := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	mean_t := mat.NewVecDense(2, []float64{1.15, 1.15})
	cov_t := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	toy, _ := newTestdata(mean_b, cov_b, mean_t, cov_t, N)

	return toy
}

func benchmarkTS2(b *testing.B, N int) {
	toy := get_benchmark_ts(N)

	for n := 0; n < b.N; n++ {
		_ = TS2(5, toy)
	}
}

func benchmarkTS(b *testing.B, N int) {
	toy := get_benchmark_ts(N)

	for n := 0; n < b.N; n++ {
		_ = TS(5, toy)
	}
}

func BenchmarkTS_10_(b *testing.B)    { benchmarkTS(b, 10) }
func BenchmarkTS_100_(b *testing.B)   { benchmarkTS(b, 100) }
func BenchmarkTS_1000_(b *testing.B)  { benchmarkTS(b, 1000) }
func BenchmarkTS_10000_(b *testing.B) { benchmarkTS(b, 10000) }

// func BenchmarkTS_100000_(b *testing.B) { benchmarkTS(b, 100000) }
// func BenchmarkTS_1000000_(b *testing.B)  { benchmarkTS(b, 1000000) }

func BenchmarkTS2_10_(b *testing.B)    { benchmarkTS2(b, 10) }
func BenchmarkTS2_100_(b *testing.B)   { benchmarkTS2(b, 100) }
func BenchmarkTS2_1000_(b *testing.B)  { benchmarkTS2(b, 1000) }
func BenchmarkTS2_10000_(b *testing.B) { benchmarkTS2(b, 10000) }

// func BenchmarkTS2_100000_(b *testing.B) { benchmarkTS2(b, 100000) }
// func BenchmarkTS2_1000000_(b *testing.B) { benchmarkTS2(b, 1000000) }

func TestJustRunTS_10_(t *testing.T)     { testJustRunTS(t, 10) }
func TestJustRunTS_1000_(t *testing.T)   { testJustRunTS(t, 1000) }
func TestJustRunTS_100000_(t *testing.T) { testJustRunTS(t, 100000) }

func testJustRunTS(t *testing.T, N int) {
	toy := get_benchmark_ts(N)
	ts_reg := TS(5, toy)
	ts_comp := TS(5, toy)
	if math.Abs(ts_reg-ts_comp)/math.Abs(ts_reg+ts_comp) > 0.01 {
		t.Error("more than 1%% difference between implementations")
	}
}
