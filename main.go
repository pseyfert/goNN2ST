package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

type testdata struct {
	B *mat.Dense
	T *mat.Dense
}

type HT struct {
	testdata
	k int
}

func (ht *HT) Dim() int {
	rows, _ := ht.T.Dims()
	return rows
}
func (ht *HT) nT() int {
	_, cols := ht.T.Dims()
	return cols
}
func (ht *HT) nB() int {
	_, cols := ht.B.Dims()
	return cols
}

func TS(k int, t *testdata) float64 {
	// original code:
	//
	// # Compute estimated density ratio on Trial points
	// r_hat = np.power(np.divide(self.r_B, self.r_T), self.D) * (self.NB/float(self.NT-1))
	//
	// # Compute test statistic over Trial points
	// TS =  np.mean( np.log(r_hat) )
	//
	// is the same as
	// TS =  np.mean(self.D * np.log(np.divide(self.r_B, self.r_T)))
	//     + np.log(self.NB/float(self.NT-1)

	nn := NewNNfromTest(k, t)
	D, nB := t.B.Dims()
	_, nT := t.T.Dims()
	r_b := nn.dists_b(NewKDPointsFromMat(t.T)) // T for both, only _b flips
	r_t := nn.dists_t(NewKDPointsFromMat(t.T))

	logratios := make([]float64, len(r_b))

	for i := 0; i < len(r_b); i++ {
		logratios[i] = math.Log(r_b[i] / r_t[i])
	}
	retval := math.Log(float64(nB)/float64(nT-1)) + float64(D)*stat.Mean(logratios, nil)

	return retval
}
func TS2(k int, t *testdata) float64 {
	// alternative implementation for benchmarking
	nn := NewNNfromTest(k, t)
	D, nB := t.B.Dims()
	_, nT := t.T.Dims()
	r_b := nn.dists_b(NewKDPointsFromMat(t.T)) // T for both, only _b flips
	r_t := nn.dists_t(NewKDPointsFromMat(t.T))

	running_numerator := 1.0
	running_denominator := 1.0
	for i := 0; i < len(r_b); i++ {
		running_numerator *= r_b[i]
		running_denominator *= r_t[i]
	}

	retval := math.Log(float64(nB)/float64(nT-1)) + float64(D)*math.Log(running_numerator/running_denominator)/float64(len(r_b))

	return retval
}

func shuffle(t *testdata) *testdata {
	r, cb := t.B.Dims()
	newB := mat.NewDense(r, cb, nil)
	_, ct := t.T.Dims()
	newT := mat.NewDense(r, ct, nil)

	shuffle_indices := make([]int, cb+ct)
	for i := 0; i < cb+ct; i++ {
		shuffle_indices[i] = i
	}

	rand.Shuffle(cb+ct, func(i, j int) {
		shuffle_indices[i], shuffle_indices[j] = shuffle_indices[j], shuffle_indices[i]
	})

	for j := 0; j < cb; j++ {
		source := shuffle_indices[j]
		var tmp mat.Vector
		if source < cb {
			tmp = t.B.ColView(source)
		} else {
			tmp = t.T.ColView(source - cb)
		}
		for i := 0; i < r; i++ {
			newB.Set(i, j, tmp.AtVec(i))
		}
	}
	for j := cb; j < cb+ct; j++ {
		source := shuffle_indices[j]
		var tmp mat.Vector
		if source < cb {
			tmp = t.B.ColView(source)
		} else {
			tmp = t.T.ColView(source - cb)
		}
		for i := 0; i < r; i++ {
			newT.Set(i, j-cb, tmp.AtVec(i))
		}
	}

	return &testdata{T: newT, B: newB}
}

func PermutationTest(nShuffle, k int, t *testdata) []float64 {
	ts_values := make([]float64, nShuffle)
	for i := 0; i < nShuffle; i++ {
		ts_values[i] = TS2(k, shuffle(t))
	}
	return ts_values
}

func gaussian_pval(randomvals []float64, obsval float64) float64 {
	mu, sigma := stat.MeanStdDev(randomvals, nil)

	// standardize TS values
	obsval = (obsval - mu) / sigma

	dist := distuv.Normal{
		Mu:    0.0,
		Sigma: 1.0,
	}
	return 2.0 * dist.Survival(obsval)
}
func teststat_pval(sortedrandomvals []float64, obsval float64) float64 {
	position := sort.SearchFloat64s(sortedrandomvals, obsval)

	percentile := float64(position) / float64(len(sortedrandomvals))

	return 2.0 * percentile
}
func significance_from_pval(pval float64) float64 {
	dist := distuv.Normal{
		Mu:    0.0,
		Sigma: 1.0,
	}
	return -dist.Quantile(0.5 * pval)
}

func Pvalue(nShuffle, k int, t *testdata) float64 {
	obs_ts := TS2(k, t)
	permutation_ts_vals := PermutationTest(nShuffle, k, t)

	sort.Float64s(permutation_ts_vals)
	return teststat_pval(permutation_ts_vals, obs_ts)
}

func main() {

	mean_b := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	mean_t := mat.NewVecDense(2, []float64{1.15, 1.15})
	cov_t := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	rand.Seed(int64(0))
	toy, _ := newTestdata(mean_b, cov_b, mean_t, cov_t, 20000)

	pval := Pvalue(100, 5, toy)
	sig := significance_from_pval(pval)

	fmt.Println(" got pval %f\n", pval)
	fmt.Println(" got significance %f\n", sig)

	// plotToy(toy)
}
