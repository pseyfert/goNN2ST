package main

import (
	"fmt"
	"math/rand"

	"github.com/hongshibao/go-kdtree"
	"gonum.org/v1/gonum/mat"
)

type testdata struct {
	B *mat.Dense
	T *mat.Dense
}

type matkdPoint struct {
	kdtree.Point
	val mat.Vector
}

func (p *matkdPoint) Dim() int {
	return p.val.Len()
}
func (p *matkdPoint) GetValue(dim int) float64 {
	return p.val.AtVec(dim)
}
func (p *matkdPoint) Distance(other matkdPoint) float64 {
	tmp := mat.NewVecDense(p.Dim(), nil)
	tmp.SubVec(other.val, p.val)
	return mat.Norm(tmp, 2)
}
func (p *matkdPoint) PlaneDistance(val float64, dim int) float64 {
	tmp := p.GetValue(dim) - val
	return tmp * tmp
}
func upgradeInterface(p mat.Vector) *matkdPoint {
	retval := &matkdPoint{}
	retval.val = p
	return retval
}

func main() {

	mean_b := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	mean_t := mat.NewVecDense(2, []float64{1.15, 1.15})
	cov_t := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	rand.Seed(int64(0))
	toy, _ := newTestdata(mean_b, cov_b, mean_t, cov_t, 200000)
	fmt.Printf(" test seed stuff %f\n", toy.T.At(1, 44))

	points_b := make([]matkdPoint, 0)
	_, cols := toy.B.Dims()
	for i := 0; i < cols; i++ {
		points_b = append(points_b, *upgradeInterface(toy.B.ColView(i)))
	}

	// plotToy(toy)
}
