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
	val *mat.Vector
}

func (p matkdPoint) Dim() int {
	return (*p.val).Len()
}
func (p matkdPoint) GetValue(dim int) float64 {
	return (*p.val).AtVec(dim)
}
func (p matkdPoint) Distance(other kdtree.Point) float64 {
	tmp := mat.NewVecDense(p.Dim(), nil)
	for i := 0; i < p.Dim(); i++ {
		tmp.SetVec(i, other.GetValue(i))
	}
	tmp.SubVec(tmp, *p.val)
	return mat.Norm(tmp, 2)
}
func (p matkdPoint) PlaneDistance(val float64, dim int) float64 {
	tmp := p.GetValue(dim) - val
	return tmp * tmp
}
func upgradeInterface(p mat.Vector) *matkdPoint {
	retval := &matkdPoint{}
	retval.val = &p
	return retval
}
func upgradeInterfaces(m *mat.Dense) []kdtree.Point {
	points := make([]kdtree.Point, 0)
	_, cols := m.Dims()
	for i := 0; i < cols; i++ {
		tmp := upgradeInterface(m.ColView(i))
		points = append(points, *tmp)
	}
	return points
}

type HT struct {
	ttree *kdtree.KDTree
	btree *kdtree.KDTree
	k     int
}

func NewHT(k int, tsample, bsample []kdtree.Point) *HT {
	retval := &HT{ttree: kdtree.NewKDTree(tsample), btree: kdtree.NewKDTree(bsample)}
	retval.k = k
	return retval
}
func NewHTfromTest(k int, t *testdata) *HT {
	return NewHT(k, upgradeInterfaces(t.T), upgradeInterfaces(t.B))
}

func (ht *HT) distt(p *kdtree.Point) float64 {
	neighbours := ht.ttree.KNN(*p, ht.k)
	lastneighbour := neighbours[ht.k-1]
	return (*p).Distance(lastneighbour)
}

func main() {

	mean_b := mat.NewVecDense(2, []float64{1.0, 1.0})
	cov_b := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	mean_t := mat.NewVecDense(2, []float64{1.15, 1.15})
	cov_t := mat.NewSymDense(2, []float64{1.0, 0.0, 0.0, 1.0})

	rand.Seed(int64(0))
	toy, _ := newTestdata(mean_b, cov_b, mean_t, cov_t, 200000)

	myht := NewHTfromTest(4, toy)
	fmt.Printf("using variable %v+\n", myht)

	// plotToy(toy)
}
