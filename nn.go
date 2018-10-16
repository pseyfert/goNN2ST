package main

import kdtree "github.com/hongshibao/go-kdtree"

type NN struct {
	ttree *kdtree.KDTree
	btree *kdtree.KDTree
	k     int
}

func NewNN(k int, tsample, bsample []kdtree.Point) *NN {
	retval := &NN{ttree: kdtree.NewKDTree(tsample), btree: kdtree.NewKDTree(bsample)}
	retval.k = k
	return retval
}
func NewNNfromTest(k int, t *testdata) *NN {
	return NewNN(k, NewKDPointsFromMat(t.T), NewKDPointsFromMat(t.B))
}

func (ht *NN) dist_t(p *kdtree.Point) float64 {
	neighbours := ht.ttree.KNN(*p, ht.k+1) // k is incremented by one for T in the original version, so here too
	lastneighbour := neighbours[ht.k-1+1]
	return (*p).Distance(lastneighbour)
}
func (ht *NN) dists_t(ps []kdtree.Point) []float64 {
	dists := make([]float64, len(ps))
	for i := 0; i < len(ps); i++ {
		dists[i] = ht.dist_t(&ps[i])
	}
	return dists
}

func (ht *NN) dist_b(p *kdtree.Point) float64 {
	neighbours := ht.btree.KNN(*p, ht.k)
	lastneighbour := neighbours[ht.k-1]
	return (*p).Distance(lastneighbour)
}
func (ht *NN) dists_b(ps []kdtree.Point) []float64 {
	dists := make([]float64, len(ps))
	for i := 0; i < len(ps); i++ {
		dists[i] = ht.dist_b(&ps[i])
	}
	return dists
}
