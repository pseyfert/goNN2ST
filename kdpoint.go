package main

import (
	kdtree "github.com/hongshibao/go-kdtree"
	"gonum.org/v1/gonum/mat"
)

type kdPoint struct {
	kdtree.Point
	val *mat.Vector
}

func (p kdPoint) Dim() int {
	return (*p.val).Len()
}
func (p kdPoint) GetValue(dim int) float64 {
	return (*p.val).AtVec(dim)
}
func (p kdPoint) Distance(other kdtree.Point) float64 {
	tmp := mat.NewVecDense(p.Dim(), nil)
	for i := 0; i < p.Dim(); i++ {
		tmp.SetVec(i, other.GetValue(i))
	}
	tmp.SubVec(tmp, *p.val)
	return mat.Norm(tmp, 2)
}
func (p kdPoint) PlaneDistance(val float64, dim int) float64 {
	tmp := p.GetValue(dim) - val
	return tmp * tmp
}
func vectorToKDPoint(p mat.Vector) *kdPoint {
	retval := &kdPoint{}
	retval.val = &p
	return retval
}
func NewKDPointsFromMat(m *mat.Dense) []kdtree.Point {
	rows, _ := m.Dims()
	points := make([]kdtree.Point, rows)
	for i := 0; i < rows; i++ {
		tmp := vectorToKDPoint(m.RowView(i))
		points[i] = *tmp
	}
	return points
}
