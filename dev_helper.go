package main

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

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
	npoints, _ := in.Dims()
	pts := make(plotter.XYs, npoints)
	for i := range pts {
		// TODO: there's got to be a nicer way
		pts[i].X = in.At(i, 0)
		pts[i].Y = in.At(i, 1)
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
