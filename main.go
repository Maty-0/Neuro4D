package main

import (
	"fmt"
	"neuro4d/core"
)

func main() {
	// Initialize the nucleus (network)
	n := core.Nucleus{}
	var x float32 = 50.0
	dimensions := []float32{x, x, x}  // 3D space
	constraints := []float32{x, x, x} // Space constraints
	n.InitializeNeurons(dimensions, constraints, 1000, 100.0, 0.5, 127, 127)
	fmt.Println("starting connections")
	n.ConnectInputs(10)
	n.ConnectNeurons(10)
	n.ConnectOutputs(30)

	// Run the network with training
	n.Start(30, 0.1, 3.0)
}
