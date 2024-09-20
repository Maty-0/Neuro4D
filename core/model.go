package core

import (
	"math"
	"math/rand/v2"
	"sort"
)

type Connection struct {
	NeuronPtr       *Neuron
	OutputPtr       *Output
	InputPtr        *Input
	ConnectionScore float32
}

type Neuron struct {
	Dimensions            []float32
	Strength              float32
	ActivationThreshold   float32
	ActivationCache       float32
	ConnectionsIn         []Connection
	ConnectionsOut        []Connection
	TrainingPossitiveCons []Connection
	TrainingNegativeCons  []Connection
	PassportStamp         int8
}

type Input struct {
	Dimensions     []float32
	ConnectionsOut []Connection
}

type Output struct {
	Dimensions    []float32
	ConnectionsIn []Connection
	Prediction    float32
}

type Nucleus struct {
	Cluster         []*Neuron
	ActivationStack []*Neuron
	Inputs          []*Input
	Outputs         []*Output
}

// CREATE Nueron network

func (n *Nucleus) InitializeNeurons(dimensions []float32, constraints []float32, neurons int, strengthM float32, thresholdM float32, inputs int, outputs int) {
	// Initialize neurons
	for i := 0; i < neurons; i++ {
		neuronA := Neuron{
			Dimensions:            make([]float32, len(dimensions)), // Allocate space for the position of the neuron
			Strength:              rand.Float32() * strengthM,       // Initial random signal strength
			ActivationThreshold:   rand.Float32() * thresholdM,      //* 0.5,             Random initial activation cap (threshold)
			ConnectionsIn:         make([]Connection, 0),
			ConnectionsOut:        make([]Connection, 0),
			TrainingPossitiveCons: make([]Connection, 0),
			TrainingNegativeCons:  make([]Connection, 0),
		}
		// Set random or constrained positions in space
		for d := range neuronA.Dimensions {
			neuronA.Dimensions[d] = rand.Float32() * constraints[d]
		}
		// Add neuron to the cluster
		var ptr *Neuron
		ptr = &neuronA
		n.Cluster = append(n.Cluster, ptr)
	}

	// Initialize inputs
	for i := 0; i < inputs; i++ {
		in := Input{
			Dimensions:     make([]float32, len(dimensions)),
			ConnectionsOut: make([]Connection, 0),
		}
		for d := range in.Dimensions {
			in.Dimensions[d] = rand.Float32() * constraints[d]
		}
		var ptr2 *Input
		ptr2 = &in
		n.Inputs = append(n.Inputs, ptr2)
	}

	// Initialize outputs
	for i := 0; i < outputs; i++ {
		out := Output{
			Dimensions:    make([]float32, len(dimensions)),
			ConnectionsIn: make([]Connection, 0),
			Prediction:    0.0,
		}
		for d := range out.Dimensions {
			out.Dimensions[d] = rand.Float32() * constraints[d]
		}
		var ptr3 *Output
		ptr3 = &out
		n.Outputs = append(n.Outputs, ptr3)
	}
}

func CalculateDistance(pos1 []float32, pos2 []float32) float32 {
	var sum float32 = 0.0

	for i := range pos1 {

		sum += (pos1[i] - pos2[i]) * (pos1[i] - pos2[i])
	}
	return float32(math.Sqrt(float64(sum)))
}

func CheckNeighbor(neuronPtr *Neuron, otherPtr *Neuron) bool {
	for _, n := range neuronPtr.ConnectionsIn {
		if otherPtr == n.NeuronPtr {
			return false
		}
	}
	return true
}

func (n *Nucleus) ConnectNeurons(maxConnections int) {
	for i, neuronPtr := range n.Cluster {
		// Compute the distance to all other neurons
		type neighbor struct {
			ptr      *Neuron
			distance float32
		}

		var neighbors []neighbor
		for j, otherPtr := range n.Cluster {
			if i == j {
				continue // Skip self
			}

			otherNeuron := otherPtr
			distance := CalculateDistance(neuronPtr.Dimensions, otherNeuron.Dimensions)

			if CheckNeighbor(neuronPtr, otherPtr) {

				neighbors = append(neighbors, neighbor{ptr: otherPtr, distance: distance})
			}
		}

		// Sort neighbors by distance
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].distance < neighbors[b].distance
		})

		// Connect to the closest `maxConnections` neurons

		for k := 0; k < maxConnections && k < len(neighbors); k++ {
			neuronPtr.ConnectionsOut = append(neuronPtr.ConnectionsOut, Connection{
				NeuronPtr: neighbors[k].ptr,
				OutputPtr: nil,
				InputPtr:  nil,
			})

			// Make the connection bidirectional by adding to the other neuron's connectionsIn
			otherNeuron := neighbors[k].ptr
			otherNeuron.ConnectionsIn = append(otherNeuron.ConnectionsIn, Connection{
				NeuronPtr: neuronPtr,
				OutputPtr: nil,
				InputPtr:  nil,
			})
		}
	}
}

func (n *Nucleus) ConnectOutputs(maxConnections int) {
	for i, outputPtr := range n.Outputs {
		// Compute the distance to all other neurons
		type neighbor struct {
			ptr      *Neuron
			distance float32
		}

		var neighbors []neighbor
		for j, otherPtr := range n.Cluster {
			if i == j {
				continue // Skip self
			}

			otherNeuron := otherPtr
			distance := CalculateDistance(outputPtr.Dimensions, otherNeuron.Dimensions)

			//if checkNeighborOutput(outputPtr, otherPtr) {
			neighbors = append(neighbors, neighbor{ptr: otherPtr, distance: distance})
			//}
		}

		// Sort neighbors by distance
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].distance < neighbors[b].distance
		})

		// Connect to the closest `maxConnections` neurons

		for k := 0; k < maxConnections && k < len(neighbors); k++ {
			outputPtr.ConnectionsIn = append(outputPtr.ConnectionsIn, Connection{
				NeuronPtr: neighbors[k].ptr,
				OutputPtr: nil,
				InputPtr:  nil,
			})

			// Make the connection bidirectional by adding to the other neuron's connectionsIn
			otherNeuron := neighbors[k].ptr
			otherNeuron.ConnectionsOut = append(otherNeuron.ConnectionsOut, Connection{
				NeuronPtr: nil,
				OutputPtr: outputPtr,
				InputPtr:  nil,
			})
		}
	}
}

func (n *Nucleus) ConnectInputs(maxConnections int) {
	for i, inputPtr := range n.Inputs {
		// Compute the distance to all other neurons
		type neighbor struct {
			ptr      *Neuron
			distance float32
		}

		var neighbors []neighbor
		for j, otherPtr := range n.Cluster {
			if i == j {
				continue // Skip self
			}

			otherNeuron := otherPtr
			distance := CalculateDistance(inputPtr.Dimensions, otherNeuron.Dimensions)

			neighbors = append(neighbors, neighbor{ptr: otherPtr, distance: distance})
		}

		// Sort neighbors by distance
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].distance < neighbors[b].distance
		})

		// Connect to the closest `maxConnections` neurons

		for k := 0; k < maxConnections && k < len(neighbors); k++ {
			inputPtr.ConnectionsOut = append(inputPtr.ConnectionsOut, Connection{
				NeuronPtr: neighbors[k].ptr,
				OutputPtr: nil,
				InputPtr:  nil,
			})

			// Make the connection bidirectional by adding to the other neuron's connectionsIn
			otherNeuron := neighbors[k].ptr
			otherNeuron.ConnectionsIn = append(otherNeuron.ConnectionsIn, Connection{
				NeuronPtr: nil,
				OutputPtr: nil,
				InputPtr:  inputPtr,
			})
		}
	}
}

/* fires each input once at same strength -> REPLACED BY LEARNING FUNCTION
func (n *nucleus) startSignal() {
	for _, input := range n.inputs {
		neuronIn := input
		for _, out := range neuronIn.connectionsOut {
			if out.outputPtr == nil {
				NeuronOut := out
				distance := calculateDistance(neuronIn.dimensions, NeuronOut.neuronPtr.dimensions)
				NeuronOut.neuronPtr.activationCache += 1 / distance //strength of 1 for the startin neurons
				n.activationStack = append(n.activationStack, out.neuronPtr)
			} else if out.outputPtr != nil {
				targetOutput := out
				distance := calculateDistance(neuronIn.dimensions, targetOutput.outputPtr.dimensions)
				targetOutput.outputPtr.prediction += 1 / distance
			}
		}
	}
} */

// PropagateSignal propagates the signal from this neuron to connected neurons.
func (n *Nucleus) PropagateSignal(neurons *Neuron, distanceLoss float32) {
	if neurons.ActivationCache > neurons.ActivationThreshold {
		for _, conn := range neurons.ConnectionsOut {
			if conn.OutputPtr != nil && conn.NeuronPtr != neurons {
				targetOutput := conn
				distance := CalculateDistance(neurons.Dimensions, targetOutput.OutputPtr.Dimensions)
				if neurons.Strength-(distance*distanceLoss) > 0 {
					targetOutput.OutputPtr.Prediction += neurons.Strength - (distance * distanceLoss)
				}
				//fmt.Println("it tickles")
			} else if conn.OutputPtr == nil && conn.NeuronPtr != neurons {
				targetNeuron := conn
				distance := CalculateDistance(neurons.Dimensions, targetNeuron.NeuronPtr.Dimensions)
				if neurons.Strength-(distance*distanceLoss) > 0 {
					targetNeuron.NeuronPtr.ActivationCache += neurons.Strength - (distance * distanceLoss)
					n.ActivationStack = append(n.ActivationStack, conn.NeuronPtr)
				}
			}
		}
		neurons.ActivationCache = 0
	}
}

func (n *Nucleus) ClearCache() {
	for _, neur := range n.Cluster {
		neur.ActivationCache = 0.0
		neur.PassportStamp = 0
	}
}

func (n *Nucleus) DequeueActivationStack() *Neuron {
	if len(n.ActivationStack) == 0 {
		return nil // or handle empty queue case appropriately
	}
	neuron := n.ActivationStack[0]
	n.ActivationStack = n.ActivationStack[1:]
	return neuron
}
