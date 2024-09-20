package core

import (
	"fmt"
	"math/rand/v2"
	"runtime"
)

func (n *Nucleus) FireQuestion(qid int, question []byte, distanceLoss float32) int {
	var neuronDeathCount int32 = 0
	//inputs need to represent all 127 ascii values
	for id, x := range question {
		var stren float32 = 0.1
		if id == qid {
			stren = 1
		}
		var outpu *Input = n.Inputs[x]
		for _, outC := range outpu.ConnectionsOut {
			if outC.OutputPtr == nil {
				NeuronOut := outC
				distance := CalculateDistance(outpu.Dimensions, NeuronOut.NeuronPtr.Dimensions)
				NeuronOut.NeuronPtr.ActivationCache += stren / distance //strength of 1 for the startin neurons
				n.ActivationStack = append(n.ActivationStack, outC.NeuronPtr)
			} else if outC.OutputPtr != nil {
				targetOutput := outC
				distance := CalculateDistance(outpu.Dimensions, targetOutput.OutputPtr.Dimensions)
				targetOutput.OutputPtr.Prediction += stren / distance
			}
		}
	}

	for len(n.ActivationStack) > 0 {
		neuron := n.DequeueActivationStack()
		if neuron != nil && neuron.PassportStamp < 50 {
			n.PropagateSignal(neuron, distanceLoss)
			neuron.PassportStamp += 1
		} else if neuron.PassportStamp >= 50 {
			neuronDeathCount += 1
		}
	}

	// Compute outputs and print results
	var sum float32 = 0.0
	for _, outputPtr := range n.Outputs {
		output := outputPtr
		/*for _, conn := range output.connectionsIn {
			neuron := conn
			output.prediction += neuron.neuronPtr.activationCache
		}*/
		//fmt.Printf("Output prediction in ST: %f\n", output.prediction)
		sum += output.Prediction
	}

	var neuronCounter = 0
	var highestChance float32 = 0
	var response = 0
	for _, outputPtr := range n.Outputs {
		prec := (outputPtr.Prediction / sum) * 100
		//fmt.Println("INFO ABOUT OUTPUT ID: ", neuronCounter)
		//fmt.Printf("Output prediction in PC: %f\n", prec)
		if prec > highestChance {
			response = neuronCounter
			highestChance = prec
		}
		outputPtr.Prediction = 0.0
		neuronCounter++
	}
	n.ClearCache()
	//println("neuron dead count: ", neuronDeathCount)
	neuronDeathCount = 0
	return response

}

func (n *Nucleus) propegateBack(correctID int) {
	//

	//we reuse the passport stamp var for this propegation
	for _, conOut := range n.Outputs {
		var familyTree []*Neuron = nil

		for _, x := range conOut.ConnectionsIn {
			familyTree = append(familyTree, x.NeuronPtr)
		}

		for id, conn := range familyTree {
			/*		if id > 3000 {
					return
				}*/
			var x *Neuron = conn
			for _, children := range conn.ConnectionsIn {
				if children.InputPtr == nil && children.NeuronPtr.PassportStamp < 50 && 20 > len(children.NeuronPtr.TrainingPossitiveCons) && 20 > len(children.NeuronPtr.TrainingNegativeCons) {
					familyTree = append(familyTree, children.NeuronPtr)
					if id == correctID {
						children.NeuronPtr.TrainingPossitiveCons = append(children.NeuronPtr.TrainingPossitiveCons, Connection{
							NeuronPtr: x,
							OutputPtr: nil,
							InputPtr:  nil,
						})
					} else {
						children.NeuronPtr.TrainingNegativeCons = append(children.NeuronPtr.TrainingNegativeCons, Connection{
							NeuronPtr: x,
							OutputPtr: nil,
							InputPtr:  nil,
						})
					}
				}
			}
			conn.PassportStamp++
			familyTree = append([]*Neuron(nil), familyTree[1:]...)

		}

	}
}

/* TODO make propegateback so it goes towards the desired neruons, then rewards the best family trees. If no connection it should create one.
func (n *Nucleus) propegateBackv2(desiredcorrrectID int, correctId int) {
	for id, outputs := range n.Outputs {
		var familyTree []*Neuron = nil
		//most important propegation
		if id == corrrectID {
			var desiredPos []float32 = n.Inputs[desiredcorrrectID].Dimensions
		}
	}
}*/

// Function will be called when the neurons cant reach the outputs anymore
func (n *Nucleus) moveRandomAll() {
	for _, neurons := range n.Cluster {
		for id, location := range neurons.Dimensions {
			neurons.Dimensions[id] = location + rand.Float32()
		}
	}
}

func (n *Neuron) trainSelf() {

	if n.TrainingNegativeCons != nil && n.TrainingPossitiveCons != nil {

		var averageCon []float32 = []float32{0, 0, 0}
		var averagePosCon []float32 = []float32{0, 0, 0} //hardcoded dimensions!!!!!!!!
		var averageNegCon []float32 = []float32{0, 0, 0}

		for _, posCons := range n.TrainingPossitiveCons {
			for id, x := range posCons.NeuronPtr.Dimensions {
				averagePosCon[id] += x
			}
		}

		for _, posCons := range n.TrainingPossitiveCons {
			for id, x := range posCons.NeuronPtr.Dimensions {
				averageNegCon[id] += x
			}
		}

		for id, dim := range averagePosCon {
			averagePosCon[id] = dim / float32(len(n.TrainingPossitiveCons))
		}

		for id, dim := range averageNegCon {
			averageNegCon[id] = dim / float32(len(n.TrainingNegativeCons))
		}

		for id := range len(averageCon) {
			averageCon[id] = averagePosCon[id] - averageNegCon[id]
		}

		for id, dir := range averageCon {
			if n.Dimensions[id]+dir/10 > 50.0 || n.Dimensions[id]+dir*0.2 < 0 {
				n.Dimensions[id] = n.Dimensions[id] + dir*0.2
			}
		}
	}
	// Promote positive connections
	if len(n.TrainingPossitiveCons) > 0 {
		n.Strength += rand.Float32() * 0.5 // boost for positive behavior
	}

	// Penalize negative connections
	if len(n.TrainingNegativeCons) > 0 {
		n.Strength -= rand.Float32() * 0.3 // reduce strength for negative behavior
	}

	// Apply more penalty for non-responses
	if len(n.TrainingPossitiveCons) == 0 && len(n.TrainingNegativeCons) == 0 {
		n.Strength -= rand.Float32() * 0.7 // stronger penalty for inactivity
	}

	// Tweak activation threshold to promote answers
	if len(n.TrainingPossitiveCons) > len(n.TrainingNegativeCons) {
		n.ActivationThreshold -= rand.Float32() * 0.1 // make it easier to activate
	} else {
		n.ActivationThreshold += rand.Float32() * 0.1 // make it harder to activate for wrong answers
	}

	// Ensure strength doesn't fall below a minimum threshold to avoid complete inactivity
	if n.Strength < 0.1 {
		n.Strength = 0.1
	}
}

func (n *Nucleus) startQandA(distanceLoss float32) {
	var question []byte = getQandA()
	var answer []byte = getQandA()
	var answerAI int = 0

	var response string = ""
	for id, char := range answer {
		answerAI = n.FireQuestion(id, question, distanceLoss)
		n.propegateBack(int(char)) //implement extra think neuron as input -> activate when answer is longer then question, relys on neurons holden cache
		response += string(answerAI)
		for _, neurons := range n.Cluster { //train neuron on data from the propegate back
			neurons.trainSelf()
			neurons.PassportStamp = 0 //clean the passport check up, the activation cache can stay the same for now.
		}
	}
	/*for _, char := range []byte(response) {
		if char == 0 {
			fmt.Println("upgrade", response)
			n.moveRandomAll()
		}
	}*/
	fmt.Println("IMPORTANT:::: ", []byte(response))
}

func (n *Nucleus) Start(epochs int, learningRate float32, distanceLoss float32) {
	for epoch := 0; epoch < epochs; epoch++ {
		runtime.GC()

		fmt.Printf("Epoch %d:\n", epoch)

		n.ClearCache()
		fmt.Println("CACHE:", n.ActivationStack)
		// Propagate signals across the network

		//q&a = training data
		for i := 0; i < 9; i++ {
			n.startQandA(distanceLoss)
		}

		n.ClearCache()

	}
}
