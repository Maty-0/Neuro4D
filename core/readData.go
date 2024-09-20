package core

import (
	"bufio"
	"log"
	"os"
)

var trainingLine int32 = 0
var traindata []string

func getQandA() []byte {
	file, err := os.Open("core/dataset.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// optionally, resize scanner's capacity for lines over 64K, see next example
	if trainingLine == 0 {
		for scanner.Scan() {
			traindata = append(traindata, scanner.Text())

		}
	}
	trainingLine++
	//fmt.Println(scanner.Text())

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	if len(traindata) > int(trainingLine) {
		return []byte(traindata[trainingLine])
	}
	return nil
}
