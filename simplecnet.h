#ifndef SIMPLECNET_H
#define SIMPLECNET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INT int
#define DOUBLE double
#define DVECTOR_1D double*
#define DVECTOR_2D double**



typedef struct NEURAL_NETWORK NEURAL_NETWORK;



/* NEURON */

struct NEURON {
	/*index for use in weighing matrix */
	INT _index;
	/*nettoInput: sum of inputs into neuron from previous layers*/
	DOUBLE _nettoInput;
	/*activation value*/
	DOUBLE _activation;
	/*output value*/
	DOUBLE _output;
	/*function pointer for activation function*/
	void (*activationFunction) (struct NEURON* neuron); 
	/*function pointer for output function (may be any output function other than "IDENTITY" only in output layer*/
	void (*outputFunction) (struct NEURON* neuron);
	/*function pointer to derivative of activation function (set by activation function)*/
	DOUBLE (*derivative)(NEURAL_NETWORK* network, INT i, INT j);
};

typedef struct NEURON NEURON;

NEURON* NEURON_INITIALIZE(INT index, DOUBLE nettoInput, void (*actf) (NEURON* neuron), void (*outf) (NEURON* neuron)) {
	/*function to initialize a neuron based on the values given as parameters*/
	/*returns pointer to neuron*/
	NEURON* neuron = malloc(sizeof(NEURON));
	if (neuron == NULL) {
		fprintf(stderr, "no memory for NEURON");
		return NULL;
	}
	neuron->_index = index;
	neuron->_nettoInput = nettoInput;
	neuron->activationFunction = actf;
	neuron->outputFunction = outf;
	return neuron;
}

/* END NEURON */




/* Training Pattern */

struct TRAINING_PATTERN {
	/*a training pattern consists of the input vector, the desired output (target) vector and the actual output*/
	/*the actual output is not set by the programmer and currently not used in any function, may be deleted*/
	DVECTOR_1D _inputVector;
	DVECTOR_1D _targetVector;
	DVECTOR_1D _actualOutput;
};

typedef struct TRAINING_PATTERN TRAINING_PATTERN;

TRAINING_PATTERN PATTERN_CREATE(DVECTOR_1D input, DVECTOR_1D target) {
	/*returns a pointer to a training pattern created from the given parameters input and target*/

	TRAINING_PATTERN* pattern = malloc(sizeof(TRAINING_PATTERN));

	pattern->_inputVector = malloc(sizeof(input));
	pattern->_targetVector = malloc(sizeof(target));
	pattern->_actualOutput = malloc(sizeof(target));
	pattern->_inputVector = input;
	pattern->_targetVector = target;

	return *pattern;
}

void TRAINING_PATTERN_FREE(TRAINING_PATTERN* pattern) {
	/*free(pattern->_inputVector);
	free(pattern->_targetVector);
	free(pattern->_actualOutput);*/
	free(pattern);
}

/* END Training Pattern */




/* Reverse DOUBLE** */

DVECTOR_2D reverseDVector_2d(DVECTOR_2D toreverse, int vecsize) {
	/*used in the training function to reverse a double** (DVECTOR_2D) */
	INT i;
	DVECTOR_2D reversed = malloc(vecsize * sizeof(DOUBLE));
	
	for (i = 0; i < vecsize; i++) {
		reversed[vecsize - 1 - i] = toreverse[i];
	}
	return reversed;
}

/* END Reverse DOUBLE** */




/* Derivatives of activation functions (DECLARATIONS)*/

DOUBLE SIGMOID_DERIV(NEURAL_NETWORK* network, INT i, INT j);

DOUBLE RELU_DERIV(NEURAL_NETWORK* network, INT i, INT j);

DOUBLE LEAKY_RELU_DERIV(NEURAL_NETWORK* network, INT i, INT j);

DOUBLE PIECEWISE_DERIV(NEURAL_NETWORK* network, INT i, INT j);

DOUBLE THRESHOLD_DERIV(NEURAL_NETWORK* network, INT i, INT j);

/* END Derivatives */




/* Activation functions*/

void SIGMOID(NEURON* neuron) {
	neuron->_activation = 1 / (1 + exp(-1 * neuron->_nettoInput));
	neuron->derivative = SIGMOID_DERIV;
}

void PIECEWISE_LINEAR(NEURON* neuron) {
	double netinput = neuron->_nettoInput;
	if (netinput >= -1 && netinput <= 1) neuron->_activation = netinput;
	else if (netinput < -1) neuron->_activation = -1.0;
	else if (netinput > 1) neuron->_activation = 1.0;
	neuron->derivative = PIECEWISE_DERIV;
}

void RELU(NEURON* neuron) {
	if (neuron->_nettoInput >= 0) {
		neuron->_activation = neuron->_nettoInput;
	}
	else {
		neuron->_activation = 0;
	}
	neuron->derivative = RELU_DERIV;
}

void LEAKY_RELU(NEURON* neuron) {
	if (neuron->_nettoInput >= 0) {
		neuron->_activation = neuron->_nettoInput;
	}
	else {
		neuron->_activation = 0.1 * neuron->_nettoInput;
	}
	neuron->derivative = LEAKY_RELU_DERIV;
}

/* END Activation functions*/




/* Output functions NOTE: no derivatives, use of output functions other than "IDENTITY" only useful in last layer*/

void THRESHOLD_OUT(NEURON* neuron) {
	if (neuron->_activation > 0.5) {
		neuron->_output = 1.0;
	}
	else {
		neuron->_output = 0.0;
	}
}

void SIGMOID_OUT(NEURON* neuron) {
	neuron->_output = 1 / (1 + exp(-1 * neuron->_activation));
}

void PIECEWISE_LINEAR_OUT(NEURON* neuron) {
	double activ = neuron->_nettoInput;
	if (activ >= -1 && activ <= 1) neuron->_output = activ;
	else if (activ < -1) neuron->_output = -1.0;
	else if (activ > 1) neuron->_output = 1.0;
}

void RELU_OUT(NEURON* neuron) {
	if (neuron->_activation >= 0) {
		neuron->_output = neuron->_activation;
	}
	else {
		neuron->_output = 0;
	}
}

void IDENTITY(NEURON* neuron) {
	neuron->_output = neuron->_activation;
}

/* END Output functions*/




/* Weighing Matrix*/

struct WEIGHING_MATRIX {
	INT _columns;
	INT _rows;
	DOUBLE **_matrix;
};

typedef struct WEIGHING_MATRIX WEIGHING_MATRIX;

WEIGHING_MATRIX* WEIGHING_MATRIX_INITIALIZE(INT columns, INT rows) {
	/*initializes a weighing matrix with columns and rows as set in parameters*/
	/*column and row numbers are always the same - one parameter would be enough actually*/

	INT i, j;
	WEIGHING_MATRIX* weMa = malloc(sizeof(WEIGHING_MATRIX));
	if (weMa == NULL) {
		fprintf(stderr, "no memory for Weighing Matrix");
		return NULL;
	}
	weMa->_columns = columns;
	weMa->_rows = rows;
	weMa->_matrix = malloc(rows*columns*sizeof(DOUBLE*));
	if (weMa->_matrix == NULL) {
		fprintf(stderr, "matrix allocation failed");
		free(weMa);
		return NULL;
	}
	for (i = 0; i < rows; i++) {
		weMa->_matrix[i] = malloc(columns*columns * sizeof(DOUBLE));
		if (weMa->_matrix[i] == NULL) {
			fprintf(stderr, "matrix allocation failed");
			free(weMa->_matrix);
			free(weMa);
			return NULL;
		}
		for (j = 0; j < columns; j++) {
			weMa->_matrix[i][j] = 0.0;
		}
	}
	return weMa;
}

void MATRIX_PRINTONCONSOLE(WEIGHING_MATRIX matrix) {
	/*helper function*/
	INT i, j;
	printf("\n");

	for (i = 0; i < matrix._rows; i++) {
		for (j = 0; j < matrix._columns; j++) {
			printf("%.3f ", matrix._matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

INT MATRIX_SAVETOFILE(WEIGHING_MATRIX matrix, const char* outfilename, const char* precision) {
	/*saves the weighing matrix to a text file*/
	INT i, j;
	FILE* outfile;

	outfile = fopen(outfilename, "w");
	if (outfile == NULL) {
		perror("output file error");
		return 0;
	}

	for (i = 0; i < matrix._rows; i++) {
		for (j = 0; j < matrix._columns; j++) {
			/*At %.Xf define needed precision in text file*/
			fprintf(outfile, precision , matrix._matrix[i][j]);
		}
		fprintf(outfile, "\n");
	}
	fclose(outfile);

	return 1;
}

INT MATRIX_READFROMFILE(WEIGHING_MATRIX matrix, const char* infilename) {
	INT i, j;
	FILE* infile;
	DOUBLE read;

	infile = fopen(infilename, "r");
	if (infile == NULL) {
		perror("input file error");
		return 0;
	}
	
	for (i = 0; i < matrix._rows; i++) {
		for (j = 0; j < matrix._columns; j++) {
			/*At %.Xf define needed precision in text file*/
			fscanf(infile, "%lf ", &read);
			matrix._matrix[i][j] = read;
		}
	}
	fclose(infile);
	return 1;
}

/*void WEIGHING_MATRIX_free(WEIGHING_MATRIX* weMa) {
	for (INT i = 0; i < weMa->_rows; i++) {
		free(weMa->_matrix[i]);
	}
	free(weMa->_matrix);
	free(weMa);
}*/

/* END Weighing Matrix*/




/* Neural Network*/

struct NEURAL_NETWORK {
	INT _numberOfLayers;
	INT* _neuronsPerLayer;
	NEURON** _neuronLayers;
	WEIGHING_MATRIX _weighingmatrix;
};

DVECTOR_1D GENERATE_OUTPUT(NEURAL_NETWORK* network, DVECTOR_1D inputVector) {
	/*generates the output for a given inputVector using forward propagation*/
	/*result can be read either by accessing the output values of the output layer of neurons*/
	/*or by retrieving the return value of this function*/

	INT i, j, a;
	INT countLayers = network->_numberOfLayers;

	INT countLastLayer = network->_neuronsPerLayer[countLayers-1];

	DVECTOR_1D output = malloc(countLastLayer*sizeof(DOUBLE));

	
	/*setting input, activation (using activation function) and output (using output function)
	  of the input layer*/
	for (i = 0; i < network->_neuronsPerLayer[0]; i++) {
		network->_neuronLayers[0][i]._nettoInput = inputVector[i];
		network->_neuronLayers[0][i].activationFunction(&network->_neuronLayers[0][i]);
		network->_neuronLayers[0][i].outputFunction(&network->_neuronLayers[0][i]);
	}

	
	/*doing the same for the following layers*/
	for (i = 1; i < countLayers; i++) {
		for (j = 0; j < network->_neuronsPerLayer[i]; j++) {
			
			/*calculating nettoInput from sums of output of the previous layer * the corresponding weighing matrix value
			  which is found using the index (_index) of each neuron*/
			network->_neuronLayers[i][j]._nettoInput = 0;
			for (a = 0; a < network->_neuronsPerLayer[i-1]; a++) {
				
				network->_neuronLayers[i][j]._nettoInput += network->_neuronLayers[i - 1][a]._output 
					* network->_weighingmatrix._matrix[network->_neuronLayers[i - 1][a]._index][network->_neuronLayers[i][j]._index];
				
			}
			
			network->_neuronLayers[i][j].activationFunction(&network->_neuronLayers[i][j]);
			network->_neuronLayers[i][j].outputFunction(&network->_neuronLayers[i][j]);
			
			/*output values of last layers make up returned vector*/
			if (i == countLayers - 1) {
				
				output[j] = network->_neuronLayers[i][j]._output;
			}
		}
	}

	return output;
}

NEURAL_NETWORK* GENERATE_NETWORK(INT numberOfLayers, INT* neuronsPerLayer, void (*actinner)(NEURON* neuron), void (*actoutput)(NEURON* neuron), void (*outinner)(NEURON* neuron), void (*outoutput)(NEURON* neuron)) {
	/*generates a neural network with a specified size, activation and output functions of inner and output layers have to be set*/
	
	NEURAL_NETWORK* network = malloc(sizeof(NEURAL_NETWORK));
	network->_neuronLayers = malloc(numberOfLayers * numberOfLayers * sizeof(NEURON*));

	network->_numberOfLayers = numberOfLayers;
	network->_neuronsPerLayer = neuronsPerLayer;

	INT a, b, c, x, y;
	INT layer, pl;
	INT counthier;
	INT x2, y2;

	INT index = 0;

	/*seed random number generator*/
	srand(time(NULL));

	for (layer = 0; layer < numberOfLayers; layer++) {
		network->_neuronLayers[layer] = malloc(neuronsPerLayer[layer]*neuronsPerLayer[layer] * sizeof(NEURON));

		for (pl = 0; pl < neuronsPerLayer[layer]; pl++) {
			if (layer == numberOfLayers - 1) {
				/*inner neuron*/
				network->_neuronLayers[layer][pl] = * NEURON_INITIALIZE(index++, 0, actoutput, outoutput);
			}
			else {
				/*output layer neuron*/
				network->_neuronLayers[layer][pl] = *NEURON_INITIALIZE(index++, 0, actinner, outinner);
			}
		}
	}

	/*initialize weighing matrix*/
	network->_weighingmatrix = *WEIGHING_MATRIX_INITIALIZE(index, index);

	/*set weighing matrix value to 1.0 where there is a connection between the neurons*/
	/*this is done using the indexes*/
	for (a = 0; a < (numberOfLayers - 1); a++) {
		for (b = 0; b < neuronsPerLayer[a]; b++) {
			for (c = 0; c < neuronsPerLayer[a + 1]; c++) {
				x = network->_neuronLayers[a][b]._index;
				y = network->_neuronLayers[a + 1][c]._index;
				network->_weighingmatrix._matrix[x][y] = 1.0;
			}
		}
	}

	/*this step could be combined with the last one:*/
	/*setting the same matrix values to a random number*/ 
	counthier = 1;
	for (x2 = 0; x2 < index-neuronsPerLayer[numberOfLayers-1]; x2++) {
		for (y2 = 0; y2 < index; y2++) {
			if (network->_weighingmatrix._matrix[x2][y2] == 1.0) {

				/*between -1 and 1*/
				/*DOUBLE random = (DOUBLE)rand() / RAND_MAX * 2.0 - 1.0;*/

				/*between 0 and 1*/
				DOUBLE random = (DOUBLE)rand() / RAND_MAX;

				network->_weighingmatrix._matrix[x2][y2] = random;
			}
		}
	}

	return network;
}

void NEURAL_NETWORK_FREE(NEURAL_NETWORK* network) {
	free(network->_neuronLayers);
	free(network->_weighingmatrix._matrix);
	free(network);
}

INT TRAIN_NETWORK(NEURAL_NETWORK * network, INT numberOfRepetitions, DOUBLE learningRate, DOUBLE tolerance, INT numberOfTP, TRAINING_PATTERN * trainingPatterns) {
	/*training the network using backpropagation*/
	INT rep,tp,out,i,j,z,d;

	INT count = 0;

	INT trainingneeded = 0;

	for (rep = 0; rep < numberOfRepetitions; rep++) {

		count++;
		trainingneeded = 0;

		/*stepping through all the training patterns in each repetition*/
		for (tp = 0; tp < numberOfTP; tp++) {
			
			/*two dimensional array to keep track of the deltas in the net*/
			DVECTOR_2D netDeltas = malloc(network->_numberOfLayers * network->_numberOfLayers * network->_numberOfLayers * sizeof(DOUBLE*));
			INT netDeltaCount = 0;

			/*generate the output to the pattern's input vector*/
			DVECTOR_1D output = GENERATE_OUTPUT(network, trainingPatterns[tp]._inputVector);

			/*out_s: "index" of last layer*/
			int out_s = network->_numberOfLayers - 1;

			/*out_n: number of output neurons*/
			int out_n = network->_neuronsPerLayer[out_s];


			DVECTOR_1D deltas = malloc(out_n * out_n * sizeof(DOUBLE));
			INT deltaCount = 0;

			for (out = 0; out < out_n; out++) {

				/*using the activation function's derivative*/
				DOUBLE deriv = network->_neuronLayers[out_s][out].derivative(network, out_s, out);

				/*to calculate each delta*/
				DOUBLE this_delta = deriv*(trainingPatterns[tp]._targetVector[out] - network->_neuronLayers[out_s][out]._activation);

				/*which is then added to the one dimensional delta vector*/
				deltas[deltaCount++] = this_delta;

				if (fabs(this_delta) > tolerance) {
					/*trainingneeded is set to 1 each time a delta is too big*/
					trainingneeded = 1;
				}
				else {
				}
			}
			/*which in turn is added to the twodimensional netDelta array*/
			netDeltas[netDeltaCount++] = deltas;


			/*trainingneeded = 1 means training is needed*/
			if (trainingneeded == 1) {

				/*p: predecessor*/
				INT p = 0;


				for (i = network->_numberOfLayers - 2; i > 0; i--) {

					DVECTOR_1D deltas_i = malloc(100 * network->_numberOfLayers * network->_numberOfLayers * sizeof(DOUBLE));
					INT icount = 0;

					for (j = 0; j < network->_numberOfLayers; j++) {

						/*using the acivation function's derivative*/
						DOUBLE deriv = network->_neuronLayers[i][j].derivative(network, i, j);

						DOUBLE sum = 0.0;
						int row = network->_neuronLayers[i][j]._index;

						for (z = 0; z < network->_neuronsPerLayer[i + 1]; z++) {

							int column = network->_neuronLayers[i + 1][z]._index;
							sum += netDeltas[p][z] * network->_weighingmatrix._matrix[row][column];

						}
						deltas_i[icount++] = deriv * sum;

					}
					netDeltas[netDeltaCount++] = deltas_i;
					p++;

				}

				netDeltas = reverseDVector_2d(netDeltas, netDeltaCount);


				/*the actual backpropagation*/
				for (i = network->_numberOfLayers - 1; i > 0; i--) {

					for (j = 0; j < network->_neuronsPerLayer[i]; j++) {

						DOUBLE delta = netDeltas[(i - 1)][j];
						INT column = network->_neuronLayers[i][j]._index;

						for (z = 0; z < network->_neuronsPerLayer[i - 1]; z++) {

							int row = network->_neuronLayers[i - 1][z]._index;
							DOUBLE weightDelta = learningRate * delta * network->_neuronLayers[i - 1][z]._activation;
							network->_weighingmatrix._matrix[row][column] += weightDelta;

						}
					}
				}

			}

			for (d = 0; d < netDeltaCount; d++) {
				free(netDeltas[d]);
			}
			free(netDeltas);
		}
		if (trainingneeded == 0) {
			break;
		}
	}
	if (trainingneeded == 0) {
		return count;
	}
	else {
		return -numberOfRepetitions;
	}
}

/* END Neural Network*/




/* Derivatives of Activation Functions (DEFINITIONS)*/

DOUBLE SIGMOID_DERIV(NEURAL_NETWORK* network, INT i, INT j) {
	return network->_neuronLayers[i][j]._activation * (1 - network->_neuronLayers[i][j]._activation);
}

DOUBLE RELU_DERIV(NEURAL_NETWORK* network, INT i, INT j) {
	if (network->_neuronLayers[i][j]._activation > 0) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

DOUBLE LEAKY_RELU_DERIV(NEURAL_NETWORK* network, INT i, INT j) {
	if (network->_neuronLayers[i][j]._activation > 0) {
		return 1.0;
	}
	else {
		return 0.1;
	}
}

DOUBLE PIECEWISE_DERIV(NEURAL_NETWORK* network, INT i, INT j) {
	if ((network->_neuronLayers[i][j]._activation <= 1) && (network->_neuronLayers[i][j]._activation >= -1)) {
		return 1.0;
	}
	else {
		return 0;
	}
}

/* END Derivatives*/

#endif
