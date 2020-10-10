# simpleCnet
A simple single-header header-only "library" for neural networks written in C (C89). 
Training is done using backpropagation.

A simple network with 3 layers of neurons with 8 input and 4 output neurons, sigmoid activation functions in each layer
and an output of either 0 or 1 in the output neurons (THRESHOLD_OUT as output funcion in the output layer) is created like this:

int neuronsPerLayer[3]= { 8,6,4 };
	network = GENERATE_NETWORK(3, neuronsPerLayer, SIGMOID, SIGMOID, IDENTITY, THRESHOLD_OUT);
  
 A simple number of training patterns can then be specified:
 
double in1[8] = { 0,0,0,0,0,0,0,1 };
	double in2[8] = { 0,0,0,0,0,0,1,1 };
	double in3[8] = { 0,0,0,0,0,1,1,1 };
	double in4[8] = { 0,0,0,0,1,1,1,1 };
	double in5[8] = { 0,0,0,1,1,1,1,1 };
	double in6[8] = { 0,0,1,1,1,1,1,1 };
	double in7[8] = { 0,1,1,1,1,1,1,1 };
	double in8[8] = { 1,1,1,1,1,1,1,1 };

	double out1[4] = { 0,0,0,1 };
	double out2[4] = { 0,0,1,0 };
	double out3[4] = { 0,0,1,1 };
	double out4[4] = { 0,1,0,0 };
	double out5[4] = { 0,1,0,1 };
	double out6[4] = { 0,1,1,0 };
	double out7[4] = { 0,1,1,1 };
	double out8[4] = { 1,0,0,0 };

	TRAINING_PATTERN patterns[8] = {
		PATTERN_CREATE(in1, out1),
		PATTERN_CREATE(in2, out2),
		PATTERN_CREATE(in3, out3),
		PATTERN_CREATE(in4, out4),
		PATTERN_CREATE(in5, out5),
		PATTERN_CREATE(in6, out6),
		PATTERN_CREATE(in7, out7),
		PATTERN_CREATE(in8, out8)
	};
  
  Using these training patterns the network is trained as follows:
  
  int count = TRAIN_NETWORK(network, 1000000, 0.6, 0.001, 8, patterns);
  
  where "count" in case of successful training returns the number of repetitions used to train the network or -1000000 (the number of max repetitions) otherwise.
  
  With
  
  double* outtest;
  outtest = GENERATE_OUTPUT(network, in1);
	printf(" %.0f %.0f %.0f %.0f\n ", outtest[0], outtest[1], outtest[2], outtest[3]);
  
  it can be checked if the network now recognizes the training samples correctly.
  
  (Note: this example is not a real use-case of a neural network, but it can be used to test and see how it performs)
  
 
 
