#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <direct.h>

#include "SimpleMath.h"


#include "timer.h"

#include <omp.h>

namespace cpu {

inline float logistic(float signal) {
	return 1.0f - 2.0f * signal * signal;
}


void calcDynamicsOneThread(
	const std::vector<float> &weightMatrix,
	const float *neuronInput,
	float *output,
	const int nNeurons
) {
	#pragma omp parallel for
	for (int neuronIdx = 0; neuronIdx < nNeurons; neuronIdx++) {
		float sum = 0.0f;
		float norm = 0.0f;
	
		for (int other = 0; other < nNeurons; other++) {
			float prev = neuronInput[other];
			float w = weightMatrix[neuronIdx * nNeurons + other];
			norm += w;
			sum  += prev * w;
		}
		output[neuronIdx] = logistic((1.0f / norm) * sum);
	}
}



void phaseSyncCheckInplace(int *currGT, int nSteps, int *hits, int nNeurons) {
	for (int step = 0; step < nSteps; step++) {
		const int shift = step * nNeurons;
		#pragma omp parallel for
		for (int neuronIdxFirst = 0; neuronIdxFirst < nNeurons; neuronIdxFirst++) {
			int first = currGT[shift + neuronIdxFirst];
			for (int neuronIdxSecond = 0; neuronIdxSecond < nNeurons; neuronIdxSecond++) {
				int second = currGT[shift + neuronIdxSecond];
				if (first == second) {
					hits[neuronIdxFirst * nNeurons + neuronIdxSecond]++;
				}
			}
		}
	}
}


/**
Простейшая функция, для учета увеличения или уменьшения выхода нейрона а также буферизирования значений.
*/
void prepareToSynchronizationCheck(
	const float *prevOutput, 
	const float *currOutput, 
	int *gt, 
	float *bufferedValues, 
	const int nNeurons
) {
	for (int neuronIdx = 0; neuronIdx < nNeurons; neuronIdx++) {
		bool value = currOutput[neuronIdx] > prevOutput[neuronIdx];
		int result = 0;
		if (value) {
			result = 1;
		}
		gt[neuronIdx] = result;
		bufferedValues[neuronIdx] = currOutput[neuronIdx];
	}
}

inline float fabsf(float val) {
	if (val < 0) {
		val = -val;
	}
	return val;
}

void fragmentaryAnalysis(float *currOutput, int nSteps, int *hits, int nNeurons, const float eps) {
	#pragma omp parallel for
	for (int neuronIdxFirst = 0; neuronIdxFirst < nNeurons; neuronIdxFirst++) {
		for (int neuronIdxSecond = 0; neuronIdxSecond < nNeurons; neuronIdxSecond++) {
			int count = 0;
			for (int step = 0; step < nSteps; step++) {
				float first = currOutput[step * nNeurons + neuronIdxFirst];
				float second = currOutput[step * nNeurons + neuronIdxSecond];
				float diff = fabsf(first - second);
				if (diff < eps) {
					count++;
				}
			}
			hits[neuronIdxFirst * nNeurons + neuronIdxSecond] += count;
		}
	}
}

inline int divRoundUp(int a, int b) {
	return (a - 1) / b + 1;
}

void randomSetHost(std::vector<float> &vals) {
	srand(23);
	for (int i = 0; i < static_cast<int>(vals.size()); i++) {
		vals[i] = (250 - rand() % 500) / 250.0f;
	}
}

void debugPrintArray(std::vector<float> &vals) {
	for (int j = 0; j < static_cast<int>(vals.size()); j++) {
		printf("%5.4f ", vals[j]);
	}
	printf("\r\n");
}

} // namespace cpu
/*
std::vector<int> processOscillatoryChaoticNetworkDynamics(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
	std::vector<float> &sheet,
	const float fragmentaryEPS,
	bool v
)
{
	return std::vector<int>(10);
}

*/

std::vector<int> processOscillatoryChaoticNetworkDynamicsCPU(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
	std::vector<float> &sheet,
	const float fragmentaryEPS
) {
	BEGIN_FUNCTION {
		check(nIterations > 0);
		check(nNeurons > 0);
		check(startObservationTime >= 0);

		check(weightMatrixHost.size() == nNeurons * nNeurons);

		std::vector<float> input(nNeurons);
		std::vector<float> output(nNeurons);
		
		cpu::randomSetHost(input);
		
		float *currInputPtr = &input[0];
		float *currOutputPtr = &output[0];

		for (int i = 0; i < startObservationTime; i++) {
			cpu::calcDynamicsOneThread(
				weightMatrixHost,
				currInputPtr,
				currOutputPtr,
				nNeurons
			);
			std::swap(currInputPtr, currOutputPtr);
		}

		std::vector<int> hits(nNeurons * nNeurons, 0);

		const int N_STEPS = 64;	
		std::vector<int> gt(nNeurons * N_STEPS, 0);
		std::vector<float> bufferedValues(nNeurons * N_STEPS, 0.0f);

		sheet.resize(nIterations * nNeurons);

		int currentStep = 0;
		for (int i = 0; i < nIterations; i++) {
			DWORD startIteration = GetTickCount();
			// fragmentary synchronization if needed
			if (syncType == FRAGMENTARY && currentStep == N_STEPS)  {
				cpu::fragmentaryAnalysis(
					&bufferedValues[0],
					N_STEPS,
					&hits[0],
					nNeurons,
					fragmentaryEPS
				);
				currentStep = 0;
			}

			// computing
			cpu::calcDynamicsOneThread(
				weightMatrixHost,
				currInputPtr,
				currOutputPtr,
				nNeurons
			);
			
			cpu::prepareToSynchronizationCheck(
				currInputPtr,
				currOutputPtr,
				&gt[0] + nNeurons * currentStep,
				&bufferedValues[0] + nNeurons * currentStep,
				nNeurons
			);
			currentStep++;

			// phase synchronization if needed
			if (syncType == PHASE) {
				if (currentStep == N_STEPS) {
					cpu::phaseSyncCheckInplace(
						&gt[0],
						N_STEPS,
						&hits[0],
						nNeurons
					);
					currentStep = 0;
				}
			}

			// copying neuron's outputs to host for sheets
			for (int j = 0; j < nNeurons; j++) {
				sheet[i * nNeurons + j] = currOutputPtr[j];
			}
			
			// swapping pointers of input/output
			std::swap(currInputPtr, currOutputPtr);
			DWORD finishIteration = GetTickCount();
		//	printf("Iteration time = %.3f\r\n", (finishIteration - startIteration) * 0.001f);
		}

		// for phase synchronization if needed to procede remained
		if (currentStep != 0) {
			if (syncType == PHASE) {
				cpu::phaseSyncCheckInplace(
					&gt[0],
					currentStep,
					&hits[0],
					nNeurons
				);
				currentStep = 0;
			} else if (syncType == FRAGMENTARY) {
				cpu::fragmentaryAnalysis(
					currInputPtr,
					currentStep,
					&hits[0],
					nNeurons,
					fragmentaryEPS
				);
			} else {
				throw std::logic_error("unknown sync type");
			}
		}
		return hits;
	} END_FUNCTION
}
