#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <direct.h>

#include "SimpleMath.h"
#include "cuda_smart_ptr.h"


__device__ float logistic(float signal) {
	return 1.0f - 2.0f * signal * signal;
}

/**
 * ������ ��������������� ����� �� ��������������� ������������,
 * ��������� � ������� ������� �� �����, ������ ����������� �������������.
 *
 * @param k
 * @param weightMatrix
 * @param neuronInput
 * @param neuronOutput
 *
 * ���������� ������ = ���-�� �������� � ��������� ����.
 */

__global__ void calcDynamicsOneThread(
	float *weightMatrix,
	float *neuronInput,
	float *output,
	int nNeurons
) {
	int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (neuronIdx >= nNeurons) {
		return;
	}

	float sum = 0.0f;
	float norm = 0.0f;
	
	for (int other = 0; other < nNeurons; other++) {
		// ��� ������ ������ ���� ������ ������ + L1 ���
		float prev = neuronInput[other];
		// ����� ��, ��������� ���������������� ������
		float w = weightMatrix[other * nNeurons + neuronIdx];
		norm += w;
		sum  += prev * w;
	}
	output[neuronIdx] = logistic((1.0f / norm) * sum);
}


// ������ ������� ������, � �� ���� �� ���������
#define SHARED_BLOCK_DIM_X 512

__global__ void calcDynamicsShared(
	const float *weightMatrix,
	const float *neuronInput,
	float *output,
	const int nNeurons
) {
	int neuronIdx = blockIdx.x;

	if (neuronIdx >= nNeurons) {
		return;
	}

	float sum = 0.0f;
	float norm = 0.0f;

	const float *wm = weightMatrix + neuronIdx * nNeurons;

	int shift = threadIdx.x;
	#pragma unroll
	for (int i = 0; i < nNeurons; i += SHARED_BLOCK_DIM_X) {
		int secondNeuron = i + shift;
		if (secondNeuron < nNeurons) {
			// ������ ���������������
			float prev = neuronInput[secondNeuron];
			float w = wm[secondNeuron];

			norm += w;
			sum  += prev * w;
		}
	}

	__shared__ float sums[SHARED_BLOCK_DIM_X];
	__shared__ float norms[SHARED_BLOCK_DIM_X];

	sums[threadIdx.x] = sum;
	norms[threadIdx.x] = norm;

	__syncthreads();
 

	// SHARED_BLOCK_DIM_X ������ ���� �������� 2.
	for (int stride = SHARED_BLOCK_DIM_X >> 1; stride >= 1; stride = stride >> 1) {
		if (threadIdx.x < stride) {
			sums[threadIdx.x]  +=  sums[threadIdx.x + stride];
			norms[threadIdx.x] += norms[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		norm = norms[0];
		sum = sums[0];
		output[neuronIdx] = logistic((1.0f / norm) * sum);
	}
}

#define DIVISOR 32

__global__ void calcDynamicsShared4(
	const float *weightMatrix,
	const float *neuronInput,
	float *output,
	const int nNeurons
) {
	const int NEURON_ZONE = SHARED_BLOCK_DIM_X / DIVISOR;
	const int INTRO_NEURON = threadIdx.x / NEURON_ZONE;
	int neuronIdx = blockIdx.x * DIVISOR + INTRO_NEURON;

	__shared__ float sums[DIVISOR][NEURON_ZONE+1];
	__shared__ float norms[DIVISOR][NEURON_ZONE+1];

	float sum = 0.0f;
	float norm = 0.0f;

	const float *wm = weightMatrix + neuronIdx * nNeurons;

	int SHIFT = threadIdx.x % NEURON_ZONE;
	
	for (int i = 0; i < nNeurons; i += NEURON_ZONE) {
		int secondNeuron = i + SHIFT;
		if (secondNeuron < nNeurons) {
			// ������ ���������������
			float prev = neuronInput[secondNeuron];
			float w = wm[secondNeuron];

			norm += w;
			sum  += prev * w;
		}
	}

	sums[INTRO_NEURON][SHIFT] = sum;
	norms[INTRO_NEURON][SHIFT] = norm;

	__syncthreads();
 
	// SHARED_BLOCK_DIM_X ������ ���� �������� 2.
	for (int stride = NEURON_ZONE >> 1; stride >= 1; stride = stride >> 1) {
		if (SHIFT < stride) {
			sums[INTRO_NEURON][SHIFT]  +=  sums[INTRO_NEURON][SHIFT + stride];
			norms[INTRO_NEURON][SHIFT] += norms[INTRO_NEURON][SHIFT + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x < DIVISOR) {
		neuronIdx = threadIdx.x;
		if (neuronIdx < nNeurons) {
			norm = norms[neuronIdx][0];
			sum = sums[neuronIdx][0];
			output[neuronIdx] = logistic((1.0f / norm) * sum);
		}
	}
}


/*
������� ������������� (���������� �����).

������������� ����� ��������� �� 1 ����� ��������� ��� ����� ������ �������� �� �������, 
����� � �������� �� ������.
*/
__global__ void phaseSyncCheckInplace(int *currGT, int nSteps, int *hits, int nNeurons) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	int count = 0;
	for (int step = 0; step < nSteps; step++) {
		int first = currGT[step * nNeurons + neuronIdxFirst];
		int second = currGT[step * nNeurons + neuronIdxSecond];
		if (first == second) {
			count++;
		}
	}
	hits[neuronIdxFirst * nNeurons + neuronIdxSecond] += count;
}


/**
���������� �������, ��� ����� ���������� ��� ���������� ������ ������� � ����� ��������������� ��������.
*/
__global__ void prepareToSynchronizationCheck(float *prevOutput, float *currOutput, int *gt, float *bufferedValues, int nNeurons) {
	int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (neuronIdx >= nNeurons) {
		return;
	}

	bool value = currOutput[neuronIdx] > prevOutput[neuronIdx];
	int result = 0;
	if (value) {
		result = 1;
	}
	gt[neuronIdx] = result;
	bufferedValues[neuronIdx] = currOutput[neuronIdx];
}


/**
����������� ��� �� ����� ����, ��� � � ������� �������������� - ��������� 
��������� ���������� �������, ��-�� ����� ���������� � ����.
*/
__global__ void fragmentaryAnalysis(float *currOutput, int nSteps, int *hits, int nNeurons, const float eps) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

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

__global__ void zeroInts(int *ar, int sizeX, int sizeY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= sizeX || y >= sizeY) {
		return;
	}
	ar[y * sizeX + x] = 0;
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

std::vector<int> processOscillatoryChaoticNetworkDynamics(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
	std::vector<float> &sheet,
	const float fragmentaryEPS,
	bool useSingleThreadPerNeuron
) {
	BEGIN_FUNCTION {
		//check(nIterations > 0);
		check(nNeurons > 0);
		check(startObservationTime >= 0);

		DeviceScopedPtr1D<float> weightMatrixDevice(nNeurons * nNeurons);
		check(weightMatrixHost.size() == nNeurons * nNeurons);
		weightMatrixDevice.copyFromHost(&weightMatrixHost[0], weightMatrixHost.size());

		DeviceScopedPtr1D<float> input(nNeurons);
		DeviceScopedPtr1D<float> output(nNeurons);

		std::vector<float> stateHost(nNeurons);
		::randomSetHost(stateHost);

		input.copyFromHost(&stateHost[0], nNeurons);
		output.copyFromHost(&stateHost[0], nNeurons);
		
		float *currInputPtr = input.getDevPtr();
		float *currOutputPtr = output.getDevPtr();

		for (int i = 0; i < startObservationTime; i++) {
			if (useSingleThreadPerNeuron) {
				dim3 blockDim(256);
				dim3 gridDim(divRoundUp(nNeurons, blockDim.x));

				checkKernelRun((
					calcDynamicsOneThread<<<gridDim, blockDim>>>(
						weightMatrixDevice.getDevPtr(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			} else {
/*
				dim3 calcBlockDim(SHARED_BLOCK_DIM_X);
				dim3 calcGridDim(nNeurons);
				checkKernelRun((
					calcDynamicsShared<<<calcGridDim, calcBlockDim>>>(
						weightMatrixDevice.getDevPtr(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
*/
				dim3 calcBlockDim4(SHARED_BLOCK_DIM_X);
				dim3 calcGridDim4(divRoundUp(nNeurons, DIVISOR));
				checkKernelRun((
					calcDynamicsShared4<<<calcGridDim4, calcBlockDim4>>>(
						weightMatrixDevice.getDevPtr(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}
			std::swap(currInputPtr, currOutputPtr);
		}

		DeviceScopedPtr1D<int> hits(nNeurons * nNeurons);
		{
			dim3 blockDimFill(32, 8);
			dim3 gridDimFill(divRoundUp(nNeurons, blockDimFill.x), divRoundUp(nNeurons, blockDimFill.y));
			checkKernelRun((
				zeroInts<<<gridDimFill, blockDimFill>>>(
					hits.getDevPtr(),
					nNeurons,
					nNeurons
				)
			));
		}

		DeviceScopedPtr1D<int> currentHits(nNeurons);
		std::vector<int> currentHitsHost(nNeurons);

		const int N_STEPS = 64;
		DeviceScopedPtr1D<int> gt(nNeurons * N_STEPS);
		DeviceScopedPtr1D<float> bufferedValues(nNeurons * N_STEPS);

		sheet.resize(nIterations * nNeurons);

		int currentStep = 0;
		for (int i = 0; i < nIterations; i++) {
			// fragmentary synchronization if needed
			if (syncType == FRAGMENTARY && currentStep == N_STEPS)  {
				dim3 blockCheck(32, 8);
				dim3 gridCheck(divRoundUp(nNeurons, blockCheck.x), divRoundUp(nNeurons, blockCheck.y));

				checkKernelRun((
					fragmentaryAnalysis<<<gridCheck, blockCheck>>>(
						bufferedValues.getDevPtr(),
						N_STEPS,
						hits.getDevPtr(),
						nNeurons,
						fragmentaryEPS
					)
				));
				
				currentStep = 0;
			}

			// computing
			if (useSingleThreadPerNeuron) {
				dim3 calcBlockDim(256);
				dim3 calcGridDim(divRoundUp(nNeurons, calcBlockDim.x));
				checkKernelRun((
					calcDynamicsOneThread<<<calcGridDim, calcBlockDim>>>(
						weightMatrixDevice.getDevPtr(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			} else {
/*
				dim3 calcBlockDim(SHARED_BLOCK_DIM_X);
				dim3 calcGridDim(nNeurons);
				checkKernelRun((
					calcDynamicsShared<<<calcGridDim, calcBlockDim>>>(
						weightMatrixDevice.getDevPtr(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
*/
				dim3 calcBlockDim4(SHARED_BLOCK_DIM_X);
				dim3 calcGridDim4(divRoundUp(nNeurons, DIVISOR));
				checkKernelRun((
					calcDynamicsShared4<<<calcGridDim4, calcBlockDim4>>>(
						weightMatrixDevice.getDevPtr(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}
		
			{
				dim3 blockDim(128);
				dim3 gridDim(divRoundUp(nNeurons, blockDim.x));
				checkKernelRun((
					prepareToSynchronizationCheck<<<gridDim, blockDim>>>(
						currInputPtr,
						currOutputPtr,
						gt.getDevPtr() + nNeurons * currentStep,
						bufferedValues.getDevPtr() + nNeurons * currentStep,
						nNeurons
					)
				));
				currentStep++;
			}

			// phase synchronization if needed
			if (syncType == PHASE) {
				
				if (currentStep == N_STEPS) {
					dim3 blockCheck(32, 8);
					dim3 gridCheck(divRoundUp(nNeurons, blockCheck.x), divRoundUp(nNeurons, blockCheck.y));
					checkKernelRun((
						phaseSyncCheckInplace<<<gridCheck, blockCheck>>>(
							gt.getDevPtr(),
							N_STEPS,
							hits.getDevPtr(),
							nNeurons
						)
					));
					currentStep = 0;
				}
			}

			// copying neuron's outputs to host for sheets
			if (output.getDevPtr() == currOutputPtr) {
				output.copyToHost(&sheet[i * nNeurons], nNeurons);
			} else {
				input.copyToHost(&sheet[i * nNeurons], nNeurons);
			}
			// swapping pointers of input/output
			std::swap(currInputPtr, currOutputPtr);
		}

		// for phase synchronization if needed to procede remained
		if (currentStep != 0) {
			if (syncType == PHASE) {
				dim3 blockCheck(32, 8);
				dim3 gridCheck(divRoundUp(nNeurons, blockCheck.x), divRoundUp(nNeurons, blockCheck.y));
				checkKernelRun((
					phaseSyncCheckInplace<<<gridCheck, blockCheck>>>(
						gt.getDevPtr(),
						currentStep,
						hits.getDevPtr(),
						nNeurons
					)
				));
				currentStep = 0;
			} else if (syncType == FRAGMENTARY) {
				dim3 fragBlockDim(128);
				dim3 fragGridDim(divRoundUp(nNeurons, fragBlockDim.x));
				
				checkKernelRun((
					fragmentaryAnalysis<<<fragGridDim, fragBlockDim>>>(
						currInputPtr,
						currentStep,
						hits.getDevPtr(),
						nNeurons,
						fragmentaryEPS
					)
				));
			} else {
				throw std::logic_error("unknown sync type");
			}
		}
		std::vector<int> hitsHost(nNeurons * nNeurons);
		hits.copyToHost(&hitsHost[0], nNeurons * nNeurons);
		return hitsHost;
	} END_FUNCTION
}
