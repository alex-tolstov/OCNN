#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <direct.h>

#include "SimpleMath.h"
#include "cuda_smart_ptr.h"
#include "timer.h"


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
	int matrixPitchElements,
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
		float prev = logistic(neuronInput[other]);
		// ����� ��, ��������� ���������������� ������
		float w = weightMatrix[other * matrixPitchElements + neuronIdx];
		norm += w;
		sum  += prev * w;
	}
	output[neuronIdx] = ((1.0f / norm) * sum);
}

__global__ void calcDynamicsOneThreadShared(
	float *weightMatrix,
	int matrixPitchElements,
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
	
	__shared__ float buf[256];

	for (int offset = 0; offset < nNeurons; offset += 256) {
		if (offset + threadIdx.x < nNeurons) {
			buf[threadIdx.x] = logistic(neuronInput[offset + threadIdx.x]);
		}
		__syncthreads();
		int after = min(nNeurons, offset + 256);
		for (int other = offset; other < after; other++) {
			float prev = buf[other - offset];
			float w = weightMatrix[other * matrixPitchElements + neuronIdx];
			norm += w;
			sum  += prev * w;
		}
	}
	output[neuronIdx] = ((1.0f / norm) * sum);
}

#define N_OUTPUTS 16
#define NEURON_ZONE 16

__global__ void calcDynamicsShared4Shared(
	const float *weightMatrix,
	int matrixPitchElements,
	const float *neuronInput,
	float *output,
	const int nNeurons
) {
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	int neuronIdx = blockIdx.x * N_OUTPUTS + ty;

	__shared__ float sums[N_OUTPUTS][NEURON_ZONE+1];
	__shared__ float norms[N_OUTPUTS][NEURON_ZONE+1];

	float sum = 0.0f;
	float norm = 0.0f;

	for (int offset = 0; offset < nNeurons; offset += NEURON_ZONE * N_OUTPUTS) {
		__shared__ float buf[NEURON_ZONE * N_OUTPUTS];
		int idx = ty * NEURON_ZONE + tx;
		if (offset + idx < nNeurons) {
			buf[idx] = logistic(neuronInput[offset + idx]);
		}
		__syncthreads();
		
		for (int i = 0; i < N_OUTPUTS; i++) {
			int internalIdx = i * NEURON_ZONE + tx;
			int other = offset + internalIdx;
			if (other < nNeurons) {
				float vectorValue = buf[internalIdx];
				float matrixValue = weightMatrix[neuronIdx * matrixPitchElements + other];
				norm += matrixValue;
				sum  += vectorValue * matrixValue;
			}
		}
	}

	sums[ty][tx] = sum;
	norms[ty][tx] = norm;

	__syncthreads();
 
	// NEURON_ZONE ������ ���� �������� 2.
	for (int stride = NEURON_ZONE >> 1; stride >= 1; stride = stride >> 1) {
		if (tx < stride) {
			sums[ty][tx] += sums[ty][tx + stride];
			norms[ty][tx] += norms[ty][tx + stride];
		}
		__syncthreads();
	}

	if (ty == 0 && tx < N_OUTPUTS) {
		neuronIdx = blockIdx.x * N_OUTPUTS + tx;
		norm = norms[tx][0];
		sum = sums[tx][0];
		if (neuronIdx < nNeurons) {
			output[neuronIdx] = ((1.0f / norm) * sum);
		}
	}
}


// ��������� ���������� � ������ 32x16, �� �������� ������� � 32x8, � 32x32.
// ���� (nNeurons + 31 / 32, 1)

#define FM_X 32
#define FM_Y 16

__global__ void calcDynamicsFm1(
   float *weightMatrix,
   int matrixPitchElements,
   float *neuronInput,
   float *output,
   int nNeurons
) {
	int neuronX = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float sums[FM_X];
	__shared__ float norms[FM_X];
	if (threadIdx.y == 0) {
		sums[threadIdx.x] = 0.0f;
		norms[threadIdx.x] = 0.0f;
	}
	float sum = 0.0f;
	float norm = 0.0f;
	if (neuronX < nNeurons) {
		int partY = (nNeurons - 1) / FM_Y + 1;
		int startY = threadIdx.y * partY;
		int afterY = min(nNeurons, startY + partY);
		for (int neuronY = startY; neuronY < afterY; neuronY++) {
			float vectorElement = logistic(neuronInput[neuronY]);
			float matrixElement = weightMatrix[neuronY * matrixPitchElements + neuronX];
			norm += matrixElement;
			sum += vectorElement * matrixElement;
		}
		atomicAdd(&sums[threadIdx.x], sum);
		atomicAdd(&norms[threadIdx.x], norm);
	}
	__syncthreads();
	if (neuronX < nNeurons && threadIdx.y == 0) {
		output[neuronX] = ((1.0f / norms[threadIdx.x]) * sums[threadIdx.x]);
	}
}



/*
������� ������������� (���������� �����).

������������� ����� ��������� �� 1 ����� ��������� ��� ����� ������ �������� �� �������, 
����� � �������� �� ������.
*/
__global__ void phaseSyncCheckInplace(
	int *currGT, 
	int gtPitchElements, 
	int nSteps, 
	int *hits, 
	int hitsPitchElements, 
	int nNeurons
) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	int count = 0;
	for (int step = 0; step < nSteps; step++) {
		int first = currGT[step * gtPitchElements + neuronIdxFirst];
		int second = currGT[step * gtPitchElements + neuronIdxSecond];
		if (first == second) {
			count++;
		}
	}
	hits[neuronIdxSecond * hitsPitchElements + neuronIdxFirst] += count;
}


/**
���������� �������, ��� ����� ���������� ��� 
���������� ������ ������� � ����� ��������������� ��������.
*/
__global__ void prepareToSynchronizationCheck(
	float *prevOutput, 
	float *currOutput, 
	int *gt, 
	float *bufferedValues, 
	int nNeurons
) {
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
__global__ void fragmentaryAnalysis(
	float *bufferedValues,
	int bvPitchElements,
	int nSteps, 
	int *hits, 
	int hitsPitchElements,
	int nNeurons, 
	const float eps
) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	int count = 0;
	for (int step = 0; step < nSteps; step++) {
		float first = bufferedValues[step * bvPitchElements + neuronIdxFirst];
		float second = bufferedValues[step * bvPitchElements + neuronIdxSecond];
		float diff = fabsf(first - second);
		if (diff < eps) {
			count++;
		}
	}
	hits[neuronIdxSecond * hitsPitchElements + neuronIdxFirst] += count;
}

__global__ void zeroInts(int *ar, int pitchElements, int sizeX, int sizeY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= sizeX || y >= sizeY) {
		return;
	}
	ar[y * pitchElements + x] = 0;
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

		DeviceScopedPtr2D<float> weightMatrixDevice(nNeurons, nNeurons);
		check(weightMatrixHost.size() == nNeurons * nNeurons);
		weightMatrixDevice.copyFromHost(&weightMatrixHost[0], nNeurons, nNeurons, nNeurons);

		DeviceScopedPtr1D<float> input(nNeurons);
		DeviceScopedPtr1D<float> output(nNeurons);

		std::vector<float> stateHost(nNeurons);
		::randomSetHost(stateHost);

		input.copyFromHost(&stateHost[0], nNeurons);
		output.copyFromHost(&stateHost[0], nNeurons);
		
		float *currInputPtr = input.getDevPtr();
		float *currOutputPtr = output.getDevPtr();

		TimerMillisPrecision timer;
		timer.start();

		for (int i = 0; i < startObservationTime; i++) {
			if (useSingleThreadPerNeuron) {
				dim3 blockDimFm1(FM_X, FM_Y);
				dim3 gridDimFm1(divRoundUp(nNeurons, blockDimFm1.x), 1);
				checkKernelRun((
					calcDynamicsFm1<<<gridDimFm1, blockDimFm1>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			} else {
				dim3 blockDim(256);
				dim3 gridDim(divRoundUp(nNeurons, blockDim.x));

				checkKernelRun((
					calcDynamicsOneThreadShared<<<gridDim, blockDim>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}
			/*
			{
				dim3 blockDim(256);
				dim3 gridDim(divRoundUp(nNeurons, blockDim.x));

				checkKernelRun((
					calcDynamicsOneThread<<<gridDim, blockDim>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}

			{
				dim3 blockDimFm1(FM_X, FM_Y);
				dim3 gridDimFm1(divRoundUp(nNeurons, blockDimFm1.x), 1);
				checkKernelRun((
					calcDynamicsFm1<<<gridDimFm1, blockDimFm1>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}

			{
				dim3 calcBlockDim4(NEURON_ZONE, N_OUTPUTS);
				dim3 calcGridDim4(divRoundUp(nNeurons, calcBlockDim4.y), 1);
				checkKernelRun((
					calcDynamicsShared4Shared<<<calcGridDim4, calcBlockDim4>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}
			*/
			std::swap(currInputPtr, currOutputPtr);
		}

		unsigned int timeMillisElapsed = timer.get_elapsed_time_ms();
		printf("%d iterations = %u ms\r\n", startObservationTime, timeMillisElapsed);

		DeviceScopedPtr2D<int> hits(nNeurons, nNeurons);
		{
			dim3 blockDimFill(32, 8);
			dim3 gridDimFill(divRoundUp(nNeurons, blockDimFill.x), divRoundUp(nNeurons, blockDimFill.y));
			checkKernelRun((
				zeroInts<<<gridDimFill, blockDimFill>>>(
					hits.getDevPtr(),
					hits.getPitchElements(),
					nNeurons,
					nNeurons
				)
			));
		}

		DeviceScopedPtr1D<int> currentHits(nNeurons);
		std::vector<int> currentHitsHost(nNeurons);

		const int N_STEPS = 64;
		DeviceScopedPtr2D<int> gt(nNeurons, N_STEPS);
		DeviceScopedPtr2D<float> bufferedValues(nNeurons, N_STEPS);

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
						bufferedValues.getPitchElements(),
						N_STEPS,
						hits.getDevPtr(),
						hits.getPitchElements(),
						nNeurons,
						fragmentaryEPS
					)
				));
				
				currentStep = 0;
			}

			// computing
			if (useSingleThreadPerNeuron) {
				// ��������� ���������� � ������ 32x16, �� �������� ������� � 32x8, � 32x32.
				// ���� (nNeurons + 31 / 32, 1)
				dim3 blockDimFm1(FM_X, FM_Y);
				dim3 gridDimFm1(divRoundUp(nNeurons, blockDimFm1.x), 1);
				checkKernelRun((
					calcDynamicsFm1<<<gridDimFm1, blockDimFm1>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			} else {
				dim3 calcBlockDim4(NEURON_ZONE, N_OUTPUTS);
				dim3 calcGridDim4(divRoundUp(nNeurons, calcBlockDim4.y), 1);
				checkKernelRun((
					calcDynamicsShared4Shared<<<calcGridDim4, calcBlockDim4>>>(
						weightMatrixDevice.getDevPtr(),
						weightMatrixDevice.getPitchElements(),
						currInputPtr,
						currOutputPtr,
						nNeurons
					)
				));
			}
		
			{
				dim3 blockDim(256);
				dim3 gridDim(divRoundUp(nNeurons, blockDim.x));
				checkKernelRun((
					prepareToSynchronizationCheck<<<gridDim, blockDim>>>(
						currInputPtr,
						currOutputPtr,
						gt.getDevPtr() + gt.getPitchElements() * currentStep,
						bufferedValues.getDevPtr() + bufferedValues.getPitchElements() * currentStep,
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
							gt.getPitchElements(),
							N_STEPS,
							hits.getDevPtr(),
							hits.getPitchElements(),
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
						gt.getPitchElements(),
						currentStep,
						hits.getDevPtr(),
						hits.getPitchElements(),
						nNeurons
					)
				));
				currentStep = 0;
			} else if (syncType == FRAGMENTARY) {
				dim3 fragBlockDim(32, 8);
				dim3 fragGridDim(divRoundUp(nNeurons, fragBlockDim.x), divRoundUp(nNeurons, fragBlockDim.y));
				
				checkKernelRun((
					fragmentaryAnalysis<<<fragGridDim, fragBlockDim>>>(
						bufferedValues.getDevPtr(),
						bufferedValues.getPitchElements(),
						currentStep,
						hits.getDevPtr(),
						hits.getPitchElements(),
						nNeurons,
						fragmentaryEPS
					)
				));
			} else {
				throw std::logic_error("unknown sync type");
			}
		}
		std::vector<int> hitsHost(nNeurons * nNeurons);
		hits.copyToHost(&hitsHost[0], nNeurons, nNeurons, nNeurons);
		return hitsHost;
	} END_FUNCTION
}
