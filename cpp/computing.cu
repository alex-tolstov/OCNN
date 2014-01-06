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
 * Множим соответствующие входы на соответствующие коэффициенты,
 * суммируем и считаем функцию от суммы, дающую хаотическое распределение.
 *
 * @param k
 * @param weightMatrix
 * @param neuronInput
 * @param neuronOutput
 *
 * Количество тредов = кол-во нейронов в нейронной сети.
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
		// все потоки читают одну ячейку памяти + L1 кэш
		float prev = neuronInput[other];
		// опять же, групповое последовательное чтение
		float w = weightMatrix[other * nNeurons + neuronIdx];
		norm += w;
		sum  += prev * w;
	}
	output[neuronIdx] = logistic((1.0f / norm) * sum);
}


// ТОЛЬКО СТЕПЕНЬ ДВОЙКИ, А ТО ЦИКЛ НЕ СРАБОТАЕТ
#define SHARED_BLOCK_DIM_X 256

__global__ void calcDynamicsShared(
	float *weightMatrix,
	float *neuronInput,
	float *output,
	int nNeurons
) {
	int neuronIdx = blockIdx.x;

	if (neuronIdx >= nNeurons) {
		return;
	}

	float sum = 0.0f;
	float norm = 0.0f;

	for (int i = 0; i < nNeurons; i += blockDim.x) {
		int secondNeuron = i + threadIdx.x;
		if (secondNeuron < nNeurons) {
			// все блоки читают этот neuronInput одновременно, оптимизация не требуется.
			float prev = neuronInput[secondNeuron];
			// читаем последовательно
			float w = weightMatrix[neuronIdx * nNeurons + secondNeuron];
			norm += w;
			sum  += prev * w;
		}
	}

	__shared__ float sums[SHARED_BLOCK_DIM_X];
	__shared__ float norms[SHARED_BLOCK_DIM_X];

	sums[threadIdx.x] = sum;
	norms[threadIdx.x] = norm;

	__syncthreads();

	// NOTE SHARED_BLOCK_DIM_X обязан быть степенью 2.
	for (int stride = SHARED_BLOCK_DIM_X / 2; stride >= 1; stride = stride / 2) {
		if (threadIdx.x < stride && threadIdx.x + stride < SHARED_BLOCK_DIM_X) {
			sums[threadIdx.x]  +=  sums[threadIdx.x + stride];
			norms[threadIdx.x] += norms[threadIdx.x + stride];
		}
		__syncthreads();
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		norm = norms[0];
		sum = sums[0];
		output[neuronIdx] = logistic((1.0f / norm) * sum);
	}
}


/*
Фазовая синхронизация (инплейсный режим).

Использование цикла сравнений на 1 выход позволяет нам очень сильно выиграть во времени, 
пусть и проиграв по памяти.
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
Простейшая функция, для учета увеличения или уменьшения выхода нейрона а также буферизирования значений.
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
Использован тот же самый трюк, что и с фазовой синхронизацией - стараемся 
уменьшить количество выходов, из-за этого выигрываем в разы.
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

__global__ void zeroInts(int *ar, int count) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= count) {
		return;
	}
	ar[idx] = 0;
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
		check(nIterations > 0);
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
			}
			std::swap(currInputPtr, currOutputPtr);
		}

		DeviceScopedPtr1D<int> hits(nNeurons * nNeurons);
		{
			dim3 blockDimFill(512);
			dim3 gridDimFill(divRoundUp(nNeurons * nNeurons, blockDimFill.x));
			checkKernelRun((
				zeroInts<<<gridDimFill, blockDimFill>>>(
					hits.getDevPtr(),
					nNeurons * nNeurons
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
