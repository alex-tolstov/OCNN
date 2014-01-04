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
#define SIZE 1

__global__ void calcDynamics(
	float *weightMatrix,
	float *neuronInput,
	float *output,
	int nNeurons
) {
	int neuronIdx[SIZE];
	#pragma unroll
	for (int i = 0; i < SIZE; i++) {
		neuronIdx[i] = threadIdx.x * SIZE + blockIdx.x * blockDim.x * SIZE + i;
	}
	if (neuronIdx[SIZE-1] >= nNeurons) {
		return;
	}

	float sum[SIZE];
	float norm[SIZE];

	#pragma unroll
	for (int i = 0; i < SIZE; i++) {
		sum[i] = 0.0f;
		norm[i] = 0.0f;
	}
	
	for (int other = 0; other < nNeurons; other++) {
		float prev = neuronInput[other];
	
		#pragma unroll
		for (int i = 0; i < SIZE; i++) {
			float w = weightMatrix[other * nNeurons + neuronIdx[i]];
			norm[i] += w;
			sum [i] += prev * w;
		}
	}
	
	#pragma unroll
	for (int i = 0; i < SIZE; i++) {
		output[neuronIdx[i]] = logistic((1.0f / norm[i]) * sum[i]);
	}
}

/*
Фазовая синхронизация (инплейсный режим).

Данная задача, возможно, эффективно решилась бы сортировкой. Но, по ходу, количество хитов
нам тут не получится редуцировать в что-то меньшее, чем O(N^2).
*/

__global__ void phaseSyncCheckInplace(int *currGT, int *hits, int nNeurons) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	// вполне себе coalesced, но тоже можно улучшить, применяя ILP.
	int first = currGT[neuronIdxFirst];
	int second = currGT[neuronIdxSecond];

	if (first == second) {
		hits[neuronIdxFirst * nNeurons + neuronIdxSecond]++;
	}
}

__global__ void phaseSyncCheckInplace(int *currGT, int nSteps, int *hits, int nNeurons) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	int count = 0;
	for (int step = 0; step < nSteps; step++) {
		int first = currGT[neuronIdxFirst];
		int second = currGT[neuronIdxSecond];
		if (first == second) {
			count++;
		}
	}
	hits[neuronIdxFirst * nNeurons + neuronIdxSecond] += count;
}


/**
Простейшая функция, работает очень быстро.
*/
__global__ void phaseSyncCheck(float *prevOutput, float *currOutput, int *gt, int nNeurons) {
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
}


/*
Имеем N нейронов, каждому thread даем 1 нейрон.
Идем по всем остальным нейронам, сравнивая текущее значение
выходов, если расхождение меньше допустимого, то увеличиваем счетчик
"хитов" данной пары нейронов.

А не решить ли ту же самую задачу сортировкой, а??
Нам надо за O(N Log N) посортировать
*/
__global__ void dynamicsAnalysis(
	float *currOutput,
	int *nHits,
	int *nCurrentStepHits,
	int nNeurons,
	const float eps
) {
	int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (neuronIdx >= nNeurons) {
		return;
	}

	int hitsCount = 0;
	float curr = currOutput[neuronIdx];
	for (int oppositeIdx = neuronIdx + 1; oppositeIdx < nNeurons; oppositeIdx++) {
		// загрузить в shared-память, может быть? но может и L1-кэш нормально справится
		float opp = currOutput[oppositeIdx];
		float diff = fabsf(opp - curr);
		// equals
		if (diff < eps) {
			// немножко греховно
			nHits[neuronIdx * nNeurons + oppositeIdx]++;
			hitsCount++;
		}
	}
	nCurrentStepHits[neuronIdx] = hitsCount;
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
	const float fragmentaryEPS
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
		printf("INITIAL\r\n");
		::debugPrintArray(stateHost);

		input.copyFromHost(&stateHost[0], nNeurons);
		output.copyFromHost(&stateHost[0], nNeurons);
		
		float *ptrEven = input.getDevPtr();
		float *ptrOdd = output.getDevPtr();

		for (int i = 0; i < startObservationTime; i++) {
			dim3 blockDim(128 / SIZE);
			dim3 gridDim(divRoundUp(nNeurons, blockDim.x * SIZE));

			checkKernelRun((
				calcDynamics<<<gridDim, blockDim>>>(
					weightMatrixDevice.getDevPtr(),
					ptrEven,
					ptrOdd,
					nNeurons
				)
			));
		//	if (output.getDevPtr() == ptrOdd) {
		//		output.copyToHost(&stateHost[0], stateHost.size());
		//	} else {
		//		input.copyToHost(&stateHost[0], stateHost.size());
		//	}
		//	::debugPrintArray(stateHost);
			std::swap(ptrEven, ptrOdd);
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

		const int N_STEPS = 32;	
		DeviceScopedPtr1D<int> gt(nNeurons * N_STEPS);

		sheet.resize(nIterations * nNeurons);

		int currentStep = 0;
		for (int i = 0; i < nIterations; i++) {
			{
				dim3 fragBlockDim(128);
				dim3 fragGridDim(divRoundUp(nNeurons, fragBlockDim.x));
				if (syncType == FRAGMENTARY) {
					checkKernelRun((
						dynamicsAnalysis<<<fragGridDim, fragBlockDim>>>(
							ptrEven,
							hits.getDevPtr(),
							currentHits.getDevPtr(),
							nNeurons,
							fragmentaryEPS
						)
					));
				}
			}

			{
				dim3 calcBlockDim(128 / SIZE);
				dim3 calcGridDim(divRoundUp(nNeurons, calcBlockDim.x * SIZE));
				checkKernelRun((
					calcDynamics<<<calcGridDim, calcBlockDim>>>(
						weightMatrixDevice.getDevPtr(),
						ptrEven,
						ptrOdd,
						nNeurons
					)
				));
			}

			if (syncType == PHASE) {
				{
					dim3 blockDim(128);
					dim3 gridDim(divRoundUp(nNeurons, blockDim.x));
					checkKernelRun((
						phaseSyncCheck<<<gridDim, blockDim>>>(
							ptrEven,
							ptrOdd,
							gt.getDevPtr() + nNeurons * currentStep,
							nNeurons
						)
					));
					currentStep++;
				}
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

			if (output.getDevPtr() == ptrOdd) {
				output.copyToHost(&sheet[i * nNeurons], nNeurons);
			} else {
				input.copyToHost(&sheet[i * nNeurons], nNeurons);
			}
			std::swap(ptrEven, ptrOdd);
		}

		if (syncType == PHASE) {
			if (currentStep != 0) {
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
			}
		}
		std::vector<int> hitsHost(nNeurons * nNeurons);
		hits.copyToHost(&hitsHost[0], nNeurons * nNeurons);
		return hitsHost;
	} END_FUNCTION
}
