
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <vector>

#include "C:\\Users\\Alex\\Desktop\\ocnn\\cpp\\SimpleMath.h"


template <typename T> 
class Ptr1D {
private:
	T *ptr;
	int size_;

	Ptr1D(const Ptr1D<T> &);
	Ptr1D operator = (const Ptr1D<T> &);
public:
	
	Ptr1D(int size) 
		: size_(size)
	{
		BEGIN_FUNCTION {
			checkCudaCall(cudaMalloc((void**)&ptr, size * sizeof(T)));
		} END_FUNCTION
	}

	int size() const {
		return this->size_;
	}

	T *getDevPtr() {
		check(ptr != NULL);
		return ptr;
	}

	void free() {
		BEGIN_FUNCTION {
			checkCudaCall(cudaFree(ptr));
		} END_FUNCTION
	}
};


template <typename T>
class DeviceScopedPtr1D {
private:
	Ptr1D<T> data;
public:
	DeviceScopedPtr1D(int size) 
		: data(size)
	{
	}

	~DeviceScopedPtr1D() {
		BEGIN_DESTRUCTOR {
			data.free();
		} END_DESTRUCTOR
	}

	void copyFromHost(const T * const host, int nElements) {
		BEGIN_FUNCTION {
			check(data.size() >= nElements); 
			checkCudaCall(cudaMemcpy(data.getDevPtr(), host, nElements * sizeof(T), cudaMemcpyHostToDevice));
		} END_FUNCTION
	}

	void copyToHost(T *host, int nElements) {
		BEGIN_FUNCTION {
			check(data.size() >= nElements); 
			checkCudaCall(cudaMemcpy(host, data.getDevPtr(), nElements * sizeof(T), cudaMemcpyDeviceToHost));
		} END_FUNCTION
	}

	T* getDevPtr() {
		return data.getDevPtr();
	}
};



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

__global__ void calcDynamics(
	float *weightMatrix,
	float *neuronInput,
	float *output,
	int nNeurons,
	const float normCoeff
) {
	int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (neuronIdx >= nNeurons) {
		return;
	}
	float sum = 0.0f;
	float norm = 0.0f;
	for (int i = 0; i < nNeurons; i++) {
		if (i != neuronIdx) {
			float w = weightMatrix[i * nNeurons + neuronIdx];
			norm += w;
			float prev = neuronInput[i];
			sum += logistic(prev) * w;
		}
	}
	output[neuronIdx] = (1.0f / norm) * sum;
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

/*
рандом в диапазоне от -1 до 1

*/
//__global__ void pseudoRandom(float *ar, int size) {
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	if (idx >= size) {
//		return;
//	}
//	ar[idx] = -1.0f + 2.0 / (idx % 3 + 1);
//}

inline int divRoundUp(int a, int b) {
	return (a - 1) / b + 1;
}

void processOscillatoryChaoticNetworkDynamics(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	const float successRate,
	const float normCoeff
) {
	BEGIN_FUNCTION {
		printf("start oscillatory process, norm coef = %5.4f\r\n", normCoeff);
		check(nIterations > 0);
		check(nNeurons > 0);
		check(startObservationTime >= 0);

		DeviceScopedPtr1D<float> weightMatrixDevice(nNeurons * nNeurons);
		check(weightMatrixHost.size() == nNeurons * nNeurons);
		weightMatrixDevice.copyFromHost(&weightMatrixHost[0], weightMatrixHost.size());

		DeviceScopedPtr1D<float> input(nNeurons);
		DeviceScopedPtr1D<float> output(nNeurons);

		std::vector<float> stateHost(nNeurons);
		srand(23);
		for (int i = 0; i < nNeurons; i++) {
			stateHost[i] = (250 - rand() % 500) / 250.0f;
		}
		printf("INITIAL\r\n");
		for (int j = 0; j < nNeurons; j++) {
			printf("%5.4f ", stateHost[j]);
		}
		printf("\r\n");
		input.copyFromHost(&stateHost[0], nNeurons);
		output.copyFromHost(&stateHost[0], nNeurons);
		dim3 blockDim(256);
		dim3 gridDim(divRoundUp(nNeurons, blockDim.x));
		float *ptrEven = input.getDevPtr();
		//checkKernelRun((pseudoRandom<<<gridDim, blockDim>>>(ptrEven, nNeurons)));
		float *ptrOdd = output.getDevPtr();
		//checkKernelRun((pseudoRandom<<<gridDim, blockDim>>>(ptrOdd, nNeurons)));
		
		for (int i = 0; i < startObservationTime; i++) {
			checkKernelRun((
				calcDynamics<<<gridDim, blockDim>>>(
					weightMatrixDevice.getDevPtr(),
					ptrEven,
					ptrOdd,
					nNeurons,
					normCoeff
				)
			));
			if (output.getDevPtr() == ptrOdd) {
				output.copyToHost(&stateHost[0], stateHost.size());
			} else {
				input.copyToHost(&stateHost[0], stateHost.size());
			}
			for (int j = 0; j < nNeurons; j++) {
				printf("%5.4f ", stateHost[j]);
			}
			printf("\r\n");
			std::swap(ptrEven, ptrOdd);
		}

		DeviceScopedPtr1D<int> hits(nNeurons * nNeurons);
		DeviceScopedPtr1D<int> currentHits(nNeurons);
		std::vector<int> currentHitsHost(nNeurons);

		for (int i = 0; i < nIterations; i++) {
			checkKernelRun((
				dynamicsAnalysis<<<gridDim, blockDim>>>(
					ptrEven,
					hits.getDevPtr(),
					currentHits.getDevPtr(),
					nNeurons,
					1e-2f
				)
			));
	//		currentHits.copyToHost(&currentHitsHost[0], nNeurons);
	//		for (int j = 0; j < nNeurons; j++) {
	//			printf("%d ", currentHitsHost[j]);
	//		}
	//		printf("\r\n");

			checkKernelRun((
				calcDynamics<<<gridDim, blockDim>>>(
					weightMatrixDevice.getDevPtr(),
					ptrEven,
					ptrOdd,
					nNeurons,
					normCoeff
				)
			));
			if (output.getDevPtr() == ptrOdd) {
				output.copyToHost(&stateHost[0], stateHost.size());
			} else {
				input.copyToHost(&stateHost[0], stateHost.size());
			}
			for (int j = 0; j < nNeurons; j++) {
				printf("%5.4f ", stateHost[j]);
			}
			printf("\r\n");
			std::swap(ptrEven, ptrOdd);
		}

	} END_FUNCTION
}



class NeuralNetwork {
private:
	std::vector<voronoi::Point> points;
	std::vector<float> weightMatrix;
	double totalAverage;
	double sumCoeffs;

	void createVoronoiDiagram() {
		voronoi::VoronoiFortuneComputing diagram(points);
		int nPoints = static_cast<int>(points.size());
		check(nPoints > 0);
		double sumAverages = 0.0;
		for (std::set<voronoi::NeighborsList>::iterator it = diagram.adjList.begin();
			it != diagram.adjList.end();
			++it
		) {
			printf("[%s]\r\n", it->site.prints().c_str());
			for (
				std::set<voronoi::Point, voronoi::PointComparatorY>::iterator i = it->nextToThis.begin();
				i != it->nextToThis.end();
				++i
			) {
				printf("%s ",i->prints().c_str());
			}
			printf("\r\n");
			sumAverages += it->getAverageDistance();
		}
		totalAverage = sumAverages / (nPoints + 0.0);
		printf("total average = %5.4f\r\n", totalAverage);
	}

	void calcWeightCoefs() {
		int nPoints = static_cast<int>(points.size());
		weightMatrix.resize(nPoints * nPoints, 0.0f);
		sumCoeffs = 0.0;
		for (int i = 0; i < nPoints; i++) {
			for (int j = i + 1; j < nPoints; j++) {
				double sqDist = points[i].squaredDistanceTo(points[j]);
				double k = sqDist / (2.0 * totalAverage);
				float result = expf(-k);
				sumCoeffs += result * 2;
				weightMatrix[i * nPoints + j] = result;
				weightMatrix[j * nPoints + i] = result;
			}
		}
	}

public:

	NeuralNetwork(const std::vector<voronoi::Point> &points) 
		: points(points)
	{
		createVoronoiDiagram();
		calcWeightCoefs();
	}

	void process() {
		::processOscillatoryChaoticNetworkDynamics(
			this->points.size(), 
			this->weightMatrix,
			10,
			40,
			0.8,
			1.0f / this->sumCoeffs
		);
		printf("GURO\n");
	}
};

int main() {
	try {
		checkCudaCall(cudaSetDevice(0));
		check(freopen("C:\\voronoi\\result.txt", "w", stdout) != NULL);
		using voronoi::Point;
		std::vector<Point> points;

		int nPoints;
		std::cin >> nPoints;
		for (int i = 0; i < nPoints; i++) {
			int x;
			int y;
			std::cin >> x >> y;
			points.push_back(Point(x, y));
		}
		NeuralNetwork network(points);
		network.process();
	} catch (const std::string &message) {
		printf("Error, message = %s", message.c_str());
	} catch (...) {
		printf("Unknown exception caught\n");
	}
	return 0;
}