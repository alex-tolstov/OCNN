
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <vector>

#define QUOTE(A) #A

#define check(EXPRESSION) \
	if (!EXPRESSION) { \
		throw std::string(QUOTE(EXPRESSION)); \
	}


#define checkKernelRun(EXEC) { \
	EXEC;\
	cudaError_t result = cudaGetLastError();\
	if (result != cudaSuccess) { \
		throw std::string("Start failure: ") + std::string(QUOTE(EXEC)) + ": " + std::string(cudaGetErrorString(result)); \
	}\
	cudaThreadSynchronize();\
	result = cudaGetLastError();\
	if (result != cudaSuccess) { \
		throw std::string("Run failure: ") + std::string(QUOTE(EXEC)) + ": " + std::string(cudaGetErrorString(result)); \
	}\
}

#define checkCudaCall(EXEC) {\
	EXEC;\
	cudaThreadSynchronize();\
	cudaError_t result = cudaGetLastError();\
	if (result != cudaSuccess) {\
		throw std::string("Run failure: ") + std::string(QUOTE(EXEC)) + ": " + std::string(cudaGetErrorString(result)); \
	}\
}

#define BEGIN_FUNCTION try

#define END_FUNCTION catch (const std::string &message) { \
	throw std::string(__FUNCTION__) + ": " + message; \
}

#define BEGIN_DESTRUCTOR try 

#define END_DESTRUCTOR catch (const std::string &message) { \
	printf("DESTRUCTOR ERROR: %s: %s\n", std::string(__FUNCTION__).c_str(), message.c_str()); \
}



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

	void copyToHost(const T *host, int nElements) const {
		BEGIN_FUNCTION {
			check(data.size() >= nElements); 
			checkCudaCall(cudaMemcpy((void*)host, (void*)data.getDevPtr(), nElements * sizeof(T), cudaMemcpyDeviceToHost));
		} END_FUNCTION
	}

	T* getDevPtr() {
		return data.getDevPtr();
	}
};



__device__ float logistic(float signal) {
	return 4.0f * signal - 1.0f;
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
	int nNeurons
) {
	int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (neuronIdx >= nNeurons) {
		return;
	}
	float sum = 0.0f;
	for (int i = 0; i < nNeurons; i++) {
		sum += neuronInput[i] * weightMatrix[i * nNeurons + neuronIdx];
	}
	output[neuronIdx] = logistic(sum);
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

void processOscillatoryChaoticNetworkDynamics(
	int nNeurons, 
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	const float successRate
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

		dim3 blockDim(256);
		dim3 gridDim(divRoundUp(nNeurons, blockDim.x));
		float *ptrEven = input.getDevPtr();
		float *ptrOdd = output.getDevPtr();
		
		for (int i = 0; i < startObservationTime; i++) {
			checkKernelRun((
				calcDynamics<<<gridDim, blockDim>>>(
					weightMatrixDevice.getDevPtr(),
					ptrEven,
					ptrOdd,
					nNeurons
				)
			));
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
					1e-4f
				)
			));
			currentHits.copyToHost(&currentHitsHost[0], nNeurons);
			for (int j = 0; j < nNeurons; j++) {
				printf("%d ", currentHitsHost[j]);
			}
			printf("\r\n");

			checkKernelRun((
				calcDynamics<<<gridDim, blockDim>>>(
					weightMatrixDevice.getDevPtr(),
					ptrEven,
					ptrOdd,
					nNeurons
				)
			));
			std::swap(ptrEven, ptrOdd);
		}

	} END_FUNCTION
}


struct Point {
	float x;
	float y;

	Point(float x, float y) 
		: x(x)
		, y(y)
	{
	}
};

class NeuralNetwork {
private:
	std::vector<Point> points;
	std::vector<float> weightMatrix;

	void createVoronoiDiagram() {}
	void calcWeightCoefs() {}
public:

	NeuralNetwork(const std::vector<Point> &points) 
		: points(points)
	{
		createVoronoiDiagram();
		calcWeightCoefs();
	}

	void process() {
		::processOscillatoryChaoticNetworkDynamics(points.size(), weightMatrix, 10, 1000, 0.74f);
		printf("GURO\n");
	}
};

int main() {
	try {
		checkCudaCall(cudaSetDevice(0));
		
		std::vector<Point> points;

		points.push_back(Point(2.5f, 3.7f));
		points.push_back(Point(0.5f, 3.7f));
		points.push_back(Point(4.5f, 3.7f));
		points.push_back(Point(2.5f, 0.7f));
		points.push_back(Point(5.5f, 0.7f));

		NeuralNetwork network(points);
		network.process();
	} catch (const std::string &message) {
		printf("Error, message = %s", message.c_str());
	} catch (...) {
		printf("Unknown exception caught\n");
	}
    return 0;
}