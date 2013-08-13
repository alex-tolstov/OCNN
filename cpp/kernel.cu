
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
	int nNeurons
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
Фазовая синхронизация (инплейсный режим).

*/

__global__ void phaseSyncCheckInplace(int *currGT, int *hits, int nNeurons) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	// вполне себе coalesced
	int first = currGT[neuronIdxFirst];
	int second = currGT[neuronIdxSecond];

	if (first == second) {
		hits[neuronIdxFirst * nNeurons + neuronIdxSecond]++;
	}
}


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

struct Group {
	int idx;
	int host;
	std::vector<int> list;

	Group(int idx) 
		: idx(idx)
		, list(1, idx) 
		, host(idx)
	{
	}

	void rebase(std::vector<Group> &groups, int newHost) {
		for (int i = 0; i < size(); i++) {
			groups[list[i]].host = newHost;
		}
	}

	void addAll(Group &second, std::vector<Group> &groups) {
		second.rebase(groups, this->idx);
		list.insert(list.end(), second.list.begin(), second.list.end());
	}

	void clear() {
		list.clear();
	}

	int size() {
		return static_cast<int>(list.size());
	}
};

enum SyncType {
	PHASE = 0,
	FRAGMENTARY
};

std::vector<Group> processOscillatoryChaoticNetworkDynamics(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
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

		std::vector<float> stateHost(nNeurons);
		::randomSetHost(stateHost);
		printf("INITIAL\r\n");
		::debugPrintArray(stateHost);

		input.copyFromHost(&stateHost[0], nNeurons);
		output.copyFromHost(&stateHost[0], nNeurons);
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
			if (output.getDevPtr() == ptrOdd) {
				output.copyToHost(&stateHost[0], stateHost.size());
			} else {
				input.copyToHost(&stateHost[0], stateHost.size());
			}
			::debugPrintArray(stateHost);
			std::swap(ptrEven, ptrOdd);
		}

		DeviceScopedPtr1D<int> hits(nNeurons * nNeurons);
		{
			dim3 blockDimFill(256);
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
		DeviceScopedPtr1D<int> gt(nNeurons);

		for (int i = 0; i < nIterations; i++) {
			if (syncType == FRAGMENTARY) {
				checkKernelRun((
					dynamicsAnalysis<<<gridDim, blockDim>>>(
						ptrEven,
						hits.getDevPtr(),
						currentHits.getDevPtr(),
						nNeurons,
						0.06f
					)
				));
			}

			checkKernelRun((
				calcDynamics<<<gridDim, blockDim>>>(
					weightMatrixDevice.getDevPtr(),
					ptrEven,
					ptrOdd,
					nNeurons
				)
			));

			if (syncType == PHASE) {
				checkKernelRun((
					phaseSyncCheck<<<gridDim, blockDim>>>(
						ptrEven,
						ptrOdd,
						gt.getDevPtr(),
						nNeurons
					)
				));
				dim3 blockCheck(32, 8);
				dim3 gridCheck(divRoundUp(nNeurons, blockCheck.x), divRoundUp(nNeurons, blockCheck.y));
				checkKernelRun((
					phaseSyncCheckInplace<<<gridCheck, blockCheck>>>(
						gt.getDevPtr(),
						hits.getDevPtr(),
						nNeurons
					)
				));
			}
			if (output.getDevPtr() == ptrOdd) {
				output.copyToHost(&stateHost[0], stateHost.size());
			} else {
				input.copyToHost(&stateHost[0], stateHost.size());
			}
			::debugPrintArray(stateHost);
			std::swap(ptrEven, ptrOdd);
		}

		std::vector<int> hitsHost(nNeurons * nNeurons);
		hits.copyToHost(&hitsHost[0], nNeurons * nNeurons);

		printf("Hits matrix: \r\n");

		
		std::vector<Group> groups;
		groups.reserve(nNeurons);
		for (int i = 0; i < nNeurons; i++) {
			groups.push_back(Group(i));
		}
		
		for (int i = 0; i < nNeurons; i++) {
			for (int j = 0; j < nNeurons; j++) {
				if (hitsHost[i * nNeurons + j] > successRate * nIterations) {
					if (groups[i].host != groups[j].host) {
						int host1 = groups[i].host;
						int host2 = groups[j].host;
						
						if (groups[host1].size() >= groups[host2].size()) {
							groups[host1].addAll(groups[host2], groups);
							groups[host2].clear();
						} else {
							groups[host2].addAll(groups[host1], groups);
							groups[host1].clear();
						}
					}
				}
			}
		}
		return groups;
	} END_FUNCTION
}

#include "EasyBMP.h"

void printPoint(BMP &bitmap, int x, int y, int r, int g, int b) {
	const int shiftX[5] = {0, 0, 0, 1, -1};
	const int shiftY[5] = {0, 1, -1, 0, 0};
	for (int i = 0; i < 5; i++) {
		int currX = min(512, max(x + shiftX[i], 0));
		int currY = min(512, max(y + shiftY[i], 0));

		bitmap(currX, currY)->Red = r;
		bitmap(currX, currY)->Green = g;
		bitmap(currX, currY)->Blue = b;
	}
}

class NeuralNetwork {
private:
	std::vector<voronoi::Point> points;
	std::vector<float> weightMatrix;
	double totalAverage;
	

	void createVoronoiDiagram(voronoi::NeighborsListContainer &diagram) {
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
		for (int i = 0; i < nPoints; i++) {
			for (int j = i + 1; j < nPoints; j++) {
				double sqDist = points[i].squaredDistanceTo(points[j]);
				double k = sqDist / (2.0 * totalAverage);
				float result = expf(-k);
				weightMatrix[i * nPoints + j] = result;
				weightMatrix[j * nPoints + i] = result;
			}
		}
	}

public:

	NeuralNetwork(const std::vector<voronoi::Point> &points, voronoi::NeighborsListContainer &diagram) 
		: points(points)
	{
		createVoronoiDiagram(diagram);
		calcWeightCoefs();
	}

	void process(const std::string &fileName, SyncType syncType, float successRate) {
		std::vector<Group> groups = ::processOscillatoryChaoticNetworkDynamics(
			this->points.size(), 
			this->weightMatrix,
			100,
			1000,
			syncType,
			successRate
		);
		BMP bitmap;
		bitmap.SetSize(512, 512);

		std::vector<bool> used(points.size(), false);

		for (int i = 0; i < static_cast<int>(groups.size()); i++) {
			if (groups[i].size() > 1) {
				printf("Got a new group: ");
				srand(i * 232);
				int red = rand()%150 + 56;
				int green = rand()%150 + 56;
				int blue = rand()%200 + 55;
				for (int j = 0; j < groups[i].size(); j++) {
					int x = points[groups[i].list[j]].x + 256;
					int y = points[groups[i].list[j]].y + 256;
					used[groups[i].list[j]] = true;
					printPoint(bitmap, x, y, red, green, blue);
					printf("[%s] ", points[groups[i].list[j]].prints().c_str());
				}
				printf("\r\n");
			}
		}

		for (int i = 0; i < points.size(); i++) {
			if (!used[i]) {
				int x = points[i].x + 256;
				int y = points[i].y + 256;
				printPoint(bitmap, x, y, 0, 0, 0);
			}
		}
		bitmap.WriteToFile(fileName.c_str());
		printf("GURO\n");
	}
};

#include "find_points.h"


int main() {
	try {
		checkCudaCall(cudaSetDevice(0));
		check(freopen((WORKING_DIR + "result.txt").c_str(), "w", stdout) != NULL);
		using voronoi::Point;
		std::vector<Point> points;

		finder::ImageToPointsConverter conv(WORKING_DIR + "input.bmp");
		conv.fillVector(points);

		printf("size = %d\n", points.size());

	//	int nPoints;
	//	std::cin >> nPoints;
		std::string syncTypeString;
		std::cin >> syncTypeString;
		float successRate = 0.f;
		std::cin >> successRate;
/*
		srand(nPoints);
		for (int i = 0; i < nPoints; i++) {
			float x = rand() % 100;
			float y = rand() % 100;
			if (i % 2 == 0) {
				x = -x - 100;
			}
			//std::cin >> x >> y;
			points.push_back(Point(x, y));
		}
*/
		SyncType syncType;
		
		if (syncTypeString == "phase") {
			syncType = PHASE;
			//successRate = 0.75f;
		} else if (syncTypeString == "fragmentary") {
			syncType = FRAGMENTARY;
			//successRate = 0.25f;
		} else {
			throw std::string("Wrong synchronization type name: " + syncTypeString);
		}

		{

			voronoi::DelaunayComputingQhull diagram(points);
			NeuralNetwork network2(points, diagram);
			network2.process(WORKING_DIR + "result_qhull.bmp", syncType, successRate);
			
		}

		{

			voronoi::VoronoiFortuneComputing diagram(points);

			NeuralNetwork network(points, diagram);
			network.process(WORKING_DIR + "result_fortu.bmp", syncType, successRate);

		}

	} catch (const std::string &message) {
		printf("Error, message = %s", message.c_str());
	} catch (...) {
		printf("Unknown exception caught\n");
	}
	return 0;
}