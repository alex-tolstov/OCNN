
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <direct.h>

#include "SimpleMath.h"


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
			// ����� ����������������, ���� ��������� �� 1 neuron per thread, � ������.
			float prev = neuronInput[i];
			sum += logistic(prev) * w;
		}
	}
	output[neuronIdx] = (1.0f / norm) * sum;
}

/*
������� ������������� (���������� �����).

������ ������, ��������, ���������� �������� �� �����������. ��, �� ����, ���������� �����
��� ��� �� ��������� ������������ � ���-�� �������, ��� O(N^2).
*/

__global__ void phaseSyncCheckInplace(int *currGT, int *hits, int nNeurons) {
	int neuronIdxFirst = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronIdxSecond = threadIdx.y + blockIdx.y * blockDim.y;

	if (neuronIdxFirst >= nNeurons || neuronIdxSecond >= nNeurons) {
		return;
	}

	// ������ ���� coalesced, �� ���� ����� ��������, �������� ILP.
	int first = currGT[neuronIdxFirst];
	int second = currGT[neuronIdxSecond];

	if (first == second) {
		hits[neuronIdxFirst * nNeurons + neuronIdxSecond]++;
	}
}


/**
���������� �������, �������� ����� ������.
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
����� N ��������, ������� thread ���� 1 ������.
���� �� ���� ��������� ��������, ��������� ������� ��������
�������, ���� ����������� ������ �����������, �� ����������� �������
"�����" ������ ���� ��������.

� �� ������ �� �� �� ����� ������ �����������, �??
��� ���� �� O(N Log N) �������������
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
		// ��������� � shared-������, ����� ����? �� ����� � L1-��� ��������� ���������
		float opp = currOutput[oppositeIdx];
		float diff = fabsf(opp - curr);
		// equals
		if (diff < eps) {
			// �������� ��������
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

	int size() const {
		return static_cast<int>(list.size());
	}
};

struct GroupComparator : public std::binary_function<Group, Group, bool> {
public:
	bool operator() (const Group &first, const Group &second) {
		return first.size() > second.size();
	}
};

enum SyncType {
	PHASE = 0,
	FRAGMENTARY
};

std::vector<int> processOscillatoryChaoticNetworkDynamics(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
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
		DeviceScopedPtr1D<int> gt(nNeurons);

		for (int i = 0; i < nIterations; i++) {
			if (syncType == FRAGMENTARY) {
				checkKernelRun((
					dynamicsAnalysis<<<gridDim, blockDim>>>(
						ptrEven,
						hits.getDevPtr(),
						currentHits.getDevPtr(),
						nNeurons,
						fragmentaryEPS
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
		//	if (output.getDevPtr() == ptrOdd) {
		//		output.copyToHost(&stateHost[0], stateHost.size());
		//	} else {
		//		input.copyToHost(&stateHost[0], stateHost.size());
		//}
		//	::debugPrintArray(stateHost);
			std::swap(ptrEven, ptrOdd);
		}

		std::vector<int> hitsHost(nNeurons * nNeurons);
		hits.copyToHost(&hitsHost[0], nNeurons * nNeurons);
		return hitsHost;
	} END_FUNCTION
}

std::vector<Group> divideOnGroups(int nNeurons, int nIterations, float successRate, std::vector<int> &hitsHost) {
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
				double sqDist = (points[i].squaredDistanceTo(points[j]));
				double k = sqDist / (2.0 * totalAverage * totalAverage);
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

	static const int N_DEFINED_COLORS = 8;
	
	void process(const std::string &fileName, SyncType syncType, std::vector<float> &successRates, float fragmentaryEPS) {
		const int nIterations = 2900;
		std::vector<int> hits = ::processOscillatoryChaoticNetworkDynamics(
			this->points.size(), 
			this->weightMatrix,
			800,
			nIterations,
			syncType,
			fragmentaryEPS
		);

		const std::string REPORT_DIR = WORKING_DIR + "report\\";
		rmdir(REPORT_DIR.c_str());
		mkdir(REPORT_DIR.c_str());
		for (int sr = 0; sr < successRates.size(); sr++) {
			std::vector<Group> groups = ::divideOnGroups(this->points.size(), nIterations, successRates[sr], hits);
			std::sort(groups.begin(), groups.end(), GroupComparator());
			BMP bitmap;
			bitmap.SetSize(512, 512);

			std::vector<bool> used(points.size(), false);

			const int COLOR_VALUES[N_DEFINED_COLORS] = {
				0x8B0000,
				0x000080,
				0xFFFF00,
				0x008000,
				0x40E0D0,
				0xDC143C,
				0x8B008B,
				0xFF00FF
			};
			for (int i = 0; i < static_cast<int>(groups.size()); i++) {
				if (groups[i].size() > 1) {
					printf("Got a new group: ");
					srand(i * 232);
					int red = rand() % 190 + 56;
					int green = rand() % 190 + 56;
					int blue = rand() % 200 + 55;
					if (i < N_DEFINED_COLORS) {
						red = (COLOR_VALUES[i] >> 16) & 0xFF;
						green = (COLOR_VALUES[i] >> 8) & 0xFF;
						blue = COLOR_VALUES[i] & 0xFF;
					}
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
			std::stringstream ss;
			ss << REPORT_DIR;
			ss << "ss_";
			ss.setf(std::ios::fixed, std::ios::floatfield); 
			ss.precision(4);
			ss << successRates[sr];
			ss << "_";
			ss << fileName;
			bitmap.WriteToFile(ss.str().c_str());
		}
	}
};

#include "find_points.h"


int main() {
	try {
		checkCudaCall(cudaSetDevice(0));

		std::cout << "[phase,p,P,1|fragmentary,f,F,2] [successRate:0..1]" << std::endl;

		std::string syncTypeString;
		std::cin >> syncTypeString;
		float successRate = 0.f;
		std::cin >> successRate;

		std::cout << "To read from " << INPUT_FILE_NAME << ", type 1" << std::endl;
		std::cout << "To read points from file, type 2" << std::endl;
		std::cout << "To use generator of random points in [-100, 100] interval, type 3" << std::endl;
		std::cout << "To read points from modified FCPS base, type 4" << std::endl;
		
		std::string type;
		std::cin >> type;

		using voronoi::Point;
		std::vector<Point> points;

		if (type == "1") {
			std::cout << "Waiting for read file " << INPUT_FILE_NAME << " with points" << std::endl;
			finder::ImageToPointsConverter conv(INPUT_FILE_NAME);
			conv.fillVector(points);
		} else if (type == "2") {
			std::cout << "Type full file name [default:input_points.txt]" << std::endl;
			std::string fileWithPoints;
			std::cin >> fileWithPoints;

			if (fileWithPoints == "") {
				fileWithPoints = "input_points.txt";
			}
			std::cout << "Using file with points: " << fileWithPoints << std::endl;

			std::ifstream pointsStream;
			pointsStream.open(fileWithPoints.c_str(), std::ifstream::in);
			if (!pointsStream.good()) {
				std::cout << "Error of opening file" << std::endl;
				throw std::string("Error of opening");
			}

			int nPoints;
			pointsStream >> nPoints;

			std::cout << "Points: " << nPoints << std::endl;

			for (int i = 0; i < nPoints; i++) {
				float x;
				float y;
				pointsStream >> x >> y;
				points.push_back(Point(x, y));
			}

			pointsStream.close();
		} else if (type == "3") {
			std::cout << "Type number of points: " << std::endl;
			int nPoints;
			std::cin >> nPoints;

			srand(nPoints);
			for (int i = 0; i < nPoints; i++) {
				float x = rand() % 100;
				float y = rand() % 100;
				if (i % 2 == 0) {
					x = -x - 100;
				}
				points.push_back(Point(x, y));
			}
		} else if (type == "4") {
			std::string files[5] = {
				"C:\\voronoi\\TwoDiamonds.lrn",
				"C:\\voronoi\\EngyTime.lrn",
				"C:\\voronoi\\Lsun.lrn",
				"C:\\voronoi\\Target.lrn",
				"C:\\voronoi\\WingNut.lrn"
			};
			std::cout << "Type index of using file: " << std::endl;
			for (int i = 0; i < 5; i++) {
				std::cout << "To use: " << files[i] << ", type " << i + 1 << std::endl;
			}
			int idx;
			std::cin >> idx;
			idx = std::min(5, std::max(0, idx));
			std::string fileWithPoints = files[idx - 1];
			std::cout << "Using file with points: " << fileWithPoints << std::endl;

			std::ifstream pointsStream;
			pointsStream.open(fileWithPoints.c_str(), std::ifstream::in);
			if (!pointsStream.good()) {
				std::cout << "Error of opening file" << std::endl;
				throw std::string("Error of opening");
			}
			
			std::string percent;
			pointsStream >> percent;
			check(percent == "%");
			int nPoints;
			pointsStream >> nPoints;

			char buf[1024];
			pointsStream.getline(buf, 1024);
			pointsStream.getline(buf, 1024);
			pointsStream.getline(buf, 1024);
			pointsStream.getline(buf, 1024);

			int scale = 20;
			std::cout << "Points: " << nPoints << ", scaling factor =" << scale << "x" << std::endl;

			for (int i = 0; i < nPoints; i++) {
				float x;
				float y;
				int idx;
				pointsStream >> idx >> x >> y;
				points.push_back(Point(x * scale, y * scale));
			}

			pointsStream.close();
		} else {
			std::cout << "Unknown token = " << type << ", exiting.." << std::endl;
			throw std::string("Unknown type");
		}
		
		SyncType syncType;
		
		if (syncTypeString == "phase" || 
			syncTypeString == "p" || 
			syncTypeString == "P" || 
			syncTypeString == "1"
		) {
			syncType = PHASE;
		} else if (
			syncTypeString == "fragmentary" || 
			syncTypeString == "f" || 
			syncTypeString == "F" ||
			syncTypeString == "2"
		) {
			syncType = FRAGMENTARY;
		} else {
			throw std::string("Wrong synchronization type name: " + syncTypeString);
		}

		check(freopen((WORKING_DIR + "result.txt").c_str(), "w", stdout) != NULL);
		
		printf("size = %d\n", points.size());

		{

			if (syncType == FRAGMENTARY) {
				voronoi::DelaunayComputingQhull diagram(points);
				for (float fragmentaryEPS = 0.01f; fragmentaryEPS <= 0.1f; fragmentaryEPS += 0.003f) {
					std::vector <float> rates;

					for (float successRateLocal = 0.10f; successRateLocal < 0.26f; successRateLocal += 0.01f) {
						rates.push_back(successRateLocal);
					}
					NeuralNetwork network2(points, diagram);
					std::stringstream ss;
					ss << "result_qhull_fragm_eps_";
					ss.setf(std::ios::fixed, std::ios::floatfield);
					ss.precision(4);
					ss << fragmentaryEPS; 
					ss << ".bmp";
					network2.process(ss.str(), syncType, rates, fragmentaryEPS);
				}
			} else {
				voronoi::DelaunayComputingQhull diagram(points);
				std::vector <float> rates;

				for (float successRateLocal = 0.50f; successRateLocal < 1.f; successRateLocal += 0.001f) {
					rates.push_back(successRateLocal);
				}
				
				NeuralNetwork network2(points, diagram);
				std::stringstream ss;
				ss << "result_qhull_phase" << ".bmp";
				network2.process(ss.str(), syncType, rates, 0);
			}

		}

		if (false) {

//			voronoi::VoronoiFortuneComputing diagram(points);

//			NeuralNetwork network(points, diagram);
//			network.process(WORKING_DIR + "result_fortu.bmp", syncType, successRate,0);

		}

	} catch (const std::string &message) {
		printf("Error, message = %s", message.c_str());
	} catch (...) {
		printf("Unknown exception caught\n");
	}
	return 0;
}