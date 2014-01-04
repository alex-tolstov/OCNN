#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <map>
#include <direct.h>

#include "SimpleMath.h"

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

void setPixelValue(BMP &bitmap, int x, int y, int r, int g, int b) {
	bitmap(x, y)->Red = r;
	bitmap(x, y)->Green = g;
	bitmap(x, y)->Blue = b;
}

void hsv2rgb(int h, int s, int v, int &r, int &g, int &b) {
	int part = h / 60;
	int vMin = (100 - s) * v / 100;
	int a = (v - vMin) * (h % 60) / 60;
	
	int vInc = vMin + a;
	int vDec = v - a;
	
	switch (part) {
		case 0:
			r = v;
			g = vInc;
			b = vMin;
			break;
		case 1:
			r = vDec;
			g = v;
			b = vMin;
			break;
		case 2:
			r = vMin;
			g = v;
			b = vInc;
			break;
		case 3:
			r = vMin;
			g = vDec;
			b = v;
			break;
		case 4:
			r = vInc;
			g = vMin;
			b = v;
			break;
		case 5:
			r = v;
			g = vMin;
			b = vDec;
			break;
		default:
			break;
	}
	r = std::max(0, std::min(r, 255));
	g = std::max(0, std::min(g, 255));
	b = std::max(0, std::min(b, 255));
}

void printPoint(BMP &bitmap, int x, int y, int r, int g, int b) {
	const int shiftX[5] = {0, 0, 0, 1, -1};
	const int shiftY[5] = {0, 1, -1, 0, 0};
	for (int i = 0; i < 5; i++) {
		int currX = std::min(512, std::max(x + shiftX[i], 0));
		int currY = std::min(512, std::max(y + shiftY[i], 0));

		setPixelValue(bitmap, currX, currY, r, g, b);
	}
}

class NeuralNetwork {
private:
	std::vector<voronoi::Point> points;
	std::map<voronoi::Point, std::set<voronoi::Point, voronoi::PointComparatorY>, voronoi::PointComparatorY> adjancent;
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
			adjancent.insert(std::pair<voronoi::Point, std::set<voronoi::Point, voronoi::PointComparatorY> >(it->site, it->nextToThis));
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
		double sumDistance = 0.0;
		std::vector<float> distances(nPoints * nPoints);

		for (int i = 0; i < nPoints; i++) {
			for (int j = 0; j < nPoints; j++) {
				float currDist = points[i].distanceTo(points[j]);
				distances[i * nPoints + j] = currDist;
				sumDistance += currDist;
			}
		}
		std::sort(distances.begin(), distances.end());
		
		float averageDistance = static_cast<float>(sumDistance / (nPoints * nPoints * 1.0));
		averageDistance = distances[distances.size() / 2]; // mediana
		
		BMP weightCoefsBitmap;
		weightCoefsBitmap.SetSize(nPoints, nPoints);

		float minCoef = 10e5f;
		float maxCoef = -10e5f;
		for (int i = 0; i < nPoints; i++) {
			const voronoi::Point &basePoint = points[i];
			
			if (adjancent.find(basePoint) != adjancent.end()) {
				const std::set<voronoi::Point, voronoi::PointComparatorY> &currAdj = adjancent.find(basePoint)->second;

				for (int j = i + 1; j < nPoints; j++) {
					const voronoi::Point &currPoint = points[j];
					if (basePoint.distanceTo(currPoint) < averageDistance) {
						double dist = sqrt(points[i].squaredDistanceTo(points[j]));
						double norm = pow(dist / totalAverage, 2);
						double k = norm / (2.0);
						float result = static_cast<float>(exp(-k));
						weightMatrix[i * nPoints + j] = result;
						weightMatrix[j * nPoints + i] = result;
					} else {
						double dist = sqrt(points[i].squaredDistanceTo(points[j]));
						double norm = pow(dist / totalAverage, 2);
						double k = norm / (2.0);
						float result = static_cast<float>(exp(-k));

						weightMatrix[i * nPoints + j] = result;
						weightMatrix[j * nPoints + i] = result;
					}
					float value = weightMatrix[i * nPoints + j];
					minCoef = std::min(value, minCoef);
					maxCoef = std::max(value, maxCoef);
/*
					if (currAdj.count(currPoint) > 0) {
						double dist = sqrt(points[i].squaredDistanceTo(points[j]));
						double norm = pow(dist / totalAverage, 2);
						double k = norm / (2.0);
						float result = static_cast<float>(exp(-k));
						weightMatrix[i * nPoints + j] = result;
						weightMatrix[j * nPoints + i] = result;
					} else {
						double dist = sqrt(points[i].squaredDistanceTo(points[j]));
						double norm = pow(dist / totalAverage, 2);
						double k = norm / (2.0);
						float result = static_cast<float>(exp(-k)) / 8;

						weightMatrix[i * nPoints + j] = result;
						weightMatrix[j * nPoints + i] = result;
					}
*/
				}
			}
		}

		for (int i = 0; i < nPoints; i++) {
			for (int j = 0; j < nPoints; j++) {
				float value = weightMatrix[i * nPoints + j];
				int repr = static_cast<int>(255 * (1.f - (value - minCoef) / (maxCoef - minCoef)));
				setPixelValue(weightCoefsBitmap, i, j, repr, repr, repr);
			}
		}
		std::cout << "max coef = " << maxCoef << ", minCoef = " << minCoef << std::endl;
		weightCoefsBitmap.WriteToFile(std::string(WORKING_DIR + "report/coefsMatrix.bmp").c_str());
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
		const int nIterations = 2000;
		std::vector<float> sheet;
		const int nNeurons = this->points.size();
		std::vector<int> hits = ::processOscillatoryChaoticNetworkDynamics(
			nNeurons, 
			this->weightMatrix,
			2,
			nIterations,
			syncType,
			sheet,
			fragmentaryEPS
		);
		check(sheet.size() == nIterations * nNeurons);


		const std::string REPORT_DIR = WORKING_DIR + "report\\";
		//_rmdir(REPORT_DIR.c_str());
		_mkdir(REPORT_DIR.c_str());
		BMP bitmapSheet;

		bitmapSheet.SetSize(nIterations, nNeurons);
		float minValue = 100;
		float maxValue = -100;
		std::vector<int> colorCount(361, 0);
		for (int iter = 0; iter < nIterations; iter++) {
			for (int neuron = 0; neuron < nNeurons; neuron++) {
				float value = sheet[iter * nNeurons + neuron];
				minValue = std::min(minValue, value);
				maxValue = std::max(maxValue, value);

				int color = static_cast<int>(360 * ((value + 1.f) / 2.f));
				color = std::min(360, std::max(0, color));
				colorCount[color]++;
				int h = color;
				int s = 100;
				int v = 120; // extra value

				int red = 0;
				int green = 0;
				int blue = 0;

				hsv2rgb(h, s, v, red, green, blue);

				setPixelValue(bitmapSheet, iter, neuron, red, green, blue);
			}
		}
		for (int i = 0; i < static_cast<int>(colorCount.size()); i++) {
			printf("clrs: %d - %d\r\n", i, colorCount[i]);
		}
		printf("min = %f, max = %f\n", minValue, maxValue);
		bitmapSheet.WriteToFile(std::string(REPORT_DIR + "sheet.bmp").c_str());

		for (int sr = 0; sr < static_cast<int>(successRates.size()); sr++) {
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
						int x = static_cast<int>(points[groups[i].list[j]].x + 256);
						int y = static_cast<int>(points[groups[i].list[j]].y + 256);
						used[groups[i].list[j]] = true;
						printPoint(bitmap, x, y, red, green, blue);
						printf("[%s] ", points[groups[i].list[j]].prints().c_str());
					}
					printf("\r\n");
				}
			}

			for (int i = 0; i < static_cast<int>(points.size()); i++) {
				if (!used[i]) {
					int x = static_cast<int>(points[i].x + 256);
					int y = static_cast<int>(points[i].y + 256);
					printPoint(bitmap, x, y, 0, 0, 0);
				}
			}
			std::stringstream ss;
			ss << REPORT_DIR;
			if (syncType == FRAGMENTARY) {
				ss << "eps_";
				ss.setf(std::ios::fixed, std::ios::floatfield); 
				ss.precision(4);
				ss << fragmentaryEPS;
				ss << "_";
			}
			ss << "corr_";
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
				float x = static_cast<float>(rand() % 100);
				float y = static_cast<float>(rand() % 100);
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
	//		std::sort(points.begin(), points.end(), voronoi::PointComparatorX());
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
			voronoi::DelaunayComputingQhull diagram(points);
			if (syncType == FRAGMENTARY) {
				for (float fragmentaryEPS = 0.0005f; fragmentaryEPS <= 0.005f; fragmentaryEPS += 0.001f) {
					std::vector <float> rates;

					for (float successRateLocal = 0.01f; successRateLocal < 0.07f; successRateLocal += 0.003f) {
						rates.push_back(successRateLocal);
					}
					NeuralNetwork network(points, diagram);
					network.process("result_fragm.bmp", syncType, rates, fragmentaryEPS);
				}
			} else {
				std::vector <float> rates;

				for (float successRateLocal = 0.60f; successRateLocal < 1.f; successRateLocal += 0.01f) {
					rates.push_back(successRateLocal);
				}
				
				NeuralNetwork network(points, diagram);
				network.process("result_phase.bmp", syncType, rates, 0);
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