#include "NeuralNetwork.h"
#include "bmpworks.h"
#include "groups.h"


#define NOMINMAX
#include "timer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

const int NeuralNetwork::N_DEFINED_COLORS = 8;

void NeuralNetwork::createVoronoiDiagram(voronoi::NeighborsListContainer &diagram) {
	int nPoints = static_cast<int>(points.size());
	check(nPoints > 0);
	double sumAverages = 0.0;
	for (std::set<voronoi::NeighborsList>::iterator it = diagram.adjList.begin();
		it != diagram.adjList.end();
		++it
	) {
		adjancent.insert(std::pair<voronoi::Point, std::set<voronoi::Point, voronoi::PointComparatorY> >(it->site, it->nextToThis));
	//	printf("[%s]\r\n", it->site.prints().c_str());
		for (
			std::set<voronoi::Point, voronoi::PointComparatorY>::iterator i = it->nextToThis.begin();
			i != it->nextToThis.end();
			++i
		) {
	//		printf("%s ",i->prints().c_str());
		}
	//	printf("\r\n");
		sumAverages += it->getAverageDistance();
	}
	totalAverage = sumAverages / (nPoints + 0.0);
	printf("total average = %5.4f\r\n", totalAverage);
}

void NeuralNetwork::calcWeightCoefs() {
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
			//const std::set<voronoi::Point, voronoi::PointComparatorY> &currAdj = adjancent.find(basePoint)->second;

			for (int j = i + 1; j < nPoints; j++) {
				const voronoi::Point &currPoint = points[j];
				if (basePoint.distanceTo(currPoint) < averageDistance) {
					double dist = sqrt(points[i].squaredDistanceTo(points[j]));
					double norm = pow(dist / totalAverage, 3.5);
					double k = norm / (2.0);
					float result = static_cast<float>(exp(-k));
					weightMatrix[i * nPoints + j] = result;
					weightMatrix[j * nPoints + i] = result;
				} else {
					double dist = sqrt(points[i].squaredDistanceTo(points[j]));
					double norm = pow(dist / totalAverage, 3.5);
					double k = norm / (2.0);
					float result = static_cast<float>(exp(-k));

					weightMatrix[i * nPoints + j] = result;
					weightMatrix[j * nPoints + i] = result;
				}
				float value = weightMatrix[i * nPoints + j];
				minCoef = std::min(value, minCoef);
				maxCoef = std::max(value, maxCoef);
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


NeuralNetwork::NeuralNetwork(const std::vector<voronoi::Point> &points, voronoi::NeighborsListContainer &diagram) 
	: points(points)
{
	createVoronoiDiagram(diagram);
	calcWeightCoefs();
}

#include "cpuimpl.h"

void NeuralNetwork::process(
	const std::string &fileName, 
	SyncType syncType, 
	std::vector<float> &successRates, 
	float fragmentaryEPS, 
	std::string singleThreadFlag,
	const int nIterations
) {
	std::vector<float> sheet;
	const int nNeurons = this->points.size();
	long long startCudaTime = GetTickCount();

	std::vector<int> hits;
	if (singleThreadFlag != "3") {
		check(singleThreadFlag == "2" || singleThreadFlag == "1");
		hits = ::processOscillatoryChaoticNetworkDynamics(
			nNeurons, 
			this->weightMatrix,
			5,
			nIterations,
			syncType,
			sheet,
			fragmentaryEPS,
			singleThreadFlag == "1"
		);
	} else {
		hits = processOscillatoryChaoticNetworkDynamicsCPU(
			nNeurons, 
			this->weightMatrix,
			5,
			nIterations,
			syncType,
			sheet,
			fragmentaryEPS
		);
	}
	check(sheet.size() == nIterations * nNeurons);
	DWORD finishCudaTime = GetTickCount();
	printf("time seconds for neural network calculation and analysis = %5.3f\r\n", (finishCudaTime - startCudaTime) * 0.001f);
	const std::string REPORT_DIR = WORKING_DIR + "report\\";
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
	DWORD preparedSheetTime = GetTickCount();
	printf("time seconds for preparing sheet = %5.3f\r\n", (preparedSheetTime - finishCudaTime) * 0.001f);
	
	bitmapSheet.WriteToFile(std::string(REPORT_DIR + "sheet.bmp").c_str());
	DWORD savedSheetTime = GetTickCount();
	printf("time seconds for saving sheet to disk = %5.3f\r\n", (savedSheetTime - preparedSheetTime) * 0.001f);
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
				//printf("Got a new group: ");
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
					printCross(bitmap, x, y, red, green, blue);
				}
			}
		}

		for (int i = 0; i < static_cast<int>(points.size()); i++) {
			if (!used[i]) {
				int x = static_cast<int>(points[i].x + 256);
				int y = static_cast<int>(points[i].y + 256);
				printCross(bitmap, x, y, 0, 0, 0);
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
	DWORD savedResultsTime = GetTickCount();
	printf("time seconds for saving results to disk = %5.3f\r\n", (savedResultsTime - savedSheetTime) * 0.001f);
}
