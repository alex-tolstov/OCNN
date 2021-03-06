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
#include "NeuralNetwork.h"
#include "find_points.h"

#define NOMINMAX

#include "timer.h"

int main() {
	try {
		checkCudaCall(cudaSetDevice(0));
		//freopen("C:/voronoi/test", "rb", stdin);

		std::cout << "[phase,p,P,1|fragmentary,f,F,2] " << std::endl;

		std::string syncTypeString = "1";
		std::cin >> syncTypeString; 
		
		//float successRate = 0.f;
		//std::cin >> successRate;

		std::cout << "To read from " << INPUT_FILE_NAME << ", type 1" << std::endl;
		std::cout << "To read points from file, type 2" << std::endl;
		std::cout << "To use generator of random points in [-100, 100] interval, type 3" << std::endl;
		std::cout << "To read points from modified FCPS base, type 4" << std::endl;
		
		std::string type="3";
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
			throw std::runtime_error("Wrong synchronization type name: " + syncTypeString);
		}

		std::cout << "TEST MODE: type 1 to use single thread computing, for slow cards " 
				  << std::endl 
				  << "type 2 to use block computing, for speedy cards" 
				  << std::endl
				  << "type 3 to use cpu (with openmp) computing, optimized by MS VC++"
				  << std::endl;

		std::string singleThreadFlag = "2";
		std::cin >> singleThreadFlag;
		if (singleThreadFlag == "1") {
			std::cout << "using single thread computing mode" << std::endl;
		} else if (singleThreadFlag == "2") {
			std::cout << "using block computing mode" << std::endl;
		} else if (singleThreadFlag == "3") {
			std::cout << "using cpu computing mode" << std::endl;
		} else {
			throw std::runtime_error("Wrong computing mode flag");
		}
		
		std::cout << "type number of algorithm's iteration to run" << std::endl;

		int nIterations = 1000;
		std::cin >> nIterations;

		const std::string REPORT_DIR = WORKING_DIR + "report\\";
		_mkdir(REPORT_DIR.c_str());
	//	check(freopen((REPORT_DIR + "result.txt").c_str(), "w", stdout) != NULL);
		
		printf("size = %d\n", points.size());

		{
			DWORD startQhull = GetTickCount();
			voronoi::DelaunayComputingQhull diagram(points);
			DWORD finishQhull = GetTickCount();
			printf("time delaunay = %.3f\r\n", (finishQhull - startQhull) * 0.001f);
			if (syncType == FRAGMENTARY) {
				for (float fragmentaryEPS = 0.20f; fragmentaryEPS <= 0.35f; fragmentaryEPS += 0.01f) {
					std::vector <float> rates;

					for (float successRateLocal = 0.5f; successRateLocal < 0.9f; successRateLocal += 0.01f) {
						rates.push_back(successRateLocal);
					}
					NeuralNetwork network(points, diagram);
					network.process("result_fragm.bmp", syncType, rates, fragmentaryEPS, singleThreadFlag, nIterations);
				}
			} else {
				std::vector <float> rates;

				for (float successRateLocal = 0.50f; successRateLocal < 1.f; successRateLocal += 0.01f) {
					rates.push_back(successRateLocal);
				}
				
				NeuralNetwork network(points, diagram);
				network.process("result_phase.bmp", syncType, rates, 0, singleThreadFlag, nIterations);
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