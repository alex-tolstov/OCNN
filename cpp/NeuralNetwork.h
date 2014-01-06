#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include <map>
#include "SimpleMath.h"

class NeuralNetwork {

private:

	std::vector<voronoi::Point> points;

	std::map<voronoi::Point, std::set<voronoi::Point, voronoi::PointComparatorY>, voronoi::PointComparatorY> adjancent;
	
	std::vector<float> weightMatrix;
	
	double totalAverage;

	void createVoronoiDiagram(voronoi::NeighborsListContainer &diagram);

	void calcWeightCoefs();
	
	static const int N_DEFINED_COLORS;

public:

	NeuralNetwork(const std::vector<voronoi::Point> &points, voronoi::NeighborsListContainer &diagram);
	
	void process(const std::string &fileName, SyncType syncType, std::vector<float> &successRates, float fragmentaryEPS, bool useSimpleComputingMode);
};

#endif // NEURALNETWORK_H_
