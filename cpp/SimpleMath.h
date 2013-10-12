#ifndef SIMPLE_MATH_H_
#define SIMPLE_MATH_H_

#include <iostream>
#include <list>
#include <algorithm>
#include <string>
#include <vector>
#include <set>

#define QUOTE(A) #A

const std::string WORKING_DIR = "C:/voronoi/";

const std::string INPUT_FILE_NAME = WORKING_DIR + "input.bmp";

enum SyncType {
	PHASE,
	FRAGMENTARY
};

std::vector<int> processOscillatoryChaoticNetworkDynamics(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
	const float fragmentaryEPS
);

inline void check(bool b) {
	if (!b) {
		throw std::string("check: error");
	}
}
//#define check(EXPRESSION) \
//	if (!EXPRESSION) { \
//		throw std::string(QUOTE(EXPRESSION)); \
//	}


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


namespace voronoi {


class Line;
class Point;
struct Breakpoint;
struct BreakpointComp;
class Triple;


class Vector {
public:
	const float x;
	const float y;
	
	Vector(const Point &base, const Point &p);
	
	Vector(float x, float y);
	
	Vector rotate90ccw() const;
};


class Line {
public:
	float a;
	float b;
	float c;
	
	Line(const Point &p1, const Point &p2);
	
	Point getIntersectionPoint(const Line &second) const;
	
	bool isParallelTo(const Line &second) const;
};

class Point {
public:
	float x;
	float y;
	
	Point(float x, float y);

	std::string prints() const;
	
	static Point getMiddlePoint(const Point &a, const Point &b);
	
	static Point getCircumcenterShiftedUpOnRadius(const Point &a, const Point &b, const Point &c);
	
	static Point getCircumcenter(const Point &a, const Point &b, const Point &c);
	
	static std::list<Point> getCircumcenters(const Point &inputPoint1, const Point &inputPoint2, float tangentY);
	
	Point add(const Vector &v) const;
	
	float squaredDistanceTo(const Point &second) const;
	
	float distanceTo(const Point &second) const;
	
	static Point getLowestPoint() {
		return Point(0.0f, -1e7f);
	}
	
	bool weakEqualsTo(const Point &that) const;
};

class Triple {
public:
	const Point *prevSite;
	const Point *mainSite;
	const Point *nextSite;
	bool isCircumcenterEvaluated;
	Point circumcenter;
	
	Triple(const Point *prev, const Point *curr, const Point *next);

	Triple(const Point *prev, const Point *curr, const Point *next, float sweepLineY);
	
	void recalcCircumcenter(float sweepLineY);
	
	bool isDegenerated(const Point &shiftedCenter) const;

	void addIfGood(std::set<Breakpoint, BreakpointComp> &breakpoints) const;
	
	void removeIfPossible(std::set<Breakpoint, BreakpointComp> &breakpoints) const;
	
	Point getCircleCenter() const;
	
	float getMaxY() const;

	std::list<float> getSortedXs(const Point &first, const Point &second) const;
				
	std::list<float> getBounds(float tangentY) const;
	
	std::string prints();
};

int signum(float value);

struct PointComparatorY : public std::binary_function<Point, Point, bool> {
	bool operator() (const Point &first, const Point &second) {
		int signY = signum(first.y - second.y);
		if (signY != 0) {
			return signY < 0;
		} else {
			return signum(first.x - second.x) < 0;
		}
	}
};

struct PointComparatorX : public std::binary_function<Point, Point, bool> {
	bool operator() (const Point &o1, const Point &o2) {
		return o1.x < o2.x;
	}
};

struct Breakpoint {
	Point point;
	Triple triple;
	
	Breakpoint(const Point &point, const Triple &triple);
};

struct NeighborsList {
	Point site;
	std::set<Point, PointComparatorY> nextToThis;

	NeighborsList(Point site)
		: site(site)
	{
	}

	void add(Point next) {
		nextToThis.insert(next);
	}

	double getAverageDistance() {
		if (nextToThis.size() == 0) {
			return 0;
		}
		double distance = 0.0;
		for (std::set<Point, PointComparatorY>::iterator it = nextToThis.begin(); it != nextToThis.end(); ++it) {
			distance += it->distanceTo(site);
		}
		return distance / (nextToThis.size() + 0.0);
	}

	bool operator < (const NeighborsList &second) const {
		return PointComparatorY()(this->site, second.site);
	}
};

struct BreakpointComp : public std::binary_function<Breakpoint, Breakpoint, bool> {
	bool operator() (const Breakpoint &first, const Breakpoint &second) const;
};

Point* LOWEST();

struct TripleComp : public std::binary_function<Triple, Triple, bool> {
	int compare(const Triple &first, const Triple &second) const;
	bool operator () (const Triple &first, const Triple &second) const;
};

class NeighborsListContainer {
public: 
	std::set<NeighborsList> adjList;

protected:
	
	void addData(const Point *main, const Point *a, const Point *b) {
		NeighborsList list(*main);
		if (adjList.find(list) != adjList.end()) {
			// gets an element
			std::set<NeighborsList>::iterator it = adjList.find(list);
			it->add(*a);
			it->add(*b);
		} else {
			list.add(*a);
			list.add(*b);
			adjList.insert(list);
		}
	}
		
	void addDataFromTriple(const Point *a, const Point *b, const Point *c) {
		addData(a, b, c);
		addData(b, a, c);
		addData(c, a, b);
	}
};

class VoronoiFortuneComputing : public NeighborsListContainer {
public:
	VoronoiFortuneComputing(std::vector<Point> &p);
	
private:

	/**
	 * С этим порой жопа. Требуется взять все точки, имеющие минимальную ординату,
	 * и сделать структуру вида
	 * (LOWEST, 1, 2)
	 * (1, 2, 3)
	 * (2, 3, 4)
	 * (3, 4, LOWEST)
	 * 
	 * Никаких событий данные фиговины не генерируют, ибо все в один ряд.
	 * */
	int processFirstLineSiteEvents(const std::vector<Point> &p, std::set<Triple, TripleComp> &beachArcs);

	void processSiteEvent(
		const Point &curr, 
		std::set<Breakpoint, BreakpointComp> &breakpoints, 
		std::set<Triple, TripleComp> &beachArcs
	);
	
	void processVertexEvent(
		std::set<Breakpoint, BreakpointComp> &breakpoints, 
		std::set<Triple, TripleComp> &beachArcs, 
		std::set<Point, PointComparatorY> &voronoi, 
		float sweepLineY, 
		Point *probablyInclude
	);
};

#include <stdio.h>

class DelaunayComputingQhull : public NeighborsListContainer {
public:
	
	DelaunayComputingQhull(std::vector<Point> &p) {
		const std::string INPUT = WORKING_DIR + "curSaves.txt";
		const std::string OUTPUT = WORKING_DIR + "curResult.txt";

		FILE *savedInput = fopen(INPUT.c_str(), "wb");
		fprintf(savedInput, "%d %d", 2, static_cast<int>(p.size()));
		fprintf(savedInput, "\r\n");
		for (int i = 0; i < static_cast<int>(p.size()); i++) {
			fprintf(savedInput, "%f %f \r\n", p[i].x, p[i].y);
		}
		fclose(savedInput);

		std::string cmd(WORKING_DIR + "qdelaunay.exe TI '" + INPUT + "' i TO '" + OUTPUT + "'");
		FILE *file = _popen(cmd.c_str(), "r");
		_pclose(file);

		FILE *out = fopen(OUTPUT.c_str(), "rb");
		if (out == NULL) {
			throw std::string("error while opening output file " + OUTPUT);
		}


		int nTriples;
		fscanf(out, "%d", &nTriples);
		for (int i = 0; i < nTriples; i++) {
			int first;
			int second;
			int third;
			fscanf(out, "%d %d %d", &first, &second, &third);
			this->addDataFromTriple(&p[first], &p[second], &p[third]);
		}
		fclose(out);
	}
private:

};

} // namespace voronoi

#endif // SIMPLE_MATH_H_
