#ifndef SIMPLE_MATH_H_
#define SIMPLE_MATH_H_

#include <iostream>
#include <list>
#include <algorithm>
#include <set>


namespace voronoi {

void check(bool e) {
	if (!e) {
		throw std::string("JOPKA");
	}
}
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

} // namespace voronoi

#endif // SIMPLE_MATH_H_
