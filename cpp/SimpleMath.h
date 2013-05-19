#ifndef SIMPLE_MATH_H_
#define SIMPLE_MATH_H_

#include <iostream>
#include <list>
#include <algorithm>
#include <set>


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

struct Breakpoint {
	Point point;
	Triple triple;
	
	Breakpoint(const Point &point, const Triple &triple);
};

struct BreakpointComp : public std::binary_function<Breakpoint, Breakpoint, bool> {
	bool operator() (const Breakpoint &first, const Breakpoint &second) const;
};

Point* LOWEST();

#endif // SIMPLE_MATH_H