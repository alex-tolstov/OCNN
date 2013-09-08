// TestingFortune.cpp : Defines the entry point for the console application.
//


#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <time.h>

#include "SimpleMath.h"

namespace voronoi {

	
	static const float EPS = 1e-5f;

	int signum(float value) {
		if (value > 0) {
			return 1;
		} else if (value < 0) {
			return -1;
		} else {
			return 0;
		}
	}
	
	
	Vector::Vector(const Point &base, const Point &p) 
		: x(p.x - base.x)
		, y(p.y - base.y)
	{
	}
		
	Vector::Vector(float x, float y)
		: x(x)
		, y(y)
	{
	}
		
	Vector Vector::rotate90ccw() const {
		return Vector(-y, x);
	}
	
	static float calcDeterminant(float a, float b, float c, float d) {
		return a*d - b*c;	
	}
	
	Line::Line(const Point &p1, const Point &p2) {
		bool equalty = p1.weakEqualsTo(p2);
		check(!equalty);
		a = p1.y - p2.y;
		b = p2.x - p1.x;
		c = -(a * p1.x + b * p1.y);
	}
		
	Point Line::getIntersectionPoint(const Line &second) const {
		check(!this->isParallelTo(second));
		float d0 = calcDeterminant(a, b, second.a, second.b);
		float dA = calcDeterminant(-c, b, -second.c, second.b);
		float dB = calcDeterminant(a, -c, second.a, -second.c);
		return Point(dA / d0, dB / d0);
	}
		
	bool Line::isParallelTo(const Line &second) const {
		return fabs(this->a * second.b - second.a * this->b) < EPS;
	}
	
		
	Point::Point(float x, float y)
		: x(x)
		, y(y)
	{
	}

	std::string Point::prints() const {
		std::stringstream stream;
		stream << x << " " << y;
		return stream.str();
	}
	
	Point Point::getMiddlePoint(const Point &a, const Point &b) {
		return Point((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
	}
		
	Point Point::getCircumcenterShiftedUpOnRadius(const Point &a, const Point &b, const Point &c) {
		Point center = Point::getCircumcenter(a, b, c);
		center.y += center.distanceTo(a);
		return center;
	}
		
	Point Point::getCircumcenter(const Point &a, const Point &b, const Point &c) {
		if (a.weakEqualsTo(b)) {
			return Point::getMiddlePoint(a, c);
		} else if (a.weakEqualsTo(c)) {
			return Point::getMiddlePoint(a, b);
		} else if (b.weakEqualsTo(c)) {
			return Point::getMiddlePoint(a, b);
		}
		Vector ab(a, b);
		Vector ac(a, c);
		
		Point centerAB = Point::getMiddlePoint(a, b);
		Point centerAC = Point::getMiddlePoint(a, c);
		
		Point abNormal = centerAB.add(ab.rotate90ccw());
		Point acNormal = centerAC.add(ac.rotate90ccw());
		return Line(centerAB, abNormal).getIntersectionPoint(Line(centerAC, acNormal));
	}
		
	std::list<Point> Point::getCircumcenters(const Point &inputPoint1, const Point &inputPoint2, float tangentY) {
		Point first = inputPoint1;
		Point second = inputPoint2;

		first.y -= tangentY;
		second.y -= tangentY;
		
		Vector firstSecond = Vector(first, second).rotate90ccw();
		
		Point centerFS = Point::getMiddlePoint(first, second);
		Point another = centerFS.add(firstSecond);
					
		float gA = centerFS.y - another.y;
		float gB = another.x - centerFS.x;
		float gC = -(gA * centerFS.x + gB * centerFS.y);
		
		if (fabs(gB) < EPS) {
			first.y += tangentY;
			second.y += tangentY;
			
			float x0 = -gC / gA;
			float y0 = ( (x0 - second.x) * x0 + 0.5f * (second.x * second.x + second.y * second.y - x0 * x0) ) / second.y;
			
			std::list<Point> result;
			result.push_back(Point(x0, y0 + tangentY));
			return result;
		} else {
			float a = 0.5f;
			float b = gA * second.y / gB - second.x;
			float c = gC * second.y / gB + (second.x * second.x + second.y * second.y) * 0.5f;
			
			first.y += tangentY;
			second.y += tangentY;
			
			float d = b * b - 4.f * a * c;
			
			float root = (float)sqrt(d);
			if (root < 0) {
				return std::list<Point>();
			}
			float x0 = (-b + root) / (2 * a);
			float x1 = (-b - root) / (2 * a);
			
			float y0 = (-gC - gA * x0) / gB;
			float y1 = (-gC - gA * x1) / gB;
			
			std::list<Point> result;
			result.push_back(Point(x0, y0 + tangentY));
			result.push_back(Point(x1, y1 + tangentY));
			return result;
		}
	}
		
	Point Point::add(const Vector &v) const {
		return Point(this->x + v.x, this->y + v.y);
	}
		
//	std::string Point::prints() const {
//		std::stringstream stream;
//		stream << x.prints() << " " << y.prints();
//		return stream.str();
//	}
		
	float Point::squaredDistanceTo(const Point &second) const {
		float dx = this->x - second.x;
		float dy = this->y - second.y;
		return dx * dx + dy * dy;
	}
		
	float Point::distanceTo(const Point &second) const {
		return (float)sqrtf(this->squaredDistanceTo(second));
	}
		
	bool Point::weakEqualsTo(const Point &that) const {
		return fabs(this->x - that.x) < EPS && fabs(this->y - that.y) < EPS;
	}

	
	
	Triple::Triple(const Point *prev, const Point *curr, const Point *next)
		: prevSite(prev)
		, mainSite(curr)
		, nextSite(next)
		, isCircumcenterEvaluated(false)
		, circumcenter(0, 0)
	{
		float maks = std::max(prev->y, std::max(curr->y, next->y));
		this->recalcCircumcenter(maks);
	}
	
	Triple::Triple(const Point *prev, const Point *curr, const Point *next, float sweepLineY) 
		: prevSite(prev)
		, mainSite(curr)
		, nextSite(next)
		, isCircumcenterEvaluated(false)
		, circumcenter(0,0)
	{
		this->recalcCircumcenter(sweepLineY);
	}
	
	void Triple::recalcCircumcenter(float sweepLineY) {
		this->isCircumcenterEvaluated = false;
		if (prevSite->x == mainSite->x && mainSite->x == nextSite->x) {
			return;
		}
		if (prevSite->y == mainSite->y && mainSite->y == nextSite->y) {
			return;
		}
		Point center = Point::getCircumcenterShiftedUpOnRadius(*prevSite, *mainSite, *nextSite);
		if (prevSite != nextSite && prevSite != LOWEST() && nextSite != LOWEST()) {
			if (!isDegenerated(center)) {
				this->isCircumcenterEvaluated = true;
				this->circumcenter = center;
			}
		}
	}
	
	bool Triple::isDegenerated(const Point &shiftedCenter) const {
		if (mainSite->y > prevSite->y && mainSite->y > nextSite->y) {
//			для обработки вырожденных случаев
//			например:
//			4
//			1 1
//			2 1
//			3 1
//			2 2
			return true;
		}
		const float WEAK_EPS = 1e-1f;
		std::list<float> boundsX = this->getBounds(shiftedCenter.y);
		float diff1 = fabs(0.0f + boundsX.front() - shiftedCenter.x);
		float diff2 = fabs(0.0f + boundsX.back() - shiftedCenter.x);
		return diff1 > WEAK_EPS || diff2 > WEAK_EPS;
	}
	
	void Triple::addIfGood(std::set<Breakpoint, BreakpointComp> &breakpoints) const {
		if (this->isCircumcenterEvaluated) {
			breakpoints.insert(Breakpoint(this->circumcenter, *this));
		}
	}
	
	void Triple::removeIfPossible(std::set<Breakpoint, BreakpointComp> &breakpoints) const {
		if (this->isCircumcenterEvaluated) {
			breakpoints.erase(Breakpoint(this->circumcenter, *this));
		}
	}
	
	Point Triple::getCircleCenter() const {
		return Point::getCircumcenter(*this->prevSite, *this->mainSite, *this->nextSite);
	}
	
	float Triple::getMaxY() const {
		return std::max(prevSite->y, std::max(mainSite->y, nextSite->y));
	}
	
	std::list<float> Triple::getSortedXs(const Point &first, const Point &second) const {
		std::list<float> resultX;
		resultX.push_back(std::min(first.x, second.x));
		resultX.push_back(std::max(first.x, second.x));
		return resultX;
	}
				
	std::list<float> Triple::getBounds(float tangentY) const {
		if (prevSite == nextSite) {
			std::list<Point> points = Point::getCircumcenters(*this->mainSite, *this->prevSite, tangentY);
			check(points.size() == 2);
			return getSortedXs(points.front(), points.back());
		} else {
			std::list<Point> withPrev = Point::getCircumcenters(*this->mainSite, *this->prevSite, tangentY);
			std::list<Point> withNext = Point::getCircumcenters(*this->mainSite, *this->nextSite, tangentY);
			
			std::vector<Point> v1(withPrev.begin(), withPrev.end());
			std::vector<Point> v2(withNext.begin(), withNext.end());
			
			std::sort(v1.begin(), v1.end(), PointComparatorX());
			std::sort(v2.begin(), v2.end(), PointComparatorX());
			
			if (mainSite->y >= prevSite->y && mainSite->y >= nextSite->y) {
				return getSortedXs(v1.front(), v2.back());
			} else if (mainSite->y <= prevSite->y && mainSite->y <= nextSite->y) {
				return getSortedXs(v1.back(), v2.front());
			} else if (prevSite->y < mainSite->y && mainSite->y < nextSite->y) {
				return getSortedXs(v1.front(), v2.front());
			} else if (nextSite->y < mainSite->y && mainSite->y < prevSite->y) {
				return getSortedXs(v1.back(), v2.back());
			} else {
				throw std::string("GAVNO");
			}
		}
	}
	
	std::string Triple::prints() {
		std::stringstream stream;
		stream  << "(" 
				<< prevSite->prints() 
				<< "), <" 
				<< mainSite->prints() 
				<< ">, (" 
				<< nextSite->prints()
				<< "), center : (" 
				<< circumcenter.prints()
				<< ")";
		return stream.str();
	}

	
	Breakpoint::Breakpoint(const Point &point, const Triple &triple)
		: point(point)
		, triple(triple)
	{
	}

	bool BreakpointComp::operator() (const Breakpoint &first, const Breakpoint &second) const {
		int signY = signum(first.point.y - second.point.y);
		if (signY != 0) {
			return signY < 0;
		} else {
			return first.point.x < second.point.x;
		}
	}
	
	bool isDivide(const Triple &divider, const Triple &lo, const Triple &hi) {
		const float LOCAL_EPS = 1e-3f;
		if (divider.prevSite != LOWEST() && divider.nextSite != LOWEST()) {
			return false;
		}
		float y = divider.mainSite->y;
		std::list<float> boundFirst = lo.getBounds(y);
		float firstHi = boundFirst.back();
		
		std::list<float> boundSecond = hi.getBounds(y);
		float secondLo = boundSecond.front();
		
		return fabs(0.0f + firstHi - secondLo) < LOCAL_EPS && fabs(0.0f + divider.mainSite->x - firstHi) < LOCAL_EPS;
	}

	static bool isBetween(float lo, float hi, float val) {
		return lo <= val && val <= hi;
	}
	

	int TripleComp::compare(const Triple &first, const Triple &second) const {
		if (first.prevSite == second.prevSite && first.mainSite == second.mainSite && first.nextSite == second.nextSite) {
			return 0;
		}
		float y = std::max(first.getMaxY(), second.getMaxY());
		if (first.prevSite == LOWEST() && first.nextSite == LOWEST()) {
			if (second.prevSite == LOWEST() && second.nextSite == LOWEST()) {
				if (first.mainSite->y != second.mainSite->y) {
					// ибо логично
					return 0;
				} else {
					return signum(first.mainSite->x - second.mainSite->x);
				}
			} else {
				std::list<float> boundSecond = second.getBounds(y);
				float secondLo = boundSecond.front();
				float secondHi = boundSecond.back();

				if (isBetween(secondHi, secondHi, first.mainSite->x)) {
					return 1;
				} else if (isBetween(secondLo, secondHi - EPS, first.mainSite->x)) {
					return 0;
				}

				if (first.mainSite->x < secondLo) {
					return -1;
				} else {
					return 1;
				}
			}
		} else if (second.prevSite == LOWEST() && second.nextSite == LOWEST()) {
			return -compare(second, first);
		}
		
		if (first.mainSite == second.prevSite && first.nextSite == second.mainSite) {
			return -1;
		} else if (first.prevSite == second.mainSite && first.mainSite == second.nextSite) {
			return 1;
		}
		
		std::list<float> boundFirst = first.getBounds(y);
		float firstLo = boundFirst.front();
		float firstHi = boundFirst.back();
		
		std::list<float> boundSecond = second.getBounds(y);
		float secondLo = boundSecond.front();
		float secondHi = boundSecond.back();
		
		if (isBetween(firstLo, firstHi, secondLo) && isBetween(firstLo, firstHi, secondHi)) {
			// вырожденный случай - когда second делит first на 2 части.
			// одновременно такие элементы не могут находиться во множестве.
			return 0;
		}
		if (isBetween(secondLo, secondHi, firstLo) && isBetween(secondLo, secondHi, firstHi)) {
			// вырожденный случай - когда first делит second на 2 части.
			// одновременно такие элементы не могут находиться во множестве.
			return 0;
		}
		bool a = firstHi < secondLo + EPS;
		bool b = secondHi < firstLo + EPS;
		check(a ^ b);
		if (firstHi < secondLo + EPS) {
			return -1;
		} else {
			return 1;
		}
	}

	bool TripleComp::operator () (const Triple &first, const Triple &second) const {
		return compare(first, second) < 0;
	}

	
	Point* LOWEST() {
		static Point ans = Point::getLowestPoint();
		return &ans;
	}

	VoronoiFortuneComputing::VoronoiFortuneComputing(std::vector<Point> &p) {
		std::sort(p.begin(), p.end(), PointComparatorY());
		std::set<Triple, TripleComp> beachArcs;
		std::set<Breakpoint, BreakpointComp> breakpoints;
		std::set<Point, PointComparatorY> ans;
		
		int currIdx = this->processFirstLineSiteEvents(p, beachArcs);
		int counter = 0;
		while (true) {
			if (breakpoints.size() != 0) {
				Breakpoint top = *(breakpoints.begin());
				if (currIdx != p.size()) {
					if (top.point.y <= p[currIdx].y) {
						this->processVertexEvent(breakpoints, beachArcs, ans, p[currIdx].y, &p[currIdx]);
					} else {
						this->processSiteEvent(p[currIdx], breakpoints, beachArcs);
						currIdx++;
					}
				} else {
					this->processVertexEvent(breakpoints, beachArcs, ans, top.point.y, NULL);
				}
			} else {
				if (currIdx == p.size()) {
					break;
				} else {
					this->processSiteEvent(p[currIdx], breakpoints, beachArcs);
					currIdx++;
				}
			}
			printf("beach step: %d:\r\n", counter);
			counter++;
		}
		printf("Voronoi diagram vertices size: %u:\r\n", ans.size());
		for (std::set<Point, PointComparatorY>::iterator it = ans.begin(); it != ans.end(); ++it) {
			printf("%s\r\n", it->prints().c_str());
		}
	}
		
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
	int VoronoiFortuneComputing::processFirstLineSiteEvents(
		const std::vector<Point> &p, 
		std::set<Triple, TripleComp> &beachArcs
	) {
		int amountTops = 0;
		while (amountTops < static_cast<int>(p.size()) && p[amountTops].y == p[0].y) {
			amountTops++;
		}
		
		if (amountTops == 1) {
			beachArcs.insert(Triple(LOWEST(), &p[0], LOWEST()));
			return amountTops;
		}
		beachArcs.insert(Triple(LOWEST(), &p[0], &p[1]));
		
		for (int i = 1; i < amountTops - 1; i++) {
			beachArcs.insert(Triple(&p[i - 1], &p[i], &p[i + 1]));
		}
		
		beachArcs.insert(Triple(&p[amountTops - 2], &p[amountTops - 1], LOWEST()));
		return amountTops;
	}
		

	void VoronoiFortuneComputing::processSiteEvent(
		const Point &curr, 
		std::set<Breakpoint, BreakpointComp> &breakpoints, 
		std::set<Triple, TripleComp> &beachArcs
	) {
		printf("Processing site event caused by point %s\r\n", curr.prints().c_str());
		Triple searchHelper(LOWEST(), &curr, LOWEST());
		if (beachArcs.size() == 0) {
			// не добавляем точку
			beachArcs.insert(searchHelper);
		} else {
			// Two elements of a set are considered equivalent if the container's comparison object 
			// returns false reflexively (i.e., no matter the order in which the elements are passed as arguments).
			std::set<Triple, TripleComp>::iterator itDividedArc = beachArcs.find(searchHelper);
			std::set<Triple, TripleComp>::iterator itDown = itDividedArc;

			Triple dividedArc = *itDividedArc;
			--itDown;

			if (itDown != beachArcs.end()) {
				Triple lowerArc = *itDown;
				printf("GURON\r\n");
				if (isDivide(searchHelper, lowerArc, dividedArc)) {
					printf("GURO\r\n");
					beachArcs.erase(lowerArc);
					beachArcs.erase(dividedArc);
					
					dividedArc.removeIfPossible(breakpoints);
					lowerArc.removeIfPossible(breakpoints);
					
					/// ------------
					
					lowerArc.nextSite = &curr;
					dividedArc.prevSite = &curr;
					searchHelper.prevSite = lowerArc.mainSite;
					searchHelper.nextSite = dividedArc.mainSite;
					
					lowerArc.recalcCircumcenter(curr.y);
					dividedArc.recalcCircumcenter(curr.y);
					searchHelper.recalcCircumcenter(curr.y);
					
					check(beachArcs.insert(lowerArc).second);
					check(beachArcs.insert(dividedArc).second);
					check(beachArcs.insert(searchHelper).second);
					
					lowerArc.addIfGood(breakpoints);
					dividedArc.addIfGood(breakpoints);
					searchHelper.addIfGood(breakpoints);
					return;
				}
			} 

			if (itDividedArc == beachArcs.end()) {
				throw "gavnen'";
				//--itDividedArc;
			}
			check(itDividedArc != beachArcs.end());
//				Triple dividedArc = *itDividedArc;
			if (!dividedArc.isCircumcenterEvaluated && dividedArc.mainSite->y == searchHelper.mainSite->y) {
				beachArcs.insert(searchHelper);
				// FIXME необходимо организовать реализацию ситуации, когда несколько точек на передовой делят горизонт.
				return;
			}
			Triple newArc(dividedArc.mainSite, &curr, dividedArc.mainSite, curr.y);
			Triple left(dividedArc.prevSite, dividedArc.mainSite, &curr, curr.y);
			Triple right(&curr, dividedArc.mainSite, dividedArc.nextSite, curr.y);
			
			beachArcs.erase(dividedArc);
			
			check(beachArcs.insert(newArc).second);
			check(beachArcs.insert(left).second);
			check(beachArcs.insert(right).second);

			left.addIfGood(breakpoints);
			right.addIfGood(breakpoints);
			dividedArc.removeIfPossible(breakpoints);
		}
	}
		
	void VoronoiFortuneComputing::processVertexEvent(
		std::set<Breakpoint, BreakpointComp> &breakpoints, 
		std::set<Triple, TripleComp> &beachArcs, 
		std::set<Point, PointComparatorY> &voronoi, 
		float sweepLineY, 
		Point *probablyInclude
	) {
		printf("Processing vertex event, sweep line y coord = %.3f\r\n", sweepLineY);
		std::set<Breakpoint, BreakpointComp>::iterator itTop = breakpoints.begin();
		check(itTop != breakpoints.end());
		
		Breakpoint top = *itTop;

		if (probablyInclude != NULL) {
			if (probablyInclude->weakEqualsTo(top.point)) {
				addDataFromTriple(probablyInclude, top.triple.mainSite, top.triple.nextSite);
				addDataFromTriple(top.triple.prevSite, probablyInclude, top.triple.nextSite);
				addDataFromTriple(top.triple.prevSite, top.triple.mainSite, probablyInclude);
			}
		}
		addDataFromTriple(top.triple.prevSite, top.triple.mainSite, top.triple.nextSite);

		breakpoints.erase(top);

		Point voronoiVertex = top.triple.getCircleCenter();
		
		printf("tangent point: %s\r\n", top.point.prints().c_str());
		
		printf("adding to voronoi diagram: %s\r\n", voronoiVertex.prints().c_str());
		voronoi.insert(voronoiVertex);
		
		std::set<Triple, TripleComp>::iterator itTriple = beachArcs.find(top.triple);
		std::set<Triple, TripleComp>::iterator it2 = itTriple;
		check(itTriple != beachArcs.end());

		Triple lo = *(--itTriple);
		Triple hi = *(++it2);
		
		beachArcs.erase(top.triple); //check(beachArcs.erase(top.triple));

		beachArcs.erase(lo);
		beachArcs.erase(hi);
		
		lo.removeIfPossible(breakpoints);
		hi.removeIfPossible(breakpoints);
		
		lo.nextSite = hi.mainSite;
		hi.prevSite = lo.mainSite;
		
		lo.recalcCircumcenter(sweepLineY);
		hi.recalcCircumcenter(sweepLineY);
		
		beachArcs.insert(lo);//check(beachArcs.add(lo));
		beachArcs.insert(hi);//check(beachArcs.add(hi));

		lo.addIfGood(breakpoints);
		hi.addIfGood(breakpoints);
	}

//	void solve() {
//		int nPoints=10;
//		std::cin >> nPoints;
//
//		std::vector<Point> pa;
//		srand(134);
//		for (int i = 0; i < nPoints; i++) {
//			float x = rand() % 500;
//			float y = rand() % 500;
//			printf("%.4f %.4f\r\n", x, y);
//			//std::cin >> x >> y;
//			pa.push_back(Point(x, y));
//		}
//		VoronoiFortuneComputing com(pa);
//	}
	
	
		
//	void run() {
//		solve();
//	}
}
