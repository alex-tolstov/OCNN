import java.io.*;
import java.util.*;


public class Main {
	
	Scanner in;
	static PrintWriter out;
	
	static final String pathIn = "C:\\Users\\Alex\\Desktop\\Voronoi Diagram\\tests\\ins\\";
	static final String pathOut = "C:\\Users\\Alex\\Desktop\\Voronoi Diagram\\tests\\outs2\\";
	static final String correctOuts = "C:\\Users\\Alex\\Desktop\\Voronoi Diagram\\tests\\correct_outs\\";
	
	static final int testTotal = 15;
	
	static final float EPS = 1e-5f;
	
	static class Vector {
		final float x;
		final float y;
		
		Vector(Point base, Point p) {
			this.x = p.x - base.x;
			this.y = p.y - base.y;
		}
		
		Vector(float x, float y) {
			this.x = x;
			this.y = y;
		}
		
		Vector rotate90ccw() {
			return new Vector(-y, x);
		}
	}
	
	static float calcDeterminant(float a, float b, float c, float d) {
		return a*d - b*c;	
	}
	
	static class Line {
		final float a;
		final float b;
		final float c;
		
		Line(Point p1, Point p2) {
			boolean equalty = p1.weakEqualsTo(p2);
			check(!equalty);
			a = p1.y - p2.y;
			b = p2.x - p1.x;
			c = -(a * p1.x + b * p1.y);
		}
		
		Point getIntersectionPoint(Line second) {
			if (this.isParallelTo(second)) {
				return null;
			}
			float d0 = calcDeterminant(a, b, second.a, second.b);
			float dA = calcDeterminant(-c, b, -second.c, second.b);
			float dB = calcDeterminant(a, -c, second.a, -second.c);
			return new Point(dA / d0, dB / d0);
		}
		
		boolean isParallelTo(Line second) {
			return Math.abs(this.a * second.b - second.a * this.b) < EPS;
		}
	}
	
	static class Point {
		static final PointComparatorX comparatorX = new PointComparatorX();
		static final PointComparatorY comparatorY = new PointComparatorY();
		
		float x;
		float y;
		
		Point(Scanner in) {
			x = in.nextFloat();
			y = in.nextFloat();
		}
		
		Point(float x, float y) {
			this.x = x;
			this.y = y;
		}
		
		static Point getMiddlePoint(Point a, Point b) {
			return new Point((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
		}
		
		static Point getCircumcenterShiftedUpOnRadius(Point a, Point b, Point c) {
			Point center = Point.getCircumcenter(a, b, c);
			center.y += center.distanceTo(a);
			return center;
		}
		
		static Point getCircumcenter(Point a, Point b, Point c) {
			if (a.weakEqualsTo(b)) {
				return Point.getMiddlePoint(a, c);
			} else if (a.weakEqualsTo(c)) {
				return Point.getMiddlePoint(a, b);
			} else if (b.weakEqualsTo(c)) {
				return Point.getMiddlePoint(a, b);
			}
			Vector ab = new Vector(a, b);
			Vector ac = new Vector(a, c);
			
			Point centerAB = Point.getMiddlePoint(a, b);
			Point centerAC = Point.getMiddlePoint(a, c);
			
			Point abNormal = centerAB.add(ab.rotate90ccw());
			Point acNormal = centerAC.add(ac.rotate90ccw());
			return new Line(centerAB, abNormal).getIntersectionPoint(new Line(centerAC, acNormal));
		}
		
		static LinkedList<Point> getCircumcenters(Point first, Point second, float tangentY) {
			first.y -= tangentY;
			second.y -= tangentY;
			
			Vector firstSecond = new Vector(first, second).rotate90ccw();
			
			Point centerFS = Point.getMiddlePoint(first, second);
			Point another = centerFS.add(firstSecond);
			
			float gA = centerFS.y - another.y;
			float gB = another.x - centerFS.x;
			float gC = -(gA * centerFS.x + gB * centerFS.y);
			
			if (Math.abs(gB) < EPS) {
				first.y += tangentY;
				second.y += tangentY;
				
				float x0 = -gC / gA;
				float y0 = ( (x0 - second.x) * x0 + 0.5f * (second.x * second.x + second.y * second.y - x0 * x0) ) / second.y;
				
				LinkedList<Point> result = new LinkedList<Main.Point>();
				result.add(new Point(x0, y0 + tangentY));
				return result;
			} else {
				float a = 0.5f;
				float b = gA * second.y / gB - second.x;
				float c = gC * second.y / gB + (second.x * second.x + second.y * second.y) * 0.5f;
				
				first.y += tangentY;
				second.y += tangentY;
				
				float d = b * b - 4.f * a * c;
				
				float root = (float)Math.sqrt(d);
				if (root < 0) {
					return new LinkedList<Point>();
				}
				float x0 = (-b + root) / (2 * a);
				float x1 = (-b - root) / (2 * a);
				
				float y0 = (-gC - gA * x0) / gB;
				float y1 = (-gC - gA * x1) / gB;
				
				LinkedList<Point> result = new LinkedList<Main.Point>();
				result.add(new Point(x0, y0 + tangentY));
				result.add(new Point(x1, y1 + tangentY));
				return result;
			}
		}
		
		Point add(Vector v) {
			return new Point(this.x + v.x, this.y + v.y);
		}
		
		public String toString() {
			return x + ", " + y;
		}
		
		float squaredDistanceTo(Point second) {
			float dx = this.x - second.x;
			float dy = this.y - second.y;
			return dx * dx + dy * dy;
		}
		
		float distanceTo(Point second) {
			return (float)Math.sqrt(this.squaredDistanceTo(second));
		}
		
		static Point getLowestPoint() {
			return new Point(0, -1e7f);
		}
		
		boolean weakEqualsTo(Point that) {
			return Math.abs(this.x - that.x) < EPS && Math.abs(this.y - that.y) < EPS;
		}
		
		static class PointComparatorY implements Comparator <Point> {
			@Override
			public int compare(Point first, Point second) {
				int signY = (int)Math.signum(first.y - second.y);
				if (signY != 0) {
					return signY;
				} else {
					return (int)Math.signum(first.x - second.x);
				}
			}
		}
		
		static class PointComparatorX implements Comparator <Point> {
			@Override
			public int compare(Point o1, Point o2) {
				return (int)Math.signum(o1.x - o2.x);
			}
		}
	}
	
	static class Triple {
		Point prevSite;
		Point mainSite;
		Point nextSite;
		Point circumcenter;
		
		Triple(Point prev, Point curr, Point next) {
			this(prev, curr, next, Math.max(prev.y, Math.max(curr.y, next.y)));
		}
		
		Triple(Point prev, Point curr, Point next, float sweepLineY) {
			this.prevSite = prev;
			this.mainSite = curr;
			this.nextSite = next;
			recalcCircumcenter(sweepLineY);
		}
		
		void recalcCircumcenter(float sweepLineY) {
			this.circumcenter = null;
			if (prevSite.x == mainSite.x && mainSite.x == nextSite.x) {
				return;
			}
			if (prevSite.y == mainSite.y && mainSite.y == nextSite.y) {
				return;
			}
			try {
				Point center = Point.getCircumcenterShiftedUpOnRadius(prevSite, mainSite, nextSite);
				if (prevSite != nextSite && prevSite != LOWEST && nextSite != LOWEST) {
					if (!isDegenerated(center)) {
						this.circumcenter = center;
					}
				}
			} catch (NullPointerException e) {
//				System.err.println("GAVNENb: " + this);
//				throw e;
				return;
			}
		}
		
		private boolean isDegenerated(Point shiftedCenter) {
			if (mainSite.y > prevSite.y && mainSite.y > nextSite.y) {
// 				для обработки вырожденных случаев
// 				например:
//				4
//				1 1
//				2 1
//				3 1
//				2 2
				return true;
			}
			final float WEAK_EPS = 1e-1f;
			LinkedList<Float> boundsX = this.getBounds(shiftedCenter.y);
			float diff1 = Math.abs(boundsX.getFirst() - shiftedCenter.x);
			float diff2 = Math.abs(boundsX.getLast() - shiftedCenter.x);
			return diff1 > WEAK_EPS || diff2 > WEAK_EPS;
		}
		
		void addIfGood(TreeSet<Breakpoint> breakpoints) {
			if (this.circumcenter != null) {
				breakpoints.add(new Breakpoint(this.circumcenter, this));
			}
		}
		
		void removeIfPossible(TreeSet<Breakpoint> breakpoints) {
			if (this.circumcenter != null) {
				breakpoints.remove(new Breakpoint(this.circumcenter, this));
			}
		}
		
		Point getCircleCenter() {
			return Point.getCircumcenter(prevSite, mainSite, nextSite);
		}
		
		float getMaxY() {
			return Math.max(prevSite.y, Math.max(mainSite.y, nextSite.y));
		}
		
		LinkedList<Float> getSortedXs(Point first, Point second) {
			LinkedList<Float> resultX = new LinkedList<Float>();
			resultX.add(Math.min(first.x, second.x));
			resultX.add(Math.max(first.x, second.x));
			return resultX;
		}
					
		LinkedList<Float> getBounds(float tangentY) {
			if (prevSite == nextSite) {
				LinkedList<Point> points = Point.getCircumcenters(mainSite, prevSite, tangentY);
				check(points.size() == 2);
				return getSortedXs(points.pollFirst(), points.pollFirst());
			} else {
				LinkedList<Point> withPrev = Point.getCircumcenters(mainSite, prevSite, tangentY);
				LinkedList<Point> withNext = Point.getCircumcenters(mainSite, nextSite, tangentY);
				Collections.sort(withPrev, Point.comparatorX);
				Collections.sort(withNext, Point.comparatorX);
				
				if (mainSite.y >= prevSite.y && mainSite.y >= nextSite.y) {
					return getSortedXs(withPrev.getFirst(), withNext.getLast());
				} else if (mainSite.y <= prevSite.y && mainSite.y <= nextSite.y) {
					return getSortedXs(withPrev.getLast(), withNext.getFirst());
				} else if (prevSite.y < mainSite.y && mainSite.y < nextSite.y) {
					return getSortedXs(withPrev.getFirst(), withNext.getFirst());
				} else if (nextSite.y < mainSite.y && mainSite.y < prevSite.y) {
					return getSortedXs(withPrev.getLast(), withNext.getLast());
				} else {
					throw new Error();
				}
			}
		}
		
		public String toString() {
			return "(" + prevSite + "), <" + mainSite + ">, (" + nextSite + "), center : (" + circumcenter + ")";
		}
	}
	
	static boolean isDivide(Triple divider, Triple lo, Triple hi) {
		final float LOCAL_EPS = 1e-3f;
		if (divider.prevSite != LOWEST && divider.nextSite != LOWEST) {
			return false;
		}
		float y = divider.mainSite.y;
		LinkedList<Float> boundFirst = lo.getBounds(y);
		float firstHi = boundFirst.getLast();
		
		LinkedList<Float> boundSecond = hi.getBounds(y);
		float secondLo = boundSecond.getFirst();
		
		return Math.abs(firstHi - secondLo) < LOCAL_EPS && Math.abs(divider.mainSite.x - firstHi) < LOCAL_EPS;
	}
	
	static class Breakpoint implements Comparable <Breakpoint> {
		Point point;
		Triple triple;
		
		Breakpoint(Point point, Triple triple) {
			this.point = point;
			this.triple = triple;
		}
		
		@Override
		public int compareTo(Breakpoint that) {
			int signY = (int)Math.signum(this.point.y - that.point.y);
			if (signY != 0) {
				return signY;
			} else {
				int signX = (int)Math.signum(this.point.x - that.point.x);
				if (signX != 0) {
					return signX;
				} else {
					return this.triple.hashCode() - that.triple.hashCode();
				}
			}
		}
	}
	
	static boolean isBetween(float lo, float hi, float val) {
		return lo <= val && val <= hi;
	}
	
	static class TripleComp implements Comparator <Triple> {
		@Override
		public int compare(Triple first, Triple second) {
			float y = Math.max(first.getMaxY(), second.getMaxY());
			if (first.prevSite == second.prevSite && first.mainSite == second.mainSite && first.nextSite == second.nextSite) {
				return 0;
			}
			if (first.prevSite == LOWEST && first.nextSite == LOWEST) {
				if (second.prevSite == LOWEST && second.nextSite == LOWEST) {
					if (first.mainSite.y != second.mainSite.y) {
						// ибо логично
						return 0;
					} else {
						return (int)Math.signum(first.mainSite.x - second.mainSite.x);
					}
				} else {
					LinkedList<Float> boundSecond = second.getBounds(y);
					float secondLo = boundSecond.pollFirst();
					float secondHi = boundSecond.pollFirst();
					
//					if (isBetween(secondLo, secondLo, first.mainSite.x)) {
//						return 0;
//					}
					if (isBetween(secondHi, secondHi, first.mainSite.x)) {
						return 1;
					} else if (isBetween(secondLo, secondHi - EPS, first.mainSite.x)) {
						return 0;
					}

					if (first.mainSite.x < secondLo) {
						return -1;
					} else {
						return 1;
					}
				}
			} else if (second.prevSite == LOWEST && second.nextSite == LOWEST) {
				return -compare(second, first);
			}
			
			if (first.mainSite == second.prevSite && first.nextSite == second.mainSite) {
				return -1;
			} else if (first.prevSite == second.mainSite && first.mainSite == second.nextSite) {
				return 1;
			}
			
			LinkedList<Float> boundFirst = first.getBounds(y);
			float firstLo = boundFirst.pollFirst();
			float firstHi = boundFirst.pollFirst();
			
			LinkedList<Float> boundSecond = second.getBounds(y);
			float secondLo = boundSecond.pollFirst();
			float secondHi = boundSecond.pollFirst();
			
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
			boolean a = firstHi < secondLo + EPS;
			boolean b = secondHi < firstLo + EPS;
			check(a ^ b);
			if (firstHi < secondLo + EPS) {
				return -1;
			} else {
				return 1;
			}
		}
	}
	
	static Point LOWEST = Point.getLowestPoint();
	
	static class NeighborsList implements Comparable<NeighborsList> {
		TreeSet<Point> nearest = new TreeSet<Point>(Point.comparatorY);
		Point site;
		
		NeighborsList(Point p) {
			this.site = p;
		}
		
		void add(Point near) {
			nearest.add(near);
		}
		
		@Override
		public int compareTo(NeighborsList that) {
			return Point.comparatorY.compare(this.site, that.site);
		}
	}
	
	
	static class VoronoiFortuneComputing {
		TreeSet<Point> ans = new TreeSet<Main.Point>(Point.comparatorY);
		TreeSet<NeighborsList> adjList = new TreeSet<Main.NeighborsList>();
		
		VoronoiFortuneComputing(Point p[]) {
			Arrays.sort(p, Point.comparatorY);
			TreeSet<Triple> beachArcs = new TreeSet<Triple>(new TripleComp());
			TreeSet<Breakpoint> breakpoints = new TreeSet<Breakpoint>();
			
			int currIdx = this.processFirstLineSiteEvents(p, breakpoints, beachArcs);
			int counter = 0;
			while (true) {
				if (breakpoints.size() != 0) {
					Breakpoint top = breakpoints.first();
					if (currIdx != p.length) {
//						if (top.point.y == p[currIdx].y && Math.abs(top.point.x - p[currIdx].x) < EPS) {
//							this.processVertexEvent(breakpoints, beachArcs, ans, p[currIdx].y, p[currIdx]);
//						} else
						if (top.point.y <= p[currIdx].y) {
							this.processVertexEvent(breakpoints, beachArcs, ans, p[currIdx].y, p[currIdx]);
						} else {
							this.processSiteEvent(p[currIdx], breakpoints, beachArcs);
							currIdx++;
						}
					} else {
						this.processVertexEvent(breakpoints, beachArcs, ans, top.point.y, null);
					}
				} else {
					if (currIdx == p.length) {
						break;
					} else {
						this.processSiteEvent(p[currIdx], breakpoints, beachArcs);
						currIdx++;
					}
				}
//				out.println("beach step: " + counter + ":");
				counter++;
			}
			out.println("Voronoi diagram vertices size: " + ans.size());
			for (Point curr : ans) {
				out.println(curr);
			}
			
			out.println("Govno");
			for (NeighborsList list : adjList) {
				out.println("Main: " + list.site);
				for (Point point : list.nearest) {
					out.print("[" + point + "]; ");
				}
				out.println();
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
		 * 
		 * */
		private int processFirstLineSiteEvents(Point[] p, TreeSet<Breakpoint> breakpoints, TreeSet<Triple> beachArcs) {
			int amountTops = 0;
			while (amountTops < p.length && p[amountTops].y == p[0].y) {
				amountTops++;
			}
			
			if (amountTops == 1) {
				beachArcs.add(new Triple(LOWEST, p[0], LOWEST));
				return 1;
			}
			beachArcs.add(new Triple(LOWEST, p[0], p[1]));
			
			for (int i = 1; i < amountTops - 1; i++) {
				beachArcs.add(new Triple(p[i - 1], p[i], p[i + 1]));
			}
			
			beachArcs.add(new Triple(p[amountTops - 2], p[amountTops - 1], LOWEST));
			return amountTops;
		}
		
		private void processSiteEvent(Point curr, TreeSet<Breakpoint> breakpoints, TreeSet<Triple> beachArcs) {
//			out.println("Processing site event caused by point " + curr);
			Triple searchHelper = new Triple(LOWEST, curr, LOWEST);
			if (beachArcs.size() == 0) {
				// не добавляем точку
				beachArcs.add(searchHelper);
			} else {
				Triple dividedArc = beachArcs.floor(searchHelper);
				Triple lowerArc = beachArcs.lower(dividedArc);
				if (lowerArc != null) {
//					out.println("GURON");
					if (isDivide(searchHelper, lowerArc, dividedArc)) {
//						out.println("GURO");
						beachArcs.remove(lowerArc);
						beachArcs.remove(dividedArc);
						
						dividedArc.removeIfPossible(breakpoints);
						lowerArc.removeIfPossible(breakpoints);
						
						/// ------------
						
						lowerArc.nextSite = curr;
						dividedArc.prevSite = curr;
						searchHelper.prevSite = lowerArc.mainSite;
						searchHelper.nextSite = dividedArc.mainSite;
						
						lowerArc.recalcCircumcenter(curr.y);
						dividedArc.recalcCircumcenter(curr.y);
						searchHelper.recalcCircumcenter(curr.y);
						
						check(beachArcs.add(lowerArc));
						check(beachArcs.add(dividedArc));
						check(beachArcs.add(searchHelper));
						
						lowerArc.addIfGood(breakpoints);
						dividedArc.addIfGood(breakpoints);
						searchHelper.addIfGood(breakpoints);
						return;
					}
				} 
				
				{
					if (dividedArc.circumcenter == null && dividedArc.mainSite.y == searchHelper.mainSite.y) {
						beachArcs.add(searchHelper);
						return;
					}
					Triple newArc = new Triple(dividedArc.mainSite, curr, dividedArc.mainSite, curr.y);
					Triple left = new Triple(dividedArc.prevSite, dividedArc.mainSite, curr, curr.y);
					Triple right = new Triple(curr, dividedArc.mainSite, dividedArc.nextSite, curr.y);
					
					beachArcs.remove(dividedArc);
					
					check(beachArcs.add(newArc));
					check(beachArcs.add(left));
					check(beachArcs.add(right));
				
					left.addIfGood(breakpoints);
					right.addIfGood(breakpoints);
					dividedArc.removeIfPossible(breakpoints);
				}
			}
		}
		
		private void addData(Point main, Point a, Point b) {
			NeighborsList list = new NeighborsList(main);
			if (adjList.contains(list)) {
				// gets an element
				list = adjList.floor(list);
			}
			list.add(a);
			list.add(b);
			adjList.add(list);
		}
		
		private void addDataFromTriple(Point a, Point b, Point c) {
			addData(a, b, c);
			addData(b, a, c);
			addData(c, a, b);
		}
		
		private void processVertexEvent(TreeSet<Breakpoint> breakpoints, TreeSet<Triple> beachArcs, TreeSet<Point> voronoi, float sweepLineY, Point probablyInclude) {
//			out.println("Processing vertex event, sweep line y coord = " + sweepLineY);
			Breakpoint top = breakpoints.pollFirst();
			Point voronoiVertex = top.triple.getCircleCenter();
			if (probablyInclude != null) {
				if (probablyInclude.weakEqualsTo(top.point)) {
					addDataFromTriple(probablyInclude, top.triple.mainSite, top.triple.nextSite);
					addDataFromTriple(top.triple.prevSite, probablyInclude, top.triple.nextSite);
					addDataFromTriple(top.triple.prevSite, top.triple.mainSite, probablyInclude);
				}
			}
			addDataFromTriple(top.triple.prevSite, top.triple.mainSite, top.triple.nextSite);
//			out.println("tangent point: " + top.point);
			
//			out.println("adding to voronoi diagram: " + voronoiVertex);
			
			voronoi.add(voronoiVertex);
			
			Triple lo = beachArcs.lower(top.triple);
			Triple hi = beachArcs.higher(top.triple);
			
			check(beachArcs.remove(top.triple));
			beachArcs.remove(lo);
			beachArcs.remove(hi);
			
			lo.removeIfPossible(breakpoints);
			hi.removeIfPossible(breakpoints);
			
			lo.nextSite = hi.mainSite;
			hi.prevSite = lo.mainSite;
			
			lo.recalcCircumcenter(sweepLineY);
			hi.recalcCircumcenter(sweepLineY);
			
			check(beachArcs.add(lo));
			check(beachArcs.add(hi));

			lo.addIfGood(breakpoints);
			hi.addIfGood(breakpoints);
		}
	}
	
	
	void solveOne(int iTest) {
		Scanner fileIn;
//		Scanner fileCorrectIn;
		PrintWriter fileOut;
		try {
			fileIn = new Scanner(new FileReader(pathIn + iTest +  ".in"));
//			fileCorrectIn = new Scanner(new FileReader(correctOuts + iTest +  ".out"));
			fileOut = new PrintWriter(new FileWriter(pathOut + iTest +  ".out"));
		} catch (IOException e) {
			throw new Error(e);
		}
		int nPoints = fileIn.nextInt();
		Point p[] = new Point[nPoints];
		for (int i = 0; i < p.length; i++) {
			p[i] = new Point(fileIn);
		}
		
		TreeSet<Point> ans = new VoronoiFortuneComputing(p).ans;
//		int correctSize = fileCorrectIn.nextInt();
//		check(ans.size() == correctSize);
		fileOut.println(ans.size());
		
		for (Point pp : ans) {
			fileOut.println(pp.x + " " + pp.y);
		}
		
		fileIn.close();
		fileOut.close();
	}
	
	void solve() {
//		for (int i = 0; i < testTotal; i++) {
//			solveOne(i + 1);
//		}
		
		int nPoints = in.nextInt();
//		Point p[] = new Point[nPoints];
		TreeSet<Point> unique = new TreeSet<Main.Point>(Point.comparatorY);
		for (int i = 0; i < nPoints; i++) {
//			p[i] = new Point(in);
//			int val1 = (int)(Math.random() * 43843938 % 700);
//			int val2 = (int)(Math.random() * 43843438 % 700);
//			p[i] = 
//			unique.add(new Point((float)val1, (float)val2));
			unique.add(new Point(in));
		}
		Point[] ar = new Point[unique.size()];
		int idx = 0;
		for (Point u : unique) {
			ar[idx] = u;
			idx++;
		}
		new VoronoiFortuneComputing(ar);
	}
	
	static void check(boolean e) {
		if (!e) {
			throw new Error();
		}
	}
		
	public void run() {
		in = new Scanner(System.in);
		out = new PrintWriter(System.out);
		
		try {
			solve();
		} finally {
			out.close();
		}
	}
	
	public static void main(String[] args) {
		new Main().run();
	}
}