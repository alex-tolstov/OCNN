#include "find_points.h"

#include <algorithm>
#include <vector>
#include <queue>

namespace finder {

const int shiftX[4] = {0,0,1,-1};
const int shiftY[4] = {1,-1,0,0};

struct Pos {
	
	int x;
	int y;

	Pos(int x, int y) 
		: x(x)
		, y(y)
	{
	}

	Pos move(int times) {
		int nx = x + shiftX[times];
		int ny = y + shiftY[times];
		nx = std::min(255, std::max(nx, 0));
		ny = std::min(255, std::max(ny, 0));
		return Pos(nx, ny);
	}
};


ImageToPointsConverter::ImageToPointsConverter(const std::string &fileName) {
	bitmap.ReadFromFile(fileName.c_str());
}

bool ImageToPointsConverter::isGoodColor(int x, int y) {
	bool isWhite = bitmap(x, y)->Red > 220 && bitmap(x, y)->Green > 220 && bitmap(x, y)->Blue > 220;
	return !isWhite;
}

void ImageToPointsConverter::fillVector(std::vector<voronoi::Point> &result) {
	int width = bitmap.TellWidth();
	int height = bitmap.TellHeight();
	if (width != height || width != 256) {
		throw std::string("bad input picture");
	}

	std::vector<bool> used(width * height, false);
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			if (!used[y * width + x] && isGoodColor(x, y)) {
				result.push_back(voronoi::Point(static_cast<float>(x), static_cast<float>(y)));
			}
		}
	}
	/*
				std::queue<Pos> q;
				q.push(Pos(x, y));
				used[y * width + x] = true;
				int count = 1;
				int sumX = x;
				int sumY = y;
				while (!q.empty()) {
					Pos curr = q.front();
					q.pop();
					for (int i = 0; i < 4; i++) {
						Pos next = curr.move(i);
						if (!used[next.y * width + next.x] && isGoodColor(next.x, next.y)) {
							used[next.y * width + next.x] = true;
							sumX += next.x;
							sumY += next.y;
							count++;
							q.push(next);
						}
					}
				}
				float aveX = sumX / (count + 0.0f);
				float aveY = sumY / (count + 0.0f);
				result.push_back(voronoi::Point(aveX, aveY));
			}
		}
	}
	*/
}

} // namespace finder