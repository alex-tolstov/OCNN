#ifndef FIND_POINTS_H_
#define FIND_POINTS_H_


#include "SimpleMath.h"

#include "EasyBMP.h"

namespace finder {
class ImageToPointsConverter {
private:
	BMP bitmap;

	bool isGoodColor(int x, int y);
public:

	ImageToPointsConverter(const std::string &fileName);

	void fillVector(std::vector<voronoi::Point> &result);
};
} // namespace finder

#endif // FIND_POINTS_H_