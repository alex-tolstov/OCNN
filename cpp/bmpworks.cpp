#include "bmpworks.h"

void setPixelValue(BMP &bitmap, int x, int y, int r, int g, int b) {
	bitmap(x, y)->Red = r;
	bitmap(x, y)->Green = g;
	bitmap(x, y)->Blue = b;
}

void hsv2rgb(int h, int s, int v, int &r, int &g, int &b) {
	int part = h / 60;
	int vMin = (100 - s) * v / 100;
	int a = (v - vMin) * (h % 60) / 60;
	
	int vInc = vMin + a;
	int vDec = v - a;
	
	switch (part) {
		case 0:
			r = v;
			g = vInc;
			b = vMin;
			break;
		case 1:
			r = vDec;
			g = v;
			b = vMin;
			break;
		case 2:
			r = vMin;
			g = v;
			b = vInc;
			break;
		case 3:
			r = vMin;
			g = vDec;
			b = v;
			break;
		case 4:
			r = vInc;
			g = vMin;
			b = v;
			break;
		case 5:
			r = v;
			g = vMin;
			b = vDec;
			break;
		default:
			break;
	}
	r = std::max(0, std::min(r, 255));
	g = std::max(0, std::min(g, 255));
	b = std::max(0, std::min(b, 255));
}

void printCross(BMP &bitmap, int x, int y, int r, int g, int b) {
	const int shiftX[5] = {0, 0, 0, 1, -1};
	const int shiftY[5] = {0, 1, -1, 0, 0};
	for (int i = 0; i < 5; i++) {
		int currX = std::min(512, std::max(x + shiftX[i], 0));
		int currY = std::min(512, std::max(y + shiftY[i], 0));

		setPixelValue(bitmap, currX, currY, r, g, b);
	}
}