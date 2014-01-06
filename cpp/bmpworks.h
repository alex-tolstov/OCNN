#ifndef BMPWORKS_H_
#define BMPWORKS_H_

#include "EasyBMP.h"

void setPixelValue(BMP &bitmap, int x, int y, int r, int g, int b);

void hsv2rgb(int h, int s, int v, int &r, int &g, int &b);

void printCross(BMP &bitmap, int x, int y, int r, int g, int b);

#endif // BMPWORKS_H_
