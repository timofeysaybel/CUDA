//
// Created by timofey
//

#ifndef IMAGECONVOLUTION_IMAGE_H
#define IMAGECONVOLUTION_IMAGE_H

#include <cuda.h>

struct Pixel
{
    unsigned char r, g, b;
    Pixel(unsigned char rr, unsigned char gg, unsigned char bb) : r(rr), g(gg), b(bb) {}
};

class Image
{

public:
    Pixel* data;

    int width, height, channels;

    Image() = default;

    Image(unsigned char* d, int w, int h, int c);

    Image(const unsigned char* &d,const int& w,const int& h,const int& c);

    Image(Pixel* d, int w, int h, int c);

    Image(const Image& an);

};


#endif //IMAGECONVOLUTION_IMAGE_H
