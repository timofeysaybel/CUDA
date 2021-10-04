//
// Created by timofey
//

#include "../../include/image/Image.h"

#include <cstring>
#include <cstdlib>

Image::Image(const unsigned char* &d,const int& w,const int& h,const int& c)
{
    data = (Pixel*)malloc(w * h * sizeof(Pixel));
    for (int i = 0; i < w * h; i++)
    {
        data[i].r = d[i];
        data[i].g = d[i + w * h];
        data[i].b = d[i + 2 * w * h];
    }

    width = w;
    height = h;
    channels = c;
}

Image::Image(Pixel* d, int w, int h, int c)
{
    data = (Pixel*)malloc(w * h * sizeof(Pixel));
    for (int i = 0; i < w * h; i++)
    {
        data[i].r = d[i].r;
        data[i].g = d[i].g;
        data[i].b = d[i].b;
    }

    width = w;
    height = h;
    channels = c;
}

Image::Image(unsigned char* d, int w, int h, int c)
{
    data = (Pixel*)malloc(w * h * sizeof(Pixel));
    for (int i = 0; i < w * h; i++)
    {
        data[i].r = d[i];
        data[i].g = d[i + w * h];
        data[i].b = d[i + 2 * w * h];
    }
    width = w;
    height = h;
    channels = c;
}

Image::Image(const Image &an)
{
    data = (Pixel*)malloc(an.width * an.height * sizeof(Pixel));
    for (int i = 0; i < an.width * an.height; i++)
    {
        data[i].r = an.data[i].r;
        data[i].g = an.data[i].g;
        data[i].b = an.data[i].b;
    }

    width = an.width;
    height = an.height;
    channels = an.channels;
}