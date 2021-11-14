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

Image::Image(unsigned char* d, int w, int h)
{
    data = (Pixel*)malloc(w * h * sizeof(Pixel));
    for (int i = 0; i < w * h; i+=3)
    {
        data[i].r = d[i];
        data[i].g = d[i + 1];
        data[i].b = d[i + 2];
    }
    width = w;
    height = h;
    channels = 3;
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

unsigned char* Image::getData() const
{
    unsigned char* resData = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    for (int i = 0; i < width * height; i++)
    {
        resData[i] = data[i].r;
        resData[i + width * height] = data[i].g;
        resData[i + 2 * width * height] = data[i].b;
    }

    return resData;
}

unsigned char* Image::getData(int pos) const
{
    unsigned char* resData = (unsigned char*) malloc(height * width * sizeof(unsigned char));

    for (int i = 0; i < width * height; i++)
    {
        switch(pos)
        {
            case 0:
                resData[i] = data[i].r;
            case 1:
                resData[i] = data[i].g;
            case 2:
                resData[i] = data[i].b;
        }
    }

    return resData;
}