//
// Created by timofey on 03.10.2021.
//
#include "../../include/io/Writer.h"

#define cimg_use_jpeg
#include "../../include/image/CImg.h"

using namespace cimg_library;

void Writer::write(const Image &image, const std::string &outFilename)
{
    const char*  out = outFilename.c_str();
    auto* data = new unsigned char[image.width * image.height * 3];

    for (int i = 0; i < image.width * image.height; i++)
    {
        data[i] = image.data[i].r;
        data[i + image.width * image.height] = image.data[i].g;
        data[i + 2 * image.width * image.height] = image.data[i].b;
    }

    CImg<unsigned char> res(data, image.width, image.height, 1, 3, false);
    res.save(out);
}