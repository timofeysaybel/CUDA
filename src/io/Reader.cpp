//
// Created by timofey
//

#include "../../include/io/Reader.h"

#include "../../include/io/InputException.h"
#include "../../include/io/Args.h"

#define cimg_use_jpeg
#include "../../include/image/CImg.h"

using namespace std;
using namespace cimg_library;

Image Reader::read(const string &filename)
{
    try
    {
        CImg<unsigned char> image(filename.c_str());
        unsigned char* img = image.data();
        int width = image.width(), height = image.height(), channels = 3;
        return {img, width, height, channels};
    }
    catch (...)
    {
        throw InputException("Error loading file " + filename);
    }
}