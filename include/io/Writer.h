//
// Created by timofey
//

#ifndef BASICIMAGECONVOLUTION_WRITER_H
#define BASICIMAGECONVOLUTION_WRITER_H

#include <string>
#include "../image/Image.h"

class Writer
{
public:
    static void write(const Image &image, const std::string &outFilename);
};


#endif //BASICIMAGECONVOLUTION_WRITER_H
