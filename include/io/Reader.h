//
// Created by timofey on 03.10.2021.
//

#ifndef BASICIMAGECONVOLUTION_READER_H
#define BASICIMAGECONVOLUTION_READER_H

#include <string>
#include "../image/Image.h"

class Reader
{
public:
    static Image read(const std::string& filename);
};


#endif //BASICIMAGECONVOLUTION_READER_H
