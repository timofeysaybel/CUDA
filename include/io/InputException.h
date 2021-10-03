//
// Created by timofey on 02.10.2021.
//

#ifndef IMAGECONVOLUTION_INPUTEXCEPTION_H
#define IMAGECONVOLUTION_INPUTEXCEPTION_H

#include "../../../../../../usr/include/c++/10/string"

class InputException
{
    std::string err;

public:
    InputException() = default;

    InputException(std::string errString);

    void handleInputException();
};

#endif //IMAGECONVOLUTION_INPUTEXCEPTION_H