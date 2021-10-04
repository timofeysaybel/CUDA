//
// Created by timofey
//

#ifndef IMAGECONVOLUTION_INPUTEXCEPTION_H
#define IMAGECONVOLUTION_INPUTEXCEPTION_H

#include <string>

class InputException
{
    std::string err;

public:
    InputException() = default;

    InputException(std::string errString);

    void handleInputException();
};

#endif //IMAGECONVOLUTION_INPUTEXCEPTION_H
