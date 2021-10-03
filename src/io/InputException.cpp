//
// Created by timofey on 02.10.2021.
//
#include <string>
#include <utility>
#include <iostream>
#include "../../include/io/InputException.h"

using namespace std;

InputException::InputException(string errString)
{
    err = move(errString);
}

void InputException::handleInputException()
{
    cerr << err << endl;

    if (err == "Wrong args")
        cerr << "Arguments: " << "filterName(gaussian/edge/sharpen) imagesNames" << endl;
}