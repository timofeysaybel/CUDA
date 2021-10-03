//
// Created by timofey on 02.10.2021.
//

#ifndef IMAGECONVOLUTION_ARGS_H
#define IMAGECONVOLUTION_ARGS_H

#include "../../../../../../usr/include/c++/10/string"
#include "../../../../../../usr/include/c++/10/vector"
#include "../../../../../../usr/include/c++/10/fstream"

class Args
{
public:
    int filter;
    std::vector<std::string> filenames;

    Args() = default;

    static Args parseArgs(int argc, char **argv);
};

#endif //IMAGECONVOLUTION_ARGS_H
