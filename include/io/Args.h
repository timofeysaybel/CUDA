//
// Created by timofey
//

#ifndef IMAGECONVOLUTION_ARGS_H
#define IMAGECONVOLUTION_ARGS_H

#include <string>
#include <vector>

class Args
{
public:
    int filter;
    std::vector<std::string> filenames;

    Args() = default;

    static Args parseArgs(int argc, char **argv);
};

#endif //IMAGECONVOLUTION_ARGS_H
