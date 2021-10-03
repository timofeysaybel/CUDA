//
// Created by timofey on 02.10.2021.
//
#include "../../include/io/Args.h"
#include "../../include/io/InputException.h"
#include "../../include/filters/Filters.h"

using namespace std;

Args Args::parseArgs(int argc, char **argv)
{
    Args res;

    if (argc < 3)
        throw InputException("Wrong args");

    string filterName(argv[1]);

    if (filterName == "gaussian")
        res.filter = Filters::GAUSSIAN;
    else if (filterName == "edge")
        res.filter = Filters::EDGE_DETECTION;
    else if (filterName == "sharpen")
        res.filter = Filters::SHARPEN;
    else
        throw InputException("Unknown filter: " + filterName);

    for (int i = 2; i < argc; i++)
        res.filenames.emplace_back(argv[i]);

    return res;
}
