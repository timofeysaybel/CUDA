//
// Created by timofey
//
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "../../include/io/Args.h"
#include "../../include/io/InputException.h"
#include "../../include/filters/Filters.h"

using namespace std;

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
    {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else
    {
        return false;
    }
}

vector <string> getJpgsFromDir(string dirname)
{
    vector<string> res;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dirname.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
            if (hasEnding(ent->d_name, ".jpg"))
                res.push_back(ent->d_name);

        closedir(dir);
    } else
        throw InputException(string("Cannot open ") + string(dirname));

    return res;
}

Args Args::parseArgs(int argc, char **argv)
{
    Args res;
    const string imageDir = "images/";

    if (argc == 2)
    {
        res.filter = Filters::ALL;
        auto jpgs = getJpgsFromDir(imageDir + string(argv[1]));
        for (auto &e : jpgs)
            res.filenames.push_back(string(argv[1]) + e);
        return res;
    }

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
    {
        if (!hasEnding(argv[i], "jpg"))
        {
            auto jpgs = getJpgsFromDir(argv[i]);
            for (auto &e : jpgs)
                res.filenames.push_back(string(argv[i]) + e);
            continue;
        }
        res.filenames.emplace_back(argv[i]);
    }

    return res;
}
