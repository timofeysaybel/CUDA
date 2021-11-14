//
// Created by timofey
//
#include <iostream>

#include "../include/io/Args.h"
#include "../include/io/InputException.h"
#include "../include/filters/Solver.cuh"
#include "../include/filters/Filters.h"

using namespace std;

const string imageDir = "images/";
const string resultDir = "res/";
const string FORMAT = ".jpg";
const string RED = "\u001B[31m";
const string RESET = "\u001B[0m";

using namespace Opt4;

int main(int argc, char **argv)
{
    try
    {
        Args args = Args::parseArgs(argc, argv);

        if (args.filter == Filters::ALL)
        {
            for (auto &filename: args.filenames)
            {
                if (filename.find(".jpg") != string::npos)
                    filename.erase(filename.begin() + filename.find(".jpg"), filename.end());

                Solver::solve(args.filter, imageDir + filename + FORMAT, resultDir + filename);
            }
        }
        else
        {
            for (auto &filename: args.filenames)
                Solver::solve(args.filter, imageDir + filename + FORMAT, resultDir + filename + argv[1] + FORMAT);
        }
    }
    catch (InputException &e)
    {
        cout << RED;
        e.handleInputException();
        cout << RESET;
    }

    return 0;
}