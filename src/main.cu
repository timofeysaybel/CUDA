//
// Created by timofey
//
#include <iostream>

#include "../include/io/Args.h"
#include "../include/io/InputException.h"
#include "../include/filters/Solver.cuh"

using namespace std;

const string imageDir = "images/";
const string resultDir = "res/";
const string FORMAT = ".jpg";
const string RED = "\u001B[31m";
const string RESET = "\u001B[0m";


int main(int argc, char **argv)
{
    try
    {
        Args args = Args::parseArgs(argc, argv);

        for (auto &filename: args.filenames)
            Solver::solve(args.filter, imageDir + filename + FORMAT, resultDir + filename + argv[1] + FORMAT);
    }
    catch (InputException &e)
    {
        cout << RED;
        e.handleInputException();
        cout << RESET;
    }

    return 0;
}