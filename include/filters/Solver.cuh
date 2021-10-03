//
// Created by timofey on 03.10.2021.
//

#ifndef BASICIMAGECONVOLUTION_SOLVER_CUH
#define BASICIMAGECONVOLUTION_SOLVER_CUH

#include <string>
#include <cuda.h>
#include "../image/Image.h"

class Solver
{
public:
    static void solve(int filter, const std::string& inFilename, const std::string& outFilename);

    static Image solveGaussian(const Image& image);

    static Image solveEdge(Image image);

    static Image solveSharpen(Image image);
};

#endif //BASICIMAGECONVOLUTION_SOLVER_CUH
