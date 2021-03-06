//
// Created by timofey
//

#ifndef BASICIMAGECONVOLUTION_SOLVER_CUH
#define BASICIMAGECONVOLUTION_SOLVER_CUH

#include <string>
#include <cuda.h>
#include <vector>
#include "../image/Image.h"


namespace Task1
{
    class Solver
    {
    public:
        static void solve(int filter, const std::string &inFilename, const std::string &outFilename);

        static Image solveGaussian(const Image &image);

        static Image solveEdge(Image image);

        static Image solveSharpen(Image image);


    };
}
//-------------------------------------------------ОПТИМИИЗАЦИЯ (TASK 2)------------------------------------------------
//-----------------------------------------1) РАЗВЕРТКА МАССИВА ИЗОБРАЖЕНИЯ---------------------------------------------
namespace Opt1
{
    class Solver
    {
    public:
        static void solve(int filter, const std::string &inFilename, const std::string &outFilename);

        static Image solveGaussian(const Image &image);

        static Image solveEdge(Image image);

        static Image solveSharpen(Image image);
    };
}
//-----------------------------------------2) ИСПОЛЬЗОВАНИЕ РАЗДЕЛЯЕМОЙ ПАМЯТИ------------------------------------------
namespace Opt2
{
    class Solver
    {
    public:
        static void solve(int filter, const std::string &inFilename, const std::string &outFilename);

        static Image solveGaussian(const Image &image);

        static Image solveEdge(Image image);

        static Image solveSharpen(Image image);
    };
}

//-----------------------------------------3) ВЫБОР ОПТИМАЛЬНОГО БЛОКА--------------------------------------------------
//----------------------------------------------------USE Opt2----------------------------------------------------------

//--------------------------------------4) ОПТИМИЗАЦИЯ ПРИМЕНЕНИЯ ФИЛЬТРА-----------------------------------------------
//--------------------------------------------4.1) РАЗВЕРКТА ЦИКЛОВ-----------------------------------------------------
//--------------------------------------------4.2) FULL UNROLL  --------------------------------------------------------
namespace Opt4
{
    class Solver
    {
    public:
        static void solve(int filter, const std::string &inFilename, const std::string &outFilename);

        static Image solveGaussian(const Image &image);

        static Image solveEdge(Image image);

        static Image solveSharpen(Image image);
    };
}
//--------------------------------------5) ОПТИМИЗАЦИЯ МАЛЕНЬКИХ ИЗОБРАЖЕНИЙ--------------------------------------------
namespace Opt5
{
    class Solver
    {
    public:
        static void solve(const std::vector<std::string> &inFilenames, std::vector<std::string> &outFilenames);

        static std::vector<Image> solveGaussian(const std::vector<Image> &image);

        static std::vector<Image> solveEdge(std::vector<Image> image);

        static std::vector<Image> solveSharpen(std::vector<Image> image);
    };
}

#endif //BASICIMAGECONVOLUTION_SOLVER_CUH
