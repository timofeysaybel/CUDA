//
// Created by timofey on 03.10.2021.
//

#include "../../include/filters/Solver.cuh"
#include "../../include/filters/Filters.h"
#include "../../include/io/Reader.h"
#include "../../include/io/Writer.h"

#include <iostream>

using namespace std;

__global__
void applyFilter(Pixel *image, Pixel *filtered, const double* kernel, int kernelCenter, int width, int height)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height)
    {
        double r = 0., g = 0., b = 0.;

        for (int i = -kernelCenter; i <= kernelCenter; i++)
        {
            for (int j = -kernelCenter; j <= kernelCenter; j++)
            {
                Pixel currentPixel = image[(width + 2 * kernelCenter) * (y + kernelCenter + j) + x  + kernelCenter + i];
                double currentKernelElement = kernel[(2 * kernelCenter + 1) * (j + kernelCenter) + i + kernelCenter];
                r += currentPixel.r * currentKernelElement;
                g += currentPixel.g * currentKernelElement;
                b += currentPixel.b * currentKernelElement;
            }
        }

        filtered[width * y + x].r = (unsigned char)round(r);
        filtered[width * y + x].g = (unsigned char)round(g);
        filtered[width * y + x].b = (unsigned char)round(b);
    }
}

void Solver::solve(int filter, const std::string& inFilename, const std::string& outFilename)
{
    Image image = Reader::read(inFilename);
    Image filtered{};

    switch(filter)
    {
        case Filters::GAUSSIAN:
            filtered = solveGaussian(image);
            break;

        case Filters::EDGE_DETECTION:
            filtered = solveEdge(image);
            break;

        case Filters::SHARPEN:
            filtered = solveSharpen(image);
            break;

        default:
            break;
    }

    Writer::write(filtered, outFilename);
}

Image Solver::solveGaussian(const Image& image)
{
    Pixel *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (Pixel *) malloc(image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dImage, image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dFiltered, image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dKernel, 25 * sizeof(double));

    cudaEvent_t start, stop, startCopy, stopCopy;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);

    cudaEventRecord(startCopy);

    cudaMemcpy(dImage, image.data, image.width * image.height, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, Filters::gaussianKernel, 25, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int blockDims = 32;
    int blocksX = ceil(image.width * 1. / blockDims);
    int blocksY = ceil(image.height * 1. / blockDims);

    cudaEventRecord(start);

    applyFilter<<<blocksX, blocksY>>>(dImage, dFiltered, dKernel, 2, image.width, image.height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(filtered, dFiltered, image.width * image.height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stopCopy);
    cudaEventSynchronize(stopCopy);

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    cudaFree(dImage);
    cudaFree(dKernel);
    cudaFree(dFiltered);

    float tmp = 0.;
    cudaEventElapsedTime(&tmp, startCopy, stopCopy);
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    cudaEventElapsedTime(&tmp, start, stop);
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}

Image Solver::solveEdge(Image image)
{
    Pixel *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (Pixel *) malloc(image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dImage, image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dFiltered, image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dKernel, 9 * sizeof(double));

    cudaEvent_t start, stop, startCopy, stopCopy;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);

    cudaEventRecord(startCopy);

    cudaMemcpy(dImage, image.data, image.width * image.height, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, Filters::edgeKernel, 9, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int blockDims = 32;
    int blocksX = ceil(image.width * 1. / blockDims);
    int blocksY = ceil(image.height * 1. / blockDims);

    cudaEventRecord(start);

    applyFilter<<<blocksX, blocksY>>>(dImage, dFiltered, dKernel, 2, image.width, image.height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(filtered, dFiltered, image.width * image.height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stopCopy);
    cudaEventSynchronize(stopCopy);

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    cudaFree(dImage);
    cudaFree(dKernel);
    cudaFree(dFiltered);

    float tmp = 0.;
    cudaEventElapsedTime(&tmp, startCopy, stopCopy);
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    cudaEventElapsedTime(&tmp, start, stop);
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}

Image Solver::solveSharpen(Image image)
{
    Pixel *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (Pixel *) malloc(image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dImage, image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dFiltered, image.width * image.height * sizeof(Pixel));
    cudaMalloc(&dKernel, 9 * sizeof(double));

    cudaEvent_t start, stop, startCopy, stopCopy;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);

    cudaEventRecord(startCopy);

    cudaMemcpy(dImage, image.data, image.width * image.height, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, Filters::sharpenKernel, 9, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int blockDims = 32;
    int blocksX = ceil(image.width * 1. / blockDims);
    int blocksY = ceil(image.height * 1. / blockDims);

    cudaEventRecord(start);

    applyFilter<<<blocksX, blocksY>>>(dImage, dFiltered, dKernel, 2, image.width, image.height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(filtered, dFiltered, image.width * image.height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stopCopy);
    cudaEventSynchronize(stopCopy);

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    cudaFree(dImage);
    cudaFree(dKernel);
    cudaFree(dFiltered);

    float tmp = 0.;
    cudaEventElapsedTime(&tmp, startCopy, stopCopy);
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    cudaEventElapsedTime(&tmp, start, stop);
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}