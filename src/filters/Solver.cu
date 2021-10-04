//
// Created by timofey
//

#include "../../include/filters/Solver.cuh"
#include "../../include/filters/Filters.h"
#include "../../include/io/Reader.h"
#include "../../include/io/Writer.h"

#include <iostream>

#define SAFE_CALL(CallInstruction) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
         throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL(KernelCallInstruction){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
        throw "error in CUDA kernel execution, aborting..."; \
    } \
}

using namespace std;

__global__
void applyFilter(Pixel *image, Pixel *filtered, const double *kernel, int kernelCenter, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    double r = 0., g = 0., b = 0.;

    for (int i = -kernelCenter; i <= kernelCenter; i++)
    {
        for (int j = -kernelCenter; j <= kernelCenter; j++)
        {
            int xx = x + i;
            int yy = y + j;
            if (xx < 0)
                xx = 0;
            if (xx >= width)
                xx = width - 1;
            if (yy < 0)
                yy = 0;
            if (yy >= height)
                yy = height - 1;

            Pixel currentPixel = image[yy * width + xx];
            double currentKernelElement = kernel[(2 * kernelCenter + 1) * (j + kernelCenter) + i + kernelCenter];
            r += currentPixel.r * currentKernelElement;
            g += currentPixel.g * currentKernelElement;
            b += currentPixel.b * currentKernelElement;
        }
    }

    filtered[width * y + x].r = (unsigned char) round(r);
    filtered[width * y + x].g = (unsigned char) round(g);
    filtered[width * y + x].b = (unsigned char) round(b);
}

void Solver::solve(int filter, const std::string &inFilename, const std::string &outFilename)
{
    Image image = Reader::read(inFilename);
    Image filtered{};

    switch (filter)
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

Image Solver::solveGaussian(const Image &image)
{
    Pixel *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (Pixel *) malloc(image.width * image.height * sizeof(Pixel));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * sizeof(Pixel)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * sizeof(Pixel)));
    SAFE_CALL(cudaMalloc(&dKernel, 25 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, image.data, image.width * image.height * sizeof(Pixel), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::gaussianKernel, 25 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(32, 32);
    dim3 blocks(image.width / threads.x + 1, image.height / threads.y + 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 2, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * sizeof(Pixel), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stopCopy));
    SAFE_CALL(cudaEventSynchronize(stopCopy));

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    SAFE_CALL(cudaFree(dImage));
    SAFE_CALL(cudaFree(dKernel));
    SAFE_CALL(cudaFree(dFiltered));

    float tmp = 0.;
    SAFE_CALL(cudaEventElapsedTime(&tmp, startCopy, stopCopy));
    cout << "Gaussian blur: " << endl;
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    SAFE_CALL(cudaEventElapsedTime(&tmp, start, stop));
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}

Image Solver::solveEdge(Image image)
{
    Pixel *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (Pixel *) malloc(image.width * image.height * sizeof(Pixel));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * sizeof(Pixel)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * sizeof(Pixel)));
    SAFE_CALL(cudaMalloc(&dKernel, 9 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, image.data, image.width * image.height * sizeof(Pixel), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::edgeKernel, 9 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(32, 32);
    dim3 blocks(image.width / threads.x + 1, image.height / threads.y + 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 1, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * sizeof(Pixel), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stopCopy));
    SAFE_CALL(cudaEventSynchronize(stopCopy));

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    SAFE_CALL(cudaFree(dImage));
    SAFE_CALL(cudaFree(dKernel));
    SAFE_CALL(cudaFree(dFiltered));

    float tmp = 0.;
    SAFE_CALL(cudaEventElapsedTime(&tmp, startCopy, stopCopy));
    cout << "Edge detection: " << endl;
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    SAFE_CALL(cudaEventElapsedTime(&tmp, start, stop));
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}

Image Solver::solveSharpen(Image image)
{
    Pixel *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (Pixel *) malloc(image.width * image.height * sizeof(Pixel));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * sizeof(Pixel)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * sizeof(Pixel)));
    SAFE_CALL(cudaMalloc(&dKernel, 9 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, image.data, image.width * image.height * sizeof(Pixel), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::sharpenKernel, 9 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(32, 32);
    dim3 blocks(image.width / threads.x + 1, image.height / threads.y + 1);
    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 1, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * sizeof(Pixel), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stopCopy));
    SAFE_CALL(cudaEventSynchronize(stopCopy));

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    SAFE_CALL(cudaFree(dImage));
    SAFE_CALL(cudaFree(dKernel));
    SAFE_CALL(cudaFree(dFiltered));

    float tmp = 0.;
    SAFE_CALL(cudaEventElapsedTime(&tmp, startCopy, stopCopy));
    cout << "Sharpen: " << endl;
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    SAFE_CALL(cudaEventElapsedTime(&tmp, start, stop));
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}