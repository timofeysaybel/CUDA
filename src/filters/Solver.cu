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

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)

const int BLOCK_SIZE = 16;

using namespace std;

//-----------------------------------------------------TASK 1-----------------------------------------------------------

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

void Task1::Solver::solve(int filter, const std::string &inFilename, const std::string &outFilename)
{
    const string FORMAT = ".jpg";
    Image image = Reader::read(inFilename);
    Image filtered{};

    switch (filter)
    {
        case Filters::GAUSSIAN:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::EDGE_DETECTION:
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::SHARPEN:
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::ALL:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename + "gaussian" + FORMAT);
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename + "edge" + FORMAT);
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename + "sharpen" + FORMAT);

        default:
            break;
    }
}

Image Task1::Solver::solveGaussian(const Image &image)
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

Image Task1::Solver::solveEdge(Image image)
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

Image Task1::Solver::solveSharpen(Image image)
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

//-------------------------------------------------ОПТИМИИЗАЦИЯ (TASK 2)------------------------------------------------
//-----------------------------------------1) РАЗВЕРТКА МАССИВА ИЗОБРАЖЕНИЯ---------------------------------------------

void Opt1::Solver::solve(int filter, const std::string &inFilename, const std::string &outFilename)
{
    const string FORMAT = ".jpg";
    Image image = Reader::read(inFilename);
    Image filtered{};

    switch (filter)
    {
        case Filters::GAUSSIAN:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::EDGE_DETECTION:
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::SHARPEN:
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::ALL:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename + "gaussian" + FORMAT);
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename + "edge" + FORMAT);
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename + "sharpen" + FORMAT);

        default:
            break;
    }
}

__global__
void applyFilter(unsigned char *image, unsigned char *filtered, const double *kernel, int kernelCenter, int width,
                 int height)
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

            double currentKernelElement = kernel[(2 * kernelCenter + 1) * (j + kernelCenter) + i + kernelCenter];
            r += image[yy * width + xx] * currentKernelElement;
            g += image[width * height + yy * width + xx] * currentKernelElement;
            b += image[2 * width * height + yy * width + xx] * currentKernelElement;
        }
    }

    filtered[width * y + x] = (unsigned char) round(r);
    filtered[width * y + x + width * height] = (unsigned char) round(g);
    filtered[width * y + x + 2 * width * height] = (unsigned char) round(b);
}

Image Opt1::Solver::solveGaussian(const Image &image)
{
    unsigned char *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dKernel, 25 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::gaussianKernel, 25 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE));

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 2, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
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

Image Opt1::Solver::solveEdge(Image image)
{
    unsigned char *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dKernel, 9 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::edgeKernel, 9 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE));

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 1, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
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

Image Opt1::Solver::solveSharpen(Image image)
{
    unsigned char *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dKernel, 9 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::sharpenKernel, 9 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE));

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 1, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
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

//-----------------------------------------2) ИСПОЛЬЗОВАНИЕ РАЗДЕЛЯЕМОЙ ПАМЯТИ------------------------------------------

void Opt2::Solver::solve(int filter, const std::string &inFilename, const std::string &outFilename)
{
    const string FORMAT = ".jpg";
    Image image = Reader::read(inFilename);
    Image filtered{};

    switch (filter)
    {
        case Filters::GAUSSIAN:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::EDGE_DETECTION:
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::SHARPEN:
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::ALL:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename + "gaussian" + FORMAT);
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename + "edge" + FORMAT);
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename + "sharpen" + FORMAT);

        default:
            break;
    }
}

namespace opt2
{
    __global__
    void applyFilter(unsigned char *image, unsigned char *filtered, const double *kernel, int kernelCenter, int width,
                     int height)
    {
        __shared__ unsigned buf[BLOCK_SIZE * BLOCK_SIZE * 3];

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        int idx = y * width + x + threadIdx.z * width * height;
        buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * blockDim.x + threadIdx.x] = image[idx];
        __syncthreads();

        if (threadIdx.x <= 1 || threadIdx.y <= 1 || threadIdx.x >= blockDim.x - 2 || threadIdx.y >= blockDim.y - 2 ||
            x > width + 1 || y > height + 1)
            return;

        double c = 0.;

        for (int i = -kernelCenter; i <= kernelCenter; i++)
        {
            for (int j = -kernelCenter; j <= kernelCenter; j++)
            {
                double currentKernelElement = kernel[(2 * kernelCenter + 1) * (j + kernelCenter) + i + kernelCenter];
                c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + i) + threadIdx.x + j] * currentKernelElement;
            }
        }

        filtered[threadIdx.z * width * height + width * y + x] = (unsigned char) round(c);
    }
}

Image Opt2::Solver::solveGaussian(const Image &image)
{
    unsigned char *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dKernel, 25 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::gaussianKernel, 25 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 3);

    dim3 blocks(FRACTION_CEILING(image.width , BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE), 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((opt2::applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 2, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
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

Image Opt2::Solver::solveEdge(Image image)
{
    unsigned char *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dKernel, 9 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::edgeKernel, 9 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 3);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE), 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((opt2::applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 1, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
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

Image Opt2::Solver::solveSharpen(Image image)
{
    unsigned char *dImage, *dFiltered, *filtered;
    double *dKernel;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dKernel, 9 * sizeof(double)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dKernel, Filters::sharpenKernel, 9 * sizeof(double), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 3);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE), 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((opt2::applyFilter<<<blocks, threads>>>(dImage, dFiltered, dKernel, 1, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
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

//-----------------------------------------3) ВЫБОР ОПТИМАЛЬНОГО БЛОКА--------------------------------------------------
//----------------------------------------------------USE Opt2----------------------------------------------------------

//--------------------------------------4) ОПТИМИЗАЦИЯ ПРИМЕНЕНИЯ ФИЛЬТРА-----------------------------------------------
//--------------------------------------------4.1) РАЗВЕРКТА ЦИКЛОВ-----------------------------------------------------
//---------------------------4.2) ИСПОЛЬЗОВАНИЕ ОТДЕЛЬНЫХ ФУНКЦИЙ ДЛЯ КАЖДОГО ФИЛЬТРА ----------------------------------

void Opt4::Solver::solve(int filter, const std::string &inFilename, const std::string &outFilename)
{
    const string FORMAT = ".jpg";
    Image image = Reader::read(inFilename);
    Image filtered{};

    switch (filter)
    {
        case Filters::GAUSSIAN:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::EDGE_DETECTION:
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::SHARPEN:
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename);
            break;

        case Filters::ALL:
            filtered = solveGaussian(image);
            Writer::write(filtered, outFilename + "gaussian" + FORMAT);
            filtered = solveEdge(image);
            Writer::write(filtered, outFilename + "edge" + FORMAT);
            filtered = solveSharpen(image);
            Writer::write(filtered, outFilename + "sharpen" + FORMAT);

        default:
            break;
    }
}

__global__
void gaussianFilter(unsigned char *image, unsigned char *filtered, int width, int height)
{
    __shared__ unsigned buf[BLOCK_SIZE * BLOCK_SIZE * 3];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * width + x + threadIdx.z * width * height;
    buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * blockDim.x + threadIdx.x] = image[idx];
    __syncthreads();

    if (threadIdx.x <= 1 || threadIdx.y <= 1 || threadIdx.x >= blockDim.x - 2 || threadIdx.y >= blockDim.y - 2 ||
        x > width + 1 || y > height + 1)
        return;

    double c = 0.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 2) + threadIdx.x - 2] *
            1 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x - 2] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x - 2] *
            7 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x - 2] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 2) + threadIdx.x - 2] *
            1 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 2) + threadIdx.x - 1] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x - 1] *
            16 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x - 1] *
            26 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x - 1] *
            16 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 2) + threadIdx.x - 1] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 2) + threadIdx.x] *
            7 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x] *
            26 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x] *
            41 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x] *
            26 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 2) + threadIdx.x] *
            7 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 2) + threadIdx.x + 1] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x + 1] *
            16 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x + 1] *
            26 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x + 1] *
            16 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 2) + threadIdx.x + 1] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 2) + threadIdx.x + 2] *
            1 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x + 2] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x + 2] *
            7 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x + 2] *
            4 / 273.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 2) + threadIdx.x + 2] *
            1 / 273.;

    filtered[threadIdx.z * width * height + width * y + x] = (unsigned char) round(c);
}

__global__
void edgeFilter(unsigned char *image, unsigned char *filtered, int width, int height)
{
    __shared__ unsigned buf[BLOCK_SIZE * BLOCK_SIZE * 3];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * width + x + threadIdx.z * width * height;
    buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * blockDim.x + threadIdx.x] = image[idx];
    __syncthreads();

    if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == blockDim.x - 1 || threadIdx.y == blockDim.y - 1 ||
        x > width || y > height)
        return;

    double c = 0.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x - 1] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x] *
         4.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x + 1] *
         -1.;

    filtered[threadIdx.z * width * height + width * y + x] = (unsigned char) round(c);
}

__global__
void sharpenFilter(unsigned char *image, unsigned char *filtered, int width, int height)
{
    __shared__ unsigned buf[BLOCK_SIZE * BLOCK_SIZE * 3];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * width + x + threadIdx.z * width * height;
    buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * blockDim.x + threadIdx.x] = image[idx];
    __syncthreads();

    if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == blockDim.x - 1 || threadIdx.y == blockDim.y - 1 ||
        x > width || y > height)
        return;

    double c = 0.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x - 1] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x - 1] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x - 1] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x] *
         9.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x - 1) + threadIdx.x + 1] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x) + threadIdx.x + 1] *
         -1.;
    c += buf[threadIdx.z * BLOCK_SIZE * BLOCK_SIZE + threadIdx.y * (blockDim.x + 1) + threadIdx.x + 1] *
         -1.;

    filtered[threadIdx.z * width * height + width * y + x] = (unsigned char) round(c);
}

Image Opt4::Solver::solveGaussian(const Image &image)
{
    unsigned char *dImage, *dFiltered, *filtered;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 3);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE), 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((gaussianFilter<<<blocks, threads>>>(dImage, dFiltered, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stopCopy));
    SAFE_CALL(cudaEventSynchronize(stopCopy));

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    SAFE_CALL(cudaFree(dImage));
    SAFE_CALL(cudaFree(dFiltered));

    float tmp = 0.;
    SAFE_CALL(cudaEventElapsedTime(&tmp, startCopy, stopCopy));
    cout << "Gaussian blur: " << endl;
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    SAFE_CALL(cudaEventElapsedTime(&tmp, start, stop));
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}

Image Opt4::Solver::solveEdge(Image image)
{
    unsigned char *dImage, *dFiltered, *filtered;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 3);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE), 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((edgeFilter<<<blocks, threads>>>(dImage, dFiltered, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stopCopy));
    SAFE_CALL(cudaEventSynchronize(stopCopy));

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    SAFE_CALL(cudaFree(dImage));
    SAFE_CALL(cudaFree(dFiltered));

    float tmp = 0.;
    SAFE_CALL(cudaEventElapsedTime(&tmp, startCopy, stopCopy));
    cout << "Edge detection: " << endl;
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    SAFE_CALL(cudaEventElapsedTime(&tmp, start, stop));
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}

Image Opt4::Solver::solveSharpen(Image image)
{
    unsigned char *dImage, *dFiltered, *filtered;

    filtered = (unsigned char *) malloc(image.width * image.height * 3 * sizeof(unsigned char));
    SAFE_CALL(cudaMalloc(&dImage, image.width * image.height * 3 * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&dFiltered, image.width * image.height * 3 * sizeof(unsigned char)));

    cudaEvent_t start, stop, startCopy, stopCopy;
    unsigned char *img = image.getData();

    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventCreate(&startCopy));
    SAFE_CALL(cudaEventCreate(&stopCopy));

    SAFE_CALL(cudaEventRecord(startCopy));

    SAFE_CALL(cudaMemcpy(dImage, img, image.width * image.height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaDeviceSynchronize());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 3);

    dim3 blocks(FRACTION_CEILING(image.width * 3, BLOCK_SIZE), FRACTION_CEILING(image.height, BLOCK_SIZE), 1);

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((sharpenFilter<<<blocks, threads>>>(dImage, dFiltered, image.width, image.height)));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    SAFE_CALL(cudaMemcpy(filtered, dFiltered, image.width * image.height * 3 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());

    SAFE_CALL(cudaEventRecord(stopCopy));
    SAFE_CALL(cudaEventSynchronize(stopCopy));

    Image res(filtered, image.width, image.height, image.channels);

    free(filtered);
    SAFE_CALL(cudaFree(dImage));
    SAFE_CALL(cudaFree(dFiltered));

    float tmp = 0.;
    SAFE_CALL(cudaEventElapsedTime(&tmp, startCopy, stopCopy));
    cout << "Sharpen: " << endl;
    cout << "Time for " << image.height << "x" << image.width << " image with copying: " << tmp << endl;
    SAFE_CALL(cudaEventElapsedTime(&tmp, start, stop));
    cout << "Time for " << image.height << "x" << image.width << " image without copying: " << tmp << endl;

    return res;
}