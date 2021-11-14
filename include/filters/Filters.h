//
// Created by timofey
//

#ifndef IMAGECONVOLUTION_FILTERS_H
#define IMAGECONVOLUTION_FILTERS_H

namespace Filters
{
    const int GAUSSIAN = 0;
    const int EDGE_DETECTION = 1;
    const int SHARPEN = 2;
    const int ALL = 3;

    const double gaussianKernel[] =
            {
            1 / 273., 4 / 273., 7 / 273., 4 / 273., 1 / 273.,
            4 / 273., 16 / 273., 26 / 273., 16 / 273., 4 / 273.,
            7 / 273., 26 / 273., 41 / 273., 26 / 273., 7 / 273.,
            4 / 273., 16 / 273., 26 / 273., 16 / 273., 4 / 273.,
            1 / 273., 4 / 273., 7 / 273., 4 / 273., 1 / 273.,
            };

    const double edgeKernel[] =
            {
            0, -1, 0,
            -1, 4, -1,
            0, -1, 0,
            };

    const double sharpenKernel[] =
            {
            -1, -1, -1,
            -1, 9, -1,
            -1, -1, -1
            };
}

#endif //IMAGECONVOLUTION_FILTERS_H
