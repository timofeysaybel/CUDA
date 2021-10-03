//
// Created by timofey on 02.10.2021.
//

#ifndef IMAGECONVOLUTION_FILTERS_H
#define IMAGECONVOLUTION_FILTERS_H

namespace Filters
{
    const int GAUSSIAN = 0;
    const int EDGE_DETECTION = 1;
    const int SHARPEN = 2;

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
            -1 / 8., -1 / 8., -1 / 8.,
            -1 / 8., 1., -1 / 8.,
            -1 / 8., -1 / 8., -1 / 8.,
            };

    const double sharpenKernel[] =
            {
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
            };
}

#endif //IMAGECONVOLUTION_FILTERS_H
