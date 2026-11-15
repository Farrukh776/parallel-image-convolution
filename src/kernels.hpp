#pragma once

#include <string>
#include <vector>

// A square convolution kernel with an odd side length (3x3, 5x5, ...),
// stored row by row.
struct Kernel {
    std::string name;
    int size = 0;
    std::vector<float> weights;

    float at(int row, int col) const { return weights[row * size + col]; }
};

Kernel edge_kernel();    // 3x3 Laplacian edge detector
Kernel blur_kernel();    // 5x5 Gaussian blur
Kernel sharpen_kernel(); // 3x3 sharpen

// Accepts "edge", "blur" or "sharpen". Throws std::invalid_argument otherwise.
Kernel kernel_by_name(const std::string& name);
