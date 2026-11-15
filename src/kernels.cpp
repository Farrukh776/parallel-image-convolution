#include "kernels.hpp"

#include <stdexcept>

// Laplacian: the weights sum to 0, so flat regions become black and only
// places where a pixel differs from its neighbours survive.
Kernel edge_kernel() {
    return {"edge", 3, {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1,
    }};
}

// Gaussian approximated by binomial coefficients (1 4 6 4 1). The weights sum
// to 256, so dividing by 256 keeps overall brightness unchanged.
Kernel blur_kernel() {
    Kernel k{"blur", 5, {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1,
    }};
    for (float& w : k.weights) w /= 256.0f;
    return k;
}

// Original pixel plus its Laplacian: the weights sum to 1 (brightness is
// preserved) while differences from the 4 neighbours are amplified.
Kernel sharpen_kernel() {
    return {"sharpen", 3, {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0,
    }};
}

Kernel kernel_by_name(const std::string& name) {
    if (name == "edge") return edge_kernel();
    if (name == "blur") return blur_kernel();
    if (name == "sharpen") return sharpen_kernel();
    throw std::invalid_argument("unknown filter '" + name +
                                "' (expected edge, blur or sharpen)");
}
