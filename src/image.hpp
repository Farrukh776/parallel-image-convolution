#pragma once

#include <cstddef>
#include <string>
#include <vector>

// An 8-bit RGB image stored as one flat array: R,G,B,R,G,B,... row by row,
// top row first. A flat array (rather than vector<vector<>>) keeps each row
// contiguous in memory, which matters for cache behaviour once threads are
// each working on their own band of rows.
struct Image {
    static constexpr int CHANNELS = 3;

    int width = 0;
    int height = 0;
    std::vector<unsigned char> data;

    Image() = default;
    Image(int w, int h)
        : width(w), height(h),
          data(static_cast<std::size_t>(w) * h * CHANNELS, 0) {}

    // Pointer to the 3 bytes of pixel (x, y).
    unsigned char* pixel(int x, int y) {
        return &data[(static_cast<std::size_t>(y) * width + x) * CHANNELS];
    }
    const unsigned char* pixel(int x, int y) const {
        return &data[(static_cast<std::size_t>(y) * width + x) * CHANNELS];
    }
};

// 24-bit uncompressed BMP. Throws std::runtime_error on failure.
Image load_bmp(const std::string& path);
void save_bmp(const Image& img, const std::string& path);

// Deterministic synthetic image (gradients, checkerboard, rings, noise) so
// that any workload size can be produced without needing a dataset on disk.
Image generate_test_image(int width, int height);
