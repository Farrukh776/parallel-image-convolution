#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {

// BMP stores integers little-endian. Reading/writing byte by byte keeps this
// independent of struct padding and of the host's byte order.
std::uint32_t read_u32(const unsigned char* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
std::uint16_t read_u16(const unsigned char* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
void write_u32(unsigned char* p, std::uint32_t v) {
    p[0] = static_cast<unsigned char>(v);
    p[1] = static_cast<unsigned char>(v >> 8);
    p[2] = static_cast<unsigned char>(v >> 16);
    p[3] = static_cast<unsigned char>(v >> 24);
}
void write_u16(unsigned char* p, std::uint16_t v) {
    p[0] = static_cast<unsigned char>(v);
    p[1] = static_cast<unsigned char>(v >> 8);
}

const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

// Each BMP row is padded to a multiple of 4 bytes.
std::size_t padded_row_size(int width) {
    return (static_cast<std::size_t>(width) * 3 + 3) & ~std::size_t(3);
}

unsigned char clamp_byte(int v) {
    return static_cast<unsigned char>(std::min(255, std::max(0, v)));
}

} // namespace

Image load_bmp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    unsigned char header[FILE_HEADER_SIZE + INFO_HEADER_SIZE];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || header[0] != 'B' || header[1] != 'M')
        throw std::runtime_error(path + " is not a BMP file");

    const std::uint32_t pixel_offset = read_u32(header + 10);
    const std::int32_t width = static_cast<std::int32_t>(read_u32(header + 18));
    const std::int32_t raw_height = static_cast<std::int32_t>(read_u32(header + 22));
    const std::uint16_t bits_per_pixel = read_u16(header + 28);
    const std::uint32_t compression = read_u32(header + 30);

    if (bits_per_pixel != 24 || compression != 0)
        throw std::runtime_error(path + ": only 24-bit uncompressed BMP is supported "
                                        "(re-save it from Paint as '24-bit Bitmap')");
    if (width <= 0 || raw_height == 0)
        throw std::runtime_error(path + ": invalid dimensions");

    // A positive height means the rows are stored bottom-to-top (the usual case).
    const bool bottom_up = raw_height > 0;
    const int height = bottom_up ? raw_height : -raw_height;

    Image img(width, height);
    std::vector<unsigned char> row(padded_row_size(width));
    in.seekg(pixel_offset);

    for (int i = 0; i < height; ++i) {
        in.read(reinterpret_cast<char*>(row.data()),
                static_cast<std::streamsize>(row.size()));
        if (!in) throw std::runtime_error(path + ": file is truncated");

        const int y = bottom_up ? height - 1 - i : i;
        unsigned char* dst = img.pixel(0, y);
        for (int x = 0; x < width; ++x) {
            // BMP stores each pixel as B,G,R.
            dst[x * 3 + 0] = row[x * 3 + 2];
            dst[x * 3 + 1] = row[x * 3 + 1];
            dst[x * 3 + 2] = row[x * 3 + 0];
        }
    }
    return img;
}

void save_bmp(const Image& img, const std::string& path) {
    const std::size_t row_size = padded_row_size(img.width);
    const std::size_t pixel_bytes = row_size * img.height;
    const std::size_t file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + pixel_bytes;
    if (file_size > 0xFFFFFFFFull)
        throw std::runtime_error("image is too large for the BMP format (4 GB limit)");

    unsigned char header[FILE_HEADER_SIZE + INFO_HEADER_SIZE] = {};
    header[0] = 'B';
    header[1] = 'M';
    write_u32(header + 2, static_cast<std::uint32_t>(file_size));
    write_u32(header + 10, FILE_HEADER_SIZE + INFO_HEADER_SIZE);
    write_u32(header + 14, INFO_HEADER_SIZE);
    write_u32(header + 18, static_cast<std::uint32_t>(img.width));
    write_u32(header + 22, static_cast<std::uint32_t>(img.height)); // bottom-up
    write_u16(header + 26, 1);  // colour planes
    write_u16(header + 28, 24); // bits per pixel
    write_u32(header + 34, static_cast<std::uint32_t>(pixel_bytes));

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char*>(header), sizeof(header));

    std::vector<unsigned char> row(row_size, 0);
    for (int i = 0; i < img.height; ++i) {
        const unsigned char* src = img.pixel(0, img.height - 1 - i);
        for (int x = 0; x < img.width; ++x) {
            row[x * 3 + 0] = src[x * 3 + 2];
            row[x * 3 + 1] = src[x * 3 + 1];
            row[x * 3 + 2] = src[x * 3 + 0];
        }
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size()));
    }
    if (!out) throw std::runtime_error("failed while writing " + path);
}

Image generate_test_image(int width, int height) {
    Image img(width, height);
    std::uint32_t seed = 12345u; // fixed seed: the same size always gives the same image

    const double cx = width / 2.0;
    const double cy = height / 2.0;
    const int max_x = std::max(1, width - 1);
    const int max_y = std::max(1, height - 1);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            seed = seed * 1664525u + 1013904223u; // linear congruential generator
            const int noise = static_cast<int>((seed >> 24) % 32) - 16;

            int r = 255 * x / max_x;                       // left-to-right gradient
            int g = 255 * y / max_y;                       // top-to-bottom gradient
            int b = (((x / 64) + (y / 64)) & 1) ? 200 : 60; // checkerboard: hard edges

            // Concentric white rings: curved edges at every orientation.
            const double dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (std::fmod(dist, 200.0) < 20.0) r = g = b = 255;

            unsigned char* p = img.pixel(x, y);
            p[0] = clamp_byte(r + noise);
            p[1] = clamp_byte(g + noise);
            p[2] = clamp_byte(b + noise);
        }
    }
    return img;
}
