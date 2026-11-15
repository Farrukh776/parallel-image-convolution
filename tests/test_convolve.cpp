// Correctness checks for the sequential baseline. Each test uses an input
// whose correct output is known in advance without running the filter.

#include "convolve.hpp"
#include "image.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "[pass] " : "[FAIL] ") << what << "\n";
    if (!ok) ++failures;
}

Image constant_image(int w, int h, unsigned char value) {
    Image img(w, h);
    std::fill(img.data.begin(), img.data.end(), value);
    return img;
}

bool all_equal_to(const Image& img, unsigned char value) {
    return std::all_of(img.data.begin(), img.data.end(),
                       [value](unsigned char v) { return v == value; });
}

} // namespace

int main() {
    const Image sample = generate_test_image(67, 45); // odd sizes on purpose

    // A kernel that is 1 in the centre and 0 elsewhere must change nothing.
    {
        const Kernel identity{"identity", 3, {0, 0, 0, 0, 1, 0, 0, 0, 0}};
        const Image out = convolve_sequential(sample, identity);
        check(out.data == sample.data, "identity kernel leaves the image unchanged");
    }

    // Blur weights sum to 1, so a flat image stays flat - including at the
    // borders, which exercises the clamp-to-edge handling.
    {
        const Image out = convolve_sequential(constant_image(40, 30, 137), blur_kernel());
        check(all_equal_to(out, 137), "blur of a constant image is the same constant");
    }

    // Sharpen weights also sum to 1.
    {
        const Image out = convolve_sequential(constant_image(40, 30, 90), sharpen_kernel());
        check(all_equal_to(out, 90), "sharpen of a constant image is the same constant");
    }

    // Edge weights sum to 0: no edges in, black out.
    {
        const Image out = convolve_sequential(constant_image(40, 30, 200), edge_kernel());
        check(all_equal_to(out, 0), "edge detection of a constant image is all black");
    }

    // An asymmetric kernel shows whether the kernel is flipped correctly.
    // With a single 1 in the middle-left cell, true convolution gives
    // out(x, y) = in(x + 1, y); cross-correlation would give in(x - 1, y).
    {
        const Kernel shift{"shift", 3, {0, 0, 0, 1, 0, 0, 0, 0, 0}};
        const Image out = convolve_sequential(sample, shift);
        bool ok = true;
        for (int y = 0; y < sample.height && ok; ++y)
            for (int x = 0; x < sample.width && ok; ++x) {
                const int sx = std::min(x + 1, sample.width - 1);
                ok = std::equal(out.pixel(x, y), out.pixel(x, y) + 3, sample.pixel(sx, y));
            }
        check(ok, "asymmetric kernel shifts the image the way convolution should");
    }

    // One bright pixel on black: blurring must reproduce the kernel itself.
    {
        Image impulse(9, 9);
        unsigned char* centre = impulse.pixel(4, 4);
        centre[0] = centre[1] = centre[2] = 255;
        const Image out = convolve_sequential(impulse, blur_kernel());
        // Centre weight is 36/256, corner weight is 1/256.
        check(out.pixel(4, 4)[0] == 36 && out.pixel(2, 2)[0] == 1 && out.pixel(1, 1)[0] == 0,
              "blur of a single bright pixel reproduces the Gaussian kernel");
    }

    // Width 67 is not a multiple of 4, so BMP row padding is exercised.
    {
        const auto path = std::filesystem::temp_directory_path() / "pdc_roundtrip_test.bmp";
        save_bmp(sample, path.string());
        const Image loaded = load_bmp(path.string());
        std::filesystem::remove(path);
        check(loaded.width == sample.width && loaded.height == sample.height &&
                  loaded.data == sample.data,
              "BMP save then load returns the identical image");
    }

    // The parallel version must give byte-for-byte the same image as the
    // sequential one, for every filter, schedule and thread count - including
    // thread counts that do not divide the row count evenly (3, 7) and more
    // threads than this machine has cores (16).
    {
        const Image big = generate_test_image(301, 203);
        const Schedule schedules[] = {Schedule::Static, Schedule::Dynamic, Schedule::Guided};
        const int thread_counts[] = {1, 2, 3, 4, 7, 8, 16};

        for (const char* name : {"edge", "blur", "sharpen"}) {
            const Kernel kernel = kernel_by_name(name);
            const Image expected = convolve_sequential(big, kernel);
            bool ok = true;
            for (Schedule s : schedules)
                for (int t : thread_counts)
                    ok = ok && convolve_parallel(big, kernel, t, s).data == expected.data;
            check(ok, std::string("parallel ") + name +
                          " matches sequential (3 schedules x 7 thread counts)");
        }
    }

    // More threads than rows: some threads get no work at all.
    {
        const Image thin = generate_test_image(50, 3);
        const Image expected = convolve_sequential(thin, blur_kernel());
        check(convolve_parallel(thin, blur_kernel(), 8).data == expected.data,
              "parallel is correct when there are fewer rows than threads");
    }

    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nSome tests FAILED.\n");
    return failures == 0 ? 0 : 1;
}
