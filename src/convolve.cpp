#include "convolve.hpp"

#include <algorithm>
#include <cstddef>

#include <omp.h>

namespace {

// Rows handed out per request under the dynamic and guided schedules.
#define CHUNK 16

// Clamp to 0..255, then round to nearest. Adding 0.5 and truncating is used
// instead of std::lround because lround is a slow library call on MSVC and
// this runs three times for every pixel.
unsigned char to_byte(float v) {
    const float clamped = std::min(255.0f, std::max(0.0f, v));
    return static_cast<unsigned char>(clamped + 0.5f);
}

// Computes one complete output row. This is the unit of work: the sequential
// version calls it for rows 0..height-1 in order, the parallel version hands
// different rows to different threads. Sharing the function guarantees both
// versions do exactly the same arithmetic.
//
// It reads `in` (never written during filtering) and writes only row `y` of
// `out`, so two threads working on different rows can never touch the same
// byte - there is nothing to lock.
//
// Plain pointers are passed instead of the Image/Kernel objects because going
// through the vectors inside the hot loop makes the compiler re-read their
// addresses on every iteration (a write through an unsigned char* is allowed
// to modify any object).
inline void convolve_row(const unsigned char* in, unsigned char* out,
                         int width, int height,
                         const float* weights, int ksize, int y) {
    const int radius = ksize / 2;

    for (int x = 0; x < width; ++x) {
        float sum[Image::CHANNELS] = {0.0f, 0.0f, 0.0f};

        for (int ky = 0; ky < ksize; ++ky) {
            // Clamp to edge: neighbours outside the image use the nearest
            // pixel that is inside it.
            const int sy = std::min(height - 1, std::max(0, y + ky - radius));

            for (int kx = 0; kx < ksize; ++kx) {
                const int sx = std::min(width - 1, std::max(0, x + kx - radius));

                // True convolution flips the kernel in both directions
                // (without the flip this would be cross-correlation). The
                // three filters here are symmetric, so it makes no visible
                // difference to them, but it keeps the maths correct.
                const float w = weights[(ksize - 1 - ky) * ksize + (ksize - 1 - kx)];

                const unsigned char* src =
                    in + (static_cast<std::size_t>(sy) * width + sx) * Image::CHANNELS;
                sum[0] += w * src[0];
                sum[1] += w * src[1];
                sum[2] += w * src[2];
            }
        }

        unsigned char* dst =
            out + (static_cast<std::size_t>(y) * width + x) * Image::CHANNELS;
        dst[0] = to_byte(sum[0]);
        dst[1] = to_byte(sum[1]);
        dst[2] = to_byte(sum[2]);
    }
}

} // namespace

const char* schedule_name(Schedule s) {
    switch (s) {
        case Schedule::Static:  return "static";
        case Schedule::Dynamic: return "dynamic";
        case Schedule::Guided:  return "guided";
    }
    return "?";
}

Image convolve_sequential(const Image& input, const Kernel& kernel) {
    const int width = input.width;
    const int height = input.height;

    // The result goes into a separate image. Filtering in place would be wrong:
    // later pixels would read neighbours that had already been overwritten.
    Image output(width, height);

    const unsigned char* const in = input.data.data();
    unsigned char* const out = output.data.data();
    const float* const weights = kernel.weights.data();

    for (int y = 0; y < height; ++y)
        convolve_row(in, out, width, height, weights, kernel.size, y);

    return output;
}

Image convolve_parallel(const Image& input, const Kernel& kernel,
                        int num_threads, Schedule schedule) {
    const int width = input.width;
    const int height = input.height;

    Image output(width, height);

    const unsigned char* const in = input.data.data();
    unsigned char* const out = output.data.data();
    const float* const weights = kernel.weights.data();
    const int ksize = kernel.size;

    // The outer (row) loop is the one parallelised: each thread gets many
    // whole rows, so threads are started once and work on large, contiguous
    // blocks of memory. Parallelising an inner loop instead would pay the
    // work-sharing overhead once per row or once per pixel.
    //
    // Data sharing:
    //   shared  - in, weights (read only) and out (each row is written by
    //             exactly one thread)
    //   private - the loop variable y (OpenMP makes it private
    //             automatically), and everything declared inside
    //             convolve_row (x, sum, ...), which lives on each thread's
    //             own stack
    // No critical sections, locks or reductions are needed.
    //
    // The three branches differ only in the schedule clause:
    //   static  - rows are split into one equal contiguous band per thread,
    //             decided up front. No bookkeeping while running.
    //   dynamic - threads take CHUNK rows at a time from a shared queue.
    //   guided  - like dynamic, but the chunks start large and shrink.
    // Every row costs the same here, so static is expected to be best; the
    // other two exist so that this can be shown by measurement. (They are
    // written out separately, rather than using schedule(runtime) with
    // omp_set_schedule(), because that needs OpenMP 3.0 and Visual Studio's
    // default OpenMP is 2.0.)
    switch (schedule) {
        case Schedule::Static:
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int y = 0; y < height; ++y)
                convolve_row(in, out, width, height, weights, ksize, y);
            break;

        case Schedule::Dynamic:
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic, CHUNK)
            for (int y = 0; y < height; ++y)
                convolve_row(in, out, width, height, weights, ksize, y);
            break;

        case Schedule::Guided:
            #pragma omp parallel for num_threads(num_threads) schedule(guided, CHUNK)
            for (int y = 0; y < height; ++y)
                convolve_row(in, out, width, height, weights, ksize, y);
            break;
    }

    return output;
}
