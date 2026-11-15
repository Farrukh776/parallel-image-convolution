#pragma once

#include "image.hpp"
#include "kernels.hpp"

// For each output pixel, the kernel is centred on the matching input pixel and
// the weighted sum of the neighbourhood is taken, separately for R, G and B.
// Pixels outside the image are replaced by the nearest edge pixel
// ("clamp to edge"), and results are rounded and clamped to 0..255.

// Sequential baseline: one thread computes every row in order.
Image convolve_sequential(const Image& input, const Kernel& kernel);

// How OpenMP divides the rows among the threads.
enum class Schedule { Static, Dynamic, Guided };
const char* schedule_name(Schedule s);

// OpenMP version: the rows are divided among `num_threads` threads.
// The result is byte-for-byte identical to convolve_sequential.
Image convolve_parallel(const Image& input, const Kernel& kernel,
                        int num_threads, Schedule schedule = Schedule::Static);
