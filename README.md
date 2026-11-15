# Parallel Image Convolution with OpenMP

Semester project for Parallel and Distributed Computing. It applies
convolution filters (edge detection, Gaussian blur, sharpening) to large
images, once sequentially and once with OpenMP, and compares the two.

## Status

| Phase | Work | State |
|---|---|---|
| 1 | Project setup | done |
| 2 | Sequential convolution | done |
| 3 | OpenMP parallelisation | done |
| 4 | Benchmark mode over workload sizes and thread counts | done |
| 5 | Speedup and efficiency graphs | to do |
| 6 | Report and demonstration | to do |

## Requirements

- CMake 3.16 or newer
- A C++17 compiler with OpenMP: Visual Studio (MSVC), or g++ / clang++

No third-party libraries are needed.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

With Visual Studio the programs end up in `build/Release/`; with g++ or
clang++ they are directly in `build/`. The commands below use the Visual
Studio paths.

## Run

Correctness tests:

```bash
./build/Release/test_convolve.exe
```

Sequential run on a generated 8000x6000 image:

```bash
./build/Release/convolve.exe --generate 8000x6000 --filter blur
```

OpenMP run with 4 threads, checked against the sequential result:

```bash
./build/Release/convolve.exe --generate 8000x6000 --filter blur --threads 4 --compare
```

Filter a real image and save the result:

```bash
./build/Release/convolve.exe --input photo.bmp --filter edge --threads 8 --output edge.bmp
```

| Option | Meaning |
|---|---|
| `--generate WxH` | create a synthetic test image of that size |
| `--input FILE` | load a 24-bit BMP instead |
| `--filter NAME` | `edge`, `blur` or `sharpen` |
| `--threads N` | use the OpenMP version with N threads (omit for sequential) |
| `--schedule NAME` | `static` (default), `dynamic` or `guided` |
| `--compare` | also run sequentially, verify identical output, print speedup |
| `--output FILE` | save the filtered image |
| `--save-input FILE` | save the unfiltered image |

Input images must be 24-bit uncompressed BMP. In Paint: *Save as* then
*24-bit Bitmap*.

## How it works

**Convolution.** For every pixel, a small grid of weights (the kernel) is
centred on it, each neighbouring pixel is multiplied by the weight above it,
and the products are added up. This is done separately for red, green and
blue. Neighbours that fall outside the image use the nearest pixel inside it.

| Filter | Kernel | Effect |
|---|---|---|
| edge | 3x3 Laplacian, weights sum to 0 | flat areas go black, boundaries stay |
| blur | 5x5 Gaussian, weights sum to 1 | each pixel becomes a weighted average of its neighbourhood |
| sharpen | 3x3, weights sum to 1 | differences from the 4 neighbours are amplified |

**Why it parallelises well.** The filter reads only the input image and
writes only the output image, which is a separate buffer. So the value of one
output pixel never depends on another output pixel, and the rows can be
computed in any order, or at the same time.

**The OpenMP version.** The loop over rows is marked with
`#pragma omp parallel for`. OpenMP splits the rows among the threads; each
thread writes its own rows of the output and nothing else, so no locks,
critical sections or reductions are needed. Both versions call the same
`convolve_row` function, so the parallel output is byte-for-byte identical to
the sequential one, and the tests check that.

## Source layout

| File | Contents |
|---|---|
| `src/convolve.cpp` | `convolve_sequential` and `convolve_parallel` |
| `src/kernels.cpp` | the three kernels |
| `src/image.cpp` | image struct, BMP load/save, test image generator |
| `src/main.cpp` | command-line program and timing |
| `src/benchmark.cpp` | benchmark over sizes, filters and thread counts |
| `tests/test_convolve.cpp` | correctness tests |

## Benchmarks

```bash
./build/Release/benchmark.exe
```

This times the sequential version and the OpenMP version for every
combination of image size, filter and thread count, checks each parallel
output against the sequential one, and writes `results/benchmark.csv`.

| Option | Default |
|---|---|
| `--sizes LIST` | `2000x1500,4000x3000,8000x6000` |
| `--filters LIST` | `edge,blur,sharpen` |
| `--threads LIST` | 1, 2, 4, ... up to the number of logical processors |
| `--schedules LIST` | `static` |
| `--repeats N` | 3 |
| `--output FILE` | `results/benchmark.csv` |

Each configuration is run `--repeats` times. The fastest run (`best_ms`) is
used for speedup, because interference from other programs can only slow a
run down; the average (`mean_ms`) is recorded too.

- Speedup = sequential time / parallel time
- Efficiency = speedup / threads

For clean numbers, close other programs, plug the laptop in, and use more
repeats, for example `--repeats 5`.

### Results so far

AMD Ryzen 5 3500U (4 cores, 8 hardware threads), MSVC Release build, static
schedule, best of 3 runs. Full data is in `results/benchmark.csv`.

Speedup on the 8000x6000 (48 megapixel) image:

| Filter | Sequential | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|---|
| edge 3x3 | 2675 ms | 1.03 | 1.76 | 2.96 | 5.12 |
| blur 5x5 | 5594 ms | 1.01 | 1.44 | 2.50 | 4.59 |
| sharpen 3x3 | 2173 ms | 0.75 | 1.29 | 2.26 | 3.81 |

Speedup at 8 threads as the image grows:

| Filter | 3 MP | 12 MP | 48 MP |
|---|---|---|---|
| edge 3x3 | 3.43 | 3.43 | 5.12 |
| blur 5x5 | 4.14 | 3.95 | 4.59 |
| sharpen 3x3 | 3.45 | 4.19 | 3.81 |

These were measured on a laptop that was in use at the time, and they are
noisy: the 1-thread OpenMP runs range from 0.75 to 1.33 of the sequential
time, when they should all be close to 1.0. They should be re-measured on a
quiet machine before going into the report.
