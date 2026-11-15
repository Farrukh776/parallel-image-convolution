#include "convolve.hpp"
#include "image.hpp"
#include "kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <omp.h>

namespace {

void print_usage() {
    std::cout <<
        "Usage:\n"
        "  convolve (--generate WxH | --input in.bmp) --filter NAME [options]\n"
        "\n"
        "  --generate WxH   create a synthetic test image, e.g. 4000x3000\n"
        "  --input FILE     load a 24-bit BMP instead\n"
        "  --filter NAME    edge | blur | sharpen\n"
        "  --threads N      use the OpenMP version with N threads\n"
        "                   (omit to run the sequential version)\n"
        "  --schedule NAME  static | dynamic | guided   (default static)\n"
        "  --compare        with --threads: also run the sequential version,\n"
        "                   check both outputs are identical, report speedup\n"
        "  --output FILE    save the filtered image as BMP\n"
        "  --save-input F   also save the unfiltered image\n";
}

Schedule parse_schedule(const std::string& name) {
    if (name == "static") return Schedule::Static;
    if (name == "dynamic") return Schedule::Dynamic;
    if (name == "guided") return Schedule::Guided;
    throw std::invalid_argument("unknown schedule '" + name +
                                "' (expected static, dynamic or guided)");
}

} // namespace

int main(int argc, char** argv) {
    std::string input_path, output_path, save_input_path, filter_name;
    int gen_width = 0, gen_height = 0;
    int threads = 0; // 0 = sequential version
    Schedule schedule = Schedule::Static;
    bool compare = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };

            if (arg == "--input") input_path = next();
            else if (arg == "--output") output_path = next();
            else if (arg == "--save-input") save_input_path = next();
            else if (arg == "--filter") filter_name = next();
            else if (arg == "--schedule") schedule = parse_schedule(next());
            else if (arg == "--compare") compare = true;
            else if (arg == "--threads") {
                threads = std::atoi(next().c_str());
                if (threads < 1) throw std::invalid_argument("--threads must be 1 or more");
            } else if (arg == "--generate") {
                const std::string size = next();
                if (std::sscanf(size.c_str(), "%dx%d", &gen_width, &gen_height) != 2 ||
                    gen_width <= 0 || gen_height <= 0)
                    throw std::invalid_argument("--generate expects WxH, e.g. 4000x3000");
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }

        const bool generate = gen_width > 0;
        if (generate == !input_path.empty())
            throw std::invalid_argument("give exactly one of --input or --generate");
        if (filter_name.empty()) throw std::invalid_argument("--filter is required");
        if (compare && threads == 0)
            throw std::invalid_argument("--compare needs --threads");

        const Kernel kernel = kernel_by_name(filter_name);
        const Image input = generate ? generate_test_image(gen_width, gen_height)
                                     : load_bmp(input_path);

        std::printf("image:    %d x %d (%.1f megapixels)\n", input.width, input.height,
                    input.width * double(input.height) / 1e6);
        std::printf("filter:   %s (%dx%d kernel)\n", kernel.name.c_str(), kernel.size,
                    kernel.size);

        // Only the convolution itself is timed. Loading, generating and saving
        // are not part of the work being parallelised, so including them would
        // hide the effect we want to measure. omp_get_wtime() is OpenMP's
        // wall-clock timer: elapsed real time, not CPU time summed over threads.
        Image output;
        if (threads == 0) {
            const double t0 = omp_get_wtime();
            output = convolve_sequential(input, kernel);
            const double seq_ms = (omp_get_wtime() - t0) * 1000.0;

            std::printf("mode:     sequential\n");
            std::printf("time:     %.2f ms\n", seq_ms);
        } else {
            const double t0 = omp_get_wtime();
            output = convolve_parallel(input, kernel, threads, schedule);
            const double par_ms = (omp_get_wtime() - t0) * 1000.0;

            std::printf("mode:     OpenMP, %d thread%s, schedule(%s)\n", threads,
                        threads == 1 ? "" : "s", schedule_name(schedule));
            std::printf("          (this machine has %d logical processors)\n",
                        omp_get_num_procs());
            std::printf("time:     %.2f ms\n", par_ms);

            if (compare) {
                const double t1 = omp_get_wtime();
                const Image reference = convolve_sequential(input, kernel);
                const double seq_ms = (omp_get_wtime() - t1) * 1000.0;

                const double speedup = seq_ms / par_ms;
                std::printf("seq time: %.2f ms\n", seq_ms);
                std::printf("speedup:  %.2fx\n", speedup);
                std::printf("efficiency: %.0f%%\n", 100.0 * speedup / threads);

                const bool same = reference.data == output.data;
                std::printf("output:   %s\n", same ? "identical to sequential"
                                                   : "DIFFERENT from sequential");
                if (!same) return 2;
            }
        }

        if (!save_input_path.empty()) save_bmp(input, save_input_path);
        if (!output_path.empty()) save_bmp(output, output_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage();
        return 1;
    }
    return 0;
}
