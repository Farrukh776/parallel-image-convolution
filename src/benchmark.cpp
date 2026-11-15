// Benchmark driver: times the sequential and OpenMP versions over a grid of
// image sizes x filters x schedules x thread counts and writes one CSV row
// per configuration.

#include "convolve.hpp"
#include "image.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

namespace {

void print_usage() {
    std::cout <<
        "Usage: benchmark [options]\n"
        "\n"
        "  --sizes LIST      image sizes        (default 2000x1500,4000x3000,8000x6000)\n"
        "  --filters LIST    filters            (default edge,blur,sharpen)\n"
        "  --threads LIST    thread counts      (default 1,2,4,... up to this machine's\n"
        "                                        logical processor count)\n"
        "  --schedules LIST  OpenMP schedules   (default static)\n"
        "  --repeats N       runs per configuration (default 3)\n"
        "  --output FILE     CSV file           (default results/benchmark.csv)\n"
        "\n"
        "LIST values are comma separated, e.g. --threads 1,2,4,8\n";
}

std::vector<std::string> split(const std::string& text, char sep = ',') {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, sep))
        if (!part.empty()) parts.push_back(part);
    return parts;
}

Schedule parse_schedule(const std::string& name) {
    if (name == "static") return Schedule::Static;
    if (name == "dynamic") return Schedule::Dynamic;
    if (name == "guided") return Schedule::Guided;
    throw std::invalid_argument("unknown schedule '" + name + "'");
}

struct Timing {
    double best_ms = 0.0;  // fastest run
    double total_ms = 0.0; // sum of all runs, for the mean
    int runs = 0;

    void add(double ms) {
        if (runs == 0 || ms < best_ms) best_ms = ms;
        total_ms += ms;
        ++runs;
    }
    double mean_ms() const { return runs ? total_ms / runs : 0.0; }
};

// Runs `work` once, stores its image in `result` and returns the elapsed
// wall-clock time in milliseconds.
template <typename Work>
double time_once(Image& result, Work work) {
    const double start = omp_get_wtime();
    result = work();
    return (omp_get_wtime() - start) * 1000.0;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> size_names = {"2000x1500", "4000x3000", "8000x6000"};
    std::vector<std::string> filter_names = {"edge", "blur", "sharpen"};
    std::vector<std::string> schedule_names = {"static"};
    std::vector<int> thread_counts;
    int repeats = 3;
    std::string output_path = "results/benchmark.csv";

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };

            if (arg == "--sizes") size_names = split(next());
            else if (arg == "--filters") filter_names = split(next());
            else if (arg == "--schedules") schedule_names = split(next());
            else if (arg == "--output") output_path = next();
            else if (arg == "--repeats") {
                repeats = std::atoi(next().c_str());
                if (repeats < 1) throw std::invalid_argument("--repeats must be 1 or more");
            } else if (arg == "--threads") {
                for (const std::string& t : split(next())) {
                    const int n = std::atoi(t.c_str());
                    if (n < 1) throw std::invalid_argument("bad thread count '" + t + "'");
                    thread_counts.push_back(n);
                }
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }

        const int processors = omp_get_num_procs();
        if (thread_counts.empty()) {
            // 1, 2, 4, ... and always finish on the full processor count.
            for (int t = 1; t < processors; t *= 2) thread_counts.push_back(t);
            thread_counts.push_back(processors);
        }

        std::vector<std::pair<int, int>> sizes;
        for (const std::string& name : size_names) {
            int w = 0, h = 0;
            if (std::sscanf(name.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0)
                throw std::invalid_argument("bad size '" + name + "' (expected WxH)");
            sizes.emplace_back(w, h);
        }

        std::vector<Kernel> kernels;
        for (const std::string& name : filter_names) kernels.push_back(kernel_by_name(name));

        std::vector<Schedule> schedules;
        for (const std::string& name : schedule_names) schedules.push_back(parse_schedule(name));

        const std::filesystem::path out_file(output_path);
        if (out_file.has_parent_path())
            std::filesystem::create_directories(out_file.parent_path());
        std::ofstream csv(out_file);
        if (!csv) throw std::runtime_error("cannot write " + output_path);
        csv << "filter,kernel_size,width,height,megapixels,mode,schedule,threads,"
               "repeats,best_ms,mean_ms,speedup,efficiency,correct\n";

        std::printf("logical processors: %d, repeats per configuration: %d\n",
                    processors, repeats);

        // OpenMP creates its worker threads the first time a parallel region
        // runs. Doing that once here keeps the start-up cost out of the first
        // timed measurement.
        #pragma omp parallel
        { }

        bool all_correct = true;

        for (const auto& [width, height] : sizes) {
            const Image input = generate_test_image(width, height);
            const double megapixels = width * double(height) / 1e6;

            for (const Kernel& kernel : kernels) {
                std::printf("\n%d x %d (%.1f MP), %s %dx%d\n", width, height, megapixels,
                            kernel.name.c_str(), kernel.size, kernel.size);
                std::printf("  %-10s %-8s %7s %11s %11s %8s %10s\n", "mode", "schedule",
                            "threads", "best ms", "mean ms", "speedup", "efficiency");

                // One entry per parallel configuration (schedule x thread count).
                struct Config {
                    Schedule schedule;
                    int threads;
                    Timing timing;
                    bool correct;
                };
                std::vector<Config> configs;
                for (Schedule schedule : schedules)
                    for (int threads : thread_counts)
                        configs.push_back({schedule, threads, Timing{}, true});

                // The runs are interleaved: each round runs the sequential
                // version once and then every parallel configuration once.
                // A laptop CPU changes speed over time (temperature, power
                // limits, background programs). If all sequential runs were
                // done first and all 8-thread runs last, that drift would be
                // mistaken for speedup or slowdown. Interleaving spreads it
                // evenly over every configuration.
                //
                // The fastest run of each configuration is the headline
                // number: disturbances can only make a run slower, never
                // faster, so the minimum is the least affected by noise.
                Timing seq;
                Image reference, result;
                for (int round = 0; round < repeats; ++round) {
                    seq.add(time_once(reference, [&] {
                        return convolve_sequential(input, kernel);
                    }));
                    for (Config& c : configs) {
                        c.timing.add(time_once(result, [&] {
                            return convolve_parallel(input, kernel, c.threads, c.schedule);
                        }));
                        c.correct = c.correct && result.data == reference.data;
                    }
                }

                // Sequential baseline: every speedup below is relative to this.
                std::printf("  %-10s %-8s %7d %11.1f %11.1f %8.2f %9.0f%%\n", "sequential",
                            "-", 1, seq.best_ms, seq.mean_ms(), 1.0, 100.0);
                csv << kernel.name << ',' << kernel.size << ',' << width << ',' << height
                    << ',' << megapixels << ",sequential,none,1," << repeats << ','
                    << seq.best_ms << ',' << seq.mean_ms() << ",1,1,yes\n";

                for (const Config& c : configs) {
                    // Speedup    S(p) = T_sequential / T_parallel(p)
                    // Efficiency E(p) = S(p) / p
                    const double speedup = seq.best_ms / c.timing.best_ms;
                    const double efficiency = speedup / c.threads;
                    all_correct = all_correct && c.correct;

                    std::printf("  %-10s %-8s %7d %11.1f %11.1f %8.2f %9.0f%%%s\n",
                                "openmp", schedule_name(c.schedule), c.threads,
                                c.timing.best_ms, c.timing.mean_ms(), speedup,
                                100.0 * efficiency, c.correct ? "" : "   OUTPUT MISMATCH");
                    csv << kernel.name << ',' << kernel.size << ',' << width << ','
                        << height << ',' << megapixels << ",openmp,"
                        << schedule_name(c.schedule) << ',' << c.threads << ',' << repeats
                        << ',' << c.timing.best_ms << ',' << c.timing.mean_ms() << ','
                        << speedup << ',' << efficiency << ',' << (c.correct ? "yes" : "NO")
                        << '\n';
                }
                csv.flush(); // keep finished rows if the run is interrupted
                std::fflush(stdout);
            }
        }

        std::printf("\nresults written to %s\n", output_path.c_str());
        if (!all_correct) {
            std::printf("ERROR: at least one parallel output differed from sequential\n");
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage();
        return 1;
    }
    return 0;
}
