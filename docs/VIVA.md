# Viva Preparation Sheet — Parallel Image Convolution

16 November 2026

## The project in one minute

If asked to summarise, say this in your own words:

1. We apply convolution filters (edge detection, Gaussian blur, sharpening) to large images.
2. We wrote a sequential version first and tested it against inputs with known answers.
3. We parallelised the loop over image rows with one OpenMP `parallel for` directive.
4. The parallel output is byte-for-byte identical to the sequential output.
5. On a 4-core, 8-thread laptop a 48-megapixel image ran about 2.5 times faster on 4 threads and 3.6 to 3.9 times faster on 8.
6. Efficiency drops as threads are added, mainly because threads 5 to 8 share physical cores.

Numbers to know by heart, from the main benchmark run (update them if you re-measure):

| Quantity | Value |
| --- | --- |
| Largest image | 8000x6000, 48 megapixels, 144 MB |
| Sequential time, blur | about 5.8 s |
| 8-thread time, blur | about 1.5 s |
| Speedup on 4 threads | about 2.5 |
| Speedup on 8 threads | 3.6 to 3.9 |
| Efficiency on 2, 4, 8 threads | about 80%, 64%, 46% |
| Tests | 11, all passing |

## Convolution and filters

**What is convolution?** Each output pixel is a weighted sum of the input pixel and its neighbours. The weights are a small grid called the kernel. We do this separately for red, green and blue.

**What are your three kernels?** Edge detection is a 3x3 Laplacian: 8 in the centre and -1 in the eight neighbours. Blur is a 5x5 Gaussian built from 1 4 6 4 1 in each direction, divided by 256. Sharpening is 3x3: 5 in the centre and -1 in the four direct neighbours.

**Why do the edge weights sum to 0 and the others to 1?** A sum of 0 means a flat area gives 0, so only changes in intensity survive. A sum of 1 keeps the overall brightness of the image unchanged.

**How do you handle pixels at the border?** Clamp to edge. A neighbour outside the image is replaced by the nearest pixel inside it. The alternatives are zero padding, which darkens the borders, and mirroring.

**What is the difference between convolution and correlation?** Convolution flips the kernel in both directions before applying it. Our code does the flip. Our three kernels are symmetric, so the output is the same either way, but one test uses an asymmetric kernel to prove the flip is correct.

**What is the time complexity?** O(W x H x K x K) for an image of W by H pixels and a kernel of side K. That is why the 5x5 blur takes more than twice as long as the 3x3 filters.

**Why can't you filter the image in place?** Later pixels would read neighbours that were already overwritten with filtered values. We write to a separate output image.

## OpenMP and parallelisation

**Show me the parallel part of your code.** It is one directive on the row loop in `src/convolve.cpp`:

```cpp
#pragma omp parallel for num_threads(num_threads) schedule(static)
for (int y = 0; y < height; ++y)
    convolve_row(in, out, width, height, weights, ksize, y);
```

**What does `parallel for` do?** `parallel` creates a team of threads. `for` divides the iterations of the loop among them. At the end of the loop there is an implicit barrier: the program continues only when every thread has finished.

**Why is there no race condition?** A race needs two threads touching the same memory with at least one writing. The input image and the kernel are only read. In the output image, each row is written by exactly one thread. No byte is written by two threads.

**Which variables are shared and which are private?** Shared: the input pointer, the output pointer, the kernel weights and the sizes. Private: the loop variable `y`, which OpenMP makes private automatically, and all local variables inside `convolve_row`, which live on each thread's own stack.

**Why did you not need `critical`, `atomic` or `reduction`?** Those protect or combine shared results. We have no shared result that several threads update. Each thread writes its own separate pixels.

**Why parallelise the outer loop and not the inner ones?** The outer loop gives each thread many whole rows, so the work-sharing cost is paid once. Parallelising the pixel loop would pay it once per row, and the kernel loops once per pixel, where the work is far too small to be worth it.

**What type of parallelism is this?** Data parallelism. The same operation is applied to different parts of the data at the same time. It is also called embarrassingly parallel, because the parts do not depend on each other.

**Explain static, dynamic and guided scheduling.** Static divides the rows into one equal band per thread before the loop starts. Dynamic lets each thread take a chunk of rows (16 in our code) from a shared queue whenever it is free. Guided is like dynamic, but the chunks start large and get smaller.

**Which schedule was fastest?** We expected static, because every row costs the same. On 8 threads dynamic was about 10% faster. Our explanation is that the threads do not all run at the same speed when they share physical cores and the laptop is busy, and dynamic lets faster threads take the remaining rows. On 1 to 4 threads there was no clear difference.

**What is false sharing, and do you have it?** False sharing happens when threads write to different variables that sit in the same cache line, so the line bounces between cores. With static scheduling each thread writes one large contiguous band, so threads can only meet at the few bytes where two bands touch. It is negligible here.

**How do you set the number of threads?** With the `num_threads` clause, from the `--threads` command-line option. The alternatives are `omp_set_num_threads()` and the `OMP_NUM_THREADS` environment variable.

**Which OpenMP version do you use?** Visual Studio's default, OpenMP 2.0. That is why the three schedules are written as three separate directives: choosing a schedule from code with `omp_set_schedule()` needs OpenMP 3.0.

## Performance and results

**Define speedup and efficiency.** Speedup is sequential time divided by parallel time. Efficiency is speedup divided by the number of threads. Ideal speedup on p threads is p, and ideal efficiency is 100%.

**Why is your speedup on 8 threads not 8?** The laptop has 4 physical cores. Threads 5 to 8 are hardware threads that share those cores, so they add much less than a full core each. Going from 4 to 8 threads gave only about 1.4 to 1.5 times more speed.

**Then why is it only 2.5 on 4 threads, not 4?** Two likely reasons, which we did not measure separately. First, a laptop processor runs one busy core at a higher clock speed than four busy cores, so the sequential baseline is faster per core. Second, all threads share the same memory channels, and this program streams 288 MB of image data.

**State Amdahl's law. Does it explain your results?** If a fraction f of the work is sequential, speedup on p threads is at most 1 / (f + (1 - f) / p). Our only sequential part inside the timed region is allocating the output image, a few percent at most. With f = 0.03 the limit on 4 threads is about 3.7, so Amdahl's law alone does not explain 2.5. The hardware does.

**What does Gustafson's law say?** If the problem grows with the number of processors, the sequential fraction matters less and speedup can keep growing. Our edge and blur results fit this loosely: speedup on 8 threads rose from about 3.2 to 3.3 on the 3 MP image to 3.8 to 3.9 on the 48 MP image.

**How did you measure time?** With `omp_get_wtime()`, which gives wall-clock time. We time only the convolution call, not loading, generating or saving the image.

**Why wall-clock time and not CPU time?** CPU time adds up the time of all threads, so it would stay the same or grow as threads are added. The user waits for wall-clock time.

**Why do you take the fastest of 5 runs instead of the average?** Other programs and the operating system can only make a run slower, never faster. The minimum is the run least disturbed. We record the mean as well.

**What do you mean by interleaved runs?** Each round runs the sequential version, then 1, 2, 4 and 8 threads, and then the next round starts. A laptop slows down as it heats up. If we ran all sequential tests first and all 8-thread tests last, that slowdown would be mixed into the speedup.

**Why does the 1-thread OpenMP version take the same time as the sequential one?** It does the same arithmetic with a team of one thread. The only extra cost is setting up the parallel region once, which is tiny compared with seconds of work. It is a useful check on measurement noise: when the two differ, the difference is noise.

**Did larger images scale better?** Slightly, for edge detection and blur. Sharpening did not follow the trend, and the smallest image runs in well under half a second, so we treat the difference as within the noise.

## Code and design choices

**How is an image stored?** As one flat array of bytes: R, G, B, R, G, B and so on, row by row from the top. Pixel (x, y) starts at index (y x width + x) x 3. A flat array keeps each row contiguous in memory, which is good for the cache.

**How do you know the sequential version is correct?** We test it on inputs whose answer is known without running the filter. An identity kernel must leave the image unchanged. Blurring a constant image must give the same constant. Edge detection of a constant image must give black. Blurring a single bright pixel must reproduce the kernel.

**How do you know the parallel version is correct?** We compare it byte for byte with the sequential output, for all three filters, all three schedules, and 1, 2, 3, 4, 7, 8 and 16 threads. The benchmark repeats this check on every run.

**Why is the output exactly identical and not just close?** Both versions call the same function for each row, and each pixel's sum is added up in the same order. Floating-point results only differ when the order of operations changes, and ours does not.

**Why BMP and not JPEG or PNG?** Uncompressed 24-bit BMP can be read and written in about a hundred lines with no external library. The image format is not what the project studies.

**Why synthetic images?** The running time of a convolution depends on the image size and kernel size, not on what the picture shows. A generator lets us produce any size on demand. The program also accepts real BMP photos.

**Why floating-point arithmetic?** The Gaussian weights are fractions. The sums are computed in `float`, then rounded and clamped to 0 to 255.

**What was the hardest part?** Getting trustworthy measurements. The first benchmark gave a sequential baseline that moved by about 30% between runs on the same laptop. Interleaving the runs made the results consistent.

**What does each file do?**

| File | Role |
| --- | --- |
| `src/convolve.cpp` | Sequential and parallel convolution |
| `src/kernels.cpp` | The three kernels |
| `src/image.cpp` | Image storage, BMP files, test image generator |
| `src/main.cpp` | Command-line program |
| `src/benchmark.cpp` | Timing over sizes, filters, schedules and threads |
| `tests/test_convolve.cpp` | The 11 tests |
| `scripts/plot_results.py` | Graphs |

## Hard questions

These are the weak points an examiner may press on. The honest answer is the strongest one.

**Your results are noisy. Can they be trusted?** The trend can: about 2.5 on 4 threads and between 3.5 and 5 on 8, with falling efficiency. The exact figures cannot. A separate session on the same laptop gave about 5.1 on 8 threads for edge detection where the main run gave 3.75. We reduced the noise with interleaving and best-of-5, and we report the remaining spread.

**Did you prove that hyper-threading or memory bandwidth causes the loss?** No. They are the likely causes, based on how the hardware works and on where the curve bends. Proving it would need hardware performance counters or a run with hyper-threading disabled.

**Is there anything sequential left in the timed region?** Yes. The output image is allocated and zero-filled by one thread before the parallel loop. It is a small fraction of the time, and parallelising it is listed as future work.

**Could the sequential version be faster?** Yes. A separable Gaussian would cut the blur from 25 to 10 multiply-adds per pixel, and SIMD instructions could process several pixels at once. We kept the simple version so that the sequential and parallel code do the same work and the comparison is fair.

**Why only one machine?** It was the hardware we had. The method and the scripts are the same on any machine, so repeating the measurements on a desktop with more cores is a matter of running one command.

**Would this scale to 64 cores?** The algorithm would, because there are no dependencies between rows. In practice memory bandwidth would become the limit well before 64 times, and on a machine with several processor sockets the single-threaded allocation of the output would place all its memory near one socket.

**Would a GPU be better?** For this workload, very likely. Convolution is the same small calculation on millions of independent pixels, which is what GPUs are built for. That would use CUDA or OpenCL and is outside the scope of an OpenMP project.

**Why OpenMP and not MPI?** The image fits in the memory of one machine and all cores share it, so shared-memory threads are the natural fit. MPI would mean copying parts of the image between processes for no benefit on a single computer.

## Demo script

The whole demo takes about three minutes. Build the project beforehand, plug the laptop in and close other programs. Run the commands from the project folder.

1. Show that the tests pass.

   ```bash
   ./build/Release/test_convolve.exe
   ```
2. Produce the three filtered images and open them next to the original.

   ```bash
   mkdir -p output
   ./build/Release/convolve.exe --generate 1200x800 --filter edge --output output/edge.bmp --save-input output/original.bmp
   ./build/Release/convolve.exe --generate 1200x800 --filter blur --output output/blur.bmp
   ./build/Release/convolve.exe --generate 1200x800 --filter sharpen --output output/sharpen.bmp
   ```
3. Run the sequential version on the large image and point out the time.

   ```bash
   ./build/Release/convolve.exe --generate 8000x6000 --filter blur
   ```
4. Run the parallel version with 8 threads. It also runs the sequential version, confirms the outputs are identical, and prints speedup and efficiency.

   ```bash
   ./build/Release/convolve.exe --generate 8000x6000 --filter blur --threads 8 --compare
   ```
5. Repeat step 4 with `--threads 4` and `--threads 2` to show how the speedup changes.
6. Open `src/convolve.cpp` and show the `#pragma omp parallel for` line and the shared `convolve_row` function.
7. Open the four graphs in the `plots` folder and explain the speedup graph first.

If a live number differs from the report, say why: timings on a laptop change from run to run, which is the reason the benchmark repeats each measurement.
