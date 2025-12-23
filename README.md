# parallel-sobel-filter
This repository provides parallel implementations of the Sobel operator for edge detection on grayscale images using OpenMP (shared-memory) and MPI (distributed-memory) in C. The programs apply a 3x3 average blurring filter to reduce noise before performing edge detection. They are designed to process square grayscale images in PGM format and output the edge-detected result as a new PGM file.
The Sobel filter computes gradients in the x and y directions using the following kernels:

- G_x kernel:
-1  0  +1
-2  0  +2
-1  0  +1

- G_y kernel:
-1 -2 -1
 0  0  0
+1 +2 +1

The gradient magnitude is calculated as G = √(G_x² + G_y²), normalized to [0, 255], with borders set to black.

The 3x3 blurring filter uses:
1/9 * [1 1 1]
      [1 1 1]
      [1 1 1]

Sample images (256x256 and 1024x1024 peppers in grayscale) are included for testing.


# Files
- sobel_omp.c: OpenMP implementation for multi-threaded execution on a single node. Uses parallel loops for blurring and Sobel application, with reduction for finding the maximum gradient.
- sobel_mpi.c: MPI implementation for distributed execution across multiple processes/nodes. Uses row-wise domain decomposition, ghost rows for boundary communication, and MPI collectives for global max gradient.
- sample_256.pgm: 256x256 grayscale test image (peppers).
- sample_1024.pgm: 1024x1024 grayscale test image (peppers).
- pgmio.h: Header file for PGM image I/O (assumed to be available or included in compilation; not provided here but standard for such tasks).


# Compilation and Execution
Prerequisites
- GCC compiler
- OpenMP support (enabled with -fopenmp)
- MPI library (e.g., OpenMPI, installed via package manager)
- PGM images for input (samples provided)

## OpenMP Version (sobel_omp.c)
- Compile: gcc -o sobel_omp sobel_omp.c -fopenmp -lm
- Run: ./sobel_omp <input_image.pgm> <num_threads>
- Example: ./sobel_omp sample_256.pgm 4
- Outputs: omp_<width>x<height>_t<threads>.pgm (e.g., omp_256x256_t4.pgm)
- Tested with thread counts like 1, 2, 4, 8, 16, 32.
- Suitable for image sizes up to 16000x16000; processes in tiles if needed for large images.

## MPI Version (sobel_mpi.c)
- Compile: mpicc -o sobel_mpi sobel_mpi.c -lm
- Run (using mpirun to specify processes): mpirun -np <num_processes> ./sobel_mpi <input_image.pgm>
- Example: mpirun -np 4 ./sobel_mpi sample_256.pgm
- Outputs: mpi_<width>x<height>_p<processes>.pgm (e.g., mpi_256x256_p4.pgm)
- Tested with 1, 2, 4 processes on one node, and 8, 16, 32 on multiple nodes.
- Uses non-blocking sends/receives for ghost row exchange and Allreduce for global max.

Notes: Measure execution time excluding I/O (using omp_get_wtime() or MPI_Wtime()).
