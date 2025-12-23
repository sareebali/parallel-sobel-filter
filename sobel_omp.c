#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "pgmio.h"

void applyBlur(float *inp, float *out, int numRows, int numCols);
void applySobel(float *inp, float *out, int numRows, int numCols);
double getExecutionTime();

int main(int argc, char *argv[]){
    // In case any argument (image file name and/or number of threads) is not given
    if (argc != 3) {
        printf("Please provide the file name followed by the number of threads as arguments.\n");
        return 1;
    }

    // Setting number of threads
    int numThreads = atoi(argv[2]);
    omp_set_num_threads(numThreads);

    float *img;
    int rows, cols;

    // Reading image
    if (pgmread(argv[1], &img, &rows, &cols) != 0) {
        fprintf(stderr, "Failed to read %s\n", argv[1]);
        return 1;
    }

    // Initializing array to store blurred and sobel filtered images
    float *blurredImg = malloc(rows*cols*sizeof(float));
    float *finalImg = malloc(rows*cols*sizeof(float));

    double startTime = omp_get_wtime();

    // Applying filters
    applyBlur(img, blurredImg, rows, cols);
    applySobel(blurredImg, finalImg, rows, cols);

    double endTime = omp_get_wtime();

    // Output image file name
    char outName[100];
    snprintf(outName, sizeof(outName), "omp_%dx%d_t%d.pgm", cols, rows, numThreads);

    // Saving output image
    if (pgmwrite(outName, finalImg, rows, cols, 1) != 0) {
        fprintf(stderr, "Failed to write %s\n", outName);
        free(img);
        free(blurredImg);
        free(finalImg);
        return 1;
    }

    // Freeing up space
    free(img);
    free(blurredImg);
    free(finalImg);

    // Final messages to user
    printf("Image Produced: %s\n", outName);
    printf("Number of Threads used: %d\n", numThreads);
    printf("Execution Time: %.5f seconds\n", endTime-startTime);

    return 0;
}

// Applies 3X3 blurring filter
void applyBlur(float *inp, float *out, int numRows, int numCols){
    #pragma omp parallel for collapse(2) // Flattening and parallelizing nested loop
    for (int i=1; i<numRows-1; i++){ // Track each pixel except the border ones
        for (int j=1; j<numCols-1; j++){
            float total = 0.0;
            for (int kr=-1; kr<2; kr++){ // 3X3 kernel/submatrix
                for (int kc=-1; kc<2; kc++){
                    total += inp[(i+kr)*numCols+(j+kc)];
                }
            }

            // Updating image pixels
            out[i*numCols+j] = total/9.0;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i=0; i<numRows; i++){ // No blurring on border pixels
        for (int j=0; j<numCols; j++){
            if (i==0 || i==numRows-1 || j==0 || j==numCols-1){
                out[i*numCols+j] = inp[i*numCols+j];
            }
        }
    }
}

// Applies Sobel filter
void applySobel(float *inp, float *out, int numRows, int numCols){
    // Sobel kernels matrices
    int matx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int maty[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    float highestTotal = 0.0;

    #pragma omp parallel for collapse(2) reduction(max:highestTotal) // reduction prevents data race for highestTotal
    for (int i=1; i<numRows-1; i++){ // Track each pixel except the border ones
        for (int j=1; j<numCols-1; j++){
            float gx = 0.0;
            float gy = 0.0;
            for (int kr=-1; kr<2; kr++){ // 3X3 kernel/submatrix
                for (int kc=-1; kc<2; kc++){
                    // calculating gx and gy for this pixel
                    gx += inp[(i+kr)*numCols+(j+kc)] * matx[kr+1][kc+1];
                    gy += inp[(i+kr)*numCols+(j+kc)] * maty[kr+1][kc+1];
                }
            }

            float g = sqrt(gx*gx + gy*gy); // Calculating G
            out[i*numCols+j] = g; // Updating image pixels
            highestTotal = g > highestTotal ? g : highestTotal; // Storing the highest value of g up till now
        }
    }

   #pragma omp parallel for collapse(2)
    for (int i=1; i<numRows-1; i++){ // Normalize each pixel (between 0 and 255 inclusive)
        for (int j=1; j<numCols-1; j++){
            out[i*numCols+j] = (out[i*numCols+j]/highestTotal)*255.0;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i=0; i<numRows; i++){ // Making borders black - no edge detection
        for (int j=0; j<numCols; j++){
            if (i==0 || i==numRows-1 || j==0 || j==numCols-1){
                out[i*numCols+j] = 0.0;
            }
        }
    }
}