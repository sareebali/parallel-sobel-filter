#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "pgmio.h"

void applyBlur(float *inp, float *out, int numRows, int numCols);
void applySobel(float *inp, float *out, int numRows, int numCols, float *localHighest, int rank, int size);
void addGhostRows(float *inp, int numRows, int numCols, int rank, int size);

int main(int argc, char *argv[]){
    // Initializing and declarations
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // In case no image name is given
    if (argc != 2) {
        if (rank==0) printf("Please provide the file name as an argument");
        MPI_Finalize();
        return 1;
    }


    float *img = NULL;
    int rows, cols;

    // Reading image
    if (rank==0){
        if (pgmread(argv[1], &img, &rows, &cols) != 0) {
            fprintf(stderr, "Failed to read %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Broadcasting to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculating number of rows in a block
    int blockRowsNum = rows/size;
    int extraRows = rows%size;
    int firstRow;
    int lastRow;
    int blockRows;

    // Allocating the remainder of rows
    if (rank<extraRows){
        blockRowsNum++;
        firstRow = rank*blockRowsNum;
    } else {
        firstRow = rank*blockRowsNum+extraRows;
    }
    lastRow = firstRow+blockRowsNum;

    // Accounting for ghost rows
    if (size==1) blockRows = rows;
    else if (rank==size-1) blockRows = blockRowsNum+1;
    else blockRows = blockRowsNum+2;

    // Initializing buffers
    float *inBuf = calloc(blockRows*cols, sizeof(float));
    float *blurBuf = calloc(blockRows*cols, sizeof(float));
    float *outBuf = calloc(blockRows*cols, sizeof(float));
    if (!inBuf || !blurBuf || !outBuf){
        fprintf(stderr, "Rank %d: Buffer Allocation failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    };

    // Preparing for scatter
    int *sCountArr = NULL;
    int *displArr = NULL;
    int rCount = blockRowsNum*cols;
    if (rank==0){
        sCountArr = malloc(size*sizeof(int));
        displArr = malloc(size*sizeof(int));
        int displ = 0;
        for (int i=0; i<size; ++i){
            int newR = (i<extraRows) ? blockRowsNum : blockRowsNum-1;
            if (i >= extraRows) newR = blockRowsNum;
            sCountArr[i] = newR*cols;
            displArr[i]=displ;
            displ += sCountArr[i];
        }
    }

    // Scattering image to all processes (by dividing them into blocks of rows)
    if (size>1) MPI_Scatterv(img, sCountArr, displArr, MPI_FLOAT, inBuf+cols, rCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    else MPI_Scatterv(img, sCountArr, displArr, MPI_FLOAT, inBuf, rCount, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank==0){
        free(sCountArr);
        free(displArr);
        free(img);
    }

    double startTime = MPI_Wtime();

    // Applying filters
    if (size>1) addGhostRows(inBuf, blockRows, cols, rank, size);
    applyBlur(inBuf, blurBuf, blockRows, cols);

    if (size>1) addGhostRows(blurBuf, blockRows, cols, rank, size);
    float localHighest = 0.0;
    applySobel(blurBuf, outBuf, blockRows, cols, &localHighest, rank, size);

    // Calculating highest value of the gradient using reduce
    float globalHighest;
    MPI_Allreduce(&localHighest, &globalHighest, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    // Normalization
    if (globalHighest>0.0){
        int normStart = 1;
        int normEnd = (size==1) ? blockRows-1 : blockRows-2; 
        // Normalize each pixel (between 0 and 255 inclusive)
        for (int i=normStart; i<=normEnd; ++i){
            for (int j=1; j<cols-1; ++j){
                outBuf[i*cols+j] = (outBuf[i*cols+j]/globalHighest)*255.0;
            }
        }
    }

    // Blacking Borders
    // Top Border
    if (firstRow==0){
        memset(outBuf, 0, cols*sizeof(float));
        if (size>1) memset(outBuf+cols, 0, cols*sizeof(float));
    }

    // Bottom Border
    if (lastRow==rows){
        memset(outBuf+(blockRows-1)*cols, 0, cols*sizeof(float));
    }

    // Side Borders
    for (int i=0; i<blockRows; ++i){
        outBuf[i*cols] = 0.0;
        outBuf[i*cols+(cols-1)] = 0.0;
    }

    float *finalImg = NULL;
    if (rank==0) finalImg = malloc(rows*cols*sizeof(float));

    // Preparing for Gather
    if (rank==0){
        sCountArr = malloc(size*sizeof(int));
        displArr = malloc(size*sizeof(int));
        int displ = 0;
        for (int i=0; i<size; ++i){
            // int newR = blockRowsNum;
            int newR = (i<extraRows) ? blockRowsNum : blockRowsNum-1;
            if (i >= extraRows) newR = blockRowsNum;
            //------------
            sCountArr[i] = newR*cols;
            displArr[i]=displ;
            displ += sCountArr[i];
        }
    }

    // Gathering results from all processes
    if (size>1) MPI_Gatherv(outBuf+cols, rCount, MPI_FLOAT, finalImg, sCountArr, displArr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    else MPI_Gatherv(outBuf, rCount, MPI_FLOAT, finalImg, sCountArr, displArr, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank==0){
        free(sCountArr);
        free(displArr);
    }

    double endTime = MPI_Wtime();

    // Writing output file
    if (rank==0){
        // Output image file name
        char outName[100];
        snprintf(outName, sizeof(outName), "mpi_%dx%d_p%d.pgm", cols, rows, size);

        // Saving output image
        if (pgmwrite(outName, finalImg, rows, cols, 1) != 0) {
            fprintf(stderr, "Failed to write %s\n", outName);
        } else printf("Image Produced: %s\n", outName);
        printf("Execution Time: %.5f s\n", endTime-startTime);
        free(finalImg);
    }

    // Freeing up space
    free(inBuf);
    free(blurBuf);
    free(outBuf);
    MPI_Finalize();
    return 0;
}

// Applies 3X3 blurring filter
void applyBlur(float *inp, float *out, int numRows, int numCols){
    // Track each pixel except the border ones
    for (int i=1; i<numRows-1; ++i){
        for (int j=1; j<numCols-1; ++j){
            float total = 0.0;
            // 3X3 kernel/submatrix
            for (int kr=-1; kr<2; ++kr){
                for (int kc=-1; kc<2; ++kc){
                    total += inp[(i+kr)*numCols+(j+kc)];
                }
            }

            // Updating image pixels
            out[i*numCols+j] = total/9.0;
        }
    }

    // No blurring on border pixels
    for (int i=0; i<numRows; ++i){
        for (int j=0; j<numCols; ++j){
            if (i==0 || i==numRows-1 || j==0 || j==numCols-1){
                out[i*numCols+j] = inp[i*numCols+j];
            }
        }
    }
}

// Applies Sobel filter
void applySobel(float *inp, float *out, int numRows, int numCols, float *localHighest, int rank, int size){
    // Sobel kernels matrices
    int matx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int maty[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    *localHighest = 0.0;

    // Track each pixel except the border ones
    for (int i=1; i<numRows-1; ++i){
        for (int j=1; j<numCols-1; ++j){
            float gx = 0.0;
            float gy = 0.0;
            // 3X3 kernel/submatrix
            for (int kr=-1; kr<2; ++kr){
                for (int kc=-1; kc<2; ++kc){
                    // calculating gx and gy for this pixel

                    gx += inp[(i+kr)*numCols+(j+kc)] * matx[kr+1][kc+1];
                    gy += inp[(i+kr)*numCols+(j+kc)] * maty[kr+1][kc+1];
                }
            }

            float g = sqrt(gx*gx + gy*gy); // Calculating G
            out[i*numCols+j] = g; // Updating image pixels
            // Highest gradient has to only come from non-ghost rows
            if (g > *localHighest && ((size != 1 && ((rank!=(size-1) && i!=1 && i!=numRows-1) || (rank==(size-1) && i!=1))) || size == 1)) *localHighest = g; // Storing the highest value of g up till now
        }
    }
}

// Adds ghost rows to blocks where suitable
void addGhostRows(float *inp, int numRows, int numCols, int rank, int size){
    MPI_Request reqArr[4];
    int ireq = 0;

    // sending top row up and recieving top ghost row
    if (rank>0){
        MPI_Isend(inp+numCols, numCols, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &reqArr[ireq++]);
        MPI_Irecv(inp, numCols, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, &reqArr[ireq++]);
    }

    // sending bottom row down and recieving bottom ghost row
    if (rank<size-1){
        MPI_Isend(inp+(numRows-2)*numCols, numCols, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &reqArr[ireq++]);
        MPI_Irecv(inp+(numRows-1)*numCols, numCols, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &reqArr[ireq++]);
    }
    MPI_Waitall(ireq, reqArr, MPI_STATUS_IGNORE);
}
