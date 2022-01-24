#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "helper.h"
extern "C"{
#include "hostFE.h"
}

__global__ void convKernel(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage){
    int halffilterSize = filterWidth / 2;
    float sum = 0;
    int k, l;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    for (k = -halffilterSize; k <= halffilterSize; k++){
	for (l = -halffilterSize; l <= halffilterSize; l++){
	    if (thisY + k >= 0 && thisY + k < imageHeight &&
		thisX + l >= 0 && thisX + l < imageWidth){
                sum += inputImage[(thisY + k) * imageWidth + thisX + l] *
                       filter[(k + halffilterSize) * filterWidth +
                              l + halffilterSize];
	    }
	}
    }

    int idx = thisX + thisY * imageWidth;
    outputImage[idx] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int ImgSize = imageHeight * imageWidth * sizeof(float);

    float *d_filter, *d_inputImage, *d_outputImage;
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_inputImage, ImgSize);
    cudaMalloc(&d_outputImage, ImgSize);

    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputImage, inputImage, ImgSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
    convKernel<<<numBlocks, threadsPerBlock>>>(filterWidth, d_filter, imageHeight, imageWidth, d_inputImage, d_outputImage);

    cudaMemcpy(outputImage, d_outputImage, ImgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_filter);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
