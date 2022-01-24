#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int ImgSize = imageHeight * imageWidth;

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);

    // Create memory buffers on the device
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, &status);
    cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY, ImgSize * sizeof(float), NULL, &status);
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, ImgSize * sizeof(float), NULL, &status);

    // Copy filter and image to their respective memory buffers
    clEnqueueWriteBuffer(command_queue, d_filter, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, d_inputImage, CL_TRUE, 0, ImgSize * sizeof(float), inputImage, 0, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_filter);
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_inputImage);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_outputImage);

    // Set local and global workgroup sizes
    size_t localws[2] = {25, 25};
    size_t globalws[2] = {imageHeight, imageWidth};
    // size_t globalItemSize = ImgSize;
    // size_t localItemSize = 8;

    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);
    // clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

    // Read the memory buffer on the device to the local variable
    clEnqueueReadBuffer(command_queue, d_outputImage, CL_TRUE, 0, ImgSize * sizeof(float), (void *)outputImage, NULL, NULL, NULL);

    // Clean up
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_outputImage);
    clReleaseKernel(kernel);

}
