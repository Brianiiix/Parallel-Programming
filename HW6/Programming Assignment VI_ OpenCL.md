# Programming Assignment VI: OpenCL
###### tags: `parallel_programming`

### <font color="#1B5875">Q1: Explain your implementation. How do you optimize the performance of convolution?</font>
* **Steps**
    1. Get information about platform and devices, and create context (initCL)
    2. Create a command queue
    3. Create memory buffers on the device
    4. Copy filter and input image to their respective memory buffers
    5. Create the OpenCL kernel
    6. Set the arguments of the kernel
    7. Set local and global workgroup sizes
        * Global workgroup is the total number of work-items, which is 600 x 400
        * Each local workgroup is set to include 25 x 25 work-items
        ```c
        size_t localws[2] = {25, 25};
        size_t globalws[2] = {imageWidth, imageHeight};
        ```
    9. Execute OpenCL kernel
    10.  Read the memory buffer on the device to the local variable
    11.  Clean up

![](https://i.imgur.com/C70r4bO.png)
There are total 240000 pixels to calculate. I divided them into 24 x 16 local work-groups with each group contains 25 x 25 work-items. The calculation of the work-items can be paralleled. 

### <font color="#1B5875">Q2: Rewrite the program using CUDA. (1) Explain your CUDA implementation, (2) plot a chart to show the performance difference between using OpenCL and CUDA, and (3) explain the result.</font>

* CUDA implementation
In order to have fair comparison, the partition strategy of the image in CUDA is same as OpenCL. Other than that, the difference is just syntaxs between the models.
```c
dim3 threadsPerBlock(25, 25);
dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
```
* OpenCL v.s. CUDA
![](https://i.imgur.com/gaOJH1f.png)

* Result Explanation
It is clear that the performance of CUDA is better than OpenCL. OpenCL is a framework that can be implemented in different hardware and software specification. CUDA is developed by Nvidia which designs the hardware for the framework. Therefore, we can conclude that CUDA has a higher integration and performs better.
