# Programming Assignment V: CUDA
###### tags: `parallel_programming`

### <font color="#1B5875">Q1: What are the pros and cons of the three methods? Give an assumption about their performances.</font>
### 1. malloc v.s. cudaHostAlloc
* malloc is a function from the c standard library, which allocates a block of memory with user defined size in bytes.
* cudaHostAlloc allocates page-locked (pinned) host memory. This specific memory is not allowed to page in and page out, in other words, it does not communicate with hard drive. Thus, the efficiency should be guaranteed.

### 2. cudaMalloc v.s. cudaMallocPitch
* cudaMalloc is typically used for allocating linear device memory.
* cudaMallocPitch is recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements, therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory.

### 3. thread processes a pixel v.s. a group of pixels
In method 3, each thread is responsible to process 4 pixels. Therefore, the utilization rate of the GPU can not be maximized.

In conclusion, I assume that the performance comparison will be Method2 > Method1 > Method3.

:::info
You should point out the advantages and disadvantages of implementing the three kernel methods.
>[name=TA]
:::

---

### <font color="#1B5875">Q2: How are the performances of the three methods? Plot a chart to show the differences among the three methods.</font>
![](https://i.imgur.com/wlKQc3i.png)
![](https://i.imgur.com/5VBtInp.png)

---

### <font color="#1B5875">Q3: Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.</font>
It is no surprise that method 3 has the worst performance because each thread has to process 4 pixels. Nevertheless, method 2 is worse than method 1. Since the two methods both processes a pixel per thread, I suspect memory allocation is the key that cause the time difference.

|		                          |time (ms)|
| ------------------------------- | ------- |
| malloc + cudaMalloc             |  0.144  |
| cudaHostAlloc + cudaMallocPitch |  2.088  |

In order to make memory page-locked, cudaHostAlloc has additional OS operations to pin each page associated to the allocation. Clearly, the cost is more significant than its advantage in this case.

---

### <font color="#1B5875">Q4: Can we do even better? Think a better approach and explain it.</font>
At the current stage, what I do is just deleting the redundant host allocation. Just directly copy the device memory to img.

---

### Reference
[1] [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)