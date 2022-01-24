# Programming Assignment II: Multi-thread
###### tags: `parallel_programming`

## <font color="#1B5875">Q1</font>
### <font color="#1B5875"> Produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used FOR VIEW 1. Is speedup linear in the number of threads used? Hypothesize why this is (or is not) the case? (You may also wish to produce a graph for VIEW 2 to help you come up with a good answer. Hint: take a careful look at the three-thread data-point.)</font>
    
* Code Segment
``` c
void workerThreadStart(WorkerArgs *const args)
{
    int startRow = args->height / args->numThreads * args->threadId;
    int totalRows = args->height / args->numThreads;
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                     args->width, args->height,
                     startRow, totalRows, args->maxIterations, args->output);
}
```

* Speedup comparison Table

|          |Serial(ms)|Thread(ms)|Speedup|
| -------- | ------- | ------- | ------- |
| 2 thread | 464.607 | 238.979 | 1.94x   |
| 3 thread | 463.587 | 284.783 | 1.63x   |
| 4 thread | 463.041 | 194.408 | 2.38x   |

* Speedup comparison line chart
![](https://i.imgur.com/cDAzdim.png)

It is obvious to see that the speedup is not linear. Particularly, when using 3 threads, the performance is worse than using 2 threads.

First, we look at the code segment for calculating the grayscale of the image.

``` c
static inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}
```

The grayscale of the image in our example is from 0 to 255, and this value of each pixel is calculated by the function ```mandel```. The pixels with white color will have to run through the loop with 256 iterations, on the other hand, pixels that are black can terminate the loop with only one iteration. Therefore, we can conclude that the white part in the MandelBrot Set has a longer run time. 

Then, let's look at how we partition the MandelBrot set to each thread.

* 2 Threads
![](https://i.imgur.com/dReFl2w.png =500x) 

* 3 Threads
![](https://i.imgur.com/ib0zyxo.png =500x)

The example with 2 threads will have equal workload on both threads. However, the second thread in the example with 3 threads will be the bottleneck in the program. It is obvious that the middle part has the most white area, and the area is even larger than the upper or lower part from the example with 2 threads. Thus, 3 threads is slower than 2 threads.


## <font color="#1B5875">Q2</font>
### <font color="#1B5875"> Measure the amount of time each thread requires to complete its work by inserting timing code at the beginning and end of workerThreadStart(). How do your measurements explain the speedup graph you previously created?</font>
**VIEW 1**
* 2 Threads
``` c
[Thread 0]: [240.014] ms
[Thread 1]: [241.662] ms
------------------------
[Thread 0]: [237.727] ms
[Thread 1]: [238.707] ms
------------------------
[Thread 0]: [244.568] ms
[Thread 1]: [249.111] ms
------------------------
[Thread 0]: [238.118] ms
[Thread 1]: [239.722] ms
------------------------
[Thread 0]: [239.161] ms
[Thread 1]: [240.605] ms
------------------------
[mandelbrot thread]:		[238.908] ms
```
Each thread has nearly equal run time, which shows that they have equal workload.
* 3 Threads
``` c
[Thread 0]: [115.959] ms
[Thread 2]: [109.720] ms
[Thread 1]: [319.421] ms
------------------------
[Thread 0]: [94.369] ms
[Thread 2]: [94.981] ms
[Thread 1]: [284.661] ms
------------------------
[Thread 0]: [94.009] ms
[Thread 2]: [94.613] ms
[Thread 1]: [286.027] ms
------------------------
[Thread 0]: [94.184] ms
[Thread 2]: [94.981] ms
[Thread 1]: [287.277] ms
------------------------
[Thread 0]: [95.086] ms
[Thread 2]: [95.039] ms
[Thread 1]: [285.702] ms
------------------------
[mandelbrot thread]:		[284.849] ms
```
The result proves that the assumption previously made was correct. Thread 1 has the most run time, which is the area with the most white color.
* 4 Threads
``` c
[Thread 0]: [50.297] ms
[Thread 3]: [51.250] ms
[Thread 1]: [198.158] ms
[Thread 2]: [199.094] ms
------------------------
[Thread 0]: [46.114] ms
[Thread 3]: [46.143] ms
[Thread 1]: [194.008] ms
[Thread 2]: [194.652] ms
------------------------
[Thread 0]: [45.879] ms
[Thread 3]: [46.603] ms
[Thread 1]: [193.504] ms
[Thread 2]: [194.130] ms
------------------------
[Thread 3]: [46.241] ms
[Thread 0]: [46.434] ms
[Thread 2]: [196.324] ms
[Thread 1]: [197.012] ms
------------------------
[Thread 0]: [64.004] ms
[Thread 3]: [58.394] ms
[Thread 2]: [195.489] ms
[Thread 1]: [204.817] ms
------------------------
[mandelbrot thread]:		[194.337] ms
```
We can see that the bottlenecks are Thread 1 and Thread 2, which are the calculations in the middle area of the Mandelbrot Set.

**VIEW 2**
The same idea can also be applied to view 2.
* 2 Threads
``` c
[Thread 1]: [127.016] ms
[Thread 0]: [177.363] ms
------------------------
[Thread 1]: [122.766] ms
[Thread 0]: [172.810] ms
------------------------
[Thread 1]: [123.041] ms
[Thread 0]: [173.080] ms
------------------------
[Thread 1]: [122.881] ms
[Thread 0]: [173.702] ms
------------------------
[Thread 1]: [122.803] ms
[Thread 0]: [173.020] ms
------------------------
[mandelbrot thread]:		[172.878] ms
```
* 3 Threads
``` c
[Thread 2]: [80.230] ms
[Thread 1]: [87.211] ms
[Thread 0]: [132.211] ms
------------------------
[Thread 2]: [80.150] ms
[Thread 1]: [87.405] ms
[Thread 0]: [132.978] ms
------------------------
[Thread 2]: [79.613] ms
[Thread 1]: [86.825] ms
[Thread 0]: [131.887] ms
------------------------
[Thread 2]: [79.619] ms
[Thread 1]: [86.733] ms
[Thread 0]: [131.481] ms
------------------------
[Thread 2]: [80.054] ms
[Thread 1]: [87.118] ms
[Thread 0]: [132.090] ms
------------------------
[mandelbrot thread]:		[131.577] ms
```
* 4 Threads
``` c
[Thread 3]: [63.287] ms
[Thread 2]: [66.524] ms
[Thread 1]: [67.761] ms
[Thread 0]: [112.599] ms
------------------------
[Thread 3]: [60.705] ms
[Thread 2]: [64.783] ms
[Thread 1]: [65.420] ms
[Thread 0]: [110.404] ms
------------------------
[Thread 3]: [60.664] ms
[Thread 2]: [64.797] ms
[Thread 1]: [65.569] ms
[Thread 0]: [110.224] ms
------------------------
[Thread 3]: [60.660] ms
[Thread 2]: [64.857] ms
[Thread 1]: [65.494] ms
[Thread 0]: [110.390] ms
------------------------
[Thread 3]: [62.028] ms
[Thread 1]: [66.175] ms
[Thread 2]: [69.855] ms
[Thread 0]: [110.611] ms
------------------------
[mandelbrot thread]:		[110.329] ms
```

## <font color="#1B5875">Q3</font>
### <font color="#1B5875"> Describe your approach to parallelization and report the final 4-thread speedup obtained.</font>

* Approach
In order to make the workload even, we divided the MandelBrot Set to the number of rows which equals its height. Then, we assign each row to the specific thread whose remainder of row number divided by the number of threads is the same as the threadID. (Value of ThreadID and row number both start at 0)

* Example
```
Thread number = 3, Height = 10

Thread 0: 0, 3, 6, 9
Thread 1: 1, 4, 7
Thread 2: 2, 5, 8
```

* Code Segment
``` c
for(int i = args->threadId; i < args->height; i += args->numThreads){
	mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
			  args->width, args->height,
			  i, 1, args->maxIterations, args->output);
}
```

* Speedup comparison Table

|          |Serial(ms)|Thread(ms)|Speedup|
| -------- | ------- | ------- | ------- |
| 2 thread | 462.313 | 236.742 | 1.95x   |
| 3 thread | 461.966 | 159.397 | 2.90x   |
| 4 thread | 462.346 | 122.104 | 3.79x   |

* Speedup comparison line chart
![](https://i.imgur.com/uvEZ3kj.png)
As we can see, the experimental result is almost the same as the theoretical speedup value. Which shows that the improved implementation is quite decent.

* VIEW 1 with 3 Threads
``` c
[Thread 1]: [158.964] ms
[Thread 0]: [159.237] ms
[Thread 2]: [159.285] ms
------------------------
[Thread 0]: [159.128] ms
[Thread 2]: [159.265] ms
[Thread 1]: [159.313] ms
------------------------
[Thread 2]: [158.911] ms
[Thread 1]: [158.917] ms
[Thread 0]: [159.332] ms
------------------------
[Thread 0]: [160.356] ms
[Thread 1]: [160.651] ms
[Thread 2]: [161.108] ms
------------------------
[Thread 1]: [158.748] ms
[Thread 0]: [159.187] ms
[Thread 2]: [159.942] ms
------------------------
[mandelbrot thread]:		[159.397] ms
```
The improved code has assigned each thread with equal workload, as the run time of each thread has almost no difference.

## <font color="#1B5875">Q4</font>
### <font color="#1B5875"> Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not? (Notice that the workstation server provides 4 cores 4 threads.)</font>

* Speedup comparison Table

|          |Serial(ms)|Thread(ms)|Speedup|
| -------- | ------- | ------- | ------- |
| 4 thread | 461.881 | 121.943 | 3.79x   |
| 8 thread | 462.727 | 126.354 | 3.66x   |

* Theory
The cpu in the workstation does not equip with intel hyperthreading technology, which means each core can only execute a thread at a time and has no ability to switch to another thread when the current thread is doing memory access or disk access. Therefore, the machine can only execute a maximum of 4 threads in a parallel fashion.
The reason why the performance of 8 threads is not equal to but worse than 4 threads is because the overhead when creating threads is expensive.