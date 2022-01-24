# Programming Assignment IV: MPI Programming
###### tags: `parallel_programming`

## <font color="#1B5875">Q1: hello</font>
### <font color="#1B5875"> 1. How do you control the number of MPI processes on each node​?</font>
```
We can specify the number of slots in the host file. Slots can be interpreted as number
of available processors on the host. If the slots are not specified, the number of slots
defaults to one.
```

### <font color="#1B5875"> 2. Which functions do you use for retrieving the rank of an MPI process and the total number of processes?</font>
```
Rank of an process: MPI_Comm_rank(MPI_COMM_WORLD, &rank);
Total number of prosesses: MPI_Comm_size(MPI_COMM_WORLD, &size);
```

## <font color="#1B5875">Q2: pi_block_linear</font>
### <font color="#1B5875"> 1. Why MPI_Send and MPI_Recv are called “blocking” communication?</font>
```
The MPI standard requires that MPI_Send send call blocks until the buffer is safe to 
reuse. Similarly, the Standard requires that MPI_Recv call blocks until the receive
buffer actually contains the intended message. To sum up, we need wait the function
to return until the communication is finished.
```

### <font color="#1B5875"> 2. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.</font>
![](https://i.imgur.com/KsGkcEx.png)


## <font color="#1B5875">Q3: pi_block_tree</font>
### <font color="#1B5875"> 1. Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.</font>
![](https://i.imgur.com/CCf7vZ0.png)



### <font color="#1B5875"> 2. How does the performance of binary tree reduction compare to the performance of linear reduction? </font>
The performances are nearly identical, only a slight difference when processes = 8.
![](https://i.imgur.com/L9hQCU0.png)


### <font color="#1B5875"> 3. Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.</font>
```
I think tree will perform better, because linear approach requires master process to 
recieve message from all other slave processes. However, in tree approach, the overhead 
of communication will reduce from O(n) to O(lg(n)).
```

## <font color="#1B5875">Q4: pi_nonblock_linear</font>
### <font color="#1B5875"> 1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.</font>
![](https://i.imgur.com/W9X0DcP.png)


### <font color="#1B5875"> 2. What are the MPI functions for non-blocking communication?</font>
```
MPI non-blocking functions usually start with the letter 'I', here are some examples.
MPI_Iallgather
MPI_Iallgatherv
MPI_Iallreduce
MPI_Ialltoall
MPI_Ialltoallv
MPI_Ibarrier
MPI_Ibsend
MPI_Igather
MPI_Igatherv
MPI_Iprobe
MPI_Irecv
MPI_Ireduce
MPI_Ireduce_scatter
MPI_Ireduce_scatter_block
MPI_Irsend
MPI_Iscatter
MPI_Iscatterv
MPI_Isend
MPI_Issend
```

### <font color="#1B5875"> 3. How the performance of non-blocking communication compares to the performance of blocking communication?</font>
The performances are nearly identical, only a slight difference when processes = 12. Theoretically, non-blocking communication has the advantage of doing other things while the non-blocking function completes.
![](https://i.imgur.com/bCnZXay.png)


## <font color="#1B5875">Q5: pi_gather</font>
### <font color="#1B5875"> 1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.</font>
![](https://i.imgur.com/XYYGK8E.png)


## <font color="#1B5875">Q6: pi_reduce</font>
### <font color="#1B5875"> 1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.</font>
![](https://i.imgur.com/OdXB7AL.png)



## <font color="#1B5875">Q7: pi_one_side</font>
### <font color="#1B5875"> 1. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.</font>
![](https://i.imgur.com/Kogk2Jn.png)



### <font color="#1B5875"> 2. Which approach gives the best performance among the 1.2.1-1.2.6 cases? What is the reason for that?</font>
Other than the one-sided communication, the implementations have nearly no difference in terms of performance. I think it is because the program doesn't heavily rely on communication between process but calculating Monte-Carlo.
![](https://i.imgur.com/TJs9M6e.png)


## <font color="#1B5875">Q8: ping-pong</font>
### <font color="#1B5875"> 1. Plot ping-pong time in function of the message size for cases 1 and 2, respectively.</font>
* **Case 1**
![](https://i.imgur.com/Wr9w4KQ.png)
* **Case 2**
![](https://i.imgur.com/mbImSap.png)

### <font color="#1B5875"> 2. Calculate the bandwidth and latency for cases 1 and 2, respectively.</font>
* **Case 1**
bandwidth = (1 / 1.5594563042333037e-10) / 1E-9 = 6.4 GB/s
latency = 0.0003616054620147061 / 1E-6 = 0.36 ms
* **Case 2**
bandwidth = (1 / 8.641607908748788e-09) / 1E-9 = 864 MB/s
latency = 0.0013578094881199364 / 1E-6 = 1.35 ms

## <font color="#1B5875">Q9: matmul</font>
### <font color="#1B5875"> 1. Describe what approach(es) were used in your MPI matrix multiplication for each data set.</font>
* Partition a_mat evenly in blocks of rows to each process. The remaining rows will be assigned to master process.
* Transpose b_mat to avoid cache miss, since accessing the matrix in row fashion increase the performance with the help of spatial locality.