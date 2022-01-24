# Programming Assignment I: SIMD Programming
###### tags: `parallel_programming`

## Q1-1
### Record the resulting vector utilization.

![](https://i.imgur.com/ogCbtQ0.png)

![](https://i.imgur.com/nRx7bjx.png)

![](https://i.imgur.com/cYjWyGm.png)

![](https://i.imgur.com/p7VAcgw.png)

### <font color="#1B5875">Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?</font>

As the VECTOR_WIDTH increases, the vector utilization decreases.

#### Reason:
Other vector lanes in the vector will be inactive and waiting for lanes with larger exponent to finish their exponential calculation. The impact will be more significant if the exponents of the vector are in a great disparity. The problem can be solved when the exponents are uniform. The larger the vector width is, the more likely the tragedy is going to happen.

#### Explanation:
exp = 6 5 5 2 1 7 9 6 6 6 8 9 0 3 5 2
<span style="color:blue">***Case 1 (width = 4):***</span>
loop | loop1 | loop2 | loop3 | loop4
------|-------|-------|-------|-------
exp|6 5 5 2|1 7 9 6|6 6 8 9|0 3 5 2
idle count|0+1+1+4|8+2+0+3|3+3+1+0|5+2+0+3

Total = 36 (*2 for vmult and vsub)

With the -l option this is the **Vector Unit Execution log** for loop 1, you can see how the idle counts are calculated.
``` c
     cntbits | ****
       vmult | ****
        vsub | ****
         vgt | ****
     cntbits | ****
       vmult | ***_
        vsub | ***_
         vgt | ****
     cntbits | ****
       vmult | ***_
        vsub | ***_
         vgt | ****
     cntbits | ****
       vmult | ***_
        vsub | ***_
         vgt | ****
     cntbits | ****
       vmult | *___
        vsub | *___
         vgt | ****
```
<span style="color:blue">***Case 2 (width = 8):***</span>
loop | loop1 | loop2
------|-------|-------
exp|6 5 5 2 1 7 9 6|6 6 8 9 0 3 5 2
idle count|3+4+4+7+8+2+0+3|3+3+1+0+9+6+4+7

Total = 64 (*2 for vmult and vsub)

## Q2-1
### <font color="#1B5875">Fix the code to make sure it uses aligned moves for the best performance.</font>
* The original assembly code
![](https://i.imgur.com/RgqCotd.png)

* From the table in [Intel AVX programming reference](https://software.intel.com/content/dam/develop/external/us/en/documents/36945), we can see that vmovaps on ymm register requires 32-byte alignment. 
![](https://i.imgur.com/e8PEu0b.png)

* Modified C code
``` c
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);
```

### Performance improvement
* Intel AVX programming reference
> Software may see performance penalties when unaligned accesses cross cacheline boundaries, so reasonable attempts to align commonly used data sets should continue to be pursued.
* [Agner Fog's "instruction tables"](https://www.agner.org/optimize/instruction_tables.pdf)
Some example comparing latency of vmovaps and vmovups on different architecture.
> Architecture|Instuction|Ops|Latency
> --|--|--|--
> Bulldozer|vmovaps|4|5
> Bulldozer|vmovups|8|6
> Piledriver|vmovaps|4|11
> Piledriver|vmovups|8|15

## Q2-2
### <font color="#1B5875">What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers.
Hint: Aside from speedup and the vectorization report, the most relevant information is that the data type for each array is float.</font>

* Experiment Results

| | Case 1 | Case 2 | Case 3 |
|-|--------|--------|--------|
|1|8.25903sec|2.54864sec|1.40038sec|
|2|8.24707sec|2.61305sec|1.41344sec|
|3|8.24214sec|2.61317sec|1.40543sec|
|medium|**8.24707sec**|**2.61305sec**|**1.40543sec**|

* unvectorized -> vectorized = 3x faster
* vectorized -> AVX2 = 2x faster
* float is 4 bytes, which is equivalent to 32 bits. Case 2 is 3 times faster than Case 1, therefore, we can guess that the bit width of the default vector register is 3 times longer than the length of float. However, most of the length of a register is power of 2, we can assume that the default vector register is 32 x 4 = 128 (bits). As for the case of AVX2 register, the same idea can be applied, the bit width of AVX2 vector register is 128 x 2 = 256 (bits).

||float|default vector register|AVX2 vector register|
|-|----|-----------------------|--------------------|
|bits|32|128                   |256                 |

## Q2-3
### <font color="#1B5875">Provide a theory for why the compiler is generating dramatically different assembly.</font>
* original (non-vectorized)
``` c
c[j] = a[j];
if (b[j] > a[j])
    c[j] = b[j];
```
* modified (vectorized)
``` c
if (b[j] > a[j]) c[j] = b[j];
else c[j] = a[j];
```
* message from compiler of the original c code with no vectorization
```
remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop [-Rpass-analysis=loop-vectorize]
```
* Compiler cannot safely vectorize a loop if there is even a potential dependency.

* [Intel® C++ Compiler Classic Developer Guide and Reference](https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/optimization-and-programming-guide/vectorization/automatic-vectorization/using-automatic-vectorization.html)
> In the example of the first version of test2, the compiler needs to determine whether, for some iteration j, c[j] might refer to the same memory location as a[j] or b[j] for a different iteration. Such memory locations are sometimes said to be aliased. For example, if a[j] pointed to the same memory location as c[j-1], there would be a read-after-write dependency as in the earlier example. If the compiler cannot exclude this possibility, it will not vectorize the loop unless you provide the compiler with hints.

* Therefore, we need to provide more information for compiler to help us.
``` c
#pragma ivdep
```
This pragma can tell compiler to safely ignore potential data dependencies.









