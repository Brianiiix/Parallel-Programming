#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_int y, count;
  __pp_vec_float result, temp;
  __pp_vec_float clamped = _pp_vset_float(9.999999f);
  __pp_vec_int oneInt = _pp_vset_int(1);
  __pp_vec_int zeroInt = _pp_vset_int(0);
  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskIsNotZero_and_CntGtZero, maskIsNotZero_and_rsltGt9;
  int cnt;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones(N-i);
    maskIsZero = _pp_init_ones(0);
    // float x = values[i];
    _pp_vload_float(x, values + i, maskAll);
    // int y = exponents[i];
    _pp_vload_int(y, exponents + i, maskAll);
    // if (y == 0) {
    _pp_veq_int(maskIsZero, y, zeroInt, maskAll);
    // output[i] = 1.f; }
    _pp_vset_float(result, 1.f, maskIsZero);
    _pp_vstore_float(output + i, result, maskIsZero);
    // else {
    maskIsNotZero = _pp_mask_not(maskIsZero);
    maskIsNotZero = _pp_mask_and(maskIsNotZero, maskAll);
    // float result = x;
    _pp_vload_float(temp, values + i, maskIsNotZero);
    // int count = y - 1;
    _pp_vsub_int(count, y, oneInt, maskIsNotZero);
    // while (count > 0) {
    _pp_vgt_int(maskIsNotZero_and_CntGtZero, count, zeroInt, maskIsNotZero);
    cnt = _pp_cntbits(maskIsNotZero_and_CntGtZero);
    while(cnt > 0) {
      // result *= x;
      _pp_vmult_float(temp, temp, x, maskIsNotZero_and_CntGtZero);
      // count--; }
      _pp_vsub_int(count, count, oneInt, maskIsNotZero_and_CntGtZero);

      _pp_vgt_int(maskIsNotZero_and_CntGtZero, count, zeroInt, maskIsNotZero);
      cnt = _pp_cntbits(maskIsNotZero_and_CntGtZero);
    }
    // if (result > 9.999999f) {
    _pp_vgt_float(maskIsNotZero_and_rsltGt9, temp, clamped, maskIsNotZero);
    // result = 9.999999f; }
    _pp_vset_float(temp, 9.999999f, maskIsNotZero_and_rsltGt9);
    // output[i] = result;
    _pp_vstore_float(output + i, temp, maskIsNotZero);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float local_sum = _pp_vset_float(0.f);
  __pp_vec_float x;
  __pp_mask maskAll;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(x, values + i, maskAll);
    _pp_vadd_float(local_sum, local_sum, x, maskAll);
  }

  for (int i = 0; i < log2(VECTOR_WIDTH); i ++)
  {
    _pp_hadd_float(local_sum, local_sum);
    _pp_interleave_float(local_sum, local_sum);
  }

  return local_sum.value[0];
}
