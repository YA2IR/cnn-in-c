#ifndef HYPERPARAMS_H
#define HYPERPARAMS_H

#define NUM_PIX_PER_DIM 28
#define NUM_PIX_TOTAL (NUM_PIX_PER_DIM * NUM_PIX_PER_DIM)

#define NUM_EXAMPLES 60000
#define NUM_VALIDATE 3000 // this is too small, but the higher NUM_TRAIN the better
#define NUM_TRAIN (NUM_EXAMPLES - NUM_VALIDATE)

#define NUM_TEST 10000

#define BATCH_SIZE 32
#define NUM_EPOCHS 5

#define ALPHA 0.006
/*#define DECAY 1.1*/
#define MOMENTUM 0.9
#define LEAKY_ALPHA 0.01
#define HIDDEN_NEURONS 256 // or 128
#define OUT_NEURONS 10

#define GLOBAL_KERNEL_SIZE 3 // this is very useful for unrolling

#ifdef _OPENMP
  #include <omp.h>
  #define OMP_PARALLEL_FOR           _Pragma("omp parallel for")
  #define OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
  #define OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for collapse(4)")
#else
  #define OMP_PARALLEL_FOR           
  #define OMP_PARALLEL_FOR_COLLAPSE2 
  #define OMP_PARALLEL_FOR_COLLAPSE4 
#endif

#ifdef __clang__
  #define CLANG_LOOP_UNROLL_ENABLE _Pragma("clang loop unroll(enable)")
#else
  #define CLANG_LOOP_UNROLL_ENABLE
#endif


#endif 
