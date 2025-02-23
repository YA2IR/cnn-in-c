#ifndef MATRIX_H
#define MATRIX_H

#include "hyperparams.h"

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

typedef struct matrix {
    int n_rows;
    int n_cols;
    float **data;
} matrix;

void init_matrix(matrix* m);
void free_matrix(matrix* m);
void init_param(matrix* p); // same as init_matrix but with proper weight initialization


float mat_sum(matrix* A);
void mat_mul(matrix* A, matrix* B, matrix* C, int batch_start, int batch_end);
void mat_mul_transpose_b(matrix* A, matrix* B, matrix* C, int batch_start, int batch_end);
void mat_mul_transpose_a(matrix* A, matrix* B, matrix* C, int batch_start, int batch_end);
void mat_mul_elemwise(matrix* A, matrix* B, matrix* C);
void mat_add(matrix* A, matrix* B, matrix* C, bool broadcast);
void zero_matrix(matrix* m);
void scale_matrix(matrix* m, float factor);

void leaky_relu(matrix* Z, matrix* C);
void leaky_relu_deriv(matrix* Z, matrix* C);
void softmax(matrix *Z_2, matrix* A_2);

#endif

