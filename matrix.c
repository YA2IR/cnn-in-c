#include "matrix.h"

void init_matrix(matrix* m) {
    float** mat = (float**)calloc(m->n_rows, sizeof(float*));
    assert(mat);
    for (int i = 0; i < m->n_rows; i++) {
        mat[i] = (float*)calloc(m->n_cols, sizeof(float));
        assert(mat[i]);
    }
    m->data = mat;
}

void free_matrix(matrix* m) {
    assert(m->data != NULL);
    for (int i = 0; i < m->n_rows; i++)
            free(m->data[i]);
    free(m->data);
    m->data = NULL;
}

void zero_matrix(matrix* m) {
    for (int i = 0; i < m->n_rows; i++)
        for (int j = 0; j < m->n_cols; j++)
            m->data[i][j] = 0.0f;
}

float mat_sum(matrix* A) {
    float res = 0.0;
    for (int r = 0; r < A->n_rows; r++)
        for (int c = 0; c < A->n_cols; c++)
            res += A->data[r][c];
    return res;
}

void mat_mul(matrix* A, matrix* B, matrix* C, int batch_start, int batch_end) {
    if (A->n_cols != B->n_rows) {
        printf("Incompatible matrices! Can't multiply %dx%d with %dx%d!\n",
               A->n_rows, A->n_cols, B->n_rows, B->n_cols);
        exit(EXIT_FAILURE);
    }

    if (batch_end > B->n_cols)
        batch_end = B->n_cols;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int i = 0; i < A->n_rows; i++) {
        for (int j = batch_start; j < batch_end; j++) { 
            float sum = 0.0f;
            for (int k = 0; k < A->n_cols; k++) {
                sum += A->data[i][k] * B->data[k][j]; 
            }
            C->data[i][j] = sum;
        }
    }
}


void mat_mul_transpose_b(matrix* A, matrix* B, matrix* C, int batch_start, int batch_end) {
    if (A->n_cols != B->n_cols) {
        printf("Incompatible matrices! Can't multiply %dx%d with %dx%d!!\n",
               A->n_rows, A->n_cols, B->n_rows, B->n_cols);
        exit(EXIT_FAILURE);
    }

    if (batch_end > A->n_cols)
        batch_end = A->n_cols;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int i = 0; i < A->n_rows; i++) {
        for (int j = 0; j < B->n_rows; j++) {
            float sum = 0.0f;
            for (int k = batch_start; k < batch_end; k++) {
                sum += A->data[i][k] * B->data[j][k];
            }
            C->data[i][j] = sum;
        }
    }
}

void mat_mul_transpose_a(matrix* A, matrix* B, matrix* C, int batch_start, int batch_end) {
    if (A->n_rows != B->n_rows) {
        printf("Incompatible matrices! Can't multiply %dx%d with %dx%d using A^T\n",
               A->n_rows, A->n_cols, B->n_rows, B->n_cols);
        exit(EXIT_FAILURE);
    }

    if (batch_end > B->n_cols)
        batch_end = B->n_cols;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int i = 0; i < A->n_cols; i++) {
        for (int j = batch_start; j < batch_end; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->n_rows; k++) {
                sum += A->data[k][i] * B->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }
}

void mat_mul_elemwise(matrix* A, matrix* B, matrix* C) {
    OMP_PARALLEL_FOR_COLLAPSE2
    for (int i = 0; i < A->n_rows; i++) {
        for (int j = 0; j < A->n_cols; j++) {
            C->data[i][j] = A->data[i][j] * B->data[i][j];
        }
    }
}

void mat_add(matrix* A, matrix* B, matrix* C, bool broadcast) {
    if ((A->n_rows != B->n_rows) ||
        ((!broadcast) && 
            (A->n_cols != B->n_cols)) ||
        (broadcast && B->n_cols != 1)) {
        printf("Incompatible matrices! Can't add %dx%d with %dx%d \n",
               A->n_rows, A->n_cols, B->n_rows, B->n_cols);
        exit(EXIT_FAILURE);
    }
        /*printf("adding %dx%d with %dx%d\n",*/
        /*   A->n_rows, A->n_cols, B->n_rows, B->n_cols);*/
    OMP_PARALLEL_FOR_COLLAPSE2
    for (int r = 0; r < A->n_rows; r++)
        for (int c = 0; c < A->n_cols; c++)
            if (B->n_cols == 1)
                C->data[r][c] = A->data[r][c] + B->data[r][0];
            else
                C->data[r][c] = A->data[r][c] + B->data[r][c];
}

void scale_matrix(matrix *m, float factor) {
    for (int r = 0; r < m->n_rows; r++)
        for (int c = 0; c < m->n_cols; c++)
            m->data[r][c] *= factor;
}


/**/
/*void relu(matrix* Z, matrix* C) {*/
/*    for (int r = 0; r < Z->n_rows; r++)*/
/*        for (int c = 0; c < Z->n_cols; c++)*/
/*            if (Z->data[r][c] < 0) */
/*                C->data[r][c] = 0; */
/*            else */
/*                C->data[r][c] = Z->data[r][c];*/
/*    printf("relu sample: 1(%f,%f), 2(%f,%f), 3(%f,%f)\n", Z->data[4][4], C->data[4][4],*/
/*           Z->data[3][3], C->data[3][3], Z->data[2][2], C->data[2][2]);*/
/*}*/

/*void relu_deriv(matrix* Z, matrix* C) {*/
/*    OMP_PARALLEL_FOR_COLLAPSE2*/
/*    for (int i = 0; i < Z->n_rows; i++) {*/
/*        for (int j = 0; j < Z->n_cols; j++) {*/
/*            C->data[i][j] = (Z->data[i][j] > 0) ? 1.0 : 0.0;*/
/*        }*/
/*    }*/
/*}*/
/**/

void leaky_relu(matrix* Z, matrix* C) {
    for (int r = 0; r < Z->n_rows; r++) {
        for (int c = 0; c < Z->n_cols; c++) {
            if (Z->data[r][c] < 0)
                C->data[r][c] = LEAKY_ALPHA * Z->data[r][c];
            else
                C->data[r][c] = Z->data[r][c];
        }
    }
}

void leaky_relu_deriv(matrix* Z, matrix* C) {
    for (int i = 0; i < Z->n_rows; i++) {
        for (int j = 0; j < Z->n_cols; j++) {
            C->data[i][j] = (Z->data[i][j] > 0) ? 1.0f : LEAKY_ALPHA;
        }
    }
}


void softmax(matrix *Z_2, matrix* A_2) {
    for (int i = 0; i < A_2->n_cols; i++) {
        float max_val = Z_2->data[0][i];
        for (int n = 1; n < OUT_NEURONS; n++) {
            max_val = fmax(max_val, Z_2->data[n][i]);
        }
        float sum = 0.0f;
        for (int n = 0; n < OUT_NEURONS; n++) {
            A_2->data[n][i] = expf(Z_2->data[n][i] - max_val);
            sum += A_2->data[n][i];
        }
        for (int n = 0; n < OUT_NEURONS; n++)
            A_2->data[n][i] /= sum;
    }
}

void init_param(matrix* p) {
    p->data = malloc(sizeof(float*)*p->n_rows);
    assert(p->data);
    
    for (int r = 0; r < p->n_rows; r++) {
        p->data[r] = malloc(sizeof(float)*p->n_cols);
        assert(p->data[r]); 
    }

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int r = 0; r < p->n_rows; r++) {
        for (int c = 0; c < p->n_cols; c++) {
            float fan_in = (float)p->n_cols;
            float std = sqrtf(2.0f / fan_in);

            float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
            float u2 = rand() / (float)RAND_MAX;
            float radius = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * M_PI * u2;

            p->data[r][c] = radius * cos(theta) * std;
        }
    }
}
