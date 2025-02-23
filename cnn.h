#ifndef CNN_H
#define CNN_H

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "matrix.h"
#include "hyperparams.h"

#ifndef MNIST_DATA_TYPE 
#define MNIST_DATA_TYPE double
#endif

#ifndef MNIST_DATA_DEFINED
#define MNIST_DATA_DEFINED 
typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28]; 
	unsigned int label; 
} mnist_data;
#endif

enum layer_type {
    INPUT_LAYER,
    CONV_LAYER,
    MAXPOOL_LAYER,
    FLATTENED_LAYER,
    FC_LAYER
};

// the core of every layer, e.g. conv->core.in_size
struct layer { 
    enum layer_type type;
    int in_chans;
    int out_chans;
    int in_size;
    int out_size;
} typedef layer;

// for convenience in forward() and backward(), see main
struct generic_layer { 
    void* layer;
    enum layer_type type;
} typedef generic_layer;

struct conv_layer {
    layer* prev;
    layer core; // ins/outs dims & sizes

    int kernel_size;
    int stride; // use 2 only, and maxpool also by 2. I haven't confirmed any other value yet
                // and also: no padding yet. 
    float**** kernels; // basically [out_chans][in_chans or depth][kernel_size][kernel_size];
    float**** v_kernels;
    float**** d_kernel;

    float* biases;
    float* v_biases;
    float* d_biases;

    float**** feature_maps;
    float**** d_feature_maps;

    float**** in;
    float**** d_in;
} typedef conv_layer;

struct {
    float average_w;
    float abs_sum_w;
} typedef kernel_stats; // for debugging purposes

struct maxpool_layer {
    conv_layer* prev;
    layer core;

    int pooling_size;
    int downsampled_size;

    float**** downsampled;
    float**** d_downsampled;

    int***** max_positions; 
} typedef maxpool_layer;

struct fully_connected {
    void* prev; 
    bool prev_is_flat; // otherwise prev is another fully connected layer
    layer core;

    matrix* w;  // weights + velocities + gradients
    matrix* v_w;
    matrix* dw;

    matrix* b;
    matrix* v_b;
    matrix* db;

    matrix* Z; // pre activation
    matrix* A;

    matrix* dZ;
    matrix* A_prev; 
} typedef fully_connected;

struct flattened_layer { // for convenience
    maxpool_layer* prev;
    layer core;
    matrix* flattened;  
    matrix* d_flattened;
} typedef flattened_layer;


fully_connected* init_fc_layer(int in_dim, int out_dim, void* prev, bool prev_is_flat);
conv_layer* init_conv_layer(layer* prev, int num_kernels, int kernel_size, int stride, float**** in); 
maxpool_layer* init_maxpool_layer(conv_layer* prev, int pooling_size); 
flattened_layer* init_flat_layer(maxpool_layer* prev);

/*
 *  the two main functions, for forward and backward pass
 *  starting & ending index of each batch is explicitly passed
 *  
 * */
void forward(generic_layer l, int start_idx, int end_idx); 
void backward(generic_layer curr, generic_layer next, int start_idx, int end_idx); 

// this is how the forward pass is done internally:
void convolute(conv_layer* conv, int start_idx, int end_idx); 
void activate_conv(conv_layer* conv, int local_batch_size); 
void maxpool(maxpool_layer* layer, int start_idx, int end_idx); 
void flatten(flattened_layer* fl, int start_idx, int end_idx); 
void forward_fc(fully_connected* fc, matrix* input, int start_idx, int end_idx); 

void conv_backward(conv_layer* conv, float**** input, int start_idx, int end_idx); 
void maxpool_backward(maxpool_layer* layer, void* next_layer, bool next_is_conv,
                        int start_idx, int end_idx);

void update_bias_fc(fully_connected *fc); 
void update_weights_fc(fully_connected *fc); 
void calc_predictions(matrix* A, unsigned int* predictions, int local_batch_size); 

unsigned int *one_hot_to_labels(matrix *Y); 
void clip_matrix(matrix* m, float threshold, int start_col, int end_col); 

void transform_to_matrix(mnist_data* mnist_array, int m, MNIST_DATA_TYPE*** X); 
bool float_eq(float a, float b); 


float**** calloc_4d(int dim1, int dim2, int dim3, int dim4); 
int***** calloc_5d_as_int(int dim1, int dim2, int dim3, int dim4, int dim5); 
void free_4d(float ****array, int dim1, int dim2, int dim3); 
void free_5d_int(int *****array, int dim1, int dim2, int dim3, int dim4); 
void free_network(generic_layer *network); 

#endif
