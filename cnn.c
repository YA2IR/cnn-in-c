#include "cnn.h"
#include "hyperparams.h"

bool float_eq(float a, float b) { 
    return fabs(a-b) < 1e-5;
}

void forward_fc(fully_connected* fc, matrix* input, int start_idx, int end_idx) {
    int local_batch_size = end_idx - start_idx; 
    fc->A_prev = input;  
    matrix Z_tmp = {fc->w->n_rows, local_batch_size, NULL};
    init_matrix(&Z_tmp);

    mat_mul(fc->w, input, &Z_tmp, 0, local_batch_size);
    const bool broadcast = true;
    mat_add(&Z_tmp, fc->b, fc->Z, broadcast);

    if (fc->core.out_size == OUT_NEURONS)
        softmax(fc->Z, fc->A);
    else
        leaky_relu(fc->Z, fc->A);
    free_matrix(&Z_tmp);

}

void update_bias_fc(fully_connected *fc) {
    OMP_PARALLEL_FOR
    for (int i = 0; i < fc->b->n_rows; i++) {
        fc->v_b->data[i][0] = MOMENTUM * fc->v_b->data[i][0] - ALPHA * fc->db->data[i][0];
        fc->b->data[i][0] += fc->v_b->data[i][0];
    }
}

void update_weights_fc(fully_connected *fc) {
    for (int r = 0; r < fc->w->n_rows; r++) {
        for (int c = 0; c < fc->w->n_cols; c++) {
            fc->v_w->data[r][c] = MOMENTUM * fc->v_w->data[r][c] - ALPHA * fc->dw->data[r][c];
            fc->w->data[r][c] += fc->v_w->data[r][c];
        }
    }
}

void calc_predictions(matrix* A, unsigned int* predictions, int local_batch_size) {
    for (int c = 0; c < local_batch_size; c++) {
        float max_val = A->data[0][c];
        int max_idx = 0;
        for (int r = 1; r < A->n_rows; r++) {
            if (A->data[r][c] > max_val) {
                max_val = A->data[r][c];
                max_idx = r;
            }
        }
        if (max_val < 0 || max_val > 1) {
            printf("invalid probability %f at column %d\n", max_val, c);
        }
        predictions[c] = max_idx;
    }
}


unsigned int *one_hot_to_labels(matrix *Y) {
    int total_examples = Y->n_cols;
    unsigned int *labels = malloc(total_examples * sizeof(unsigned int));
    assert(labels);
    for (int j = 0; j < total_examples; j++) {
        for (int i = 0; i < Y->n_rows; i++) {
            if (Y->data[i][j] == 1.0f) {
                labels[j] = i;
                break;
            }
        }
    }
    return labels;
}

void activate_conv(conv_layer* conv, int local_batch_size) {
    OMP_PARALLEL_FOR_COLLAPSE2
    for (int b = 0; b < local_batch_size; b++) 
        for (int out_chan = 0; out_chan < conv->core.out_chans; out_chan++)
            for (int y = 0; y < conv->core.out_size; y++)
                for (int x = 0; x < conv->core.out_size; x++) {
                    if (conv->feature_maps[b][out_chan][y][x] < 0)
                        conv->feature_maps[b][out_chan][y][x] *= LEAKY_ALPHA;
                }
}

void convolute(conv_layer* conv, int start_idx, int end_idx) {
    float**** input = conv->in;
    int in_chans = (conv->core.in_chans == 0) ? 1 : conv->core.in_chans;
    int out_size = conv->core.out_size;
    int local_batch_size = end_idx-start_idx;
    const bool first_conv_layer = (conv->prev->in_chans == 0);

    OMP_PARALLEL_FOR
    for (int b = 0; b < local_batch_size; b++) {
        for (int out_chan = 0; out_chan < conv->core.out_chans; out_chan++) { // for each output featuremap
            for (int y_out = 0; y_out < out_size; y_out++){ 
                for (int x_out = 0; x_out < out_size; x_out++){
                    float sum = 0.0;
                    int y_in = y_out * conv->stride;
                    int x_in = x_out * conv->stride;
                    for (int depth = 0; depth < in_chans; depth++) {
                        CLANG_LOOP_UNROLL_ENABLE
                        for (int ky = 0; ky < GLOBAL_KERNEL_SIZE; ky++) { 
                        CLANG_LOOP_UNROLL_ENABLE
                            for (int kx = 0; kx < GLOBAL_KERNEL_SIZE; kx++) {
                                    if (first_conv_layer) // this is ugly, TODO: reomve this
                                        sum += input[start_idx+b] [depth] [y_in+ky] [x_in+kx] *
                                           conv->kernels[out_chan][depth][ky][kx];
                                    else
                                        sum += input[b][depth] [y_in+ky] [x_in+kx] *
                                           conv->kernels[out_chan][depth][ky][kx];
                            }
                        }
                    }
                    conv->feature_maps[b][out_chan][y_out][x_out] = sum;
                }
            }
        }
    }

    /*log_printf("sample conv pre-activation values %f %f %f\n",*/
           /*conv->feature_maps[0][0][2][1],*/
           /*conv->feature_maps[1][0][1][1],*/
           /*conv->feature_maps[2][1][2][0]*/
           /*);*/

    OMP_PARALLEL_FOR_COLLAPSE2 
    for (int b = 0; b < local_batch_size; b++) 
        for (int k = 0; k < conv->core.out_chans; k++)
            for (int y = 0; y < conv->core.out_size; y++)
                for (int x = 0; x < conv->core.out_size; x++) {
                    conv->feature_maps[b][k][y][x] += conv->biases[k];
                }

    // leaky relu-ing:
    activate_conv(conv, local_batch_size);

}

float**** calloc_4d(int dim1, int dim2, int dim3, int dim4) {
    float**** array = calloc(dim1, sizeof(float***));
    assert(array); 
    for (int i = 0; i < dim1; i++) {
        array[i] = calloc(dim2, sizeof(float**));
        assert(array[i]);
        for (int j = 0; j < dim2; j++) {
            array[i][j] = calloc(dim3, sizeof(float*));
            assert(array[i][j]);
            for (int k = 0; k < dim3; k++) {
                array[i][j][k] = calloc(dim4, sizeof(float));
                assert(array[i][j][k]);
            }
        }
    }
    return array;
}

int***** calloc_5d_as_int(int dim1, int dim2, int dim3, int dim4, int dim5) {
    int***** array = calloc(dim1, sizeof(int****));
    for (int i = 0; i < dim1; i++) {
        array[i] = calloc(dim2, sizeof(int***));
        assert(array[i] != NULL);
        for (int j = 0; j < dim2; j++) {
            array[i][j] = calloc(dim3, sizeof(int**));
            assert(array[i][j] != NULL);
            for (int k = 0; k < dim3; k++) {
                array[i][j][k] = calloc(dim4, sizeof(int*));
                assert(array[i][j][k] != NULL);
                for (int l = 0; l < dim4; l++) {
                    array[i][j][k][l] = calloc(dim5, sizeof(int));
                    assert(array[i][j][k][l] != NULL);
                }
            }
        }
    }
    return array;
}

fully_connected* init_fc_layer(int in_dim, int out_dim, void* prev, bool prev_is_flat) {
    fully_connected* layer = (fully_connected*)malloc(sizeof(fully_connected));
    assert(layer);

    layer->w = (matrix*)malloc(sizeof(matrix)); assert(layer->w);
    *(layer->w) = (matrix){ out_dim, in_dim, NULL };
    init_param(layer->w);

    layer->b = (matrix*)malloc(sizeof(matrix)); assert(layer->b);
    *(layer->b) = (matrix){ out_dim, 1, NULL };
    init_param(layer->b); 

    layer->v_w = (matrix*)malloc(sizeof(matrix)); assert(layer->v_w);
    *(layer->v_w) = (matrix){ out_dim, in_dim, NULL };
    init_matrix(layer->v_w); 

    layer->v_b = (matrix*)malloc(sizeof(matrix)); assert(layer->v_b);
    *(layer->v_b) = (matrix){ out_dim, 1, NULL };
    init_matrix(layer->v_b);

    layer->dw = (matrix*)malloc(sizeof(matrix)); assert(layer->dw);
    *(layer->dw) = (matrix){ out_dim, in_dim, NULL };
    init_matrix(layer->dw);

    layer->db = (matrix*)malloc(sizeof(matrix)); assert(layer->db);
    *(layer->db) = (matrix){ out_dim, 1, NULL };
    init_matrix(layer->db);

    layer->Z = (matrix*)malloc(sizeof(matrix)); assert(layer->Z);
    *(layer->Z) = (matrix){ out_dim, BATCH_SIZE, NULL };
    init_matrix(layer->Z);

    layer->A = (matrix*)malloc(sizeof(matrix)); assert(layer->A);
    *(layer->A) = (matrix){ out_dim, BATCH_SIZE, NULL };
    init_matrix(layer->A);

    layer->dZ = (matrix*)malloc(sizeof(matrix)); assert(layer->dZ);
    *(layer->dZ) = (matrix){ out_dim, BATCH_SIZE, NULL };
    init_matrix(layer->dZ);

    layer->prev = prev;
    layer->prev_is_flat = prev_is_flat;

    layer->core.in_chans  = 1;  
    layer->core.in_size   = in_dim;
    layer->core.out_chans = 1;
    layer->core.out_size  = out_dim;

    return layer;
}

flattened_layer* init_flat_layer(maxpool_layer* prev) {
    int flattened_size = prev->core.in_chans * prev->core.out_size * prev->core.out_size;

    matrix* flattened = malloc(sizeof(matrix));
    assert(flattened);
    *flattened = (matrix){ flattened_size, BATCH_SIZE, NULL };
    init_matrix(flattened);

    matrix* d_flattened = malloc(sizeof(matrix));
    assert(d_flattened);
    *d_flattened = (matrix){flattened_size, BATCH_SIZE, NULL};
    init_matrix(d_flattened);

    flattened_layer* layer = (flattened_layer*)malloc(sizeof(flattened_layer));
    assert(layer);
    layer->prev = prev; 
    layer->flattened = flattened;
    layer->d_flattened = d_flattened;

    return layer;

}

conv_layer* init_conv_layer(layer* prev, int num_kernels, int kernel_size, int stride, float**** in) {
    conv_layer* layer = malloc(sizeof(conv_layer));
    assert(layer != NULL);
    assert(in != NULL);
   
    layer->in = in;
    layer->prev = prev;

    layer->core.type = CONV_LAYER;

    layer->core.in_size = prev->out_size;
    layer->core.in_chans = prev->out_chans;

    layer->core.out_chans = num_kernels; 
    layer->core.out_size = ((prev->out_size - kernel_size ) / stride) + 1; // aka feature size, no padding

    layer->kernel_size = kernel_size;
    layer->stride = stride;

    layer->kernels = calloc_4d(layer->core.out_chans, layer->core.in_chans, kernel_size, kernel_size);
    layer->v_kernels = calloc_4d(layer->core.out_chans, layer->core.in_chans, kernel_size, kernel_size);

    layer->biases = calloc(layer->core.out_chans, sizeof(float));
    layer->v_biases = calloc(layer->core.out_chans, sizeof(float));
    assert(layer->biases && layer->v_biases);

    for (int k = 0; k < layer->core.out_chans; k++) {
        for (int d = 0; d < layer->core.in_chans; d++) {
            float fan_in = layer->core.in_chans * layer->kernel_size * layer->kernel_size;
            float std = sqrtf(2.0f / fan_in);  
            for (int i = 0; i < layer->kernel_size; i++) {
                for (int j = 0; j < layer->kernel_size; j++) {
                    float u1 = rand() / (float)RAND_MAX;
                    float u2 = rand() / (float)RAND_MAX;
                    float radius = sqrtf(-2.0f * logf(u1));
                    float theta = 2.0f * M_PI * u2;

                    layer->kernels[k][d][i][j] = radius * cosf(theta) * std;
                }
            }
        }
    }


    layer->d_in = calloc_4d(BATCH_SIZE, layer->core.in_chans, layer->core.in_size, layer->core.in_size);
    layer->d_kernel = calloc_4d(layer->core.out_chans, layer->core.in_chans, kernel_size, kernel_size);
    layer->feature_maps = calloc_4d(BATCH_SIZE, layer->core.out_chans, layer->core.out_size, layer->core.out_size);
    layer->d_feature_maps = calloc_4d(BATCH_SIZE, layer->core.out_chans, layer->core.out_size, layer->core.out_size);


    layer->d_biases = malloc(layer->core.out_chans * sizeof(float)); 

    return layer;
}

void conv_backward(conv_layer* conv, float**** input, int start_idx, int end_idx) {
    /*struct timespec start_time, end_time;                          */
    /*clock_gettime(CLOCK_MONOTONIC, &start_time);                   */
    const int out_chans = conv->core.out_chans;
    const int in_chans = conv->core.in_chans;
    const int out_size = conv->core.out_size;
    const int kernel_size = conv->kernel_size;
    const int stride = conv->stride;
    const int input_size = conv->core.in_size;
    const int local_batch_size = end_idx - start_idx;
    const bool first_conv_layer = (conv->prev->in_chans == 0);


    OMP_PARALLEL_FOR_COLLAPSE2
    for (int b = 0; b < local_batch_size; b++) {
        for (int oc = 0; oc < conv->core.out_chans; oc++) {
            for (int y = 0; y < conv->core.out_size; y++) {
                for (int x = 0; x < conv->core.out_size; x++) {
                    float val = conv->feature_maps[b][oc][y][x];
                    float deriv = (val > 0.0f) ? 1.0f : LEAKY_ALPHA;
                    conv->d_feature_maps[b][oc][y][x] *= deriv;
                }
            }
        }
    }

    for (int oc = 0; oc < out_chans; oc++) {
        conv->d_biases[oc] = 0.0f;
        for (int ic = 0; ic < in_chans; ic++) {
            CLANG_LOOP_UNROLL_ENABLE
            for (int ky = 0; ky < GLOBAL_KERNEL_SIZE; ky++) {
                CLANG_LOOP_UNROLL_ENABLE
                for (int kx = 0; kx < GLOBAL_KERNEL_SIZE; kx++) {
                    conv->d_kernel[oc][ic][ky][kx] = 0.0f;
                }
            }
        }
    }

    /*
     *  this if-else branching can be combined in a single one
     *  which is cleaner, but I separated them because
     *  under higher throughput and deeper networks, they resulted
     *  in a significant improvement in performance by avoiding 
     *  unnecessary checks inside the loop
     * */

    /*
     *  the following loop does the following: 
     *  for each output featmap (output channel),
     *      accumulate its gradient by convoluting through ALL incoming channels 
     *      (each kernel in the output convolves over all input
     *      channels) which are typically feature maps,
     *  unless in this case where the input is from 
     *  input_layer rather than another conv/maxpool
     * */

    if (first_conv_layer) {
        OMP_PARALLEL_FOR_COLLAPSE2
        for (int b = 0; b < local_batch_size; b++) { // for each image in the batch

            for (int oc = 0; oc < out_chans; oc++) {
                for (int y_out = 0; y_out < out_size; y_out++) { // for each position in the feature maps
                    for (int x_out = 0; x_out < out_size; x_out++) {
                        const float dz_val = conv->d_feature_maps[b][oc][y_out][x_out];
                        conv->d_biases[oc] += dz_val;
                        const int y_start = y_out * stride;
                        const int x_start = x_out * stride;
                        for (int ic = 0; ic < in_chans; ic++) { 
                            CLANG_LOOP_UNROLL_ENABLE
                            for (int ky = 0; ky < GLOBAL_KERNEL_SIZE; ky++) { 
                                CLANG_LOOP_UNROLL_ENABLE
                                for (int kx = 0; kx < GLOBAL_KERNEL_SIZE; kx++) {
                                    conv->d_kernel[oc][ic][ky][kx] += 
                                                    input[start_idx + b][ic][y_start+ky][x_start+kx] * dz_val;
                                }
                            }
                        }
                    }
                }
            }
        } 
    } else {

        OMP_PARALLEL_FOR_COLLAPSE2
        for (int b = 0; b < local_batch_size; b++) {
            for (int oc = 0; oc < out_chans; oc++) {
                for (int y_out = 0; y_out < out_size; y_out++) {
                    for (int x_out = 0; x_out < out_size; x_out++) {
                        const float dz_val = conv->d_feature_maps[b][oc][y_out][x_out];
                        conv->d_biases[oc] += dz_val;
                        const int y_start = y_out * stride;
                        const int x_start = x_out * stride;
                        CLANG_LOOP_UNROLL_ENABLE
                        for (int ic = 0; ic < in_chans; ic++) {
                            CLANG_LOOP_UNROLL_ENABLE
                            for (int ky = 0; ky < GLOBAL_KERNEL_SIZE; ky++) {
                                CLANG_LOOP_UNROLL_ENABLE
                                for (int kx = 0; kx < GLOBAL_KERNEL_SIZE; kx++) {
                                        conv->d_kernel[oc][ic][ky][kx] += 
                                                    input[b][ic][y_start+ky][x_start+kx] * dz_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    float batch_norm = 1.0f / (end_idx-start_idx); 
    OMP_PARALLEL_FOR
    for (int oc = 0; oc < out_chans; oc++) {
        conv->d_biases[oc] *= (1.0f/(end_idx-start_idx));
        for (int ic = 0; ic < in_chans; ic++) {
            CLANG_LOOP_UNROLL_ENABLE
            for (int ky = 0; ky < GLOBAL_KERNEL_SIZE; ky++) {
                CLANG_LOOP_UNROLL_ENABLE
                for (int kx = 0; kx < GLOBAL_KERNEL_SIZE; kx++) {
                    conv->d_kernel[oc][ic][ky][kx] *= batch_norm;
                }
            }
        }
    }

    
    OMP_PARALLEL_FOR 
    for (int oc=0; oc<out_chans; oc++)
        for (int ic=0; ic<in_chans; ic++)
            CLANG_LOOP_UNROLL_ENABLE
            for (int ky=0; ky<GLOBAL_KERNEL_SIZE; ky++)
                CLANG_LOOP_UNROLL_ENABLE
                for (int kx=0; kx<GLOBAL_KERNEL_SIZE; kx++)
                    conv->d_kernel[oc][ic][ky][kx] = 
                        fminf(fmaxf(conv->d_kernel[oc][ic][ky][kx], -5.0f), 5.0f);



    OMP_PARALLEL_FOR
    for (int k = 0; k < conv->core.out_chans; k++) {
        for (int d = 0; d < conv->core.in_chans; d++)
            for (int i = 0; i < conv->kernel_size; i++) 
                for (int j = 0; j < conv->kernel_size; j++) {
                    conv->v_kernels[k][d][i][j] = MOMENTUM * conv->v_kernels[k][d][i][j] - ALPHA * conv->d_kernel[k][d][i][j];
                    conv->kernels[k][d][i][j] += conv->v_kernels[k][d][i][j]; 
                }
        conv->v_biases[k] = MOMENTUM * conv->v_biases[k] - ALPHA * conv->d_biases[k];
        conv->biases[k] += conv->v_biases[k];
    }

    OMP_PARALLEL_FOR_COLLAPSE4
    for (int b = 0; b < local_batch_size; b++) 
        for (int ic = 0; ic < in_chans; ic++)
            for (int iy = 0; iy < input_size; iy++)
                for (int ix = 0; ix < input_size; ix++)
                    conv->d_in[b][ic][iy][ix] = 0.0f;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int b = 0; b < local_batch_size; b++) {
        for (int oc = 0; oc < out_chans; oc++) {
            for (int y_out = 0; y_out < out_size; y_out++) {
                for (int x_out = 0; x_out < out_size; x_out++) {
                    float dz_val = conv->d_feature_maps[b][oc][y_out][x_out];
                    for (int ic = 0; ic < in_chans; ic++) {
                        CLANG_LOOP_UNROLL_ENABLE
                        for (int ky = 0; ky < GLOBAL_KERNEL_SIZE; ky++) {
                            CLANG_LOOP_UNROLL_ENABLE
                            for (int kx = 0; kx < GLOBAL_KERNEL_SIZE; kx++) {
                                conv->d_in[b][ic][y_out * stride + ky][x_out * stride + kx] +=
                                        conv->kernels[oc][ic][kernel_size - 1 - ky][kernel_size - 1 - kx] * dz_val;
                            } 
                        }

                    }
                }
            }
        }
    }

        /*clock_gettime(CLOCK_MONOTONIC, &end_time);                */
    /*double elapsed = (end_time.tv_sec - start_time.tv_sec) +       */
                     /*(end_time.tv_nsec - start_time.tv_nsec) / 1e9;*/

        /*printf("[%s] took %.6f seconds\n", "conv_back", elapsed);*/
}






void maxpool_backward(maxpool_layer* layer, void* next_layer, bool next_is_conv,
                        int start_idx, int end_idx) {
    const int num_channels = layer->core.in_chans;
    const int downsampled_size = layer->downsampled_size;
    const int feature_size = layer->prev->core.out_size;
    const int local_batch_size = end_idx - start_idx;

    OMP_PARALLEL_FOR
    for (int b = 0; b < local_batch_size; b++) 
        for (int k = 0; k < num_channels; k++) 
            for (int i = 0; i < feature_size; i++) 
                for (int j = 0; j < feature_size; j++) 
                    layer->prev->d_feature_maps[b][k][i][j] = 0; 

    if (next_is_conv) {
        conv_layer* next_conv = (conv_layer*)next_layer;
        for (int b = 0; b < local_batch_size; b++)
            for (int k = 0; k < next_conv->core.in_chans; k++)
                for (int i = 0; i < downsampled_size; i++)
                    for (int j = 0; j < downsampled_size; j++)
                        layer->d_downsampled[b][k][i][j] = next_conv->d_in[b][k][i][j];
    } else {
        flattened_layer* flat_layer = (flattened_layer*)next_layer;
        for (int b = 0; b < local_batch_size; b++) {
            int flat_idx = 0; 
            for (int k = 0; k < layer->core.out_chans; k++) {
                for (int y = 0; y < layer->downsampled_size; y++) {
                    for (int x = 0; x < layer->downsampled_size; x++) {
                        layer->d_downsampled[b][k][y][x] = 
                                flat_layer->d_flattened->data[flat_idx][b];
                        flat_idx++;
                    }
                }
            }
        }
    }

    OMP_PARALLEL_FOR 
    for (int b = 0; b < local_batch_size; b++) {
        for (int k = 0; k < layer->core.out_chans; k++) {
            for (int i = 0; i < layer->downsampled_size; i++) {
                for (int j = 0; j < layer->downsampled_size; j++) {
                    int max_i = layer->max_positions[b][k][i][j][0];
                    int max_j = layer->max_positions[b][k][i][j][1];
                    layer->prev->d_feature_maps[b][k][max_i][max_j] += 
                        layer->d_downsampled[b][k][i][j];
                }
            }
        }
    }
}

void flatten(flattened_layer* fl, int start_idx, int end_idx) { 
    maxpool_layer* maxpool = fl->prev;
    const int local_batch_size = end_idx - start_idx;
    float** flattened = fl->flattened->data;
    for (int b = 0; b < local_batch_size; b++) {
        int feature_idx = 0;
        for (int k = 0; k < maxpool->core.in_chans; k++)
            for (int i = 0; i < maxpool->core.out_size; i++)
                for (int j = 0; j < maxpool->core.out_size; j++)
                    flattened[feature_idx++][b] = maxpool->downsampled[b][k][i][j];
    }
}

maxpool_layer* init_maxpool_layer(conv_layer* prev, int pooling_size) {
    maxpool_layer* layer = malloc(sizeof(maxpool_layer));
    assert(layer);
    layer->prev = prev;

    layer->pooling_size = pooling_size;
    layer->downsampled_size = (prev->core.out_size / pooling_size);

    layer->core.in_chans = prev->core.out_chans;
    layer->core.out_chans = prev->core.out_chans; // doesn't change, just the size gets smaller
    layer->core.type = MAXPOOL_LAYER;
    layer->core.in_size = prev->core.out_size;
    layer->core.out_size = layer->downsampled_size;

    layer->downsampled = calloc_4d(BATCH_SIZE, layer->core.out_chans, layer->downsampled_size, layer->downsampled_size);
    layer->d_downsampled = calloc_4d(BATCH_SIZE, layer->core.out_chans, layer->downsampled_size, layer->downsampled_size);
    layer->max_positions = calloc_5d_as_int(BATCH_SIZE, layer->core.out_chans, layer->downsampled_size, layer->downsampled_size, 2);

    return layer;
}

void maxpool(maxpool_layer* layer, int start_idx, int end_idx) {
    const int ps = layer->pooling_size;
    float**** featmaps = layer->prev->feature_maps;
    const int local_batch_size = end_idx - start_idx;

    for (int b = 0; b < local_batch_size; b++) {
        for (int k = 0; k < layer->core.in_chans; k++) {
            for (int i = 0; i < layer->core.in_size - ps + 1; i += ps) {
                for (int j = 0; j < layer->core.in_size - ps + 1; j += ps) {
                    // this looks more complicated than it is
                    float max = featmaps[b][k][i][j];
                    int max_x = i; 
                    int max_y = j; 
                    for (int w = 0; w < ps; w++) 
                        for (int h = 0; h < ps; h++)
                            if (featmaps[b][k][w+i][h+j] > max) {
                                max = featmaps[b][k][w+i][h+j];
                                max_x = w+i;
                                max_y = h+j;
                           }
                    layer->downsampled[b][k][i/ps][j/ps] = max;
                    layer->max_positions[b][k][i/ps][j/ps][0] = max_x;
                    layer->max_positions[b][k][i/ps][j/ps][1] = max_y;
                }
            }
        }
    }
}

void clip_matrix(matrix* m, float threshold, int start_col, int end_col) {
    if (end_col > m->n_cols) 
        end_col = m->n_cols; 
    OMP_PARALLEL_FOR_COLLAPSE2
    for (int r = 0; r < m->n_rows; r++) {
        for (int c = start_col; c < end_col; c++) {
            if (m->data[r][c] > threshold)
                m->data[r][c] = threshold;
            else if (m->data[r][c] < -threshold) 
                m->data[r][c] = -threshold;
        }
    }
}

void forward(generic_layer l, int start_idx, int end_idx) {
    switch (l.type) {
        case CONV_LAYER:
            convolute((conv_layer*)l.layer, start_idx, end_idx);
            break;
        case MAXPOOL_LAYER:
            maxpool((maxpool_layer*)l.layer, start_idx, end_idx);
            break;
        case FLATTENED_LAYER:
            flatten((flattened_layer*)l.layer, start_idx, end_idx);
            break;
        case FC_LAYER: {
                fully_connected *fc = (fully_connected*)l.layer;
                matrix *in = fc->prev_is_flat ?
               ((flattened_layer*)fc->prev)->flattened :
               ((fully_connected*)fc->prev)->A;
                forward_fc(fc, in, start_idx, end_idx);
                break;
        }
        default:
            printf("\nERROR: invalid layer type in forward() \n");
            exit(EXIT_FAILURE);
    }
}

void backward(generic_layer curr, generic_layer next, int start_idx, int end_idx) {
    const int local_batch_size = end_idx - start_idx;
    switch (curr.type) {
        case FC_LAYER: {
            fully_connected *fc = (fully_connected*)curr.layer;

            if (next.type == FC_LAYER) {
                fully_connected *next_fc = (fully_connected*)next.layer;
                matrix temp = { next_fc->w->n_cols, next_fc->dZ->n_cols, NULL };
                init_matrix(&temp);
                mat_mul_transpose_a(next_fc->w, next_fc->dZ, &temp, start_idx, end_idx);

                matrix deriv = { fc->Z->n_rows, fc->Z->n_cols, NULL };
                init_matrix(&deriv);
                leaky_relu_deriv(fc->Z, &deriv);
                
                zero_matrix(fc->dZ);
                mat_mul_elemwise(&temp, &deriv, fc->dZ);
                
                free_matrix(&temp);
                free_matrix(&deriv);
            }
            clip_matrix(fc->dZ, 3.0f, start_idx, end_idx);

            matrix dW = { fc->w->n_rows, fc->w->n_cols, NULL };
            init_matrix(&dW);
            mat_mul_transpose_b(fc->dZ, fc->A_prev, &dW, 0, local_batch_size);
            scale_matrix(&dW, 1.0f / local_batch_size);

            for (int i = 0; i < fc->dw->n_rows; i++) {
                for (int j = 0; j < fc->dw->n_cols; j++) {
                    fc->dw->data[i][j] = dW.data[i][j];
                }
            }

            free_matrix(&dW);
            clip_matrix(fc->dw, 3.0f, start_idx, end_idx);

            for (int i = 0; i < fc->db->n_rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < fc->dZ->n_cols; j++) {
                    sum += fc->dZ->data[i][j];
                }
                fc->db->data[i][0] = sum / (end_idx - start_idx);
            }

            matrix dA_prev = { fc->w->n_cols, fc->dZ->n_cols, NULL};
            init_matrix(&dA_prev);
            mat_mul_transpose_a(fc->w, fc->dZ, &dA_prev, start_idx, end_idx);

            OMP_PARALLEL_FOR
            for (int i=0; i<dA_prev.n_rows; i++)
                for (int j=0; j<dA_prev.n_cols; j++)
                    dA_prev.data[i][j] = fminf(fmaxf(dA_prev.data[i][j], -3.0f), 3.0f);

            if (fc->prev_is_flat) {
                flattened_layer *fl = (flattened_layer*)fc->prev;
                for (int i = 0; i < fl->d_flattened->n_rows; i++) {
                    for (int j = 0; j < fl->d_flattened->n_cols; j++) {
                        fl->d_flattened->data[i][j] = dA_prev.data[i][j];
                    }
                }
            } else {
                fully_connected *prev_fc = (fully_connected*)fc->prev;
                for (int i = 0; i < prev_fc->dZ->n_rows; i++) {
                    for (int j = 0; j < prev_fc->dZ->n_cols; j++) {
                        prev_fc->dZ->data[i][j] = dA_prev.data[i][j];
                    }
                }
            }
            free_matrix(&dA_prev);

            update_bias_fc(fc);
            update_weights_fc(fc);
            break;
        }
        case FLATTENED_LAYER: {
            flattened_layer *fl = (flattened_layer*)curr.layer;
            maxpool_layer *mp = fl->prev;
            int flat_idx;
            for (int b = 0; b < local_batch_size; b++) {
                flat_idx = 0;
                for (int k = 0; k < mp->core.out_chans; k++)
                    for (int i = 0; i < mp->downsampled_size; i++)
                        for (int j = 0; j < mp->downsampled_size; j++)
                            mp->d_downsampled[b][k][i][j] = fl->d_flattened->data[flat_idx++][b];
            }
            break;
        }
        case MAXPOOL_LAYER: {
            maxpool_layer *mp = (maxpool_layer*)curr.layer;
            bool next_is_conv = (next.type == CONV_LAYER);
            maxpool_backward(mp, next.layer, next_is_conv, start_idx, end_idx);
            break;
        }
        case CONV_LAYER: {
            conv_layer *conv = (conv_layer*)curr.layer;
            if (next.type == CONV_LAYER) { // TODO: I don't remember if this is totally correct
                OMP_PARALLEL_FOR 
                for (int b = 0; b < local_batch_size; b++) 
                    for (int oc = 0; oc < conv->core.out_chans; oc++)
                        for (int i = 0; i < conv->core.out_size; i++)
                            for (int j = 0; j < conv->core.out_size; j++)
                                conv->d_feature_maps[b][oc][i][j] = 0.0f;

                conv_layer *next_conv = (conv_layer*)next.layer;
                OMP_PARALLEL_FOR 
                for (int b = 0; b < local_batch_size; b++) 
                    for (int oc = 0; oc < conv->core.out_chans; oc++) {
                        for (int i = 0; i < conv->core.out_size; i++) {
                            for (int j = 0; j < conv->core.out_size; j++) {
                                conv->d_feature_maps[b][oc][i][j] += next_conv->d_in[b][oc][i][j];
                            }
                        }
                    }
            }
            conv_backward(conv, conv->in, start_idx, end_idx);
            break;
        }
        default:
            printf("\nERROR: invalid layer type in backward() \n");
            exit(EXIT_FAILURE);
    }
}

void transform_to_matrix(mnist_data* mnist_array, int m, MNIST_DATA_TYPE*** X) {
    *X = malloc(NUM_PIX_TOTAL * sizeof(double*));
    for (int i = 0; i < NUM_PIX_TOTAL; i++) {
        (*X)[i] = malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            (*X)[i][j] = (float)mnist_array[j].data[i/NUM_PIX_PER_DIM][i%NUM_PIX_PER_DIM];
        }
    }
}

void free_4d(float ****array, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                free(array[i][j][k]);
            }
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

void free_5d_int(int *****array, int dim1, int dim2, int dim3, int dim4) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int l = 0; l < dim4; l++) {
                    free(array[i][j][k][l]);
                }
                free(array[i][j][k]);
            }
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

void free_network(generic_layer *network) {
    for (int i = 0; network[i].layer != NULL; i++) {
        switch (network[i].type) {
            case CONV_LAYER: {
                conv_layer* layer = (conv_layer*)network[i].layer;
                free_4d(layer->kernels, layer->core.out_chans, layer->core.in_chans, layer->kernel_size);
                free_4d(layer->v_kernels, layer->core.out_chans, layer->core.in_chans, layer->kernel_size);
                free(layer->biases);
                free(layer->d_biases);
                free(layer->v_biases);
                free_4d(layer->d_kernel, layer->core.out_chans, layer->core.in_chans, layer->kernel_size);
                free_4d(layer->feature_maps, BATCH_SIZE, layer->core.out_chans, layer->core.out_size);
                free_4d(layer->d_feature_maps, BATCH_SIZE, layer->core.out_chans, layer->core.out_size);
                free_4d(layer->d_in, BATCH_SIZE, layer->core.in_chans, layer->core.in_size);
                free(layer);
                break;
            }
            case MAXPOOL_LAYER: {
                maxpool_layer* layer = (maxpool_layer*)network[i].layer;
                free_4d(layer->downsampled, BATCH_SIZE, layer->core.out_chans, layer->downsampled_size);
                free_4d(layer->d_downsampled, BATCH_SIZE, layer->core.out_chans, layer->downsampled_size);
                free_5d_int(layer->max_positions, BATCH_SIZE, layer->core.out_chans, layer->downsampled_size, layer->downsampled_size);
                free(layer);
                break;
            }
            case FC_LAYER: {
                fully_connected* layer = (fully_connected*)network[i].layer;
                free_matrix(layer->w); free(layer->w);
                free_matrix(layer->v_w); free(layer->v_w);
                free_matrix(layer->b); free(layer->b);
                free_matrix(layer->v_b); free(layer->v_b);
                free_matrix(layer->dw); free(layer->dw);
                free_matrix(layer->db); free(layer->db);
                free_matrix(layer->Z); free(layer->Z);
                free_matrix(layer->A); free(layer->A);
                free_matrix(layer->dZ); free(layer->dZ);
                free(layer);
                break;
            }
            case FLATTENED_LAYER: { 
                flattened_layer* layer = (flattened_layer*)network[i].layer;
                free_matrix(layer->flattened);
                free_matrix(layer->d_flattened);
                free(layer->flattened);
                free(layer->d_flattened);
                free(layer);
                break;
            }
            default:
                break;
        }
    }
}


