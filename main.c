#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#define MNIST_DATA_DEFINED 

#include "mnist.h"
#include "cnn.h"
#include "matrix.h"
#include "hyperparams.h"

#include <stdio.h>
#include <stdlib.h>
/*#define NDEBUG*/
#include <time.h>

int main() {

    srand(time(NULL)); 

    /***** preproccessing *****/

    mnist_data* data;
    int return_code;
    unsigned int loaded_count;
    if ((return_code = mnist_load("mnist_dataset/train-images.idx3-ubyte", "mnist_dataset/train-labels.idx1-ubyte", &data, &loaded_count))) {
        printf("error loading data, code: %d\n", return_code);
        return EXIT_FAILURE;
    }

    unsigned int Y[NUM_EXAMPLES] ;
    matrix Y_one_hot = {OUT_NEURONS, NUM_EXAMPLES, NULL};
    init_matrix(&Y_one_hot);

    assert(loaded_count == NUM_EXAMPLES);

    for (unsigned int i = 0; i < NUM_EXAMPLES; i++) {
        Y[i] = (data[i]).label;
        Y_one_hot.data[(data[i]).label][i] = 1;
    }

    for (int i = 0; i < NUM_EXAMPLES; i++)
        assert(Y_one_hot.data[Y[i]][i] == 1);

    MNIST_DATA_TYPE** mat;
    transform_to_matrix(data, loaded_count, &mat);

    matrix X = {NUM_PIX_TOTAL, NUM_EXAMPLES, NULL}; 
    init_matrix(&X);
    for (int r = 0; r < NUM_PIX_TOTAL; r++) {
        for (int c = 0; c < NUM_EXAMPLES; c++) {
            X.data[r][c] = (float)mat[r][c];
        }
    }

    for (int i = 0; i < NUM_PIX_TOTAL; i++) {
        free(mat[i]);
    }
    free(mat);

    float**** images = calloc_4d(NUM_EXAMPLES, 1, NUM_PIX_PER_DIM, NUM_PIX_PER_DIM);
    for (int i = 0; i < NUM_EXAMPLES; i++) {
        for (int r = 0; r < NUM_PIX_PER_DIM; r++) {
            for (int c = 0; c < NUM_PIX_PER_DIM; c ++) {
                images[i][0][r][c] = (float)data[i].data[r][c]; 
            }
        }
    }

    free(data);
    data = NULL;

    /***** configuring & initializing Layers: *****/
    
    layer input_layer = {
        .in_chans = 0,
        .out_chans = 1, // i.e. grayscale
        .in_size = 0,
        .out_size = NUM_PIX_PER_DIM
    };

    conv_layer* conv1 = init_conv_layer(&input_layer, 4,
                                        3, 1, images);
    maxpool_layer* maxpool1 = init_maxpool_layer(conv1, 2);

    conv_layer* conv2 = init_conv_layer(&maxpool1->core, 8,
                                        3, 1, maxpool1->downsampled);
    maxpool_layer* maxpool2 = init_maxpool_layer(conv2, 2);


    flattened_layer* flat_layer = init_flat_layer(maxpool2);



    const bool prev_is_flat = true;
    fully_connected* fc_layer1 = init_fc_layer(
       flat_layer->d_flattened->n_rows,
        HIDDEN_NEURONS, (void*)flat_layer, prev_is_flat);

    fully_connected* fc_layer2 = init_fc_layer(
        HIDDEN_NEURONS, OUT_NEURONS, (void*)fc_layer1, !prev_is_flat);

    matrix* A_2 = fc_layer2->A;

    generic_layer network[] = {
        {conv1, CONV_LAYER},
        {maxpool1, MAXPOOL_LAYER},
        {conv2, CONV_LAYER},
        {maxpool2, MAXPOOL_LAYER},
        {flat_layer, FLATTENED_LAYER},
        {fc_layer1, FC_LAYER},
        {fc_layer2, FC_LAYER},
        (generic_layer){NULL, -1}, // dummy layer, not calculated in num_layers
    };
    int num_layers = 7;


    /***** Main Training Loop *****/

    printf("\n------------------------\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        /*ALPHA = ALPHA * DECAY;*/ // decay-ing was not very useful in my case
        float epoch_correct = 0;
        float epoch_total = 0;
        for (int b = 0; b < NUM_TRAIN; b += BATCH_SIZE) {
        /* struct timespec start_time, end_time;*/ /*clock_gettime(CLOCK_MONOTONIC, &start_time);*/
            int start_idx = b;
            int end_idx = (b + BATCH_SIZE > NUM_TRAIN) ? NUM_TRAIN : b + BATCH_SIZE;
            int local_batch_size = end_idx - start_idx;

            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
                forward(network[layer_idx], start_idx, end_idx);

            fully_connected *fc_out = (fully_connected*)network[num_layers - 1].layer;
            for (int i = 0; i < fc_out->A->n_rows; i++) {
                for (int bn = 0; bn < local_batch_size; bn++) {
                    int global_j = start_idx + bn;
                    fc_out->dZ->data[i][bn] = fc_out->A->data[i][bn] - Y_one_hot.data[i][global_j];
                }
            }

            for (int layer_idx = num_layers-1; layer_idx >= 0; layer_idx--)
                    backward(network[layer_idx], network[layer_idx+1], start_idx, end_idx);

            unsigned int batch_preds[BATCH_SIZE];
            calc_predictions(A_2, batch_preds, local_batch_size);

            int batch_correct = 0;
            for (int i = 0; i < local_batch_size; i++) {
                if (batch_preds[i] == Y[start_idx + i])
                    batch_correct++;
            }
            epoch_correct += batch_correct;
            epoch_total   += local_batch_size;
        }


        // validation acc:
        float train_acc = epoch_correct / epoch_total;
        float val_correct = 0.0f;
        float val_total   = 0.0f;
        for (int b = NUM_TRAIN; b < NUM_EXAMPLES; b += BATCH_SIZE) {
            int start_idx = b;
            int end_idx   = (b + BATCH_SIZE > NUM_EXAMPLES) ? NUM_EXAMPLES : (b + BATCH_SIZE);
            int local_batch_size = end_idx - start_idx;

            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
                forward(network[layer_idx], start_idx, end_idx);
            }

            unsigned int batch_preds[BATCH_SIZE];
            calc_predictions(A_2, batch_preds, local_batch_size);

            int batch_correct = 0;
            for (int i = 0; i < local_batch_size; i++)
                if (batch_preds[i] == Y[start_idx + i])
                    batch_correct++;

            val_correct += batch_correct;
            val_total   += local_batch_size;
        }
        float val_acc = val_correct / val_total;

        printf("Epoch %d: train_acc = %.1f%%, val_acc = %.1f%%\n", epoch + 1, train_acc * 100, val_acc * 100);

    }


    /* --- test set --- */
    mnist_data* test_data;
    unsigned int test_count;
    int test_ret = mnist_load("mnist_dataset/t10k-images.idx3-ubyte",
                              "mnist_dataset/t10k-labels.idx1-ubyte",
                              &test_data, &test_count);
    if (test_ret) {
        printf("Error loading test data, code: %d\n", test_ret);
        return EXIT_FAILURE;
    }

    assert(test_count == NUM_TEST);

    unsigned int test_Y[NUM_TEST];
    for (unsigned int i = 0; i < test_count; i++)
        test_Y[i] = test_data[i].label;

    float**** test_images = calloc_4d(test_count, 1, NUM_PIX_PER_DIM, NUM_PIX_PER_DIM);
    for (int i = 0; i < test_count; i++) {
        for (int r = 0; r < NUM_PIX_PER_DIM; r++) {
            for (int c = 0; c < NUM_PIX_PER_DIM; c++) {
                test_images[i][0][r][c] = (float)test_data[i].data[r][c];
            }
        }
    }

    conv1->in = test_images;
    int correct_test = 0;
    for (int b = 0; b < NUM_TEST; b += BATCH_SIZE) {
        int start_idx = b;
        int end_idx = (b + BATCH_SIZE > NUM_TEST) ? NUM_TEST : b + BATCH_SIZE;
        int local_batch_size = end_idx - start_idx;

        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
            forward(network[layer_idx], start_idx, end_idx);

        unsigned int batch_preds[BATCH_SIZE];
        calc_predictions(A_2, batch_preds, local_batch_size);

        for (int i = 0; i < local_batch_size; i++)
            if (batch_preds[i] == test_Y[start_idx + i])
                correct_test++;
    }
    float test_acc = (float)correct_test / NUM_TEST;
    printf("\n------------------------\n");
    printf("test accuracy: %.1f%%\n", test_acc * 100);


    /* * 
     *      -- cleanup -- 
     * */

    free_4d(test_images, test_count, 1, NUM_PIX_PER_DIM);
    free(test_data);

    free_network(network);
    free_4d(images, NUM_EXAMPLES, 1, NUM_PIX_PER_DIM);
    free_matrix(&X);
    free_matrix(&Y_one_hot);
    
    printf("\nExit success.\n");
    return 0;
}
