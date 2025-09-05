// Layer sizes: 1024, 256, 64, 64, 32, 10
// > Average time with column major weights: 23us
// > Average time with row major weights: 20us
// ----------------------------------------------------------------
// Layer sizes: 1024, 256, 256, 256, 64, 10
// > Average time with column major weights: 21us
// > Average time with row major weights: 26us
// ----------------------------------------------------------------
// Note: must run a few times to make sure no dependence on the seeding
// Results: idk

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "../include/network.h"
#include "../include/activation.h"
#include "../include/initializers.h"
#include "../include/evaluation.h"

#include <windows.h>

int main() {
    uint32_t input_count = 1024;
    uint32_t eval_rounds = 1000000;
    float test_pair[1024 + 10];

    struct tinynn_network_layout_t layout = {
        .input_node_count = 1024,
        .layer_count = 5,
        .layers = (struct tinynn_layer_t[]){
            {
                .node_count = 256,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 256,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 256,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 64,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 10,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            }
        }
    };

    struct tinynn_network_t network;
    tinynn_create_network(&network, &layout);
    tinynn_init_params_random_normalized(&network, time(NULL));

    {
        struct tinynn_evaluation_ctx_t eval_ctx;
        tinynn_create_evaluation_ctx(&eval_ctx, &network);

        {
            LARGE_INTEGER start_time, end_time, average_us, frequency;
            QueryPerformanceFrequency(&frequency);
            QueryPerformanceCounter(&start_time);

            volatile float dont_optimize = 0.0f;
            for (volatile uint32_t i = 0; i < eval_rounds; i++) {
                volatile float* inputs = &test_pair[0];
                volatile float* outputs = &test_pair[input_count];

                for (volatile uint32_t j = 0; j < input_count; j++) {
                    inputs[j] = (float)rand() / (float)RAND_MAX;
                }

                tinynn_evaluate(&eval_ctx, (float*)inputs, (float*)outputs);
                dont_optimize += outputs[0];
            }

            QueryPerformanceCounter(&end_time);
            average_us.QuadPart = end_time.QuadPart - start_time.QuadPart;
            average_us.QuadPart *= 1000000;
            average_us.QuadPart /= frequency.QuadPart;
            printf("Total time: %zuus\n", average_us.QuadPart);

            average_us.QuadPart /= eval_rounds;
            printf("Average time: %zuus\n", average_us.QuadPart);

            printf("afihwaifhaofihaw %f\n", dont_optimize);
        }

        tinynn_destroy_evaluation_ctx(&eval_ctx);
    }

    tinynn_destroy_network(&network);
    return 0;
}