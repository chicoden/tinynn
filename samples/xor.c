#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "../include/network.h"
#include "../include/activation.h"
#include "../include/initializers.h"
#include "../include/evaluation.h"

#define COUNTOF(x) (sizeof(x) / sizeof(x[0]))

int main() {
    uint32_t input_count = 2;
    uint32_t output_count = 1;
    float test_data[][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f, 0.0f},
    };

    struct tinynn_network_t network;
    tinynn_create_network(&network, (struct tinynn_network_layout_t){
        .input_node_count = 2,
        .layer_count = 5,
        .layers = (struct tinynn_layer_t[]){
            {
                .node_count = 4,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 4,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 3,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 8,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 1,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            }
        }
    });

    {
        tinynn_init_params_random_normalized(&network, /*4079*/ time(NULL));

        struct tinynn_evaluation_ctx_t eval_ctx;
        tinynn_create_evaluation_ctx(&eval_ctx, &network);

        {
            FILE* save_file = fopen("test_eval.bin", "wb");
            if (save_file != NULL) {
                const struct tinynn_network_layout_t* layout = &network.layout;

                fwrite(&layout->layer_count, sizeof(layout->layer_count), 1, save_file);
                fwrite(&layout->input_node_count, sizeof(layout->input_node_count), 1, save_file);
                for (uint32_t l = 0; l < layout->layer_count; l++) {
                    fwrite(&layout->layers[l].node_count, sizeof(layout->layers[l].node_count), 1, save_file);
                }

                fwrite(network.biases, sizeof(float), network.bias_count, save_file);
                fwrite(network.weights, sizeof(float), network.weight_count, save_file);

                uint32_t example_count = COUNTOF(test_data);
                fwrite(&example_count, sizeof(example_count), 1, save_file);
                for (uint32_t i = 0; i < COUNTOF(test_data); i++) {
                    float* inputs = &test_data[i][0];
                    float* outputs = &test_data[i][input_count];
                    tinynn_evaluate(&eval_ctx, inputs, outputs);
                    fwrite(&test_data[i][0], sizeof(float), input_count + output_count, save_file);
                }

                fclose(save_file);
            } else {
                printf("Failed to save\n");
            }
        }

        for (uint32_t i = 0; i < COUNTOF(test_data); i++) {
            float* inputs = &test_data[i][0];
            float* outputs = &test_data[i][input_count];
            tinynn_evaluate(&eval_ctx, inputs, outputs);
            printf("%f, %f -> %f\n", inputs[0], inputs[1], outputs[0]);
        }

        tinynn_destroy_evaluation_ctx(&eval_ctx);
    }

    tinynn_destroy_network(&network);
    return 0;
}