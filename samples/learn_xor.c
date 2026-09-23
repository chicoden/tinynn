#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "../include/network.h"
#include "../include/activation.h"
#include "../include/cost.h"
#include "../include/initializers.h"
#include "../include/training.h"
#include "../include/evaluation.h"

int main() {
    uint32_t example_count = 4;
    float example_inputs[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    float example_outputs[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    struct tinynn_network_t network;
    tinynn_create_network(&network, (struct tinynn_network_layout_t){
        .input_node_count = 2,
        .layer_count = 2,
        .layers = (struct tinynn_layer_t[]){
            {
                .node_count = 3,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            },
            {
                .node_count = 1,
                .activation = &TINYNN_ACTIVATION_SIGMOID
            }
        }
    });
    tinynn_init_params_random_normalized(&network, time(NULL));

    struct tinynn_evaluation_ctx_t evaluation_ctx;
    tinynn_create_evaluation_ctx(&evaluation_ctx, &network);

    float* inputs = example_inputs;
    for (uint32_t i = 0; i < example_count; i++) {
        float output;
        tinynn_evaluate(&evaluation_ctx, inputs, &output);
        printf("%f, %f -> %f\n", inputs[0], inputs[1], output);
        inputs += network.layout.input_node_count;
    }
    char _;
    scanf("%c", &_);

    struct tinynn_training_ctx_t training_ctx;
    struct tinynn_training_params_t training_params;
    tinynn_create_training_ctx(&training_ctx, &network);
    training_params.cost = TINYNN_COST_QUADRATIC;
    training_params.learning_rate = 0.1f;
    training_params.regularization_factor = 0.0f;
    tinynn_train(&training_ctx, training_params, example_count, example_inputs, example_outputs, 200000, 1);
    tinynn_destroy_training_ctx(&training_ctx);

    inputs = example_inputs;
    for (uint32_t i = 0; i < example_count; i++) {
        float output;
        tinynn_evaluate(&evaluation_ctx, inputs, &output);
        printf("%f, %f -> %f\n", inputs[0], inputs[1], output);
        inputs += network.layout.input_node_count;
    }

    tinynn_destroy_evaluation_ctx(&evaluation_ctx);
    tinynn_destroy_network(&network);
    return 0;
}