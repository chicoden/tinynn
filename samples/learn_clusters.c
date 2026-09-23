#include <stdio.h>
#include <stdint.h>
#include "../include/network.h"
#include "../include/activation.h"
#include "../include/cost.h"
#include "../include/initializers.h"
#include "../include/training.h"

const uint32_t EXAMPLE_COUNT = 32;
static const float EXAMPLE_INPUTS[] = {
    0.181259f, 0.600128f,
    0.209161f, 0.686168f,
    0.272544f, 0.760272f,
    0.210224f, 0.875845f,
    0.045871f, 0.691806f,
    0.107407f, 0.879798f,
    0.096386f, 0.464752f,
    0.705028f, 0.667955f,
    0.561318f, 0.717127f,
    0.551332f, 0.755783f,
    0.642029f, 0.859787f,
    0.578756f, 0.661293f,
    0.625287f, 0.561804f,
    0.993646f, 0.878986f,
    0.905712f, 0.376894f,
    0.828445f, 0.289356f,
    0.900651f, 0.442489f,
    0.979855f, 0.145529f,
    0.535972f, 0.125601f,
    0.535338f, 0.286151f,
    0.422854f, 0.193598f,
    0.596321f, 0.269657f,
    0.616100f, 0.290497f,
    0.540227f, 0.336999f,
    0.632601f, 0.092070f,
    0.386532f, 0.083337f,
    0.707578f, 0.041996f,
    0.308483f, 0.217245f,
    0.380038f, 0.107003f,
    0.325411f, 0.257011f,
    0.592546f, 0.425580f,
    0.415377f, 0.017189f
};
static const float EXAMPLE_OUTPUTS[] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

int main() {
    struct tinynn_network_t network;
    tinynn_create_network(&network, (struct tinynn_network_layout_t){
        .input_node_count = 2,
        .layer_count = 3,
        .layers = (struct tinynn_layer_t[]){
            {
                .activation = &TINYNN_ACTIVATION_SIGMOID,
                .node_count = 4
            },
            {
                .activation = &TINYNN_ACTIVATION_SIGMOID,
                .node_count = 4
            },
            {
                .activation = &TINYNN_ACTIVATION_SIGMOID,
                .node_count = 4
            }
        }
    });
    tinynn_init_params_random_normalized(&network, 4079);

    struct tinynn_training_ctx_t training_ctx;
    tinynn_create_training_ctx(&training_ctx, &network);

    struct tinynn_training_params_t training_params;
    training_params.cost = TINYNN_COST_QUADRATIC;
    training_params.learning_rate = 0.1f;

    tinynn_train(&training_ctx, training_params, EXAMPLE_COUNT, EXAMPLE_INPUTS, EXAMPLE_OUTPUTS, 1000000, 1);

    printf("%f", network.biases[0]);
    for (uint32_t i = 1; i < network.bias_count; i++) {
        printf(", %f", network.biases[i]);
    }
    printf("\n%f", network.weights[0]);
    for (uint32_t i = 1; i < network.weight_count; i++) {
        printf(", %f", network.weights[i]);
    }
    fputc('\n', stdout);

    tinynn_destroy_training_ctx(&training_ctx);
    tinynn_destroy_network(&network);
    return 0;
}