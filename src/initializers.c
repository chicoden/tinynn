#include <stdlib.h>
#include <stdint.h>
#include "../include/umath.h"
#include "../include/initializers.h"

void tinynn_init_params_random(struct tinynn_network_t* network, uint32_t seed) {
    srand(seed);

    for (uint32_t i = 0; i < network->bias_count; i++) {
        network->biases[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }

    for (uint32_t i = 0; i < network->weight_count; i++) {
        network->weights[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

void tinynn_init_params_random_normalized(struct tinynn_network_t* network, uint32_t seed) {
    tinynn_init_params_random(network, seed);

    float* layer_weights = network->weights;
    uint32_t prev_layer_size = network->layout.input_node_count;
    for (uint32_t i = 0; i < network->layout.layer_count; i++) {
        uint32_t this_layer_size = network->layout.layers[i].node_count;
        uint32_t layer_weight_count = this_layer_size * prev_layer_size;

        float normalization_factor = umath_recip_sqrt(prev_layer_size);
        for (uint32_t w = 0; w < layer_weight_count; w++) {
            layer_weights[w] *= normalization_factor;
        }

        layer_weights += layer_weight_count;
        prev_layer_size = this_layer_size;
    }
}