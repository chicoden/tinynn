#include <stdlib.h>
#include <stdint.h>
#include "../include/network.h"

void tinynn_create_network(struct tinynn_network_t* network, const struct tinynn_network_layout_t* layout) {
    uint32_t bias_count = 0;
    uint32_t weight_count = 0;
    uint32_t prev_layer_size = layout->input_node_count;
    for (uint32_t l = 0; l < layout->layer_count; l++) {
        uint32_t this_layer_size = layout->layers[l].node_count;
        bias_count += this_layer_size;
        weight_count += this_layer_size * prev_layer_size;
        prev_layer_size = this_layer_size;
    }

    network->layout = *layout;
    network->bias_count = bias_count;
    network->weight_count = weight_count;
    network->biases = (float*)malloc(bias_count * sizeof(float));
    network->weights = (float*)malloc(weight_count * sizeof(float));
}

void tinynn_destroy_network(struct tinynn_network_t* network) {
    free(network->biases);
    free(network->weights);
}