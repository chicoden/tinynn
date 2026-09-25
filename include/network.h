#pragma once

#include <stdint.h>
#include "../include/activation.h"

struct tinynn_layer_t {
    uint32_t node_count;
    const struct tinynn_activation_t* activation;
};

struct tinynn_network_layout_t {
    uint32_t input_node_count;
    uint32_t layer_count;
    struct tinynn_layer_t* layers;
};

struct tinynn_network_t {
    struct tinynn_network_layout_t layout;
    uint32_t bias_count;
    uint32_t weight_count;
    float* biases;
    float* weights;
};

void tinynn_create_network(struct tinynn_network_t* network, struct tinynn_network_layout_t layout);
void tinynn_destroy_network(struct tinynn_network_t* network);
int tinynn_save_network(
    const struct tinynn_network_t* network,
    const char* path,
    uint32_t activation_table_entry_count,
    const struct tinynn_activation_t** activation_table
);
int tinynn_load_network(
    struct tinynn_network_t* network,
    const char* path,
    uint32_t activation_table_entry_count,
    const struct tinynn_activation_t** activation_table
);