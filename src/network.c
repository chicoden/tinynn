#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../include/network.h"
#include "../include/activation.h"

#define COUNTOF(x) (sizeof(x) / sizeof(x[0]))

static const struct tinynn_activation_t* TINYNN_ACTIVATION_TABLE[] = {
    &TINYNN_ACTIVATION_SIGMOID,
    &TINYNN_ACTIVATION_SOFTMAX
};

void tinynn_create_network(struct tinynn_network_t* network, struct tinynn_network_layout_t layout) {
    uint32_t bias_count = 0;
    uint32_t weight_count = 0;
    uint32_t prev_layer_size = layout.input_node_count;
    for (uint32_t l = 0; l < layout.layer_count; l++) {
        uint32_t this_layer_size = layout.layers[l].node_count;
        bias_count += this_layer_size;
        weight_count += this_layer_size * prev_layer_size;
        prev_layer_size = this_layer_size;
    }

    network->layout.input_node_count = layout.input_node_count;
    network->layout.layer_count = layout.layer_count;
    network->layout.layers = (struct tinynn_layer_t*)malloc(layout.layer_count * sizeof(struct tinynn_layer_t));
    memcpy(network->layout.layers, layout.layers, layout.layer_count * sizeof(struct tinynn_layer_t));

    network->bias_count = bias_count;
    network->weight_count = weight_count;
    network->biases = (float*)malloc(bias_count * sizeof(float));
    network->weights = (float*)malloc(weight_count * sizeof(float));
}

void tinynn_destroy_network(struct tinynn_network_t* network) {
    free(network->layout.layers);
    free(network->biases);
    free(network->weights);
}

int tinynn_save_network(
    const struct tinynn_network_t* network,
    const char* path,
    uint32_t activation_table_entry_count,
    const struct tinynn_activation_t** activation_table
) {
    FILE* file = fopen(path, "wb");
    if (file == NULL) return 0;

    if (activation_table == NULL) {
        activation_table_entry_count = COUNTOF(TINYNN_ACTIVATION_TABLE);
        activation_table = TINYNN_ACTIVATION_TABLE;
    }

    const struct tinynn_network_layout_t* layout = &network->layout;
    fwrite(&layout->input_node_count, sizeof(layout->input_node_count), 1, file);
    fwrite(&layout->layer_count, sizeof(layout->layer_count), 1, file);
    for (uint32_t l = 0; l < layout->layer_count; l++) {
        const struct tinynn_layer_t* layer = &layout->layers[l];

        uint32_t activation_id = 0xFFFFFFFF;
        for (uint32_t i = 0; i < activation_table_entry_count; i++) {
            if (layer->activation == activation_table[i]) {
                activation_id = i;
                break;
            }
        }

        fwrite(&layer->node_count, sizeof(layer->node_count), 1, file);
        fwrite(&activation_id, sizeof(activation_id), 1, file);
    }

    fwrite(network->biases, sizeof(float), network->bias_count, file);
    fwrite(network->weights, sizeof(float), network->weight_count, file);

    fclose(file);
    return 1;
}

int tinynn_load_network(
    struct tinynn_network_t* network,
    const char* path,
    uint32_t activation_table_entry_count,
    const struct tinynn_activation_t** activation_table
) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) return 0;

    if (activation_table == NULL) {
        activation_table_entry_count = COUNTOF(TINYNN_ACTIVATION_TABLE);
        activation_table = TINYNN_ACTIVATION_TABLE;
    }

    struct tinynn_network_layout_t* layout = &network->layout;
    fread(&layout->input_node_count, sizeof(layout->input_node_count), 1, file);
    fread(&layout->layer_count, sizeof(layout->layer_count), 1, file);
    layout->layers = (struct tinynn_layer_t*)malloc(layout->layer_count * sizeof(struct tinynn_layer_t));
    uint32_t bias_count = 0;
    uint32_t weight_count = 0;
    uint32_t prev_layer_size = layout->input_node_count;
    for (uint32_t l = 0; l < layout->layer_count; l++) {
        struct tinynn_layer_t* layer = &layout->layers[l];
        fread(&layer->node_count, sizeof(layer->node_count), 1, file);
        layer->activation = NULL;

        uint32_t activation_id = 0xFFFFFFFF;
        fread(&activation_id, sizeof(activation_id), 1, file);
        if (activation_id < activation_table_entry_count) {
            layer->activation = activation_table[activation_id];
        }

        bias_count += layer->node_count;
        weight_count += layer->node_count * prev_layer_size;
        prev_layer_size = layer->node_count;
    }

    network->bias_count = bias_count;
    network->biases = (float*)malloc(bias_count * sizeof(float));
    fread(network->biases, sizeof(float), bias_count, file);

    network->weight_count = weight_count;
    network->weights = (float*)malloc(weight_count * sizeof(float));
    fread(network->weights, sizeof(float), weight_count, file);

    fclose(file);
    return 1;
}