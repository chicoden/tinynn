#include <stdlib.h>
#include <stdint.h>
#include "../include/network.h"
#include "../include/evaluation.h"

void tinynn_create_evaluation_ctx(struct tinynn_evaluation_ctx_t* evaluation_ctx, const struct tinynn_network_t* network) {
    evaluation_ctx->network = network;
    evaluation_ctx->output_buffers[0] = (float*)malloc(network->max_layer_size * sizeof(float));
    evaluation_ctx->output_buffers[1] = (float*)malloc(network->max_layer_size * sizeof(float));
}

void tinynn_destroy_evaluation_ctx(struct tinynn_evaluation_ctx_t* evaluation_ctx) {
    free(evaluation_ctx->output_buffers[0]);
    free(evaluation_ctx->output_buffers[1]);
}

void tinynn_evaluate(struct tinynn_evaluation_ctx_t* evaluation_ctx, const float* inputs, float* outputs) {
    const struct tinynn_network_t* network = evaluation_ctx->network;
    float* prev_layer_output = evaluation_ctx->output_buffers[0];
    float* this_layer_output = evaluation_ctx->output_buffers[1];
    uint32_t prev_layer_size = network->layout.input_node_count;

    for (uint32_t i = 0; i < prev_layer_size; i++) {
        prev_layer_output[i] = inputs[i];
    }

    float* layer_biases = network->biases;
    float* layer_weights = network->weights;
    for (uint32_t l = 0; l < network->layout.layer_count; l++) {
        const struct tinynn_layer_t* layer = &network->layout.layers[l];
        uint32_t this_layer_size = layer->node_count;

        // Initialize with biases
        for (uint32_t b = 0; b < this_layer_size; b++) {
            this_layer_output[b] = *(layer_biases++);
        }

        // Accumulate weighted input from previous layer
        for (uint32_t i = 0; i < prev_layer_size; i++) {
            float input = prev_layer_output[i];
            for (uint32_t w = 0; w < this_layer_size; w++) {
                this_layer_output[w] += *(layer_weights++) * input;
            }
        }

        // Apply activation function, storing the result so that it feeds into the next layer
        layer->activation->eval_elementwise(this_layer_size, this_layer_output, prev_layer_output);
        prev_layer_size = this_layer_size;
    }

    for (uint32_t i = 0; i < prev_layer_size; i++) {
        outputs[i] = prev_layer_output[i];
    }
}