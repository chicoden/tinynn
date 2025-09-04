#include <stdlib.h>
#include <stdint.h>
#include "../include/network.h"
#include "../include/evaluation.h"

void tinynn_create_evaluation_ctx(struct tinynn_evaluation_ctx_t* evaluation_ctx, const struct tinynn_network_t* network) {
    evaluation_ctx->network = network;
    evaluation_ctx->preactivation_outputs = (float*)malloc(network->bias_count * sizeof(float));
    evaluation_ctx->postactivation_outputs = (float*)malloc(network->bias_count * sizeof(float));
}

void tinynn_destroy_evaluation_ctx(struct tinynn_evaluation_ctx_t* evaluation_ctx) {
    free(evaluation_ctx->preactivation_outputs);
    free(evaluation_ctx->postactivation_outputs);
}

void tinynn_evaluate(struct tinynn_evaluation_ctx_t* evaluation_ctx, const float* inputs, float* outputs) {
    const struct tinynn_network_t* network = evaluation_ctx->network;

    const float* prev_layer_postactivation = inputs;
    float* this_layer_preactivation = evaluation_ctx->preactivation_outputs;
    float* this_layer_postactivation = evaluation_ctx->postactivation_outputs;

    float* layer_biases = network->biases;
    float* layer_weights = network->weights;

    uint32_t prev_layer_size = network->layout.input_node_count;
    for (uint32_t l = 0; l < network->layout.layer_count; l++) {
        const struct tinynn_layer_t* layer = &network->layout.layers[l];
        uint32_t this_layer_size = layer->node_count;

        // Initialize with biases
        for (uint32_t i = 0; i < this_layer_size; i++) {
            this_layer_preactivation[i] = *(layer_biases++);
        }

        // Accumulate weighted input from previous layer
        for (uint32_t i = 0; i < prev_layer_size; i++) {
            float input = prev_layer_postactivation[i];
            for (uint32_t j = 0; j < this_layer_size; j++) {
                this_layer_preactivation[j] += *(layer_weights++) * input;
            }
        }

        // Apply activation function
        layer->activation->eval_elementwise(this_layer_size, this_layer_preactivation, this_layer_postactivation);

        prev_layer_postactivation = this_layer_postactivation;
        this_layer_preactivation += this_layer_size;
        this_layer_postactivation += this_layer_size;
        prev_layer_size = this_layer_size;
    }

    if (outputs != NULL) {
        for (uint32_t i = 0; i < prev_layer_size; i++) {
            outputs[i] = prev_layer_postactivation[i];
        }
    }
}