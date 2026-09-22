#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../include/network.h"
#include "../include/training.h"
#include "../include/evaluation.h"

static void backpropagate(
    struct tinynn_training_ctx_t* training_ctx,
    struct tinynn_training_params_t training_params,
    const float* input,
    const float* target
) {
    const struct tinynn_network_t* network = training_ctx->evaluation_ctx.network;
    struct tinynn_evaluation_ctx_t* evaluation_ctx = &training_ctx->evaluation_ctx;
    tinynn_evaluate(evaluation_ctx, input, NULL);

    const struct tinynn_layer_t* this_layer = &network->layout.layers[network->layout.layer_count - 1];
    uint32_t this_layer_size = this_layer->node_count;
    uint32_t last_layer_first_node_offset = network->bias_count - this_layer_size;
    float* this_layer_preactivation = evaluation_ctx->preactivation_outputs + last_layer_first_node_offset;
    float* this_layer_postactivation = evaluation_ctx->postactivation_outputs + last_layer_first_node_offset;
    float* this_layer_bias_gradients = training_ctx->bias_gradients + last_layer_first_node_offset;
    float* next_layer_weights = network->weights + network->weight_count;
    float* next_layer_weight_gradients = training_ctx->weight_gradients + network->weight_count;

    // compute bias gradients for last layer
    training_params.cost.eval_gradient(this_layer_size, this_layer_postactivation, target, this_layer_bias_gradients);
    this_layer->activation->map_derivative(this_layer_size, this_layer_preactivation, this_layer_preactivation);
    for (uint32_t i = 0; i < this_layer_size; i++) {
        this_layer_bias_gradients[i] *= this_layer_preactivation[i];
    }

    // iterate backward over remaining layers
    for (uint32_t l = 0; l < network->layout.layer_count - 1; l++) {
        uint32_t next_layer_size = this_layer_size;
        float* next_layer_bias_gradients = this_layer_bias_gradients;

        this_layer--;
        this_layer_size = this_layer->node_count;
        this_layer_preactivation -= this_layer_size;
        this_layer_postactivation -= this_layer_size;
        this_layer_bias_gradients -= this_layer_size;

        uint32_t next_layer_weight_count = this_layer_size * next_layer_size;
        next_layer_weights -= next_layer_weight_count;
        next_layer_weight_gradients -= next_layer_weight_count;

        // compute weight gradients for the layer we last computed bias gradients for
        float* weight_gradients = next_layer_weight_gradients;
        for (uint32_t i = 0; i < next_layer_size; i++) {
            for (uint32_t j = 0; j < this_layer_size; j++) {
                *(weight_gradients++) = this_layer_postactivation[j] * next_layer_bias_gradients[i];
            }
        }

        // compute bias gradients for the next layer back
        memset(this_layer_bias_gradients, 0, this_layer_size * sizeof(float));
        float* weights = next_layer_weights;
        for (uint32_t n = 0; n < next_layer_size; n++) {
            for (uint32_t i = 0; i < this_layer_size; i++) {
                this_layer_bias_gradients[i] += *(weights++) * next_layer_bias_gradients[n];
            }
        }

        this_layer->activation->map_derivative(this_layer_size, this_layer_preactivation, this_layer_preactivation);
        for (uint32_t i = 0; i < this_layer_size; i++) {
            this_layer_bias_gradients[i] *= this_layer_preactivation[i];
        }
    }

    // compute weight gradients for first layer
    float* weight_gradients = training_ctx->weight_gradients;
    for (uint32_t i = 0; i < this_layer_size; i++) {
        for (uint32_t j = 0; j < network->layout.input_node_count; j++) {
            *(weight_gradients++) = input[j] * this_layer_bias_gradients[i];
        }
    }
}

void tinynn_create_training_ctx(struct tinynn_training_ctx_t* training_ctx, const struct tinynn_network_t* network) {
    tinynn_create_evaluation_ctx(&training_ctx->evaluation_ctx, network);
    training_ctx->bias_gradients = (float*)malloc(network->bias_count * sizeof(float));
    training_ctx->weight_gradients = (float*)malloc(network->weight_count * sizeof(float));
}

void tinynn_destroy_training_ctx(struct tinynn_training_ctx_t* training_ctx) {
    tinynn_destroy_evaluation_ctx(&training_ctx->evaluation_ctx);
    free(training_ctx->bias_gradients);
    free(training_ctx->weight_gradients);
}

void tinynn_train(
    struct tinynn_training_ctx_t* training_ctx,
    struct tinynn_training_params_t training_params,
    uint32_t example_count,
    const float* example_inputs,
    const float* target_outputs,
    uint32_t epochs,
    int monitor_accuracy
) {
    const struct tinynn_network_t* network = training_ctx->evaluation_ctx.network;

    uint32_t input_node_count = network->layout.input_node_count;
    uint32_t output_node_count = network->layout.layers[network->layout.layer_count - 1].node_count;
    uint32_t last_layer_first_node_offset = network->bias_count - output_node_count;
    float* output = training_ctx->evaluation_ctx.postactivation_outputs + last_layer_first_node_offset;
    float step_factor = training_params.learning_rate / (float)example_count;

    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
        const float* input = example_inputs;
        const float* target = target_outputs;
        float total_cost = 0.0f;
        for (uint32_t i = 0; i < example_count; i++) {
            backpropagate(training_ctx, training_params, input, target);
            if (monitor_accuracy) {
                total_cost += training_params.cost.eval(output_node_count, output, target);
            }

            for (uint32_t i = 0; i < network->bias_count; i++) {
                network->biases[i] -= training_ctx->bias_gradients[i] * step_factor;
            }

            for (uint32_t i = 0; i < network->weight_count; i++) {
                network->weights[i] -= training_ctx->weight_gradients[i] * step_factor;
            }

            input += input_node_count;
            target += output_node_count;
        }

        printf("Epoch %u of %u complete", epoch + 1, epochs);
        if (monitor_accuracy) {
            printf(", cost = %f", total_cost);
        }
        fputc('\n', stdout);
    }
}