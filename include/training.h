#pragma once

#include <stdint.h>
#include "../include/network.h"
#include "../include/evaluation.h"
#include "../include/cost.h"

struct tinynn_training_params_t {
    struct tinynn_cost_fn_t cost;
    float learning_rate;
    float regularization_factor;
};

struct tinynn_training_ctx_t {
    struct tinynn_evaluation_ctx_t evaluation_ctx;
    float* bias_gradients;
    float* weight_gradients;
    float* batch_bias_gradients;
    float* batch_weight_gradients;
    float* delta;
};

void tinynn_create_training_ctx(struct tinynn_training_ctx_t* training_ctx, const struct tinynn_network_t* network);
void tinynn_destroy_training_ctx(struct tinynn_training_ctx_t* training_ctx);
void tinynn_train(
    struct tinynn_training_ctx_t* training_ctx,
    struct tinynn_training_params_t training_params,
    uint32_t example_count,
    float* example_inputs,
    float* target_outputs,
    uint32_t batch_size,
    uint32_t epochs
);