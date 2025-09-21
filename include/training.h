/*#pragma once

#include "../include/network.h"
#include "../include/evaluation.h"
#include "../include/cost.h"

struct tinynn_training_params_t {
    struct tinynn_cost_fn_t cost;
    void (*regularization_fn)(uint32_t weight_count, float* weights, float regularization_rate);
    float learning_rate;
    float regularization_rate;
};

struct tinynn_training_ctx_t {
    struct tinynn_evaluation_ctx_t evaluation_ctx;
    struct tinynn_training_params_t training_params;
    float* bias_gradients;
    float* weight_gradients;
};*/