#pragma once

#include "../include/network.h"

struct tinynn_evaluation_ctx_t {
    const struct tinynn_network_t* network;
    float* preactivation_outputs;
    float* postactivation_outputs;
};

void tinynn_create_evaluation_ctx(struct tinynn_evaluation_ctx_t* evaluation_ctx, const struct tinynn_network_t* network);
void tinynn_destroy_evaluation_ctx(struct tinynn_evaluation_ctx_t* evaluation_ctx);
void tinynn_evaluate(struct tinynn_evaluation_ctx_t* evaluation_ctx, const float* inputs, float* outputs);