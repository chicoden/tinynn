#include <stdint.h>
#include "../include/umath.h"
#include "../include/activation.h"

static void eval_sigmoid(uint32_t element_count, const float* x, float* y) {
    for (uint32_t i = 0; i < element_count; i++) {
        y[i] = 1.0f / (1.0f + umath_exp(-x[i]));
    }
}

static void backpropagate_sigmoid(uint32_t element_count, const float* x, const float* y, const float* grad_out, float* grad_in) {
    (void)x;
    for (uint32_t i = 0; i < element_count; i++) {
        grad_in[i] = grad_out[i] * y[i] * (1.0f - y[i]);
    }
}

const struct tinynn_activation_t TINYNN_ACTIVATION_SIGMOID = {
    .eval = eval_sigmoid,
    .backpropagate = backpropagate_sigmoid
};