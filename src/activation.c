#include <stdint.h>
#include <string.h>
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

static void eval_softmax(uint32_t element_count, const float* x, float* y) {
    float total = 0.0f;
    for (uint32_t i = 0; i < element_count; i++) {
        total += (y[i] = umath_exp(x[i]));
    }

    for (uint32_t i = 0; i < element_count; i++) {
        y[i] /= total;
    }
}

static void backpropagate_softmax(uint32_t element_count, const float* x, const float* y, const float* grad_out, float* grad_in) {
    (void)x;
    memset(grad_in, 0, element_count * sizeof(float));
    for (uint32_t i = 0; i < element_count; i++) {
        for (uint32_t j = 0; j < element_count; j++) {
            float dai_dzj = (i == j ? 1.0f - y[j] : -y[j]) * y[i];
            grad_in[j] += dai_dzj * grad_out[i];
        }
    }
}

const struct tinynn_activation_t TINYNN_ACTIVATION_SIGMOID = {
    .eval = eval_sigmoid,
    .backpropagate = backpropagate_sigmoid
};

const struct tinynn_activation_t TINYNN_ACTIVATION_SOFTMAX = {
    .eval = eval_softmax,
    .backpropagate = backpropagate_softmax
};