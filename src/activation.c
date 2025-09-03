#include <stdint.h>
#include "../include/umath.h"
#include "../include/activation.h"

static float sigmoid(float x) {
    return 1.0f / (1.0f + umath_exp(-x));
}

static float sigmoid_derivative(float x) {
    float y = sigmoid(x);
    return y * (1.0f - y);
}

static void elementwise_sigmoid(uint32_t element_count, const float* x, float* y) {
    for (uint32_t i = 0; i < element_count; i++) {
        y[i] = sigmoid(x[i]);
    }
}

static void elementwise_sigmoid_derivative(uint32_t element_count, const float* x, float* y) {
    for (uint32_t i = 0; i < element_count; i++) {
        y[i] = sigmoid_derivative(x[i]);
    }
}

const struct tinynn_activation_t TINYNN_ACTIVATION_SIGMOID = {
    .eval_elementwise = elementwise_sigmoid,
    .compute_derivative_elementwise = elementwise_sigmoid_derivative
};