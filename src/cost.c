#include <stdint.h>
#include "../include/cost.h"
#include "../include/umath.h"

static float quadratic_cost(uint32_t element_count, const float* result, const float* target) {
    float cost = 0.0f;
    for (uint32_t i = 0; i < element_count; i++) {
        float error = result[i] - target[i];
        cost += error * error;
    }

    return 0.5f * cost;
}

static void quadratic_cost_gradient(
    uint32_t element_count,
    const float* result,
    const float* target,
    float* gradient
) {
    for (uint32_t i = 0; i < element_count; i++) {
        gradient[i] = result[i] - target[i];
    }
}

static float cross_entropy_cost(uint32_t element_count, const float* result, const float* target) {
    float cost = 0.0f;
    for (uint32_t i = 0; i < element_count; i++) {
        float a = result[i];
        float y = target[i];
        cost += y * umath_ln(a) + (1.0f - y) * umath_ln(1.0f - a);
    }

    return -cost;
}

static void cross_entropy_cost_gradient(
    uint32_t element_count,
    const float* result,
    const float* target,
    float* gradient
) {
    for (uint32_t i = 0; i < element_count; i++) {
        float a = result[i];
        float y = target[i];
        gradient[i] = (a - y) / (a * (1.0f - a));
    }
}

const struct tinynn_cost_fn_t TINYNN_COST_QUADRATIC = {
    .eval = quadratic_cost,
    .eval_gradient = quadratic_cost_gradient
};

const struct tinynn_cost_fn_t TINYNN_COST_CROSS_ENTROPY = {
    .eval = cross_entropy_cost,
    .eval_gradient = cross_entropy_cost_gradient
};