#include <stdint.h>
#include "../include/cost.h"

static float quadratic_cost(uint32_t element_count, const float* result, const float* target) {
    float cost = 0.0f;
    for (uint32_t i = 0; i < element_count; i++) {
        float error = result[i] - target[i];
        cost += error * error;
    }

    return 0.5f * cost;
}

static void quadratic_cost_gradient(uint32_t element_count, const float* result, const float* target, float* gradient) {
    for (uint32_t i = 0; i < element_count; i++) {
        gradient[i] = result[i] - target[i];
    }
}

const struct tinynn_cost_fn_t TINYNN_COST_QUADRATIC = {
    .eval = quadratic_cost,
    .eval_gradient = quadratic_cost_gradient
};