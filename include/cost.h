#pragma once

#include <stdint.h>

struct tinynn_cost_fn_t {
    float (*eval)(uint32_t element_count, const float* result, const float* target);
    void (*eval_gradient)(uint32_t element_count, const float* result, const float* target, float* gradient);
};

extern const struct tinynn_cost_fn_t TINYNN_COST_QUADRATIC;