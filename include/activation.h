#pragma once

#include <stdint.h>

struct tinynn_activation_t {
    void (*eval_elementwise)(uint32_t element_count, const float* x, float* y);
    void (*compute_derivative_elementwise)(uint32_t element_count, const float* x, float* y);
};

extern const struct tinynn_activation_t TINYNN_ACTIVATION_SIGMOID;