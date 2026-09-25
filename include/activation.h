#pragma once

#include <stdint.h>

struct tinynn_activation_t {
    void (*eval)(uint32_t element_count, const float* x, float* y);
    void (*backpropagate)(uint32_t element_count, const float* x, const float* y, const float* grad_out, float* grad_in);
};

extern const struct tinynn_activation_t TINYNN_ACTIVATION_SIGMOID;