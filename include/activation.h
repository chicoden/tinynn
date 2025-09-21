#pragma once

#include <stdint.h>

struct tinynn_activation_t {
    void (*map)(uint32_t element_count, const float* x, float* y);
    void (*map_derivative)(uint32_t element_count, const float* x, float* y);
};

extern const struct tinynn_activation_t TINYNN_ACTIVATION_SIGMOID;