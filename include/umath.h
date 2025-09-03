#pragma once

#include <stdint.h>

static const float UMATH_LOGE_2 = 0.6931471824645996f;
static const float UMATH_RECIP_LOGE_2 = 1.4426950216293335f;
static const float UMATH_RECIP_SQRT_2 = 0.7071067690849304f;

static int32_t umath_extract_exponent(float x) {
    union { float value; uint32_t bits; } view = { .value = x };
    return (int32_t)((view.bits >> 23) & 0xFF) - 127;
}

static float umath_set_exponent(float x, int32_t exponent) {
    exponent += 127;
    if (exponent < 0) exponent = 0;
    if (exponent > 255) exponent = 255;
    union { float value; uint32_t bits; } view = { .value = x };
    view.bits = (view.bits & ~(0xFF << 23)) | (exponent << 23);
    return view.value;
}

static float umath_recip_sqrt(float x) {
    static const uint32_t N = 8;

    if (x <= 0.0f) return 0.0f;
    int32_t exponent = umath_extract_exponent(x);
    x = umath_set_exponent(x, 0);

    float y = -0.225577937896f * x + 1.15044748327f;
    for (uint32_t n = 0; n < N; n++) {
        y = -0.5f * x * y * y + y + 0.5f;
    }

    y = umath_set_exponent(y, umath_extract_exponent(y) - (exponent & ~1) / 2);
    if (exponent & 1) y *= UMATH_RECIP_SQRT_2;
    return y;
}

static float umath_exp(float x) {
    static const float A[] = {1.0f/30240, 1.0f/1008, 1.0f/72, 1.0f/9, 1.0f/2, 1.0f};

    int32_t shift = (int32_t)(x * UMATH_RECIP_LOGE_2);
    x -= UMATH_LOGE_2 * (float)shift;

    float u = (((( A[0] * x + A[1]) * x +  A[2]) * x + A[3]) * x +  A[4]) * x + A[5];
    float v = ((((-A[0] * x + A[1]) * x + -A[2]) * x + A[3]) * x + -A[4]) * x + A[5];
    float y = u / v;

    y = umath_set_exponent(y, umath_extract_exponent(y) + shift);
    return y;
}

static float umath_ln(float x) {
    static const uint32_t N = 8;

    if (x <= 0.0f) return 0.0f;
    int32_t exponent = umath_extract_exponent(x);
    x = umath_set_exponent(x, 0);

    float t = 1.0f - 1.0f / x;
    float y = (1.0f/N) * t;
    for (uint32_t n = N - 1; n > 0; n--) {
        y = (y + 1.0f/n) * t;
    }

    y += UMATH_LOGE_2 * (float)exponent;
    return y;
}