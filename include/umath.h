#pragma once

#include <stdint.h>

static const float UMATH_LOGE_2 = 0.6931471824645996f;
static const float UMATH_RECIP_LOGE_2 = 1.4426950216293335f;
static const float UMATH_RECIP_SQRT_2 = 0.7071067690849304f;
static const float UMATH_PI = 3.1415926535897932f;
static const float UMATH_HALF_PI = 1.5707963267948966f;
static const float UMATH_RECIP_PI = 0.3183098861837907f;

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

static int umath_is_odd(float x) {
    union { float value; uint32_t bits; } view = { .value = x };
    int32_t exponent = umath_extract_exponent(x);
    if (exponent < 0) {
        return 0;
    } else if (exponent == 0) {
        return 1;
    } else if (exponent <= 23) {
        return (view.bits >> (23 - exponent)) & 1;
    } else {
        return 0;
    }
}

static float umath_trunc(float x) {
    union { float value; uint32_t bits; } view = { .value = x };
    int32_t exponent = umath_extract_exponent(x);
    if (exponent < 0) {
        return 0.0f;
    } else {
        if (exponent < 23) {
            uint32_t bits_to_clear = 23 - exponent;
            view.bits = view.bits >> bits_to_clear << bits_to_clear;
        }
        return view.value;
    }
}

static float umath_floor(float x) {
    float whole_part = umath_trunc(x);
    float frac = x - whole_part;
    if (x < 0.0f) {
        return frac != 0.0f ? whole_part - 1.0f : whole_part;
    } else {
        return whole_part;
    }
}

static float umath_ceil(float x) {
    float whole_part = umath_trunc(x);
    float frac = x - whole_part;
    if (x < 0.0f) {
        return whole_part;
    } else {
        return frac != 0.0f ? whole_part + 1.0f : whole_part;
    }
}

static float umath_round(float x) {
    float whole_part = umath_trunc(x);
    float frac = x - whole_part;
    if (x < 0.0f) {
        return frac > -0.5f ? whole_part : whole_part - 1.0f;
    } else {
        return frac < 0.5f ? whole_part : whole_part + 1.0f;
    }
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

static float umath_sin(float x) {
    float i = umath_round(x * UMATH_RECIP_PI);
    x -= UMATH_PI * i;

    float y = x;
    float xx = x * x;
    for (uint32_t n = 11; n > 1; n -= 2) {
        y = x - (1.0f/((n-1)*n)) * xx * y;
    }

    if (umath_is_odd(i)) y = -y;
    return y;
}

static float umath_cos(float x) {
    float i = umath_round(x * UMATH_RECIP_PI);
    x -= UMATH_PI * i;

    float y = 1.0f;
    float xx = x * x;
    for (uint32_t n = 10; n > 0; n -= 2) {
        y = 1.0f - (1.0f/((n-1)*n)) * xx * y;
    }

    if (umath_is_odd(i)) y = -y;
    return y;
}