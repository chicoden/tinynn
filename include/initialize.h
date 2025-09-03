#pragma once

#include <stdint.h>
#include "../include/network.h"

void tinynn_init_params_random(struct tinynn_network_t* network, uint32_t seed);
void tinynn_init_params_random_normalized(struct tinynn_network_t* network, uint32_t seed);