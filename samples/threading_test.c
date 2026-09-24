#include <stdio.h>
#include <stdint.h>
#include "../include/threading.h"

#define COUNTOF(x) (sizeof(x) / sizeof(x[0]))

struct thread_params_t {
    barrier_t* barrier;
    int data;
};

thread_routine_decl(my_routine) {
    struct thread_params_t* params = (struct thread_params_t*)thread_get_payload();
    enter_barrier(params->barrier);
    printf("%i\n", params->data);
    thread_exit();
}

int main() {
    struct thread_params_t params[10];
    barrier_t barrier;
    if (!create_barrier(&barrier, COUNTOF(params) + 1)) {
        printf("failed to create barrier\n");
    }

    thread_t threads[COUNTOF(params)];
    for (uint32_t i = 0; i < COUNTOF(params); i++) {
        params[i].barrier = &barrier;
        params[i].data = i;
        if (!create_thread(my_routine, &params[i], &threads[i])) {
            printf("failed to create a thread\n");
        }
    }

    Sleep(3000);
    enter_barrier(&barrier);

    for (uint32_t i = 0; i < COUNTOF(params); i++) {
        join_thread(threads[i]);
    }

    destroy_barrier(&barrier);
    return 0;
}