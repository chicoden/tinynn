#pragma once

#include <stdint.h>
#include <windows.h>

typedef HANDLE thread_t;
typedef DWORD WINAPI (*thread_routine_t)(LPVOID);

typedef SYNCHRONIZATION_BARRIER barrier_t;

#define thread_routine_decl(name) DWORD WINAPI name(LPVOID payload)
#define thread_get_payload() payload
#define thread_exit() ExitThread(0)

static int create_thread(thread_routine_t routine, void* payload, thread_t* thread) {
    *thread = CreateThread(NULL, 0, routine, payload, 0, NULL);
    return *thread != NULL;
}

static void join_thread(thread_t thread) {
    (void)WaitForSingleObject(thread, INFINITE);
    (void)CloseHandle(thread);
}

static int create_barrier(barrier_t* barrier, uint32_t thread_count) {
    return InitializeSynchronizationBarrier(barrier, thread_count, -1) == TRUE;
}

static void destroy_barrier(barrier_t* barrier) {
    (void)DeleteSynchronizationBarrier(barrier);
}

static void enter_barrier(barrier_t* barrier) {
    (void)EnterSynchronizationBarrier(barrier, 0);
}