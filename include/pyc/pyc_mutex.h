#ifndef PYC_MUTEX_H
#define PYC_MUTEX_H

/* Minimal portable mutex used to guard the engine's process-global mutable
 * state (compile cache, kernel symbol table, CUDA workspace, comm-backend
 * loader registry). Statically initializable so file-scope globals need no
 * runtime init step. */

#if defined(_WIN32)
#include <windows.h>

typedef SRWLOCK pyc_mutex;
#define PYC_MUTEX_INIT SRWLOCK_INIT

static __inline void pyc_mutex_lock(pyc_mutex* m) {
    AcquireSRWLockExclusive(m);
}
static __inline void pyc_mutex_unlock(pyc_mutex* m) {
    ReleaseSRWLockExclusive(m);
}

#else
#include <pthread.h>

typedef pthread_mutex_t pyc_mutex;
#define PYC_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER

static inline void pyc_mutex_lock(pyc_mutex* m) {
    pthread_mutex_lock(m);
}
static inline void pyc_mutex_unlock(pyc_mutex* m) {
    pthread_mutex_unlock(m);
}

#endif

#endif /* PYC_MUTEX_H */
