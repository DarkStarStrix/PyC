#include "pyc/collective_comm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef struct pyc_comm_loader_entry {
    pyc_collective_comm* comm;
    pyc_comm_backend_destroy_fn destroy_fn;
#if defined(_WIN32)
    HMODULE module;
#else
    void* module;
#endif
    struct pyc_comm_loader_entry* next;
} pyc_comm_loader_entry;

static pyc_comm_loader_entry* g_loader_entries;
static char g_loader_error[256];

static void set_loader_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_loader_error, sizeof(g_loader_error), fmt, args);
    g_loader_error[sizeof(g_loader_error) - 1] = '\0';
    va_end(args);
}

static int validate_comm_interface(const pyc_collective_comm* comm) {
    if (!comm) {
        return 0;
    }
    return comm->all_reduce &&
        comm->all_gather &&
        comm->reduce_scatter &&
        comm->broadcast &&
        comm->send &&
        comm->recv &&
        comm->barrier;
}

const char* pyc_comm_loader_last_error(void) {
    return g_loader_error;
}

pyc_collective_comm* pyc_load_comm_backend(const char* backend_path, const char* config_json) {
    pyc_collective_comm* comm;
    pyc_comm_backend_create_fn create_fn;
    pyc_comm_backend_destroy_fn destroy_fn;
    pyc_comm_loader_entry* entry;

#if defined(_WIN32)
    HMODULE module;
#else
    void* module;
#endif

    if (!backend_path || backend_path[0] == '\0') {
        set_loader_error("backend path is required");
        return NULL;
    }

    g_loader_error[0] = '\0';

#if defined(_WIN32)
    module = LoadLibraryA(backend_path);
    if (!module) {
        set_loader_error("failed to load backend: %s", backend_path);
        return NULL;
    }
    create_fn = (pyc_comm_backend_create_fn)GetProcAddress(module, PYC_COMM_BACKEND_CREATE_SYMBOL);
    destroy_fn = (pyc_comm_backend_destroy_fn)GetProcAddress(module, PYC_COMM_BACKEND_DESTROY_SYMBOL);
#else
    module = dlopen(backend_path, RTLD_NOW);
    if (!module) {
        const char* dl_err = dlerror();
        set_loader_error("failed to load backend: %s (%s)", backend_path, dl_err ? dl_err : "unknown");
        return NULL;
    }
    create_fn = (pyc_comm_backend_create_fn)dlsym(module, PYC_COMM_BACKEND_CREATE_SYMBOL);
    destroy_fn = (pyc_comm_backend_destroy_fn)dlsym(module, PYC_COMM_BACKEND_DESTROY_SYMBOL);
#endif

    if (!create_fn || !destroy_fn) {
        set_loader_error("backend missing required symbols: %s", backend_path);
#if defined(_WIN32)
        FreeLibrary(module);
#else
        dlclose(module);
#endif
        return NULL;
    }

    comm = create_fn(config_json);
    if (!validate_comm_interface(comm)) {
        if (comm) {
            destroy_fn(comm);
        }
        set_loader_error("backend returned invalid comm interface: %s", backend_path);
#if defined(_WIN32)
        FreeLibrary(module);
#else
        dlclose(module);
#endif
        return NULL;
    }

    entry = (pyc_comm_loader_entry*)malloc(sizeof(*entry));
    if (!entry) {
        destroy_fn(comm);
        set_loader_error("out of memory while tracking backend: %s", backend_path);
#if defined(_WIN32)
        FreeLibrary(module);
#else
        dlclose(module);
#endif
        return NULL;
    }

    entry->comm = comm;
    entry->destroy_fn = destroy_fn;
    entry->module = module;
    entry->next = g_loader_entries;
    g_loader_entries = entry;
    return comm;
}

void pyc_unload_comm_backend(pyc_collective_comm* comm) {
    pyc_comm_loader_entry* prev = NULL;
    pyc_comm_loader_entry* cur = g_loader_entries;

    while (cur) {
        if (cur->comm == comm) {
            if (cur->destroy_fn) {
                cur->destroy_fn(comm);
            }
#if defined(_WIN32)
            if (cur->module) {
                FreeLibrary(cur->module);
            }
#else
            if (cur->module) {
                dlclose(cur->module);
            }
#endif
            if (prev) {
                prev->next = cur->next;
            } else {
                g_loader_entries = cur->next;
            }
            free(cur);
            return;
        }
        prev = cur;
        cur = cur->next;
    }
}
