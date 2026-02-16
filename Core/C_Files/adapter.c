#include "adapter.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int adapter_read_file(const char* path, char** out_source, size_t* out_size, char* err, size_t err_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        snprintf(err, err_size, "failed to open '%s': %s", path, strerror(errno));
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    *out_source = (char*)malloc((size_t)n + 1);
    if (!*out_source) {
        fclose(f);
        snprintf(err, err_size, "out of memory");
        return -1;
    }
    size_t r = fread(*out_source, 1, (size_t)n, f);
    fclose(f);
    (*out_source)[r] = '\0';
    *out_size = r;
    return 0;
}

int adapter_write_file(const char* path, const char* contents, char* err, size_t err_size) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        snprintf(err, err_size, "failed to write '%s': %s", path, strerror(errno));
        return -1;
    }
    if (fputs(contents, f) < 0) {
        fclose(f);
        snprintf(err, err_size, "write error for '%s'", path);
        return -1;
    }
    fclose(f);
    return 0;
}

AdapterResult adapter_run_command(const char* const argv[]) {
    AdapterResult result = {0, ""};
    pid_t pid = fork();
    if (pid < 0) {
        result.exit_code = -1;
        snprintf(result.stderr_msg, sizeof(result.stderr_msg), "fork failed");
        return result;
    }

    if (pid == 0) {
        execvp(argv[0], (char* const*)argv);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        result.exit_code = -1;
        snprintf(result.stderr_msg, sizeof(result.stderr_msg), "waitpid failed");
        return result;
    }

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else {
        result.exit_code = -1;
        snprintf(result.stderr_msg, sizeof(result.stderr_msg), "subprocess terminated unexpectedly");
    }
    return result;
}
