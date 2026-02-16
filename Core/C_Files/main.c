#include <stdio.h>
#include <string.h>

// Forward declarations for external functions from ir_generator.c
extern void generate_ir(ASTNode* ast_root);
extern void cleanup_ir_generator(void);

static void usage(const char* argv0) {
    printf("Usage: %s <command> <input> [-o <output>] [--jit]\n", argv0);
    printf("Commands: build, optimize, visualize, kernel\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const char* command = argv[1];
    const char* input = argv[2];
    const char* output = "a.o";
    BackendOutputMode mode = BACKEND_OBJECT;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output = argv[++i];
        else if (strcmp(argv[i], "--jit") == 0) mode = BACKEND_JIT;
    }

    ApiStatus status = API_STATUS_INVALID_ARGUMENT;
    if (strcmp(command, "build") == 0) {
        BuildConfig cfg = {.output_mode = mode, .output_path = output};
        status = build_script(input, &cfg);
    } else if (strcmp(command, "optimize") == 0) {
        status = optimize_script(input, 1);
    } else if (strcmp(command, "visualize") == 0) {
        status = visualize_graph(input);
    } else if (strcmp(command, "kernel") == 0) {
        status = register_kernel(input);
    } else {
        usage(argv[0]);
        return 1;
    }

    if (status == API_STATUS_FEATURE_DISABLED) {
        fprintf(stderr, "feature for '%s' is disabled at compile time\n", command);
        cleanup_api();
        return 2;
    }

    cleanup_api();
    return status == API_STATUS_OK ? 0 : 1;
}
