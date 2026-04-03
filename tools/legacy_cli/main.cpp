#include "CLI11.hpp"
#include "api.h"
#include "error_handler.h"
#include "graph_compiler.h"
#include <iostream>
#include <string>
#include <memory>
#include <filesystem>

class CompilerContext {
private:
    bool verbose = false;
    std::string output_file = "a.out";
    bool enable_optimizations = false;

public:
    void set_verbose(bool v) { verbose = v; }
    void set_output_file(const std::string& out) { output_file = out; }
    void set_optimizations(bool opt) { enable_optimizations = opt; }

    void log(const std::string& message) const {
        if (verbose) {
            std::cout << "[INFO] " << message << std::endl;
        }
    }
};

class CompilerDriver {
private:
    std::unique_ptr<CompilerContext> context;

public:
    CompilerDriver() : context(std::make_unique<CompilerContext>()) {}

    void compile_file(const std::string& filename) {
        context->log("Starting compilation of " + filename);

        if (!std::filesystem::exists(filename)) {
            std::cerr << "Error: File '" << filename << "' not found" << std::endl;
            return;
        }

        try {
            compile_script(filename.c_str());
            context->log("Compilation completed successfully");
        }
        catch (const std::exception& e) {
            std::cerr << "Compilation failed: " << e.what() << std::endl;
        }
    }

    void optimize_file(const std::string& filename, bool graph_opt) {
        context->log("Optimizing " + filename + (graph_opt ? " with graph optimizations" : ""));

        try {
            optimize_script(filename.c_str(), graph_opt);
            context->log("Optimization completed successfully");
        }
        catch (const std::exception& e) {
            std::cerr << "Optimization failed: " << e.what() << std::endl;
        }
    }

    void visualize_file(const std::string& filename) {
        context->log("Generating visualization for " + filename);

        try {
            visualize_graph(filename.c_str());
            context->log("Visualization generated as 'graph.png'");
        }
        catch (const std::exception& e) {
            std::cerr << "Visualization failed: " << e.what() << std::endl;
        }
    }

    void run_file(const std::string& filename) {
        context->log("Executing " + filename);

        try {
            run_script(filename.c_str());
            context->log("Execution completed successfully");
        }
        catch (const std::exception& e) {
            std::cerr << "Execution failed: " << e.what() << std::endl;
        }
    }

    void register_kernel(const std::string& kernel_file) {
        context->log("Registering CUDA kernel from " + kernel_file);

        if (!std::filesystem::exists(kernel_file)) {
            std::cerr << "Error: Kernel file '" << kernel_file << "' not found" << std::endl;
            return;
        }

        try {
            register_kernel(kernel_file.c_str());
            context->log("Kernel registration successful");
        }
        catch (const std::exception& e) {
            std::cerr << "Kernel registration failed: " << e.what() << std::endl;
        }
    }

    void set_context_options(bool verbose, const std::string& output_file, bool optimizations) {
        context->set_verbose(verbose);
        context->set_output_file(output_file);
        context->set_optimizations(optimizations);
    }
};

int main(int argc, char** argv) {
    CLI::App app{"PyC Compiler - Python to CUDA Compiler"};
    CompilerDriver driver;

    // Global options
    bool verbose = false;
    std::string output_file = "a.out";
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");
    app.add_option("-o,--output", output_file, "Output file name");

    // Build command
    auto* build_cmd = app.add_subcommand("build", "Compile and optimize a PyC script");
    std::string build_file;
    bool build_optimize = false;
    build_cmd->add_option("file", build_file, "Input PyC script (*.pc)")->required();
    build_cmd->add_flag("-O,--optimize", build_optimize, "Enable optimizations");

    // Optimize command
    auto* optimize_cmd = app.add_subcommand("optimize", "Apply optimizations to a PyC script");
    std::string optimize_file;
    bool graph_flag = false;
    optimize_cmd->add_option("file", optimize_file, "Input PyC script (*.pc)")->required();
    optimize_cmd->add_flag("--graph", graph_flag, "Apply graph-based optimizations");

    // Visualize command
    auto* visualize_cmd = app.add_subcommand("visualize", "Generate computational graph visualization");
    std::string visualize_file;
    visualize_cmd->add_option("file", visualize_file, "Input PyC script (*.pc)")->required();

    // Run command
    auto* run_cmd = app.add_subcommand("run", "Execute optimized PyC script");
    std::string run_file;
    run_cmd->add_option("file", run_file, "Input PyC script (*.pc)")->required();

    // Kernel command
    auto* kernel_cmd = app.add_subcommand("kernel", "Manage CUDA kernels");
    auto* register_cmd = kernel_cmd->add_subcommand("register", "Register CUDA kernel");
    std::string kernel_file;
    register_cmd->add_option("file", kernel_file, "CUDA kernel file (*.cu)")->required();

    CLI11_PARSE(app, argc, argv);

    // Set global context options
    driver.set_context_options(verbose, output_file, build_optimize);

    try {
        if (*build_cmd) {
            driver.compile_file(build_file);
        } else if (*optimize_cmd) {
            driver.optimize_file(optimize_file, graph_flag);
        } else if (*visualize_cmd) {
            driver.visualize_file(visualize_file);
        } else if (*run_cmd) {
            driver.run_file(run_file);
        } else if (*register_cmd) {
            driver.register_kernel(kernel_file);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    cleanup_api();
    return 0;
}
