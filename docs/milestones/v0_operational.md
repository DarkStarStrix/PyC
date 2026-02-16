# Milestone: v0 operational

Acceptance criteria:

1. CLI command `build` is guaranteed working end-to-end.
2. `build` pipeline includes:
   - source loading
   - lexing
   - parsing
   - minimal semantic checks
   - LLVM IR text generation
   - object-like or JIT-like backend output artifact
3. Build failures return non-zero process code with explicit diagnostics.
4. `optimize`, `visualize`, and `kernel` remain feature-gated until contracts are complete.
5. Core API uses adapter layer functions for I/O/external command boundaries with explicit status codes.
