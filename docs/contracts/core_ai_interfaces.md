# Core Compiler IR ↔ AI Graph Module Interfaces

This document defines stable contracts between the core compiler pipeline and AI modules.

## 1) Graph extraction format

The canonical C types live in `Core/Header_Files/ai_integration.h`:
- `GraphExtractionNode`
- `GraphExtractionPayload`

Contract:
- Core compiler emits one `GraphExtractionNode` per lowered IR op.
- `op_name` uses lower-case op IDs (`add`, `matmul`, `relu`, ...).
- `input_ids` and `output_id` are SSA-like tensor IDs.
- Nodes are topologically sorted.

## 2) Memory planning input/output

Canonical C types:
- `MemoryPlanningInput`
- `MemoryPlanningOutput`

Contract:
- Input contains tensor lifetime (`first_use_index`, `last_use_index`) and size in bytes.
- Output returns assigned `offset` in a contiguous arena.
- Planner must preserve tensor size and ID identity.

## 3) Optimizer pass registration

Canonical C types:
- `AIPassFn`
- `AIPassRegistration`

Contract:
- Each pass registers a unique `pass_name`.
- Passes receive immutable graph payload and write error details into `(err, err_size)`.
- Non-zero return code indicates pass failure and aborts execution chain.
