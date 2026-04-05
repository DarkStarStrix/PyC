# Phantom Graph Rationale

This note explains why the phantom-graph feature is worth keeping in PyC even when it does not directly improve peak throughput.

## Short Answer

The main value of phantom graph is runtime control, not raw speed.

If it helps PyC:

- avoid failures when workload shape families drift,
- reduce needless fallback or guard-miss churn,
- keep the GPU hot path on a prepared execution family,
- and give the runtime better signals about what is coming next,

then it is justified.

That is the correct bar for this feature. FLOPS wins are optional.

## Problem With The Old Path

The old no-speculative path is purely reactive.

It waits for the actual runtime inputs, then decides what to do after the fact:

- shape matches: run
- shape mismatch: miss the current path
- unsupported case: fall back or fail

That is simple, but it gives the runtime almost no predictive control surface.

The result is that:

- the planner has weaker information,
- the runtime discovers drift late,
- allocator and kernel decisions are more myopic,
- graph replay and workspace reuse are more fragile,
- and debugging workload instability is harder.

## What Phantom Graph Adds

Phantom graph gives the runtime an expected graph family and a way to compare it against reality.

In the current PyC design that means:

- expected shape bucket
- expected shape signature
- expected graph skeleton fingerprint
- expected confidence
- mismatch counters
- reshape counters

The runtime can then classify each run as:

- exact match
- family match
- drift
- failure mismatch

That is better than the old path because it turns runtime execution into:

1. expect
2. compare
3. classify
4. adapt safely

instead of only:

1. run
2. react

## Ranked Reasons To Keep It

### 1. Control stability and failure avoidance

This is the strongest reason.

Phantom graph is useful because it helps the runtime absorb shape-family drift without immediately falling into guard misses or hard planner churn. Even if the current run still succeeds without it, the predictive state reduces the chance that the next run lands on the wrong path cold.

Why it is better than the old path:

- old path reacts only after mismatch
- phantom graph notices drift as part of the control loop
- phantom graph can reshape expectation after successful runs instead of treating every change as unrelated

### 2. Better runtime planning

Memory reuse, workspace sizing, graph replay eligibility, and execution-family selection all benefit from knowing what the runtime expects to happen next.

Why it is better than the old path:

- old path plans from the current run only
- phantom graph gives a short-horizon prediction surface
- that prediction can inform future planning even when immediate throughput is unchanged

### 3. Lower latency variance

A system can be fast on average and still be operationally bad if it thrashes between paths or behaves inconsistently under drifting shapes.

Phantom graph helps reduce that by making adaptation more structured.

Why it is better than the old path:

- fewer cold misses
- fewer abrupt path switches
- better chance of stable reuse behavior

### 4. Faster recovery from workload drift

The feature is not about perfect prediction. It is about cheap recovery when the workload moves but stays near a known family.

Why it is better than the old path:

- old path treats drift as something to survive
- phantom graph treats drift as something to learn from

### 5. Better observability

Phantom telemetry gives a direct signal about workload stability:

- how often the graph matched expectation
- how often it drifted
- how often the runtime reshaped its expectation
- what the runtime expected versus what it saw

Why it is better than the old path:

- old path mostly exposes failures, misses, and slowdowns
- phantom graph exposes the structure of runtime drift itself

### 6. Better future controller inputs

If PyC grows a stronger runtime controller, phantom graph gives it better state than pure post-hoc counters.

Why it is better than the old path:

- old path has weak predictive signal
- phantom graph gives the controller a measure of graph stability and drift

### 7. Energy and wasted-work reduction

This is a secondary justification, but still real.

Avoiding repeated cold-path behavior, poor planning choices, and needless fallback work can reduce wasted GPU and CPU effort even if the final model throughput looks similar.

### 8. Foundation for future runtime ideas

Phantom graph is a useful precursor for:

- true what-if simulation
- shape-clustered plan reuse
- guarded prewarming
- controller-led plan switching
- kernel plus allocator co-selection informed by predicted workload family

Without a predictive graph concept, those ideas are much harder to structure cleanly.

## Practical Success Criteria

For PyC, phantom graph is worth it if it can do some combination of:

- reduce guard misses
- reduce fallback churn
- improve graph replay or workspace reuse stability
- improve planning decisions under drifting shapes
- surface useful runtime drift telemetry
- make long-lived services more robust under non-static shape families

It does not need to be a direct throughput feature to be valuable.

## Current PyC Position

In the current repository, phantom graph should be treated as:

- an experimental runtime-control feature,
- a reliability and observability layer first,
- an optimization surface second.

That matches the current implementation and the current evidence.

The strongest present-case justification is:

phantom graph helps the runtime stay safer and more informed when the computation graph drifts during real execution.

That is enough to justify keeping and extending it in this repo.
