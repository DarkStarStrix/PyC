import {
    byId,
    fetchJson,
    formatNumber,
    initThemeToggle,
    setImageWithFallback,
} from "./shared.js";

type AdapterRow = {
    adapter?: string;
    display_name?: string;
    mode?: string;
    mean_ms?: number;
    p50_ms?: number;
    p95_ms?: number;
    throughput_tokens_per_sec?: number;
};

type LatestSummary = {
    cpu?: {
        adapters?: AdapterRow[];
    };
    gpu?: {
        adapters?: AdapterRow[];
    };
};

type DistributedRun = {
    run_id?: string;
    world_size?: number;
    samples_per_sec?: number;
    tokens_per_sec?: number;
    gpu_util_mean?: number;
    compute_time_ms_mean?: number;
    comm_time_ms_mean?: number;
    idle_gap_ms_mean?: number;
    published?: {
        summary_svg?: string;
    };
};

type DistributedPayload = {
    latest?: DistributedRun;
    runs?: DistributedRun[];
    visuals?: {
        throughput_svg?: string;
        pipeline_svg?: string;
    };
};

function renderAdapterRows(bodyId: string, rows: AdapterRow[]): void {
    const body = byId<HTMLTableSectionElement>(bodyId);
    if (!body) {
        return;
    }

    body.innerHTML = "";
    if (rows.length === 0) {
        body.innerHTML = `<tr><td colspan="6">No rows available.</td></tr>`;
        return;
    }

    for (const rowData of rows) {
        const row = document.createElement("tr");
        const values = [
            rowData.display_name ?? rowData.adapter ?? "-",
            rowData.mode ?? "-",
            formatNumber(rowData.mean_ms, 2),
            formatNumber(rowData.p50_ms, 2),
            formatNumber(rowData.p95_ms, 4),
            formatNumber(rowData.throughput_tokens_per_sec, 4),
        ];
        for (const value of values) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        }
        body.appendChild(row);
    }
}

function renderDistributedRows(runs: DistributedRun[]): void {
    const body = byId<HTMLTableSectionElement>("dist-body");
    if (!body) {
        return;
    }

    body.innerHTML = "";
    if (runs.length === 0) {
        body.innerHTML = `<tr><td colspan="6">No rows available.</td></tr>`;
        return;
    }

    for (const run of runs) {
        const row = document.createElement("tr");
        const values = [
            run.run_id ?? "-",
            String(run.world_size ?? "-"),
            formatNumber(run.samples_per_sec, 2),
            formatNumber(run.tokens_per_sec, 2),
            formatNumber(run.comm_time_ms_mean, 4),
            formatNumber(run.idle_gap_ms_mean, 4),
        ];
        for (const value of values) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        }
        body.appendChild(row);
    }
}

function renderKpis(latest?: DistributedRun): void {
    const grid = byId<HTMLDivElement>("kpi-grid");
    if (!grid || !latest) {
        return;
    }

    grid.innerHTML =
        `<div class="kpi"><p class="label">Samples/s</p><p class="value">${formatNumber(latest.samples_per_sec, 2)}</p></div>` +
        `<div class="kpi"><p class="label">Tokens/s</p><p class="value">${formatNumber(latest.tokens_per_sec, 2)}</p></div>` +
        `<div class="kpi"><p class="label">GPU Util Mean</p><p class="value">${formatNumber(latest.gpu_util_mean, 2)}%</p></div>` +
        `<div class="kpi"><p class="label">Compute (ms)</p><p class="value">${formatNumber(latest.compute_time_ms_mean, 4)}</p></div>` +
        `<div class="kpi"><p class="label">Comm (ms)</p><p class="value">${formatNumber(latest.comm_time_ms_mean, 4)}</p></div>` +
        `<div class="kpi"><p class="label">Idle (ms)</p><p class="value">${formatNumber(latest.idle_gap_ms_mean, 4)}</p></div>`;
}

async function loadResultsPage(): Promise<void> {
    const runStatus = byId<HTMLParagraphElement>("run-status");

    try {
        const [summary, distributed] = await Promise.all([
            fetchJson<LatestSummary>("./results/latest-summary.json", { cache: "no-store" }),
            fetchJson<DistributedPayload>("./results/distributed-latest.json", { cache: "no-store" }),
        ]);

        renderAdapterRows("cpu-body", summary.cpu?.adapters ?? []);
        renderAdapterRows("gpu-body", summary.gpu?.adapters ?? []);
        renderDistributedRows(distributed.runs ?? []);
        renderKpis(distributed.latest);

        if (runStatus) {
            const latest = distributed.latest;
            runStatus.textContent =
                `Latest distributed run: ${latest?.run_id ?? "unknown"} | world size: ${latest?.world_size ?? "-"} | ` +
                `tokens/s: ${formatNumber(latest?.tokens_per_sec, 2)} | comm ms: ${formatNumber(latest?.comm_time_ms_mean, 4)}`;
        }

        setImageWithFallback(
            byId<HTMLImageElement>("latest-dist-summary"),
            distributed.latest?.published?.summary_svg ?? null,
            "./results/artifacts/images/latest_distributed_pipeline.svg"
        );
        setImageWithFallback(
            byId<HTMLImageElement>("latest-dist-throughput"),
            distributed.visuals?.throughput_svg ?? null,
            "./results/artifacts/images/latest_distributed_throughput.svg"
        );
        setImageWithFallback(
            byId<HTMLImageElement>("latest-dist-pipeline"),
            distributed.visuals?.pipeline_svg ?? null,
            "./results/artifacts/images/latest_distributed_pipeline.svg"
        );
    } catch (_error) {
        if (runStatus) {
            runStatus.textContent = "Failed to load results payloads.";
        }
        renderAdapterRows("cpu-body", []);
        renderAdapterRows("gpu-body", []);
        renderDistributedRows([]);
    }
}

void (async function init(): Promise<void> {
    initThemeToggle();
    await loadResultsPage();
})();
