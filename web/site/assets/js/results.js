import {
    byId,
    fetchJson,
    formatNumber,
    initThemeToggle,
    setImageWithFallback,
} from "./shared.js";

function renderAdapterRows(bodyId, rows) {
    const body = byId(bodyId);
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

function renderDistributedRows(runs) {
    const body = byId("dist-body");
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

function renderKpis(latest) {
    const grid = byId("kpi-grid");
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

async function loadResultsPage() {
    const runStatus = byId("run-status");
    try {
        const [summary, distributed] = await Promise.all([
            fetchJson("./results/latest-summary.json", { cache: "no-store" }),
            fetchJson("./results/distributed-latest.json", { cache: "no-store" }),
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
        setImageWithFallback(byId("latest-dist-summary"), distributed.latest?.published?.summary_svg ?? null, "./results/artifacts/images/latest_distributed_pipeline.svg");
        setImageWithFallback(byId("latest-dist-throughput"), distributed.visuals?.throughput_svg ?? null, "./results/artifacts/images/latest_distributed_throughput.svg");
        setImageWithFallback(byId("latest-dist-pipeline"), distributed.visuals?.pipeline_svg ?? null, "./results/artifacts/images/latest_distributed_pipeline.svg");
    } catch (_error) {
        if (runStatus) {
            runStatus.textContent = "Failed to load results payloads.";
        }
        renderAdapterRows("cpu-body", []);
        renderAdapterRows("gpu-body", []);
        renderDistributedRows([]);
    }
}

void (async function init() {
    initThemeToggle();
    await loadResultsPage();
})();
