import {
    byId,
    fetchJson,
    fetchJsonWithFallback,
    formatNumber,
    initThemeToggle,
    setImageWithFallback,
    toSiteHref,
} from "./shared.js";

type ReleaseAsset = {
    name: string;
    browser_download_url: string;
};

type ReleaseResponse = {
    html_url: string;
    tag_name: string;
    assets?: ReleaseAsset[];
};

type AdapterRow = {
    adapter?: string;
    display_name?: string;
    mode?: string;
    mean_ms?: number;
    p50_ms?: number;
    p95_ms?: number;
    throughput_tokens_per_sec?: number;
};

type SummarySection = {
    run_id?: string;
    rows?: AdapterRow[];
    adapters?: AdapterRow[];
};

type LatestSummary = {
    run_id?: string;
    cpu?: SummarySection;
    gpu?: SummarySection;
};

type DistributedLatest = {
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
    latest?: DistributedLatest;
    visuals?: {
        throughput_svg?: string;
        pipeline_svg?: string;
    };
};

const OWNER = "DarkStarStrix";
const REPO = "PyC";
const RELEASE_API = `https://api.github.com/repos/${OWNER}/${REPO}/releases/latest`;
const DEFAULT_DOWNLOADS = {
    linux: `https://github.com/${OWNER}/${REPO}/releases/latest/download/pyc-linux-x86_64.tar.gz`,
    macos: `https://github.com/${OWNER}/${REPO}/releases/latest/download/pyc-macos-arm64.tar.gz`,
    windows: `https://github.com/${OWNER}/${REPO}/releases/latest/download/pyc-windows-x86_64.zip`,
};
const RAW_RESULTS_BASE = `https://raw.githubusercontent.com/${OWNER}/${REPO}/main/web/site/results`;

function findAsset(assets: ReleaseAsset[], os: keyof typeof DEFAULT_DOWNLOADS): ReleaseAsset | null {
    const patterns = {
        windows: /windows/i,
        macos: /macos|darwin/i,
        linux: /linux/i,
    };
    const pattern = patterns[os];
    return assets.find((asset) => pattern.test(asset.name)) ?? null;
}

function setLink(
    element: HTMLAnchorElement | null,
    asset: ReleaseAsset | null,
    fallback: string,
    fallbackText: string
): void {
    if (!element) {
        return;
    }

    if (asset) {
        element.textContent = asset.name;
        element.href = asset.browser_download_url;
        element.removeAttribute("aria-disabled");
        return;
    }

    element.textContent = fallbackText;
    element.href = fallback;
    element.removeAttribute("aria-disabled");
}

function renderReleaseAssets(assets: ReleaseAsset[]): void {
    const assetList = byId<HTMLUListElement>("asset-list");
    if (!assetList) {
        return;
    }

    assetList.innerHTML = "";
    if (assets.length === 0) {
        const empty = document.createElement("li");
        empty.textContent = "No binary assets found in latest release.";
        assetList.appendChild(empty);
        return;
    }

    for (const asset of assets) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = asset.browser_download_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = asset.name;
        item.appendChild(link);
        assetList.appendChild(item);
    }
}

function renderAdapterRows(tbodyId: string, rows: AdapterRow[]): void {
    const tbody = byId<HTMLTableSectionElement>(tbodyId);
    if (!tbody) {
        return;
    }

    tbody.innerHTML = "";
    if (rows.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.textContent = "No adapter rows available.";
        row.appendChild(cell);
        tbody.appendChild(row);
        return;
    }

    for (const entry of rows) {
        const row = document.createElement("tr");
        const columns = [
            entry.display_name ?? entry.adapter ?? "-",
            entry.mode ?? "unknown",
            formatNumber(entry.mean_ms, 4),
            formatNumber(entry.p50_ms, 4),
            formatNumber(entry.p95_ms, 4),
            formatNumber(entry.throughput_tokens_per_sec, 2),
        ];
        for (const value of columns) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        }
        tbody.appendChild(row);
    }
}

function summaryRows(section?: SummarySection): AdapterRow[] {
    if (!section) {
        return [];
    }
    if (Array.isArray(section.rows)) {
        return section.rows;
    }
    if (Array.isArray(section.adapters)) {
        return section.adapters;
    }
    return [];
}

function renderDistributedKpis(latest?: DistributedLatest): void {
    const grid = byId<HTMLDivElement>("dist-kpi-grid");
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

async function loadRelease(): Promise<void> {
    const releaseLink = byId<HTMLAnchorElement>("release-link");
    const linuxLink = byId<HTMLAnchorElement>("download-linux");
    const macosLink = byId<HTMLAnchorElement>("download-macos");
    const windowsLink = byId<HTMLAnchorElement>("download-windows");
    const status = byId<HTMLParagraphElement>("status");
    const fallback = `https://github.com/${OWNER}/${REPO}/releases/latest`;

    if (!releaseLink || !status) {
        return;
    }

    try {
        const release = await fetchJson<ReleaseResponse>(RELEASE_API);
        const assets = release.assets ?? [];
        releaseLink.href = release.html_url;
        status.textContent = `Latest release: ${release.tag_name}`;
        renderReleaseAssets(assets);
        setLink(linuxLink, findAsset(assets, "linux"), DEFAULT_DOWNLOADS.linux, "pyc-linux-x86_64.tar.gz");
        setLink(macosLink, findAsset(assets, "macos"), DEFAULT_DOWNLOADS.macos, "pyc-macos-arm64.tar.gz");
        setLink(windowsLink, findAsset(assets, "windows"), DEFAULT_DOWNLOADS.windows, "pyc-windows-x86_64.zip");
    } catch (_error) {
        status.textContent = "Release metadata unavailable. Direct download links are still active.";
        releaseLink.href = fallback;
        setLink(linuxLink, null, DEFAULT_DOWNLOADS.linux, "pyc-linux-x86_64.tar.gz");
        setLink(macosLink, null, DEFAULT_DOWNLOADS.macos, "pyc-macos-arm64.tar.gz");
        setLink(windowsLink, null, DEFAULT_DOWNLOADS.windows, "pyc-windows-x86_64.zip");
    }
}

async function loadPublishedResults(): Promise<void> {
    const status = byId<HTMLParagraphElement>("results-status");
    try {
        const summary = await fetchJsonWithFallback<LatestSummary>(
            [
                "./results/latest-summary.json",
                `${RAW_RESULTS_BASE}/latest-summary.json`,
            ],
            { cache: "no-store" }
        );
        const cpuRows = summaryRows(summary.cpu);
        const gpuRows = summaryRows(summary.gpu);
        const latestRun = summary.run_id ?? summary.cpu?.run_id ?? summary.gpu?.run_id ?? "unknown";

        renderAdapterRows("cpu-results-body", cpuRows);
        renderAdapterRows("gpu-results-body", gpuRows);
        if (status) {
            status.textContent =
                `Latest baseline run: ${latestRun} | CPU adapters: ${cpuRows.length} | GPU adapters: ${gpuRows.length}`;
        }
    } catch (_error) {
        renderAdapterRows("cpu-results-body", []);
        renderAdapterRows("gpu-results-body", []);
        if (status) {
            status.textContent = "Baseline adapter summary unavailable.";
        }
    }
}

function renderBaselineCharts(): void {
    const cpuImage = byId<HTMLImageElement>("latest-cpu-svg");
    const gpuImage = byId<HTMLImageElement>("latest-gpu-svg");
    setImageWithFallback(cpuImage, "./results/artifacts/latest/latest_cpu.svg", `${RAW_RESULTS_BASE}/artifacts/latest/latest_cpu.svg`);
    setImageWithFallback(gpuImage, "./results/artifacts/latest/latest_gpu.svg", `${RAW_RESULTS_BASE}/artifacts/latest/latest_gpu.svg`);
}

async function loadDistributedInsights(): Promise<void> {
    const status = byId<HTMLParagraphElement>("distributed-status");
    try {
        const payload = await fetchJsonWithFallback<DistributedPayload>(
            [
                "./results/distributed-latest.json",
                `${RAW_RESULTS_BASE}/distributed-latest.json`,
            ],
            { cache: "no-store" }
        );
        const latest = payload.latest;
        if (!latest) {
            if (status) {
                status.textContent = "Distributed training insights unavailable.";
            }
            return;
        }

        renderDistributedKpis(latest);
        if (status) {
            status.textContent =
                `Latest distributed run: ${latest.run_id ?? "unknown"} | world size: ${latest.world_size ?? "-"} | ` +
                `samples/s: ${formatNumber(latest.samples_per_sec, 2)} | tokens/s: ${formatNumber(latest.tokens_per_sec, 2)} | ` +
                `comm ms: ${formatNumber(latest.comm_time_ms_mean, 4)} | idle ms: ${formatNumber(latest.idle_gap_ms_mean, 4)}`;
        }

        setImageWithFallback(
            byId<HTMLImageElement>("latest-dist-summary-main"),
            latest.published?.summary_svg ?? null,
            "./results/artifacts/images/latest_distributed_pipeline.svg"
        );
        setImageWithFallback(
            byId<HTMLImageElement>("latest-dist-throughput-main"),
            payload.visuals?.throughput_svg ?? null,
            "./results/artifacts/images/latest_distributed_throughput.svg"
        );
        setImageWithFallback(
            byId<HTMLImageElement>("latest-dist-pipeline-main"),
            payload.visuals?.pipeline_svg ?? null,
            "./results/artifacts/images/latest_distributed_pipeline.svg"
        );
    } catch (_error) {
        if (status) {
            status.textContent = "Distributed training insights unavailable.";
        }
    }
}

void (async function init(): Promise<void> {
    initThemeToggle();
    renderBaselineCharts();
    await Promise.all([
        loadRelease(),
        loadPublishedResults(),
        loadDistributedInsights(),
    ]);
})();
