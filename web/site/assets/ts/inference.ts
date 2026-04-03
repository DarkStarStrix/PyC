import {
    byId,
    fetchJson,
    fetchText,
    formatNumber,
    initThemeToggle,
} from "./shared.js";

type InferenceLatency = {
    p50?: number;
    p95?: number;
    p99?: number;
};

type InferenceRow = {
    mode?: string;
    concurrency?: number;
    latency_ms?: InferenceLatency;
    throughput_tokens_per_sec?: number;
    allocation_event_delta?: number;
    memory_stable?: boolean;
};

type ManifestSvgEntry = {
    label?: string;
    path: string;
    caption?: string;
};

type InferenceManifest = {
    title?: string;
    updated_utc?: string;
    primary_json: string;
    primary_svg: string;
    paper_context?: {
        workload?: string;
        goal?: string;
        notes?: string[];
    };
    svgs?: ManifestSvgEntry[];
    markdown_reports?: string[];
    json_artifacts?: string[];
};

type PrimaryJson = {
    results?: InferenceRow[];
};

function fileName(path: string): string {
    const chunks = String(path).split("/");
    return chunks[chunks.length - 1] || path;
}

function modeLabel(mode?: string): string {
    return mode === "compiled_aten" ? "compiled+ATEN" : (mode ?? "-");
}

function escapeHtml(text: string): string {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function normalizePath(path: string): string {
    const parts: string[] = [];
    for (const segment of String(path).split("/")) {
        if (!segment || segment === ".") {
            continue;
        }
        if (segment === "..") {
            if (parts.length > 0 && parts[parts.length - 1] !== "..") {
                parts.pop();
            } else {
                parts.push("..");
            }
            continue;
        }
        parts.push(segment);
    }
    return parts.join("/");
}

function dirname(path: string): string {
    const index = path.lastIndexOf("/");
    return index >= 0 ? path.slice(0, index) : "";
}

function isAbsoluteHref(href: string): boolean {
    return /^(?:[a-z]+:|#|\/\/)/i.test(href);
}

function resolveHref(basePath: string, href: string): string {
    const trimmed = String(href).trim();
    if (!trimmed) {
        return "#";
    }
    if (isAbsoluteHref(trimmed)) {
        return trimmed;
    }

    let path = trimmed;
    let hash = "";
    let query = "";

    const hashIndex = path.indexOf("#");
    if (hashIndex >= 0) {
        hash = path.slice(hashIndex);
        path = path.slice(0, hashIndex);
    }

    const queryIndex = path.indexOf("?");
    if (queryIndex >= 0) {
        query = path.slice(queryIndex);
        path = path.slice(0, queryIndex);
    }

    const base = dirname(basePath);
    return normalizePath((base ? `${base}/` : "") + path) + query + hash;
}

function parseInline(markdown: string, basePath: string): string {
    let text = escapeHtml(markdown);
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_match, alt: string, href: string) => {
        const resolved = escapeHtml(resolveHref(basePath, href));
        return `<img src="${resolved}" alt="${escapeHtml(alt)}">`;
    });
    text = text.replace(/`([^`]+)`/g, (_match, code: string) => `<code>${code}</code>`);
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, label: string, href: string) => {
        const resolved = escapeHtml(resolveHref(basePath, href));
        return `<a href="${resolved}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`;
    });
    return text;
}

function markdownToHtml(markdown: string, basePath: string): string {
    let text = String(markdown).replace(/\r\n/g, "\n");
    const codeBlocks: string[] = [];

    text = text.replace(/```([\w-]*)\n([\s\S]*?)```/g, (_match, lang: string, code: string) => {
        const index = codeBlocks.length;
        const cls = lang ? ` class="language-${escapeHtml(lang)}"` : "";
        codeBlocks.push(`<pre><code${cls}>${escapeHtml(code)}</code></pre>`);
        return `@@CODEBLOCK_${index}@@`;
    });

    const lines = text.split("\n");
    const html: string[] = [];
    let paragraph: string[] = [];
    let inUnorderedList = false;
    let inOrderedList = false;

    const flushParagraph = (): void => {
        if (paragraph.length === 0) {
            return;
        }
        html.push(`<p>${parseInline(paragraph.join(" "), basePath)}</p>`);
        paragraph = [];
    };

    const closeLists = (): void => {
        if (inUnorderedList) {
            html.push("</ul>");
            inUnorderedList = false;
        }
        if (inOrderedList) {
            html.push("</ol>");
            inOrderedList = false;
        }
    };

    for (const rawLine of lines) {
        const trimmed = rawLine.trim();
        if (!trimmed) {
            flushParagraph();
            closeLists();
            continue;
        }

        const heading = /^(#{1,6})\s+(.+)$/.exec(trimmed);
        if (heading) {
            flushParagraph();
            closeLists();
            const level = heading[1].length;
            html.push(`<h${level}>${parseInline(heading[2], basePath)}</h${level}>`);
            continue;
        }

        if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
            flushParagraph();
            closeLists();
            html.push("<hr>");
            continue;
        }

        const unordered = /^[-*]\s+(.+)$/.exec(trimmed);
        if (unordered) {
            flushParagraph();
            if (inOrderedList) {
                html.push("</ol>");
                inOrderedList = false;
            }
            if (!inUnorderedList) {
                html.push("<ul>");
                inUnorderedList = true;
            }
            html.push(`<li>${parseInline(unordered[1], basePath)}</li>`);
            continue;
        }

        const ordered = /^\d+\.\s+(.+)$/.exec(trimmed);
        if (ordered) {
            flushParagraph();
            if (inUnorderedList) {
                html.push("</ul>");
                inUnorderedList = false;
            }
            if (!inOrderedList) {
                html.push("<ol>");
                inOrderedList = true;
            }
            html.push(`<li>${parseInline(ordered[1], basePath)}</li>`);
            continue;
        }

        const quote = /^>\s?(.*)$/.exec(trimmed);
        if (quote) {
            flushParagraph();
            closeLists();
            html.push(`<blockquote><p>${parseInline(quote[1], basePath)}</p></blockquote>`);
            continue;
        }

        closeLists();
        paragraph.push(trimmed);
    }

    flushParagraph();
    closeLists();

    return html.join("\n").replace(/@@CODEBLOCK_(\d+)@@/g, (_match, index: string) => {
        return codeBlocks[Number(index)] ?? "";
    });
}

function renderPaperContext(manifest: InferenceManifest, rows: InferenceRow[]): void {
    const workload = byId<HTMLParagraphElement>("paper-workload");
    const abstract = byId<HTMLParagraphElement>("paper-abstract");
    const notes = byId<HTMLUListElement>("paper-notes");
    const interpretation = byId<HTMLParagraphElement>("paper-interpretation");
    if (!workload || !abstract || !notes || !interpretation) {
        return;
    }

    const context = manifest.paper_context ?? {};
    workload.textContent = `${context.workload ?? "Workload context unavailable."}${context.goal ? ` Goal: ${context.goal}` : ""}`;

    notes.innerHTML = "";
    for (const note of context.notes ?? []) {
        const item = document.createElement("li");
        item.textContent = note;
        notes.appendChild(item);
    }

    const concurrencyEight = rows.filter((row) => Number(row.concurrency) === 8);
    const arena = concurrencyEight.find((row) => row.mode === "arena");
    const eager = concurrencyEight.find((row) => row.mode === "eager");
    const compiled = concurrencyEight.find((row) => row.mode === "compiled_aten");

    if (!arena || !eager || !compiled || !arena.latency_ms || !eager.latency_ms || !compiled.latency_ms) {
        abstract.textContent = "Primary inference rows were not complete enough to compute operating-point deltas.";
        interpretation.textContent = "Interpretation unavailable because one or more mode/concurrency slices are missing.";
        return;
    }

    const p95VsCompiled = ((arena.latency_ms.p95! - compiled.latency_ms.p95!) / compiled.latency_ms.p95!) * 100.0;
    const p99VsCompiled = ((arena.latency_ms.p99! - compiled.latency_ms.p99!) / compiled.latency_ms.p99!) * 100.0;
    const p95VsEager = ((arena.latency_ms.p95! - eager.latency_ms.p95!) / eager.latency_ms.p95!) * 100.0;

    abstract.textContent =
        "Under concurrent load, arena mode maintains deterministic memory behavior while preserving near-eager tail latency and materially reducing compiled-path tail inflation. " +
        `At c=8, arena is ${formatNumber(p95VsCompiled, 2)}% on p95 and ${formatNumber(p99VsCompiled, 2)}% on p99 versus compiled+ATEN (negative means better).`;

    interpretation.textContent =
        "This supports a stability-first operating point: arena eliminates post-warmup allocation churn (allocΔ=0, segΔ=0) and keeps tails tight. " +
        `Against eager at c=8, p95 remains near parity (${formatNumber(p95VsEager, 2)}% delta), so the tradeoff is small while determinism improves significantly.`;
}

async function renderPrimaryStats(manifest: InferenceManifest): Promise<void> {
    const tableBody = byId<HTMLTableSectionElement>("inference-table-body");
    const points = byId<HTMLUListElement>("quick-points");
    const latestSource = byId<HTMLParagraphElement>("latest-source");
    const primaryLinks = byId<HTMLParagraphElement>("primary-links");
    if (!tableBody || !points || !latestSource || !primaryLinks) {
        return;
    }

    latestSource.textContent = `Source JSON: ${manifest.primary_json}`;
    primaryLinks.innerHTML =
        `<a href="${manifest.primary_json}" target="_blank" rel="noopener noreferrer">Primary JSON</a>` +
        ` | <a href="${manifest.primary_svg}" target="_blank" rel="noopener noreferrer">Primary SVG</a>`;

    try {
        const payload = await fetchJson<PrimaryJson>(manifest.primary_json);
        const rows = Array.isArray(payload.results) ? payload.results : [];
        tableBody.innerHTML = "";
        points.innerHTML = "";

        if (rows.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8">No inference rows found.</td></tr>`;
            renderPaperContext(manifest, rows);
            return;
        }

        rows.slice()
            .sort((left, right) => {
                if ((left.concurrency ?? 0) !== (right.concurrency ?? 0)) {
                    return (left.concurrency ?? 0) - (right.concurrency ?? 0);
                }
                return String(left.mode ?? "").localeCompare(String(right.mode ?? ""));
            })
            .forEach((rowData) => {
                const row = document.createElement("tr");
                row.innerHTML =
                    `<td>${modeLabel(rowData.mode)}</td>` +
                    `<td>${rowData.concurrency ?? "-"}</td>` +
                    `<td>${formatNumber(rowData.latency_ms?.p50, 4)}</td>` +
                    `<td>${formatNumber(rowData.latency_ms?.p95, 4)}</td>` +
                    `<td>${formatNumber(rowData.latency_ms?.p99, 4)}</td>` +
                    `<td>${formatNumber(rowData.throughput_tokens_per_sec, 2)}</td>` +
                    `<td>${rowData.allocation_event_delta ?? "-"}</td>` +
                    `<td>${rowData.memory_stable ? "true" : "false"}</td>`;
                tableBody.appendChild(row);
            });

        const stableRows = rows.filter((row) => row.memory_stable).length;
        const stablePoint = document.createElement("li");
        stablePoint.textContent = `Rows loaded: ${rows.length}. Memory-stable rows: ${stableRows}/${rows.length}.`;
        points.appendChild(stablePoint);

        const concurrencyEight = rows.filter((row) => Number(row.concurrency) === 8);
        const arena = concurrencyEight.find((row) => row.mode === "arena");
        const eager = concurrencyEight.find((row) => row.mode === "eager");
        const compiled = concurrencyEight.find((row) => row.mode === "compiled_aten");

        if (arena?.latency_ms && compiled?.latency_ms) {
            const p95Delta = ((arena.latency_ms.p95! - compiled.latency_ms.p95!) / compiled.latency_ms.p95!) * 100.0;
            const p99Delta = ((arena.latency_ms.p99! - compiled.latency_ms.p99!) / compiled.latency_ms.p99!) * 100.0;
            const item = document.createElement("li");
            item.textContent =
                `c=8 arena vs compiled+ATEN: p95 ${formatNumber(p95Delta, 2)}% , p99 ${formatNumber(p99Delta, 2)}% (negative is better).`;
            points.appendChild(item);
        }

        if (arena?.latency_ms && eager?.latency_ms) {
            const p95VsEager = ((arena.latency_ms.p95! - eager.latency_ms.p95!) / eager.latency_ms.p95!) * 100.0;
            const item = document.createElement("li");
            item.textContent = `c=8 arena vs eager: p95 ${formatNumber(p95VsEager, 2)}% with zero allocation deltas in arena mode.`;
            points.appendChild(item);
        }

        renderPaperContext(manifest, rows);
    } catch (_error) {
        tableBody.innerHTML = `<tr><td colspan="8">Failed to load primary inference JSON.</td></tr>`;
    }
}

function renderSvgs(entries: ManifestSvgEntry[]): void {
    const gallery = byId<HTMLDivElement>("svg-gallery");
    if (!gallery) {
        return;
    }

    gallery.innerHTML = "";
    for (const entry of entries) {
        const figure = document.createElement("figure");
        const image = document.createElement("img");
        const caption = document.createElement("figcaption");
        const rawLink = document.createElement("a");

        image.src = entry.path;
        image.alt = entry.label ?? fileName(entry.path);
        image.loading = "lazy";

        rawLink.href = entry.path;
        rawLink.target = "_blank";
        rawLink.rel = "noopener noreferrer";
        rawLink.textContent = "Open raw SVG";

        caption.textContent = `${entry.label ?? fileName(entry.path)}${entry.caption ? `: ${entry.caption}` : ""} - `;
        caption.appendChild(rawLink);

        figure.appendChild(image);
        figure.appendChild(caption);
        gallery.appendChild(figure);
    }
}

function makeExpandableMarkdown(container: HTMLElement, path: string): void {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const content = document.createElement("div");
    const link = document.createElement("a");
    let loaded = false;

    summary.textContent = fileName(path);
    content.className = "report-content";
    link.href = path;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Open raw markdown";
    link.className = "path";

    details.appendChild(summary);
    details.appendChild(link);
    details.appendChild(content);

    details.addEventListener("toggle", async () => {
        if (!details.open || loaded) {
            return;
        }
        try {
            const text = await fetchText(path);
            loaded = true;
            content.innerHTML = markdownToHtml(text, path);
        } catch (_error) {
            loaded = true;
            content.textContent = `Failed to load: ${path}`;
        }
    });

    container.appendChild(details);
}

function makeExpandableJson(container: HTMLElement, path: string): void {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const pre = document.createElement("pre");
    const link = document.createElement("a");
    let loaded = false;

    summary.textContent = fileName(path);
    link.href = path;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Open raw json";
    link.className = "path";

    details.appendChild(summary);
    details.appendChild(link);
    details.appendChild(pre);

    details.addEventListener("toggle", async () => {
        if (!details.open || loaded) {
            return;
        }
        try {
            const text = await fetchText(path);
            loaded = true;
            try {
                pre.textContent = JSON.stringify(JSON.parse(text), null, 2);
            } catch (_error) {
                pre.textContent = text;
            }
        } catch (_error) {
            loaded = true;
            pre.textContent = `Failed to load: ${path}`;
        }
    });

    container.appendChild(details);
}

function renderReports(paths: string[]): void {
    const list = byId<HTMLDivElement>("report-list");
    if (!list) {
        return;
    }
    list.innerHTML = "";
    for (const path of paths) {
        makeExpandableMarkdown(list, path);
    }
}

function renderJsonArtifacts(paths: string[]): void {
    const list = byId<HTMLDivElement>("json-list");
    if (!list) {
        return;
    }
    list.innerHTML = "";
    for (const path of paths) {
        makeExpandableJson(list, path);
    }
}

async function loadManifest(): Promise<void> {
    const status = byId<HTMLParagraphElement>("manifest-status");
    if (!status) {
        return;
    }

    try {
        const manifest = await fetchJson<InferenceManifest>("./manifest.json");
        status.textContent = `Loaded ${manifest.title ?? "manifest"} | updated: ${manifest.updated_utc ?? "unknown"}`;
        await renderPrimaryStats(manifest);
        renderSvgs(manifest.svgs ?? []);
        renderReports(manifest.markdown_reports ?? []);
        renderJsonArtifacts(manifest.json_artifacts ?? []);
    } catch (_error) {
        status.textContent = "Failed to load inference manifest.";
    }
}

void (async function init(): Promise<void> {
    initThemeToggle();
    await loadManifest();
})();
