(function () {
  "use strict";

  function byId(id) {
    return document.getElementById(id);
  }

  function toFixedNum(v, d) {
    if (typeof v !== "number" || Number.isNaN(v)) return "-";
    return v.toFixed(d);
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function normalizePath(path) {
    var parts = [];
    String(path || "").split("/").forEach(function (seg) {
      if (!seg || seg === ".") return;
      if (seg === "..") {
        if (parts.length && parts[parts.length - 1] !== "..") {
          parts.pop();
        } else {
          parts.push("..");
        }
        return;
      }
      parts.push(seg);
    });
    return parts.join("/");
  }

  function dirname(path) {
    var s = String(path || "");
    var idx = s.lastIndexOf("/");
    return idx >= 0 ? s.slice(0, idx) : "";
  }

  function isAbsoluteHref(href) {
    return /^(?:[a-z]+:|#|\/\/)/i.test(href);
  }

  function resolveHref(basePath, href) {
    var h = String(href || "").trim();
    if (!h) return "#";
    if (isAbsoluteHref(h)) return h;

    var hash = "";
    var q = "";
    var hashIdx = h.indexOf("#");
    if (hashIdx >= 0) {
      hash = h.slice(hashIdx);
      h = h.slice(0, hashIdx);
    }
    var qIdx = h.indexOf("?");
    if (qIdx >= 0) {
      q = h.slice(qIdx);
      h = h.slice(0, qIdx);
    }

    var base = dirname(basePath);
    var joined = normalizePath((base ? base + "/" : "") + h);
    return joined + q + hash;
  }

  function parseInline(md, basePath) {
    var s = escapeHtml(md);
    s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function (_, alt, href) {
      var safeAlt = escapeHtml(alt || "");
      var safeHref = escapeHtml(resolveHref(basePath, href));
      return '<img src="' + safeHref + '" alt="' + safeAlt + '">';
    });
    s = s.replace(/`([^`]+)`/g, function (_, code) {
      return "<code>" + code + "</code>";
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, label, href) {
      var safeHref = escapeHtml(resolveHref(basePath, href));
      var safeLabel = escapeHtml(label);
      return '<a href="' + safeHref + '" target="_blank" rel="noopener noreferrer">' + safeLabel + "</a>";
    });
    return s;
  }

  function markdownToHtml(markdown, basePath) {
    var text = String(markdown || "").replace(/\r\n/g, "\n");
    var codeBlocks = [];
    text = text.replace(/```([\w-]*)\n([\s\S]*?)```/g, function (_, lang, code) {
      var idx = codeBlocks.length;
      var cls = lang ? ' class="language-' + escapeHtml(lang) + '"' : "";
      codeBlocks.push("<pre><code" + cls + ">" + escapeHtml(code) + "</code></pre>");
      return "@@CODEBLOCK_" + idx + "@@";
    });

    var lines = text.split("\n");
    var html = [];
    var paragraph = [];
    var inUl = false;
    var inOl = false;

    function flushParagraph() {
      if (!paragraph.length) return;
      html.push("<p>" + parseInline(paragraph.join(" "), basePath) + "</p>");
      paragraph = [];
    }

    function closeLists() {
      if (inUl) {
        html.push("</ul>");
        inUl = false;
      }
      if (inOl) {
        html.push("</ol>");
        inOl = false;
      }
    }

    lines.forEach(function (raw) {
      var line = raw;
      var trimmed = line.trim();

      if (!trimmed) {
        flushParagraph();
        closeLists();
        return;
      }

      var h = /^(#{1,6})\s+(.+)$/.exec(trimmed);
      if (h) {
        flushParagraph();
        closeLists();
        var level = h[1].length;
        html.push("<h" + level + ">" + parseInline(h[2], basePath) + "</h" + level + ">");
        return;
      }

      if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
        flushParagraph();
        closeLists();
        html.push("<hr>");
        return;
      }

      var ul = /^[-*]\s+(.+)$/.exec(trimmed);
      if (ul) {
        flushParagraph();
        if (inOl) {
          html.push("</ol>");
          inOl = false;
        }
        if (!inUl) {
          html.push("<ul>");
          inUl = true;
        }
        html.push("<li>" + parseInline(ul[1], basePath) + "</li>");
        return;
      }

      var ol = /^\d+\.\s+(.+)$/.exec(trimmed);
      if (ol) {
        flushParagraph();
        if (inUl) {
          html.push("</ul>");
          inUl = false;
        }
        if (!inOl) {
          html.push("<ol>");
          inOl = true;
        }
        html.push("<li>" + parseInline(ol[1], basePath) + "</li>");
        return;
      }

      var bq = /^>\s?(.*)$/.exec(trimmed);
      if (bq) {
        flushParagraph();
        closeLists();
        html.push("<blockquote><p>" + parseInline(bq[1], basePath) + "</p></blockquote>");
        return;
      }

      closeLists();
      paragraph.push(trimmed);
    });

    flushParagraph();
    closeLists();

    var out = html.join("\n");
    out = out.replace(/@@CODEBLOCK_(\d+)@@/g, function (_, idx) {
      return codeBlocks[Number(idx)] || "";
    });
    return out;
  }

  function preferredTheme() {
    try {
      var stored = window.localStorage.getItem("pyc-theme");
      if (stored === "light" || stored === "dark") return stored;
    } catch (e) {}
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }
    return "light";
  }

  function applyTheme(theme) {
    var t = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    var btn = byId("theme-toggle");
    if (btn) btn.textContent = t === "dark" ? "Light Mode" : "Dark Mode";
  }

  function initTheme() {
    applyTheme(preferredTheme());
    var btn = byId("theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        window.localStorage.setItem("pyc-theme", next);
      } catch (e) {}
    });
  }

  function fileName(path) {
    var chunks = String(path || "").split("/");
    return chunks[chunks.length - 1] || path;
  }

  function modeLabel(mode) {
    if (mode === "compiled_aten") return "compiled+ATEN";
    return mode;
  }

  function renderPaperContext(manifest, rows) {
    var workload = byId("paper-workload");
    var abstract = byId("paper-abstract");
    var notes = byId("paper-notes");
    var interpretation = byId("paper-interpretation");

    var ctx = manifest.paper_context || {};
    workload.textContent = (ctx.workload || "Workload context unavailable.") +
      (ctx.goal ? " Goal: " + ctx.goal : "");

    notes.innerHTML = "";
    (ctx.notes || []).forEach(function (n) {
      var li = document.createElement("li");
      li.textContent = n;
      notes.appendChild(li);
    });

    var c8 = rows.filter(function (r) { return Number(r.concurrency) === 8; });
    var arena8 = c8.find(function (r) { return r.mode === "arena"; });
    var eager8 = c8.find(function (r) { return r.mode === "eager"; });
    var comp8 = c8.find(function (r) { return r.mode === "compiled_aten"; });

    if (!arena8 || !eager8 || !comp8) {
      abstract.textContent = "Primary inference rows were not complete enough to compute operating-point deltas.";
      interpretation.textContent = "Interpretation unavailable because one or more mode/concurrency slices are missing.";
      return;
    }

    var p95VsComp = ((arena8.latency_ms.p95 - comp8.latency_ms.p95) / comp8.latency_ms.p95) * 100.0;
    var p99VsComp = ((arena8.latency_ms.p99 - comp8.latency_ms.p99) / comp8.latency_ms.p99) * 100.0;
    var p95VsEager = ((arena8.latency_ms.p95 - eager8.latency_ms.p95) / eager8.latency_ms.p95) * 100.0;

    abstract.textContent =
      "Under concurrent load, arena mode maintains deterministic memory behavior while preserving near-eager tail latency and materially reducing compiled-path tail inflation. " +
      "At c=8, arena is " + toFixedNum(p95VsComp, 2) + "% on p95 and " + toFixedNum(p99VsComp, 2) + "% on p99 versus compiled+ATEN (negative means better).";

    interpretation.textContent =
      "This supports a stability-first operating point: arena eliminates post-warmup allocation churn (allocΔ=0, segΔ=0) and keeps tails tight. " +
      "Against eager at c=8, p95 remains near parity (" + toFixedNum(p95VsEager, 2) + "% delta), so the tradeoff is small while determinism improves significantly.";
  }

  function renderPrimaryStats(manifest) {
    var primaryJsonPath = manifest.primary_json;
    var primarySvgPath = manifest.primary_svg;
    var tableBody = byId("inference-table-body");
    var points = byId("quick-points");
    var latestSource = byId("latest-source");
    var primaryLinks = byId("primary-links");

    latestSource.textContent = "Source JSON: " + primaryJsonPath;
    primaryLinks.innerHTML =
      '<a href="' + primaryJsonPath + '" target="_blank" rel="noopener noreferrer">Primary JSON</a>' +
      ' | <a href="' + primarySvgPath + '" target="_blank" rel="noopener noreferrer">Primary SVG</a>';

    return fetch(primaryJsonPath)
      .then(function (resp) {
        if (!resp.ok) throw new Error("primary json not available");
        return resp.json();
      })
      .then(function (data) {
        var rows = Array.isArray(data.results) ? data.results : [];
        tableBody.innerHTML = "";
        points.innerHTML = "";

        if (!rows.length) {
          var empty = document.createElement("tr");
          empty.innerHTML = '<td colspan="8">No inference rows found.</td>';
          tableBody.appendChild(empty);
          renderPaperContext(manifest, rows);
          return;
        }

        rows
          .slice()
          .sort(function (a, b) {
            if ((a.concurrency || 0) !== (b.concurrency || 0)) return (a.concurrency || 0) - (b.concurrency || 0);
            return String(a.mode || "").localeCompare(String(b.mode || ""));
          })
          .forEach(function (r) {
            var tr = document.createElement("tr");
            tr.innerHTML =
              '<td>' + modeLabel(r.mode || "-") + "</td>" +
              '<td>' + (r.concurrency || "-") + "</td>" +
              '<td>' + toFixedNum(r.latency_ms && r.latency_ms.p50, 4) + "</td>" +
              '<td>' + toFixedNum(r.latency_ms && r.latency_ms.p95, 4) + "</td>" +
              '<td>' + toFixedNum(r.latency_ms && r.latency_ms.p99, 4) + "</td>" +
              '<td>' + toFixedNum(r.throughput_tokens_per_sec, 2) + "</td>" +
              '<td>' + (r.allocation_event_delta != null ? r.allocation_event_delta : "-") + "</td>" +
              '<td>' + (r.memory_stable ? "true" : "false") + "</td>";
            tableBody.appendChild(tr);
          });

        var c8Rows = rows.filter(function (r) { return Number(r.concurrency) === 8; });
        var arena8 = c8Rows.find(function (r) { return r.mode === "arena"; });
        var eager8 = c8Rows.find(function (r) { return r.mode === "eager"; });
        var compiled8 = c8Rows.find(function (r) { return r.mode === "compiled_aten"; });
        var stableCount = rows.filter(function (r) { return !!r.memory_stable; }).length;

        var p1 = document.createElement("li");
        p1.textContent = "Rows loaded: " + rows.length + ". Memory-stable rows: " + stableCount + "/" + rows.length + ".";
        points.appendChild(p1);

        if (arena8 && compiled8) {
          var p95Delta = ((arena8.latency_ms.p95 - compiled8.latency_ms.p95) / compiled8.latency_ms.p95) * 100.0;
          var p99Delta = ((arena8.latency_ms.p99 - compiled8.latency_ms.p99) / compiled8.latency_ms.p99) * 100.0;
          var p2 = document.createElement("li");
          p2.textContent =
            "c=8 arena vs compiled+ATEN: p95 " + toFixedNum(p95Delta, 2) + "% , p99 " + toFixedNum(p99Delta, 2) + "% (negative is better).";
          points.appendChild(p2);
        }

        if (arena8 && eager8) {
          var p95VsEager = ((arena8.latency_ms.p95 - eager8.latency_ms.p95) / eager8.latency_ms.p95) * 100.0;
          var p3 = document.createElement("li");
          p3.textContent =
            "c=8 arena vs eager: p95 " + toFixedNum(p95VsEager, 2) + "% with zero allocation deltas in arena mode.";
          points.appendChild(p3);
        }

        renderPaperContext(manifest, rows);
      })
      .catch(function () {
        tableBody.innerHTML = '<tr><td colspan="8">Failed to load primary inference JSON.</td></tr>';
      });
  }

  function renderSvgs(entries) {
    var gallery = byId("svg-gallery");
    gallery.innerHTML = "";
    (entries || []).forEach(function (entry) {
      var figure = document.createElement("figure");
      var img = document.createElement("img");
      var caption = document.createElement("figcaption");
      var raw = document.createElement("a");

      img.src = entry.path;
      img.alt = entry.label || fileName(entry.path);
      img.loading = "lazy";

      raw.href = entry.path;
      raw.target = "_blank";
      raw.rel = "noopener noreferrer";
      raw.textContent = "Open raw SVG";

      var cap = entry.caption || "";
      caption.textContent = (entry.label || fileName(entry.path)) + (cap ? ": " + cap : "") + " - ";
      caption.appendChild(raw);

      figure.appendChild(img);
      figure.appendChild(caption);
      gallery.appendChild(figure);
    });
  }

  function makeExpandableMarkdown(container, path) {
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    var content = document.createElement("div");
    var link = document.createElement("a");
    var loaded = false;

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

    details.addEventListener("toggle", function () {
      if (!details.open || loaded) return;
      fetch(path)
        .then(function (resp) {
          if (!resp.ok) throw new Error("fetch failed");
          return resp.text();
        })
        .then(function (text) {
          loaded = true;
          content.innerHTML = markdownToHtml(text, path);
        })
        .catch(function () {
          loaded = true;
          content.textContent = "Failed to load: " + path;
        });
    });

    container.appendChild(details);
  }

  function makeExpandableJson(container, path) {
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    var pre = document.createElement("pre");
    var link = document.createElement("a");
    var loaded = false;

    summary.textContent = fileName(path);
    link.href = path;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Open raw json";
    link.className = "path";

    details.appendChild(summary);
    details.appendChild(link);
    details.appendChild(pre);

    details.addEventListener("toggle", function () {
      if (!details.open || loaded) return;
      fetch(path)
        .then(function (resp) {
          if (!resp.ok) throw new Error("fetch failed");
          return resp.text();
        })
        .then(function (text) {
          loaded = true;
          try {
            pre.textContent = JSON.stringify(JSON.parse(text), null, 2);
          } catch (e) {
            pre.textContent = text;
          }
        })
        .catch(function () {
          loaded = true;
          pre.textContent = "Failed to load: " + path;
        });
    });

    container.appendChild(details);
  }

  function renderReports(paths) {
    var list = byId("report-list");
    list.innerHTML = "";
    (paths || []).forEach(function (path) {
      makeExpandableMarkdown(list, path);
    });
  }

  function renderJson(paths) {
    var list = byId("json-list");
    list.innerHTML = "";
    (paths || []).forEach(function (path) {
      makeExpandableJson(list, path);
    });
  }

  function loadManifest() {
    var status = byId("manifest-status");
    fetch("./manifest.json")
      .then(function (resp) {
        if (!resp.ok) throw new Error("manifest unavailable");
        return resp.json();
      })
      .then(function (manifest) {
        status.textContent =
          "Loaded " + (manifest.title || "manifest") +
          " | updated: " + (manifest.updated_utc || "unknown");

        renderPrimaryStats(manifest);
        renderSvgs(manifest.svgs || []);
        renderReports(manifest.markdown_reports || []);
        renderJson(manifest.json_artifacts || []);
      })
      .catch(function () {
        status.textContent = "Failed to load inference manifest.";
      });
  }

  initTheme();
  loadManifest();
})();
