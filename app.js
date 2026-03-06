(function () {
  "use strict";

  var owner = "DarkStarStrix";
  var repo = "PyC";
  var api = "https://api.github.com/repos/" + owner + "/" + repo + "/releases/latest";
  var defaultDownloadLinks = {
    linux: "https://github.com/" + owner + "/" + repo + "/releases/latest/download/pyc-linux-x86_64.tar.gz",
    macos: "https://github.com/" + owner + "/" + repo + "/releases/latest/download/pyc-macos-arm64.tar.gz",
    windows: "https://github.com/" + owner + "/" + repo + "/releases/latest/download/pyc-windows-x86_64.zip"
  };

  var LATEST_BENCH = {
    runId: "20260219T164800Z_opt5_full_v2",
    cpuSvg: "website/results/artifacts/latest/latest_cpu.svg",
    gpuSvg: "website/results/artifacts/latest/latest_gpu.svg",
    cpuSvgRemote: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/website/results/artifacts/latest/latest_cpu.svg",
    gpuSvgRemote: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/website/results/artifacts/latest/latest_gpu.svg",
    summaryJson: "website/results/latest-summary.json",
    summaryJsonRemote: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/website/results/latest-summary.json"
  };

  var releaseLink = document.getElementById("release-link");
  var linuxLink = document.getElementById("download-linux");
  var macosLink = document.getElementById("download-macos");
  var windowsLink = document.getElementById("download-windows");
  var status = document.getElementById("status");
  var assetList = document.getElementById("asset-list");
  var themeToggle = document.getElementById("theme-toggle");

  var resultsStatus = document.getElementById("results-status");
  var distributedStatus = document.getElementById("distributed-status");
  var cpuBody = document.getElementById("cpu-results-body");
  var gpuBody = document.getElementById("gpu-results-body");
  var latestCpuSvg = document.getElementById("latest-cpu-svg");
  var latestGpuSvg = document.getElementById("latest-gpu-svg");

  var distSummaryMain = document.getElementById("latest-dist-summary-main");
  var distThroughputMain = document.getElementById("latest-dist-throughput-main");
  var distPipelineMain = document.getElementById("latest-dist-pipeline-main");
  var distKpiGrid = document.getElementById("dist-kpi-grid");

  function siteBaseHref() {
    var origin = window.location.origin || "";
    var pathname = window.location.pathname || "/";
    var basePath = pathname;

    if (!basePath.endsWith("/")) {
      var slash = basePath.lastIndexOf("/");
      var hasExt = slash >= 0 && basePath.slice(slash + 1).indexOf(".") !== -1;
      basePath = hasExt ? basePath.slice(0, slash + 1) : basePath + "/";
    }
    return origin + basePath;
  }

  function toHref(path) {
    if (!path) return "#";
    if (/^https?:\/\//i.test(path)) return path;
    return new URL(path, siteBaseHref()).toString();
  }

  function preferredTheme() {
    var stored = null;
    try {
      stored = window.localStorage.getItem("pyc-theme");
    } catch (e) {}
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    var resolved = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolved);
    if (themeToggle) {
      themeToggle.textContent = resolved === "dark" ? "Light Mode" : "Dark Mode";
      themeToggle.setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
    }
  }

  function initThemeToggle() {
    var initial = preferredTheme();
    applyTheme(initial);

    if (!themeToggle) return;
    themeToggle.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
      var next = current === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        window.localStorage.setItem("pyc-theme", next);
      } catch (e) {}
    });
  }

  function findAsset(assets, os) {
    var patterns = {
      windows: /windows/i,
      macos: /macos|darwin/i,
      linux: /linux/i
    };
    var re = patterns[os];
    if (!re) return null;
    for (var i = 0; i < assets.length; i++) {
      if (re.test(assets[i].name)) return assets[i];
    }
    return null;
  }

  function setLink(el, asset, fallback, fallbackText) {
    if (!el) return;
    if (asset) {
      el.textContent = asset.name;
      el.href = asset.browser_download_url;
      el.removeAttribute("aria-disabled");
    } else {
      if (fallbackText) {
        el.textContent = fallbackText;
      }
      el.href = fallback;
      el.removeAttribute("aria-disabled");
    }
  }

  function renderReleaseAssets(assets) {
    if (!assetList) return;
    assetList.innerHTML = "";
    if (!assets.length) {
      var empty = document.createElement("li");
      empty.textContent = "No binary assets found in latest release.";
      assetList.appendChild(empty);
      return;
    }
    assets.forEach(function (asset) {
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = asset.browser_download_url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = asset.name;
      li.appendChild(a);
      assetList.appendChild(li);
    });
  }

  function fmt(value, digits) {
    if (value === null || value === undefined) return "-";
    if (typeof value !== "number") return String(value);
    return value.toFixed(digits);
  }

  function renderRows(tbody, rows) {
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!rows || !rows.length) {
      var tr = document.createElement("tr");
      var td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = "No adapter rows available.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      var cols = [
        row.display_name || row.adapter,
        row.mode || "unknown",
        fmt(row.mean_ms, 4),
        fmt(row.p50_ms, 4),
        fmt(row.p95_ms, 4),
        fmt(row.throughput_tokens_per_sec, 2)
      ];
      cols.forEach(function (value) {
        var td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  function setImageWithFallback(img, src, fallback) {
    if (!img) return;
    if (src) {
      img.src = toHref(src);
    }
    img.onerror = function () {
      if (fallback) {
        img.src = fallback;
      }
    };
  }

  function renderBaselineCharts() {
    setImageWithFallback(latestCpuSvg, LATEST_BENCH.cpuSvg, LATEST_BENCH.cpuSvgRemote);
    setImageWithFallback(latestGpuSvg, LATEST_BENCH.gpuSvg, LATEST_BENCH.gpuSvgRemote);
  }

  function summaryRows(section) {
    if (!section) return [];
    if (Array.isArray(section.rows)) return section.rows;
    if (Array.isArray(section.adapters)) return section.adapters;
    return [];
  }

  function loadRelease() {
    if (!releaseLink || !status) return;
    fetch(api)
      .then(function (resp) {
        if (!resp.ok) throw new Error("Failed to fetch release");
        return resp.json();
      })
      .then(function (release) {
        releaseLink.href = release.html_url;
        status.textContent = "Latest release: " + release.tag_name;
        renderReleaseAssets(release.assets || []);
        setLink(linuxLink, findAsset(release.assets || [], "linux"), defaultDownloadLinks.linux || release.html_url, "pyc-linux-x86_64.tar.gz");
        setLink(macosLink, findAsset(release.assets || [], "macos"), defaultDownloadLinks.macos || release.html_url, "pyc-macos-arm64.tar.gz");
        setLink(windowsLink, findAsset(release.assets || [], "windows"), defaultDownloadLinks.windows || release.html_url, "pyc-windows-x86_64.zip");
      })
      .catch(function () {
        status.textContent = "Release metadata unavailable. Direct download links are still active.";
        var fallback = "https://github.com/" + owner + "/" + repo + "/releases/latest";
        setLink(linuxLink, null, defaultDownloadLinks.linux || fallback, "pyc-linux-x86_64.tar.gz");
        setLink(macosLink, null, defaultDownloadLinks.macos || fallback, "pyc-macos-arm64.tar.gz");
        setLink(windowsLink, null, defaultDownloadLinks.windows || fallback, "pyc-windows-x86_64.zip");
        releaseLink.href = fallback;
      });
  }

  function loadPublishedResults() {
    fetch(toHref(LATEST_BENCH.summaryJson), { cache: "no-store" })
      .then(function (resp) {
        if (!resp.ok) throw new Error("local latest summary unavailable");
        return resp.json();
      })
      .catch(function () {
        return fetch(LATEST_BENCH.summaryJsonRemote, { cache: "no-store" }).then(function (resp) {
          if (!resp.ok) throw new Error("remote latest summary unavailable");
          return resp.json();
        });
      })
      .then(function (latestSummary) {
        latestSummary = latestSummary || {};
        var latestRun =
          latestSummary.run_id ||
          (latestSummary.cpu && latestSummary.cpu.run_id) ||
          (latestSummary.gpu && latestSummary.gpu.run_id) ||
          LATEST_BENCH.runId;
        var cpuRows = summaryRows(latestSummary.cpu);
        var gpuRows = summaryRows(latestSummary.gpu);
        renderRows(cpuBody, cpuRows);
        renderRows(gpuBody, gpuRows);

        if (resultsStatus) {
          resultsStatus.textContent =
            "Latest baseline run: " + latestRun +
            " | CPU adapters: " + cpuRows.length +
            " | GPU adapters: " + gpuRows.length;
        }
      })
      .catch(function () {
        renderRows(cpuBody, []);
        renderRows(gpuBody, []);
        if (resultsStatus) {
          resultsStatus.textContent = "Baseline adapter summary unavailable.";
        }
      });
  }

  function renderDistKpis(latest) {
    if (!distKpiGrid || !latest) return;
    distKpiGrid.innerHTML =
      "<div class='kpi'><p class='label'>Samples/s</p><p class='value'>" + fmt(latest.samples_per_sec, 2) + "</p></div>" +
      "<div class='kpi'><p class='label'>Tokens/s</p><p class='value'>" + fmt(latest.tokens_per_sec, 2) + "</p></div>" +
      "<div class='kpi'><p class='label'>GPU Util Mean</p><p class='value'>" + fmt(latest.gpu_util_mean, 2) + "%</p></div>" +
      "<div class='kpi'><p class='label'>Compute (ms)</p><p class='value'>" + fmt(latest.compute_time_ms_mean, 4) + "</p></div>" +
      "<div class='kpi'><p class='label'>Comm (ms)</p><p class='value'>" + fmt(latest.comm_time_ms_mean, 4) + "</p></div>" +
      "<div class='kpi'><p class='label'>Idle (ms)</p><p class='value'>" + fmt(latest.idle_gap_ms_mean, 4) + "</p></div>";
  }

  function loadDistributedInsights() {
    fetch(toHref("website/results/distributed-latest.json"), { cache: "no-store" })
      .then(function (resp) {
        if (!resp.ok) throw new Error("local distributed summary unavailable");
        return resp.json();
      })
      .then(function (payload) {
        var latest = payload && payload.latest ? payload.latest : null;
        var visuals = payload && payload.visuals ? payload.visuals : {};
        if (!latest) {
          if (distributedStatus) distributedStatus.textContent = "Distributed training insights unavailable.";
          return;
        }

        if (distributedStatus) {
          distributedStatus.textContent =
            "Latest distributed run: " + (latest.run_id || "unknown") +
            " | world size: " + (latest.world_size || "-") +
            " | samples/s: " + fmt(latest.samples_per_sec, 2) +
            " | tokens/s: " + fmt(latest.tokens_per_sec, 2) +
            " | comm ms: " + fmt(latest.comm_time_ms_mean, 4) +
            " | idle ms: " + fmt(latest.idle_gap_ms_mean, 4);
        }

        renderDistKpis(latest);

        var summarySvg = latest.published && latest.published.summary_svg ? latest.published.summary_svg : null;
        setImageWithFallback(distSummaryMain, summarySvg, "./website/results/artifacts/images/latest_distributed_pipeline.svg");
        setImageWithFallback(distThroughputMain, visuals.throughput_svg, "./website/results/artifacts/images/latest_distributed_throughput.svg");
        setImageWithFallback(distPipelineMain, visuals.pipeline_svg, "./website/results/artifacts/images/latest_distributed_pipeline.svg");
      })
      .catch(function () {
        if (distributedStatus) distributedStatus.textContent = "Distributed training insights unavailable.";
      });
  }

  initThemeToggle();
  renderBaselineCharts();
  loadRelease();
  loadPublishedResults();
  loadDistributedInsights();
})();
