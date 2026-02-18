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

  var PHASE4 = {
    runId: "20260218T023355Z_phase4_final",
    cpuSvg: "benchmark/benchmarks/results/remote_results/host89/images/20260218T023355Z_phase4_final__cpu.svg",
    gpuSvg: "benchmark/benchmarks/results/remote_results/host89/images/20260218T023355Z_phase4_final__gpu.svg",
    cpuSvgRemote: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/benchmark/benchmarks/results/remote_results/host89/images/20260218T023355Z_phase4_final__cpu.svg",
    gpuSvgRemote: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/benchmark/benchmarks/results/remote_results/host89/images/20260218T023355Z_phase4_final__gpu.svg",
    cpuJson: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/benchmark/benchmarks/results/remote_results/host89/json/20260218T023355Z_phase4_final__cpu.json",
    gpuJson: "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/benchmark/benchmarks/results/remote_results/host89/json/20260218T023355Z_phase4_final__gpu.json"
  };

  var PHASE4_CPU_FALLBACK_ROWS = [
    { display_name: "PyC CUDA", mode: "native", mean_ms: 24.0459, p50_ms: 24.0220, p95_ms: 24.1800, throughput_tokens_per_sec: 5450908.47 },
    { display_name: "TensorRT", mode: "proxy", mean_ms: 24.4121, p50_ms: 9.3322, p95_ms: 68.6434, throughput_tokens_per_sec: 5369130.72 },
    { display_name: "PyTorch Compile", mode: "native", mean_ms: 26.7237, p50_ms: 10.0514, p95_ms: 72.1404, throughput_tokens_per_sec: 4904708.07 },
    { display_name: "XLA", mode: "proxy", mean_ms: 30.2406, p50_ms: 12.8271, p95_ms: 72.3854, throughput_tokens_per_sec: 4334310.06 },
    { display_name: "Glow", mode: "proxy", mean_ms: 34.8523, p50_ms: 15.7008, p95_ms: 75.1887, throughput_tokens_per_sec: 3760782.37 },
    { display_name: "PyTorch Eager", mode: "native", mean_ms: 36.9971, p50_ms: 15.4305, p95_ms: 75.9299, throughput_tokens_per_sec: 3542766.92 }
  ];

  var PHASE4_GPU_FALLBACK_ROWS = [
    { display_name: "PyTorch Eager", mode: "native", mean_ms: 0.1154, p50_ms: 0.1135, p95_ms: 0.1285, throughput_tokens_per_sec: 1135905416.11 },
    { display_name: "XLA", mode: "proxy", mean_ms: 0.1157, p50_ms: 0.1138, p95_ms: 0.1263, throughput_tokens_per_sec: 1133048992.77 },
    { display_name: "Glow", mode: "proxy", mean_ms: 0.1314, p50_ms: 0.1190, p95_ms: 0.1332, throughput_tokens_per_sec: 997355892.15 },
    { display_name: "PyTorch Compile", mode: "native", mean_ms: 0.1551, p50_ms: 0.1544, p95_ms: 0.1665, throughput_tokens_per_sec: 844896466.85 },
    { display_name: "TensorRT", mode: "proxy", mean_ms: 0.1598, p50_ms: 0.1560, p95_ms: 0.1743, throughput_tokens_per_sec: 820055823.27 },
    { display_name: "PyC CUDA", mode: "proxy", mean_ms: 25.5228, p50_ms: 25.3830, p95_ms: 27.3270, throughput_tokens_per_sec: 5135486.70 }
  ];

  var releaseLink = document.getElementById("release-link");
  var linuxLink = document.getElementById("download-linux");
  var macosLink = document.getElementById("download-macos");
  var windowsLink = document.getElementById("download-windows");
  var status = document.getElementById("status");
  var assetList = document.getElementById("asset-list");
  var themeToggle = document.getElementById("theme-toggle");

  var resultsStatus = document.getElementById("results-status");
  var cpuBody = document.getElementById("cpu-results-body");
  var gpuBody = document.getElementById("gpu-results-body");
  var latestCpuSvg = document.getElementById("latest-cpu-svg");
  var latestGpuSvg = document.getElementById("latest-gpu-svg");
  var svgGallery = document.getElementById("svg-gallery");

  function toHref(path) {
    if (!path) return "#";
    if (/^https?:\/\//i.test(path)) return path;
    return new URL(path, window.location.href).toString();
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

  function assetHref(path) {
    if (!path) return "#";
    if (/^https?:\/\//i.test(path)) return path;
    if (path.indexOf("website/results/") === 0) {
      return "https://raw.githubusercontent.com/DarkStarStrix/PyC/main/" + path;
    }
    return toHref(path);
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

  function renderPinnedPhase4Charts() {
    if (latestCpuSvg) {
      latestCpuSvg.src = toHref(PHASE4.cpuSvg);
      latestCpuSvg.onerror = function () {
        latestCpuSvg.src = PHASE4.cpuSvgRemote;
      };
    }
    if (latestGpuSvg) {
      latestGpuSvg.src = toHref(PHASE4.gpuSvg);
      latestGpuSvg.onerror = function () {
        latestGpuSvg.src = PHASE4.gpuSvgRemote;
      };
    }
  }

  function renderSvgGallery(entries) {
    if (!svgGallery) return;
    svgGallery.innerHTML = "";
    if (!entries.length) {
      var empty = document.createElement("p");
      empty.textContent = "No SVG artifacts found.";
      svgGallery.appendChild(empty);
      return;
    }

    entries.forEach(function (entry) {
      var card = document.createElement("figure");
      card.className = "svg-preview";

      var item = document.createElement("a");
      item.href = assetHref(entry.published);
      item.target = "_blank";
      item.rel = "noopener noreferrer";

      var img = document.createElement("img");
      img.src = assetHref(entry.published);
      img.alt = entry.source;
      img.loading = "lazy";

      var label = document.createElement("figcaption");
      label.textContent = captionFromSource(entry.source);

      item.appendChild(img);
      card.appendChild(item);
      card.appendChild(label);
      svgGallery.appendChild(card);
    });
  }

  function captionFromSource(source) {
    var file = String(source || "").split("/").pop().replace(/\.svg$/i, "");
    if (!file) return "Chart";
    return file.replace(/__/g, " | ").replace(/_/g, " ");
  }

  function adaptersToRows(payload) {
    var rows = [];
    if (!payload || !payload.adapters) return rows;

    Object.keys(payload.adapters).forEach(function (key) {
      var entry = payload.adapters[key];
      var latency = entry && entry.latency_ms ? entry.latency_ms : {};
      if (!entry || entry.status !== "ok") return;
      rows.push({
        adapter: key,
        display_name: entry.display_name || key,
        mode: entry.mode || "unknown",
        mean_ms: latency.mean,
        p50_ms: latency.p50,
        p95_ms: latency.p95,
        throughput_tokens_per_sec: entry.throughput_tokens_per_sec
      });
    });

    rows.sort(function (a, b) {
      if (a.mean_ms === undefined || a.mean_ms === null) return 1;
      if (b.mean_ms === undefined || b.mean_ms === null) return -1;
      return a.mean_ms - b.mean_ms;
    });
    return rows;
  }

  function loadPhase4Stats() {
    Promise.all([
      fetch(toHref(PHASE4.cpuJson)).then(function (resp) {
        if (!resp.ok) throw new Error("phase4 cpu json unavailable");
        return resp.json();
      }),
      fetch(toHref(PHASE4.gpuJson)).then(function (resp) {
        if (!resp.ok) throw new Error("phase4 gpu json unavailable");
        return resp.json();
      })
    ])
      .then(function (payload) {
        var cpuRows = adaptersToRows(payload[0]);
        var gpuRows = adaptersToRows(payload[1]);
        renderRows(cpuBody, cpuRows.length ? cpuRows : PHASE4_CPU_FALLBACK_ROWS);
        renderRows(gpuBody, gpuRows.length ? gpuRows : PHASE4_GPU_FALLBACK_ROWS);
      })
      .catch(function () {
        renderRows(cpuBody, PHASE4_CPU_FALLBACK_ROWS);
        renderRows(gpuBody, PHASE4_GPU_FALLBACK_ROWS);
      });
  }

  function loadRelease() {
    fetch(api)
      .then(function (resp) {
        if (!resp.ok) throw new Error("Failed to fetch release");
        return resp.json();
      })
      .then(function (release) {
        releaseLink.href = release.html_url;
        status.textContent = "Latest release: " + release.tag_name;
        renderReleaseAssets(release.assets || []);
        setLink(
          linuxLink,
          findAsset(release.assets || [], "linux"),
          defaultDownloadLinks.linux || release.html_url,
          "pyc-linux-x86_64.tar.gz"
        );
        setLink(
          macosLink,
          findAsset(release.assets || [], "macos"),
          defaultDownloadLinks.macos || release.html_url,
          "pyc-macos-arm64.tar.gz"
        );
        setLink(
          windowsLink,
          findAsset(release.assets || [], "windows"),
          defaultDownloadLinks.windows || release.html_url,
          "pyc-windows-x86_64.zip"
        );
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
    fetch(toHref("website/results/manifest.json"))
      .then(function (resp) {
        if (!resp.ok) throw new Error("local manifest unavailable");
        return resp.json();
      })
      .catch(function () {
        return fetch("https://raw.githubusercontent.com/DarkStarStrix/PyC/main/website/results/manifest.json")
          .then(function (resp) {
            if (!resp.ok) throw new Error("remote manifest unavailable");
            return resp.json();
          });
      })
      .then(function (manifest) {
        resultsStatus.textContent =
          "Pinned benchmark run: " + PHASE4.runId +
          " | published artifacts: " +
          manifest.counts.total +
          " (" + manifest.counts.images + " SVG, " + manifest.counts.metadata + " metadata JSON)";

        var svgs = (manifest.artifacts || []).filter(function (entry) {
          return entry.kind === "image_svg";
        });
        renderSvgGallery(svgs);
      })
      .catch(function () {
        resultsStatus.textContent = "Pinned benchmark run: " + PHASE4.runId + " (manifest unavailable; showing pinned charts and fallback stats).";
      });
  }

  initThemeToggle();
  renderPinnedPhase4Charts();
  loadPhase4Stats();
  loadRelease();
  loadPublishedResults();
})();
