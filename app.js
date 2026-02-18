(function () {
  "use strict";

  var owner = "DarkStarStrix";
  var repo = "PyC";
  var api = "https://api.github.com/repos/" + owner + "/" + repo + "/releases/latest";

  var releaseLink = document.getElementById("release-link");
  var linuxLink = document.getElementById("download-linux");
  var macosLink = document.getElementById("download-macos");
  var windowsLink = document.getElementById("download-windows");
  var status = document.getElementById("status");
  var assetList = document.getElementById("asset-list");

  var resultsStatus = document.getElementById("results-status");
  var cpuBody = document.getElementById("cpu-results-body");
  var gpuBody = document.getElementById("gpu-results-body");
  var svgList = document.getElementById("svg-list");
  var metadataList = document.getElementById("metadata-list");
  var latestCharts = document.getElementById("latest-charts");
  var latestCpuSvg = document.getElementById("latest-cpu-svg");
  var latestGpuSvg = document.getElementById("latest-gpu-svg");
  var svgGallery = document.getElementById("svg-gallery");

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

  function setLink(el, asset, fallback) {
    if (!el) return;
    if (asset) {
      el.textContent = asset.name;
      el.href = asset.browser_download_url;
      el.removeAttribute("aria-disabled");
    } else {
      el.textContent = "latest release assets";
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

  function appendLinks(container, entries) {
    container.innerHTML = "";
    if (!entries.length) {
      var li = document.createElement("li");
      li.textContent = "None published.";
      container.appendChild(li);
      return;
    }

    entries.forEach(function (entry) {
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "./" + entry.published;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = entry.source;
      li.appendChild(a);
      container.appendChild(li);
    });
  }

  function findLatestChart(manifest, suffix) {
    if (!manifest || !manifest.artifacts) return null;
    var charts = manifest.artifacts.filter(function (entry) {
      return entry.kind === "image_svg" && entry.source.indexOf("__" + suffix + ".svg") !== -1;
    });
    if (!charts.length) return null;
    charts.sort(function (a, b) {
      return a.source < b.source ? -1 : a.source > b.source ? 1 : 0;
    });
    return charts[charts.length - 1];
  }

  function renderLatestCharts(manifest) {
    latestCharts.innerHTML = "";

    var cpu = findLatestChart(manifest, "cpu");
    var gpu = findLatestChart(manifest, "gpu");
    [cpu, gpu].forEach(function (entry) {
      if (!entry) return;
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "./" + entry.published;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = entry.source;
      li.appendChild(a);
      latestCharts.appendChild(li);
    });

    if (cpu && latestCpuSvg) {
      latestCpuSvg.src = "./" + cpu.published;
    }
    if (gpu && latestGpuSvg) {
      latestGpuSvg.src = "./" + gpu.published;
    }

    if (!latestCharts.children.length) {
      var empty = document.createElement("li");
      empty.textContent = "No latest CPU/GPU charts found.";
      latestCharts.appendChild(empty);
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
      var item = document.createElement("a");
      item.className = "svg-preview";
      item.href = "./" + entry.published;
      item.target = "_blank";
      item.rel = "noopener noreferrer";

      var img = document.createElement("img");
      img.src = "./" + entry.published;
      img.alt = entry.source;
      img.loading = "lazy";

      var label = document.createElement("span");
      label.textContent = entry.source;

      item.appendChild(img);
      item.appendChild(label);
      svgGallery.appendChild(item);
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
        setLink(linuxLink, findAsset(release.assets || [], "linux"), release.html_url);
        setLink(macosLink, findAsset(release.assets || [], "macos"), release.html_url);
        setLink(windowsLink, findAsset(release.assets || [], "windows"), release.html_url);
      })
      .catch(function () {
        status.textContent = "Could not load release metadata. Open latest release manually.";
        var fallback = "https://github.com/" + owner + "/" + repo + "/releases/latest";
        setLink(linuxLink, null, fallback);
        setLink(macosLink, null, fallback);
        setLink(windowsLink, null, fallback);
        releaseLink.href = fallback;
      });
  }

  function loadPublishedResults() {
    Promise.all([
      fetch("./website/results/manifest.json").then(function (resp) {
        if (!resp.ok) throw new Error("manifest unavailable");
        return resp.json();
      }),
      fetch("./website/results/latest-summary.json").then(function (resp) {
        if (!resp.ok) throw new Error("latest summary unavailable");
        return resp.json();
      })
    ])
      .then(function (payload) {
        var manifest = payload[0];
        var latest = payload[1];

        resultsStatus.textContent =
          "Published artifacts: " +
          manifest.counts.total +
          " (" + manifest.counts.images + " SVG, " + manifest.counts.metadata + " metadata JSON)";

        renderRows(cpuBody, latest.cpu ? latest.cpu.adapters : []);
        renderRows(gpuBody, latest.gpu ? latest.gpu.adapters : []);

        renderLatestCharts(manifest);

        var svgs = (manifest.artifacts || []).filter(function (entry) {
          return entry.kind === "image_svg";
        });
        var metadata = (manifest.artifacts || []).filter(function (entry) {
          return entry.kind === "metadata_json";
        });

        appendLinks(svgList, svgs);
        appendLinks(metadataList, metadata);
        renderSvgGallery(svgs);
      })
      .catch(function () {
        resultsStatus.textContent = "Published benchmark data could not be loaded.";
      });
  }

  loadRelease();
  loadPublishedResults();
})();
