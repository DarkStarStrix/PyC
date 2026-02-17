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

  function renderAssets(assets) {
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

  fetch(api)
    .then(function (resp) {
      if (!resp.ok) throw new Error("Failed to fetch release");
      return resp.json();
    })
    .then(function (release) {
      releaseLink.href = release.html_url;
      status.textContent = "Latest release: " + release.tag_name;
      renderAssets(release.assets || []);
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
})();
