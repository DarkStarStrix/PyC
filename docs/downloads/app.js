(function () {
  "use strict";

  var owner = "DarkStarStrix";
  var repo = "PyC";
  var api = "https://api.github.com/repos/" + owner + "/" + repo + "/releases/latest";

  var recommended = document.getElementById("recommended-link");
  var releaseLink = document.getElementById("release-link");
  var status = document.getElementById("status");
  var assetList = document.getElementById("asset-list");

  function detectOs() {
    var ua = navigator.userAgent.toLowerCase();
    if (ua.indexOf("win") !== -1) return "windows";
    if (ua.indexOf("mac") !== -1) return "macos";
    if (ua.indexOf("linux") !== -1 || ua.indexOf("x11") !== -1) return "linux";
    return "unknown";
  }

  function preferredAsset(assets, os) {
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

  function setRecommended(label, href) {
    recommended.textContent = label;
    recommended.href = href;
    recommended.classList.remove("disabled");
    recommended.removeAttribute("aria-disabled");
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
      var os = detectOs();
      var asset = preferredAsset(release.assets || [], os);

      releaseLink.href = release.html_url;
      status.textContent = "Latest release: " + release.tag_name;
      renderAssets(release.assets || []);

      if (asset) {
        setRecommended("Download for " + os + " (" + asset.name + ")", asset.browser_download_url);
      } else {
        setRecommended("Open latest release assets", release.html_url);
      }
    })
    .catch(function () {
      status.textContent = "Could not load release metadata. Open latest release manually.";
      setRecommended("Open latest release", "https://github.com/" + owner + "/" + repo + "/releases/latest");
    });
})();
