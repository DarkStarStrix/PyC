export function byId(id) {
    return document.getElementById(id);
}

export function formatNumber(value, digits) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }
    return value.toFixed(digits);
}

export function toSiteHref(path) {
    if (!path) {
        return "#";
    }
    if (/^https?:\/\//i.test(path)) {
        return path;
    }
    return new URL(path, window.location.href).toString();
}

export async function fetchJson(path, init) {
    const response = await fetch(path, init);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${path}: ${response.status}`);
    }
    return await response.json();
}

export async function fetchText(path, init) {
    const response = await fetch(path, init);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${path}: ${response.status}`);
    }
    return await response.text();
}

export async function fetchJsonWithFallback(paths, init) {
    let lastError = null;
    for (const path of paths) {
        try {
            return await fetchJson(toSiteHref(path), init);
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError ?? new Error("No fetch paths succeeded.");
}

export function preferredTheme() {
    try {
        const stored = window.localStorage.getItem("pyc-theme");
        if (stored === "light" || stored === "dark") {
            return stored;
        }
    } catch (_error) {
    }
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
    }
    return "light";
}

export function applyTheme(theme, button) {
    const resolved = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolved);
    if (button) {
        button.textContent = resolved === "dark" ? "Light Mode" : "Dark Mode";
        button.setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
    }
}

export function initThemeToggle(buttonId = "theme-toggle") {
    const button = byId(buttonId);
    applyTheme(preferredTheme(), button);
    if (!button) {
        return;
    }
    button.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next, button);
        try {
            window.localStorage.setItem("pyc-theme", next);
        } catch (_error) {
        }
    });
}

export function setImageWithFallback(image, source, fallback) {
    if (!image) {
        return;
    }
    const fallbackHref = fallback ? toSiteHref(fallback) : null;
    if (source) {
        image.src = toSiteHref(source);
    } else if (fallbackHref) {
        image.src = fallbackHref;
    }
    if (fallbackHref) {
        image.onerror = () => {
            if (image.src !== fallbackHref) {
                image.src = fallbackHref;
            }
        };
    }
}
