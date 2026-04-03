export type ThemeMode = "light" | "dark";

export function byId<T extends HTMLElement>(id: string): T | null {
    return document.getElementById(id) as T | null;
}

export function formatNumber(value: unknown, digits: number): string {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }
    return value.toFixed(digits);
}

export function toSiteHref(path: string): string {
    if (!path) {
        return "#";
    }
    if (/^https?:\/\//i.test(path)) {
        return path;
    }
    return new URL(path, window.location.href).toString();
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(path, init);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${path}: ${response.status}`);
    }
    return await response.json() as T;
}

export async function fetchText(path: string, init?: RequestInit): Promise<string> {
    const response = await fetch(path, init);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${path}: ${response.status}`);
    }
    return await response.text();
}

export async function fetchJsonWithFallback<T>(paths: string[], init?: RequestInit): Promise<T> {
    let lastError: unknown = null;
    for (const path of paths) {
        try {
            return await fetchJson<T>(toSiteHref(path), init);
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError ?? new Error("No fetch paths succeeded.");
}

export function preferredTheme(): ThemeMode {
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

export function applyTheme(theme: ThemeMode, button: HTMLButtonElement | null): void {
    const resolved: ThemeMode = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolved);
    if (button) {
        button.textContent = resolved === "dark" ? "Light Mode" : "Dark Mode";
        button.setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
    }
}

export function initThemeToggle(buttonId = "theme-toggle"): void {
    const button = byId<HTMLButtonElement>(buttonId);
    applyTheme(preferredTheme(), button);
    if (!button) {
        return;
    }

    button.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
        const next: ThemeMode = current === "dark" ? "light" : "dark";
        applyTheme(next, button);
        try {
            window.localStorage.setItem("pyc-theme", next);
        } catch (_error) {
        }
    });
}

export function setImageWithFallback(
    image: HTMLImageElement | null,
    source: string | null | undefined,
    fallback?: string | null
): void {
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
