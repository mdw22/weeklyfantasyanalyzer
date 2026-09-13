const PREFIX = "wfa:";

// localStorage is the only persistence mechanism in this app (no backend,
// no accounts) -- wrap every access so a private-browsing / blocked-storage
// environment degrades to "nothing persists" instead of throwing.
export function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function saveJSON(key, value) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // ignore -- e.g. private browsing with storage disabled
  }
}
