/**
 * OceanEmbedAPI - tiny fetch wrapper shared by every frontend variant.
 *
 * Design intent (matches backend/app/services/gateway.py):
 *   - Every call resolves to { ok: true, data, status } on success, where
 *     `status` is whatever the backend gateway put there ("live" | "cached"
 *     | "demo" | "computed"), or { ok: false, error } if the backend itself
 *     is unreachable (e.g. running this HTML file directly with no backend
 *     running at all).
 *   - Callers are expected to fall back to their own embedded demo JSON on
 *     `ok:false` -- this file never invents data, only relays or reports
 *     failure, so "no fake data" holds at this layer too.
 *   - Base URL is configurable via `window.OCEANEMBED_API_BASE` (defaults
 *     to same-origin /api, which is correct once the FastAPI app also
 *     serves the built frontend; override for local dev against
 *     `http://localhost:8000/api`).
 */
(function (global) {
  const DEFAULT_BASE = (global.OCEANEMBED_API_BASE || "http://localhost:8000/api");

  async function request(path, { method = "GET", body, timeoutMs = 6000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(`${DEFAULT_BASE}${path}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (!resp.ok) {
        return { ok: false, error: `HTTP ${resp.status}` };
      }
      const data = await resp.json();
      return { ok: true, data };
    } catch (err) {
      clearTimeout(timer);
      return { ok: false, error: err && err.message ? err.message : String(err) };
    }
  }

  const OceanEmbedAPI = {
    setBase(url) { this._base = url; },
    health: () => request("/health"),
    regions: () => request("/regions"),
    locateRegion: (lat, lon) => request(`/regions/locate?lat=${lat}&lon=${lon}`),
    overview: (region) => request(`/overview?region=${encodeURIComponent(region)}`),
    map: (region, depth) => request(`/ocean/map?region=${encodeURIComponent(region)}&depth=${depth}`),
    profile: (region, lat, lon) => request(`/ocean/profile?region=${encodeURIComponent(region)}&lat=${lat}&lon=${lon}`),
    argoNearby: (lat, lon) => request(`/argo?lat=${lat}&lon=${lon}`),
    activeLearningRecommendations: (region, params = {}) => {
      const qs = new URLSearchParams({ region, ...params }).toString();
      return request(`/active-learning/recommendations?${qs}`);
    },
    activeLearningMap: (region) => request(`/active-learning/map?region=${encodeURIComponent(region)}`),
    cyclones: () => request("/cyclones"),
    timeline: (region, days = 30) => request(`/timeline?region=${encodeURIComponent(region)}&days=${days}`),
    generateReport: (region, lat, lon) => request("/report", { method: "POST", body: { region, lat, lon } }),

    /** Small colored pill for a gateway status string -- shared styling
     * across all three frontends so "live" always reads the same way. */
    statusBadgeHTML(status) {
      const map = {
        live: ["LIVE", "#5BC98A"],
        cached: ["CACHED", "#F5C451"],
        demo: ["DEMO", "#E8D8A8"],
        computed: ["COMPUTED", "#4FC3F7"],
        error: ["OFFLINE", "#E2665B"],
      };
      const [label, color] = map[status] || ["UNKNOWN", "#9FBAC7"];
      return `<span class="oe-badge" style="display:inline-flex;align-items:center;gap:4px;` +
        `font-size:10.5px;font-weight:700;letter-spacing:.03em;padding:2px 7px;border-radius:100px;` +
        `background:${color}22;color:${color};border:1px solid ${color}55;">● ${label}</span>`;
    },
  };

  global.OceanEmbedAPI = OceanEmbedAPI;
})(window);
