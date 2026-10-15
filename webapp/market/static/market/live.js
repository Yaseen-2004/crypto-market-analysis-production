/*
 * Live refresh indicator.
 *
 * Polls the status endpoint, shows when the data was last refreshed and counts
 * down to the next scheduled refresh. When the server finishes a new refresh the
 * page reloads onto the new data - after asking the page to save its state
 * (window.__captureState), so the selected coin and filters survive.
 */
(function () {
  "use strict";

  const script = document.currentScript;
  const STATUS_URL = (script && script.dataset.statusUrl) || "/api/status/";
  const REFRESH_URL = (script && script.dataset.refreshUrl) || "/api/refresh/";
  const POLL_MS = 20000;
  const RELOAD_GUARD = "live-reloaded-for-run";

  /* ---------- styles: borrow whichever token set the host page defines ---------- */
  const css = `
  .live-pill {
    position: fixed; right: 16px; bottom: calc(16px + env(safe-area-inset-bottom, 0px)); z-index: 2147483000;
    display: flex; align-items: center; gap: 10px; max-width: calc(100vw - 32px);
    padding: 7px 8px 7px 12px; border-radius: 999px;
    background: var(--panel, var(--surface-2, #ffffff));
    color: var(--ink, #0b0f14);
    border: 1px solid var(--line, var(--hairline, rgba(0,0,0,.12)));
    box-shadow: 0 6px 22px rgba(0,0,0,.22);
    font: 500 12px/1.2 "IBM Plex Sans", "Figtree", system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--gain, #0a9d4a); }
  .live-pill[data-state="running"] .live-dot { background: var(--accent, #2a78d6); animation: live-pulse 1.1s ease-in-out infinite; }
  .live-pill[data-state="failed"] .live-dot { background: var(--loss, #d03b3b); }
  .live-pill[data-state="offline"] .live-dot, .live-pill[data-state="waiting"] .live-dot { background: var(--warn, #b7791f); }
  @keyframes live-pulse { 50% { opacity: .35; transform: scale(.75); } }
  .live-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .live-text b { font-weight: 600; }
  .live-muted { color: var(--ink-3, #7d8ba0); font-variant-numeric: tabular-nums; }
  .live-btn {
    flex: none; border: 1px solid var(--line, var(--hairline, rgba(0,0,0,.12))); border-radius: 999px;
    background: var(--panel-2, var(--surface-3, #eef1f5)); color: inherit; cursor: pointer;
    font: 600 11.5px/1 inherit; padding: 6px 10px;
  }
  .live-btn:hover:not(:disabled), .live-btn:focus-visible { border-color: var(--accent, #2a78d6); outline: none; }
  .live-btn:disabled { opacity: .5; cursor: default; }
  .live-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
  @media (max-width: 680px) { .live-pill { bottom: calc(88px + env(safe-area-inset-bottom, 0px)); } }
  @media (prefers-reduced-motion: reduce) { .live-pill[data-state="running"] .live-dot { animation: none; } }`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const pill = document.createElement("div");
  pill.className = "live-pill";
  pill.dataset.state = "waiting";
  pill.innerHTML =
    '<span class="live-dot" aria-hidden="true"></span>' +
    '<span class="live-text" aria-hidden="true">Connecting…</span>' +
    '<button class="live-btn" type="button">Refresh now</button>' +
    '<span class="live-sr" role="status" aria-live="polite"></span>';
  const text = pill.querySelector(".live-text");
  const button = pill.querySelector(".live-btn");
  const announcer = pill.querySelector(".live-sr");
  document.body.appendChild(pill);

  /* ---------- helpers ---------- */
  let status = null;
  let clockOffset = 0;          // server time minus local time
  let lastAnnounced = "";
  let flash = null;             // temporary message after pressing the button

  const serverNow = () => Date.now() + clockOffset;
  const ago = (iso) => {
    const s = Math.max(0, Math.round((serverNow() - Date.parse(iso)) / 1000));
    if (s < 60) return "just now";
    const m = Math.round(s / 60);
    return m < 60 ? m + " min ago" : Math.floor(m / 60) + " h " + (m % 60) + " min ago";
  };
  const nextRefresh = (iso) => {
    const s = Math.round((Date.parse(iso) - serverNow()) / 1000);
    if (s <= 0) return "next refresh due now";
    return "next refresh in " + Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  };
  const csrf = () => (document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/) || [])[1] || "";
  const announce = (msg) => { if (msg !== lastAnnounced) { announcer.textContent = msg; lastAnnounced = msg; } };

  function pageRunId() {
    const meta = window.CRYPTO_DATA && window.CRYPTO_DATA.meta;
    return meta ? meta.runId : undefined;   // undefined = page has no data payload (e.g. the index)
  }
  let baseline = null;                       // first success id seen, for pages without a payload

  function maybeReload() {
    const success = status && status.lastSuccess;
    if (!success) return;
    const current = pageRunId();
    let stale;
    if (current === undefined) {
      if (baseline === null) { baseline = success.id; return; }
      stale = success.id !== baseline;
    } else {
      stale = current !== success.id;        // includes current === null (static fallback data)
    }
    if (!stale) return;
    let guard = null;
    try { guard = sessionStorage.getItem(RELOAD_GUARD); } catch (e) { /* storage unavailable */ }
    if (guard === String(success.id)) return; // already reloaded once for this run - avoid a loop
    try { sessionStorage.setItem(RELOAD_GUARD, String(success.id)); } catch (e) { /* ignore */ }
    if (typeof window.__captureState === "function") {
      try { window.__captureState(); } catch (e) { /* page state is a convenience, never block */ }
    }
    announce("New market data arrived. Reloading.");
    window.location.reload();
  }

  function render() {
    if (flash && Date.now() < flash.until) {
      text.innerHTML = flash.html;
      return;
    }
    flash = null;
    if (!status) return;
    const success = status.lastSuccess;
    const last = status.lastRun;
    const sched = status.scheduler || {};
    const next = sched.alive && sched.nextRunAt
      ? ' <span class="live-muted">· ' + nextRefresh(sched.nextRunAt) + "</span>"
      : ' <span class="live-muted">· scheduler offline</span>';

    let state, html, spoken;
    if (status.running) {
      state = "running";
      html = "<b>Refreshing market data…</b>" +
        (status.runningSince ? ' <span class="live-muted">started ' + ago(status.runningSince) + "</span>" : "");
      spoken = "Refreshing market data.";
    } else if (last && last.status === "failed" && (!success || last.id > success.id)) {
      state = "failed";
      html = "<b>Last refresh failed</b>" + next;
      spoken = "The last refresh failed.";
      pill.title = last.error || "";
    } else if (success) {
      state = sched.alive ? "ok" : "offline";
      html = "Updated <b>" + ago(success.finishedAt) + "</b>" + next;
      spoken = sched.alive ? "Market data is up to date." : "Scheduler is offline.";
      pill.title = "Refresh #" + success.id + " · " + success.coins + " coins · every " + status.intervalMinutes + " minutes";
    } else {
      state = "waiting";
      html = "<b>Waiting for the first refresh</b>" + next;
      spoken = "Waiting for the first refresh.";
    }
    pill.dataset.state = state;
    text.innerHTML = html;
    button.disabled = !!status.running;
    announce(spoken);
  }

  async function poll() {
    try {
      const res = await fetch(STATUS_URL, { headers: { Accept: "application/json" }, cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      status = await res.json();
      if (status.serverTime) clockOffset = Date.parse(status.serverTime) - Date.now();
      render();
      maybeReload();
    } catch (e) {
      pill.dataset.state = "offline";
      text.innerHTML = "<b>Server unreachable</b> <span class=\"live-muted\">· retrying</span>";
      announce("Server unreachable.");
    }
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const res = await fetch(REFRESH_URL, {
        method: "POST",
        headers: { "X-CSRFToken": csrf(), Accept: "application/json" },
        credentials: "same-origin",
      });
      const body = await res.json().catch(() => ({}));
      const msg = res.status === 202 ? "<b>Refresh started</b>"
        : "<b>" + (body.reason || "Could not start a refresh") + "</b>";
      flash = { html: msg, until: Date.now() + 4000 };
      render();
    } catch (e) {
      flash = { html: "<b>Could not reach the server</b>", until: Date.now() + 4000 };
      render();
    }
    setTimeout(poll, 1500);
  });

  poll();
  setInterval(poll, POLL_MS);
  setInterval(render, 1000);   // keeps the countdown ticking between polls
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
})();
