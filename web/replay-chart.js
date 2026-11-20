import { $, el, fmt, pctf, secs } from "./ui.js";

export const tierStyles = {
  gpu: { name: "GPU KV cache", color: "#1247ee" },
  cpu: { name: "CPU memory", color: "#2778b7" },
  disk: { name: "Local SSD", color: "#7952ab" },
  p2p: { name: "Peer CPU cache (P2P)", color: "#ac5b16" },
  shared: { name: "Object storage", color: "#098d81" },
};
const left = 72, right = 15;

function niceMax(value) {
  const base = 10 ** Math.floor(Math.log10(Math.max(0.001, value)));
  return [1, 1.5, 2, 2.5, 4, 5, 7.5, 10].find(n => n * base >= value) * base;
}

// Every track uses the same minute-end x coordinates. Occupancy starts at zero;
// the latency line starts at the first minute with requests and breaks at idle minutes.
function drawTrack(track, selected) {
  const { canvas, values, max, unit, color, firstRead, occupancy } = track;
  const width = canvas.clientWidth, height = canvas.clientHeight, ratio = devicePixelRatio || 1;
  if (!width || !height) return;
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const top = 11, bottom = height - 23;
  const x = minute => left + minute / values.length * (width - left - right);
  const y = value => bottom - value / max * (bottom - top);
  ctx.font = '12px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = "right";
  for (const fraction of [0, 0.5, 1]) {
    const value = fraction * max, py = y(value);
    ctx.strokeStyle = "#dee3eb"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(width - right, py); ctx.stroke();
    ctx.fillStyle = "#516075"; ctx.fillText(fmt(value) + unit, left - 5, py + 4);
  }
  ctx.textAlign = "center";
  const interval = width < 480 ? 15 : 10;
  for (let i = 0; i <= values.length; i += interval) {
    ctx.fillText(String(i), x(i), height - 5);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
  let connected = occupancy;
  if (occupancy) ctx.moveTo(x(0), y(0));
  values.forEach((v, i) => {
    if (v == null) { connected = false; return; }
    if (connected) ctx.lineTo(x(i + 1), y(v));
    else ctx.moveTo(x(i + 1), y(v));
    connected = true;
  });
  ctx.stroke();
  const dot = (i, radius, hollow = false) => {
    if (values[i] == null) return;
    ctx.beginPath(); ctx.arc(x(i + 1), y(values[i]), radius, 0, 2 * Math.PI);
    ctx.fillStyle = hollow ? "white" : color; ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
  };
  for (let i = 0; i < values.length; i++) {
    if (values[i] != null && values[i-1] == null && values[i+1] == null) dot(i, 2);
  }
  if (firstRead >= 0) dot(firstRead, 4, true);
  ctx.strokeStyle = "#788599"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(x(selected + 1), top); ctx.lineTo(x(selected + 1), bottom); ctx.stroke();
  ctx.setLineDash([]); dot(selected, 3);
  for (const marker of track.markers ?? []) {
    const px = x(marker.time / 60);
    ctx.strokeStyle = marker.color; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = marker.color;
    // Opposite ends keep simultaneous SSD and S3 reads individually visible.
    ctx.fillRect(px - 3, (marker.key === "disk" ? top : bottom) - 3, 6, 6);
  }
}

export function timeline() {
  const slider = $("inspect-minute"), chart = $("chart"), container = $("tier-tracks");
  const scope = $("reader-scope"), events = $("read-events");
  // selected starts unset; the first replay opens on its last minute, where occupancy is highest.
  let minutes = [], tracks = [], selected = -1, runStats = {};
  const resize = new ResizeObserver(() => tracks.forEach(track => drawTrack(track, selected)));

  function showReaders() {
    const stats = scope.value === "replay" ? runStats : minutes[selected]?.readerStats;
    for (const track of tracks) {
      if (!track.cells) continue;
      const sample = stats?.[track.key];
      const values = [(sample?.count ?? 0).toLocaleString("en-US"), secs(sample?.medianTTFT), fmt(sample?.medianMiB)];
      track.cells.forEach((cell, i) => { cell.textContent = values[i]; });
    }
  }
  scope.onchange = showReaders;

  function inspect(index) {
    if (!minutes.length) return;
    selected = Math.max(0, Math.min(minutes.length - 1, index));
    const m = minutes[selected];
    showReaders();
    slider.value = selected + 1;
    slider.style.setProperty("--fill", (selected / Math.max(1, minutes.length - 1) * 100) + "%");
    $("minute-label").textContent = selected + 1;
    $("minute-heading").textContent = "Minute " + (selected + 1) + " average TTFT";
    scope.options[0].textContent = "Minute " + (selected + 1);
    $("minute-ttft").textContent = secs(m.mean);
    $("minute-time").textContent = m.count
      ? m.count + " requests · " + secs(m.load) + " loading cache · " + secs(m.transfer) + " transferring KV to decode · " + secs(m.wait) + " queueing per request"
      : "No requests arrived in this minute.";
    for (const track of tracks) {
      drawTrack(track, selected);
      if (!track.key) continue;
      const readers = m.reads[track.key] ?? 0;
      if (track.key === "p2p") {
        track.amount.textContent = m.count ? pctf(readers / m.count, 1) + " of requests" : "No requests";
      } else {
        const used = m.usedGiB[track.key], cap = track.capacity;
        track.amount.textContent = cap == null ? fmt(used) + " GiB stored · uncapped"
        : fmt(used) + " / " + fmt(cap) + " GiB · " + (used > 0 && used / cap < 0.0005 ? "<0.1%" : pctf(used / cap, 1)) + " full";
      }
      const peerVolume = track.key === "p2p" && readers
        ? ` · ${fmt(m.peerBytes / 2 ** 20)} MiB transferred` : "";
      const summary = m.count ? readers + " of " + m.count + " requests read this tier" : "No requests";
      track.reads.textContent = readers ? summary + peerVolume + " · " + secs(m.readerStats[track.key]?.medianTTFT) + " median TTFT for those requests" : summary;
    }
  }
  function addTrack(track) {
    tracks.push(track);
    resize.observe(track.canvas);
    const pick = e => {
      const box = track.canvas.getBoundingClientRect();
      const minute = Math.round((e.clientX - box.left - left) / (box.width - left - right) * minutes.length);
      inspect(minute - 1);
    };
    // Selection follows a press or drag, not a passing pointer, so the reader table stays put while scrolling.
    track.canvas.onpointermove = e => { if (e.buttons) pick(e); };
    track.canvas.onpointerdown = pick;
    track.canvas.onkeydown = e => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      e.preventDefault(); inspect(selected + (e.key === "ArrowLeft" ? -1 : 1));
    };
  }
  function setOccupancyScale(track) {
    const asPercent = track.capacity != null && $("occupancy-scale").value === "percent";
    track.values = minutes.map(m => asPercent ? m.usedGiB[track.key] / track.capacity * 100 : m.usedGiB[track.key]);
    track.unit = asPercent ? "%" : " GiB";
    // A tier that stays nearly empty gets a zoomed percent axis rather than a flat line at 0%.
    track.max = asPercent ? Math.min(100, niceMax(Math.max(1, ...track.values))) : niceMax(Math.max(...track.values));
  }
  $("occupancy-scale").onchange = () => {
    tracks.filter(track => track.occupancy).forEach(setOccupancyScale);
    inspect(selected);
  };
  slider.oninput = () => inspect(slider.valueAsNumber - 1);

  return result => {
    resize.disconnect();
    container.replaceChildren();
    events.replaceChildren();
    $("reader-rows").replaceChildren();
    minutes = result?.minutes ?? [];
    tracks = [];
    runStats = result?.readerStats ?? {};
    $("reader-summary").hidden = !minutes.length;
    slider.disabled = !minutes.length;
    if (!minutes.length) {
      chart.getContext("2d").clearRect(0, 0, chart.width, chart.height);
      $("minute-label").textContent = $("minute-ttft").textContent = "—";
      $("minute-heading").textContent = "Average TTFT by minute";
      $("minute-time").textContent = "";
      return;
    }
    slider.max = minutes.length;
    if (selected < 0 || selected >= minutes.length) selected = minutes.length - 1;
    const peak = Math.max(...minutes.map(m => m.mean ?? 0));
    const scale = peak < 2 ? 1000 : 1;
    const markers = [];
    for (const [key, name] of [["disk", "physical SSD read"], ["shared", "S3 read"]]) {
      if (!(key in result.tiers)) continue;
      const time = result.firstPayloadReadS[key];
      if (time == null) {
        events.append(el("span", { text: "No " + name + " during replay" }));
        continue;
      }
      const minute = Math.floor(time / 60);
      const stamp = minute + ":" + String(Math.floor(time % 60)).padStart(2, "0");
      const label = "First " + name + " · " + stamp;
      if (minute >= minutes.length) {
        events.append(el("span", { text: label + " (after timeline)" }));
        continue;
      }
      const color = tierStyles[key].color;
      markers.push({ key, time, color });
      events.append(el("button", { type: "button", onclick: () => { scope.value = "minute"; inspect(minute); } },
        el("i", { style: "background:" + color }), label));
    }
    addTrack({ canvas: chart, values: minutes.map(m => m.mean == null ? null : m.mean * scale),
      max: niceMax(peak * scale), unit: scale === 1000 ? " ms" : " s", color: "#1247ee", markers });

    for (const key of result.order) {
      const peer = key === "p2p";
      if (!peer && !(key in result.tiers)) continue;
      const capacity = result.tiers[key];
      const { name, color } = tierStyles[key];
      const amount = el("strong"), reads = el("span");
      const canvas = el("canvas", { role: "img", tabIndex: 0,
        "aria-label": name + (peer ? " share of requests by minute." : " occupancy by minute.") + " Use the arrow keys to inspect a minute." });
      const firstRead = minutes.findIndex(m => m.reads[key] > 0);
      const event = firstRead < 0 ? el("span", { text: "No reads during replay" })
        : el("button", { type: "button", text: "First reader arrived · min " + (firstRead + 1),
          "aria-label": "Inspect first " + name + " reader's arrival in minute " + (firstRead + 1), onclick: () => inspect(firstRead) });
      let eviction;
      if (key === "disk") {
        const minute = Math.floor(result.firstDiskEviction / 60);
        eviction = result.firstDiskEviction == null ? el("span", { text: "No SSD evictions" })
          : minute < 0 ? el("span", { text: "SSD evictions began during warm-up" })
          : el("button", { type: "button", text: "First SSD eviction · min " + (minute + 1),
            onclick: () => inspect(minute) });
      }
      let note;
      if (key === "gpu") {
        note = `KV budget is ${result.instances} instance${result.instances === 1 ? "" : "s"} × ${fmt(result.gpuGiB, 4)} GiB after weights and runtime reserve. Global KV uses ${fmt(result.kvBytesPerToken, 5)} bytes/token.${result.replayTokens ? ` Each hit rebuilds up to ${result.replayTokens} tokens of live window (${fmt(result.windowBytes / 2 ** 20, 3)} MiB per active request).` : ""} Replay chunks contain ${result.blockTokens} tokens.`;
      } else if (peer) {
        note = "Reads from another node’s CPU cache over the interconnect. Its bytes are already included in CPU memory.";
      }
      container.append(el("section", { class: "tier-track" },
        el("div", { class: "track-heading" }, el("h3", {}, el("i", { style: "background:" + color }), name), amount),
        canvas, el("div", { class: "track-note" }, reads, eviction, event), note && el("p", { class: "track-note", text: note })));
      const cells = Array.from({ length: 3 }, () => el("td"));
      $("reader-rows").append(el("tr", {}, el("th", { scope: "row", text: name }), cells));
      const track = { key, capacity, canvas, amount, reads, cells, occupancy: !peer, color, firstRead };
      if (peer) {
        track.values = minutes.map(m => m.count ? 100 * (m.reads.p2p ?? 0) / m.count : null);
        track.max = Math.min(100, niceMax(Math.max(1, ...track.values)));
        track.unit = "%";
      } else setOccupancyScale(track);
      addTrack(track);
    }
    inspect(selected);
  };
}
