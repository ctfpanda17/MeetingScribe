// Reads "who is speaking" from Google Meet / Microsoft Teams (web) and reports it to the background worker.
// Two independent cues, used together:
//   1. the participant tile the meeting app highlights while someone talks ("tile")
//   2. live captions, which carry the speaker's name next to the words ("captions")
// Meeting apps change their markup often, so every rule is a small, separate heuristic.
(() => {
  console.log("[MeetingScribe] connector loaded on", location.host);
  const PLATFORM = location.host.includes("meet.google") ? "meet" : "teams";
  const SPEAKING_WORDS = /(正在說話|正在讲话|說話中|发言中|is speaking|speaking|is talking)/i;
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();

  // ---------------------------------------------------------------- 1. highlighted tiles / aria labels
  function speakingFromAria() {
    const names = new Set();
    for (const el of document.querySelectorAll("[aria-label]")) {
      const label = el.getAttribute("aria-label");
      if (label && label.length < 120 && SPEAKING_WORDS.test(label)) {
        const name = clean(label.split(/[,，、]/)[0].replace(SPEAKING_WORDS, ""));
        if (name && name.length <= 40) names.add(name);
      }
    }
    return names;
  }

  function speakingMeetTiles() {
    // Meet: each participant tile carries data-participant-id; the speaking one shows an animated
    // audio indicator. Visible indicator = an element whose class list changes while audio plays,
    // so look for the indicator Meet marks with data-audio-level / the "speaking" border class.
    const names = new Set();
    for (const tile of document.querySelectorAll("[data-participant-id]")) {
      const nameEl = tile.querySelector("[data-self-name], [data-participant-name]");
      const name = clean(nameEl ? nameEl.getAttribute("data-self-name") || nameEl.textContent : "");
      if (!name) continue;
      const indicator = tile.querySelector("[data-audio-level], [data-is-speaking='true']");
      if (indicator && indicator.getAttribute("data-is-speaking") === "true") names.add(name);
    }
    return names;
  }

  function speakingTeamsTiles() {
    // Teams (verified on the web client, Oct 2026): every remote participant's tile is
    // [data-stream-type][data-tid="<display name>"] and contains [data-tid="voice-level-stream-outline"].
    // While that person talks, the outline gets the class "vdi-frame-occlusion" and its ::before border turns
    // visible (opacity 1). Your own tile has no outline element (you're covered by the microphone).
    const names = new Set();
    for (const tile of document.querySelectorAll("[data-stream-type][data-tid]")) {
      const ring = tile.querySelector("[data-tid='voice-level-stream-outline']");
      if (!ring) continue;
      const before = getComputedStyle(ring, "::before");
      const on = String(ring.className).includes("vdi-frame-occlusion") || parseFloat(before.opacity) > 0.5;
      const name = clean(tile.getAttribute("data-tid"));
      if (on && name) names.add(name);
    }
    return names;
  }

  // ---------------------------------------------------------------- 2. live captions
  const captionSeen = new Map(); // caption block element -> last text
  const recent = new Map(); // name -> time its caption text last grew

  function captionBlocks() {
    if (PLATFORM === "teams") {
      return [...document.querySelectorAll("[data-tid='closed-caption-text']")].map((t) => {
        const item = t.closest("li, [role='listitem'], .ui-chat__item") || t.parentElement;
        const author = item && item.querySelector("[data-tid='author'], .ui-chat__message__author");
        return { el: t, name: clean(author && author.textContent), text: clean(t.textContent) };
      });
    }
    // Meet: the captions region holds one block per speaker turn: avatar image + name + text
    const region = document.querySelector("[role='region'][aria-label*='字幕'], [role='region'][aria-label*='aption'], div[jsname='dsyhDe']");
    if (!region) return [];
    const out = [];
    for (const img of region.querySelectorAll("img")) {
      const block = img.closest("div[class]")?.parentElement;
      if (!block) continue;
      const parts = [...block.querySelectorAll("div, span")].map((e) => clean(e.textContent)).filter(Boolean);
      const name = parts.find((p) => p.length <= 40) || "";
      const text = parts.sort((a, b) => b.length - a.length)[0] || "";
      if (name && text && text !== name) out.push({ el: block, name, text });
    }
    return out;
  }

  function speakingFromCaptions() {
    const now = Date.now();
    for (const b of captionBlocks()) {
      if (!b.name) continue;
      const prev = captionSeen.get(b.el);
      if (prev !== b.text) {
        captionSeen.set(b.el, b.text);
        if (prev !== undefined) recent.set(b.name, now); // text grew: this person is talking
      }
    }
    const names = new Set();
    for (const [name, t] of recent) if (now - t < 1500) names.add(name);
    return names;
  }

  // ---------------------------------------------------------------- my own name in the meeting
  // The page is briefly inconsistent while it loads, so report the value seen most often.
  const selfVotes = new Map();
  function selfName() {
    const now = rawSelfName();
    if (now && !/個人資料圖片|profile picture|頭像|avatar/i.test(now)) selfVotes.set(now, (selfVotes.get(now) || 0) + 1);
    let best = "", n = 0;
    for (const [k, v] of selfVotes) if (v > n) [best, n] = [k, v];
    return n >= 5 ? best : "";
  }
  function rawSelfName() {
    if (PLATFORM === "meet") {
      const el = document.querySelector("[data-self-name]");
      return clean(el && el.getAttribute("data-self-name"));
    }
    // Teams: the only participant tile without a voice-level outline is your own
    const tiles = [...document.querySelectorAll("[data-stream-type][data-tid]")];
    const mine = tiles.filter((t) => !t.querySelector("[data-tid='voice-level-stream-outline']"));
    return mine.length === 1 ? clean(mine[0].getAttribute("data-tid")) : "";
  }

  // ---------------------------------------------------------------- report loop
  let lastKey = "";
  let lastSent = 0;
  const diag = { v: chrome.runtime.getManifest().version, sent: 0, app: null, last: "" };
  const mark = () => {
    document.documentElement.setAttribute("data-meetingscribe", JSON.stringify(diag));
    if (document.body) document.body.setAttribute("data-meetingscribe", JSON.stringify(diag));
  };
  mark();
  function tick() {
    let speaking = new Set();
    let source = "tile";
    try {
      for (const n of speakingFromAria()) speaking.add(n);
      for (const n of PLATFORM === "meet" ? speakingMeetTiles() : speakingTeamsTiles()) speaking.add(n);
      if (!speaking.size) {
        speaking = speakingFromCaptions();
        source = "captions";
      }
    } catch (e) {
      /* markup changed under us: try again next tick */
    }
    const list = [...speaking].slice(0, 8);
    const key = list.join("|");
    const now = Date.now();
    if (key !== lastKey || now - lastSent > 2000) {
      lastKey = key;
      lastSent = now;
      diag.sent += 1;
      diag.last = list.join(",");
      chrome.runtime.sendMessage({ type: "speakers", platform: PLATFORM, speaking: list, source, self: selfName(), t: now }, (r) => {
        diag.app = r ? r.app : false;
        mark();
      });
    }
  }
  setInterval(tick, 250);
})();
