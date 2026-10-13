// Relays "who is speaking" from the meeting tab to the desktop app on this computer (127.0.0.1 only).
const APP = "http://127.0.0.1:47833";
const state = { app: false, recording: false, last: null, error: "" };

async function post(payload) {
  try {
    const r = await fetch(`${APP}/speakers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    state.app = true;
    state.recording = !!j.recording;
    state.error = "";
  } catch (e) {
    state.app = false;
    state.error = "找不到會議記錄助手（請先開啟桌面程式）";
  }
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.type === "speakers") {
    state.last = { ...msg, at: Date.now(), tab: sender.tab && sender.tab.title };
    post({ platform: msg.platform, speaking: msg.speaking, source: msg.source, self: msg.self, t: msg.t });
    reply({ ok: true, app: state.app });
  } else if (msg.type === "status") {
    reply(state);
  }
  return false;
});
