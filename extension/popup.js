function show() {
  chrome.runtime.sendMessage({ type: "status" }, (s) => {
    if (!s) return;
    const app = document.getElementById("app");
    app.textContent = s.app ? "已連接" : "未連接";
    app.className = s.app ? "ok" : "bad";
    document.getElementById("rec").textContent = s.app ? (s.recording ? "是" : "否（請在桌面程式按開始記錄）") : "—";
    const fresh = s.last && Date.now() - s.last.at < 5000;
    document.getElementById("meet").textContent = fresh ? (s.last.platform === "meet" ? "Google Meet" : "Microsoft Teams") : "未偵測到會議分頁";
    document.getElementById("who").textContent = fresh && s.last.speaking.length ? s.last.speaking.join("、") : "—";
    if (!s.app && s.error) document.getElementById("who").textContent = s.error;
  });
}
show();
setInterval(show, 1000);
