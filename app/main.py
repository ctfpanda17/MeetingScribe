"""MeetingScribe — live meeting transcription with speaker separation (desktop app)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if getattr(sys, "frozen", False):
    # packaged build: core modules live next to the exe, and there is no console for logs/progress bars
    sys.path.insert(0, os.path.join(sys._MEIPASS, "core"))
    from paths import LOG_PATH

    sys.stdout = sys.stderr = open(LOG_PATH, "a", encoding="utf-8", buffering=1)

from PySide6.QtCore import QObject, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QAction, QGuiApplication, QIcon  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
    QWidget,
)

import theme  # noqa: E402
from paths import APP_NAME, SETTINGS_PATH, TRANSCRIPT_DIR  # noqa: E402
from widgets import Avatar, SegmentCard, initials  # noqa: E402

LANGUAGES = [("中文", "zh"), ("自動偵測", "auto"), ("English", "en"), ("日本語", "ja"), ("한국어", "ko")]
SOURCES = [("線上會議 Meet / Teams（對方 + 我）", "both"), ("線上會議（只記對方）", "system"),
           ("實體會議（麥克風）", "mic"), ("匯入錄音檔", "file")]
SOURCE_HINTS = {
    "both": "先加入 Meet / Teams 會議再按開始。對方的聲音從電腦收，你的聲音從麥克風收。",
    "system": "只記錄電腦播出的聲音（會議中的其他人），不收你的麥克風。",
    "mic": "大家對著同一支麥克風說話，適合會議室。",
    "file": "選一個錄音 / 錄影檔，程式會快速播放並整理成逐字稿。",
}


def fmt(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    except Exception:
        pass


def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    v = QVBoxLayout(f)
    v.setContentsMargins(16, 14, 16, 16)
    v.setSpacing(10)
    if title:
        v.addWidget(label(title, "cardTitle"))
    return f, v


def repolish(w: QWidget, **props):
    for k, v in props.items():
        w.setProperty(k, v)
    w.style().unpolish(w)
    w.style().polish(w)


class Bridge(QObject):
    """Carries events from worker threads to the GUI thread."""
    event = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("會議記錄助手")
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)
        self.bridge = Bridge()
        self.bridge.event.connect(self.on_event)
        self.engine = self.encoder = self.turns = None
        self.session = None
        self.capture = None
        self.recording = False
        self.started_at = 0.0
        self.cards: dict[int, SegmentCard] = {}
        self.segs: dict[int, dict] = {}
        self.speakers: dict[str, dict] = {}
        self.colors: dict[str, int] = {}
        self.settings = load_settings()
        self.dark = self.settings.get("dark", QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark)

        self._build_ui()
        self._apply_theme()
        self._restore_settings()
        self.clock = QTimer(self, interval=500, timeout=self._tick)
        self._load_models()
        self._start_extension_link()

    # ================================================================== layout
    def _build_ui(self):
        root = QWidget(objectName="root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(18, 16, 18, 14)
        body.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(self._build_settings())
        left.addWidget(self._build_transcript(), 1)
        body.addLayout(left, 1)
        body.addWidget(self._build_sidebar())
        outer.addLayout(body, 1)
        self.setCentralWidget(root)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().setSizeGripEnabled(False)

    def _build_header(self) -> QWidget:
        head = QFrame(objectName="header")
        h = QHBoxLayout(head)
        h.setContentsMargins(20, 12, 20, 12)
        h.setSpacing(14)
        logo = QLabel()
        icon = os.path.join(HERE, "icon.ico")
        if os.path.exists(icon):
            logo.setPixmap(QIcon(icon).pixmap(36, 36))
        h.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("會議記錄助手", "appName"))
        self.title = QLineEdit("會議紀錄", objectName="title")
        self.title.setPlaceholderText("輸入會議名稱")
        self.title.setMinimumWidth(320)
        self.title.editingFinished.connect(lambda: self.session and self.session.control(
            {"type": "title", "title": self.title.text()}))
        titles.addWidget(self.title)
        h.addLayout(titles)
        h.addStretch()

        self.level = QProgressBar(objectName="level", maximum=100, textVisible=False)
        self.level.setFixedSize(110, 6)
        self.level.setToolTip("音量")
        h.addWidget(self.level)
        self.ext_chip = QLabel("", objectName="extChip")
        self.ext_chip.setToolTip("Chrome 擴充功能回報的會議發言者（Google Meet / Teams 網頁版）")
        self.ext_chip.hide()
        h.addWidget(self.ext_chip, 0, Qt.AlignVCenter)
        self.rec = QLabel("00:00:00", objectName="recPill")
        self.rec.setFixedHeight(34)
        h.addWidget(self.rec, 0, Qt.AlignVCenter)
        self.theme_btn = QPushButton(objectName="iconBtn")
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.setToolTip("切換深色 / 淺色")
        self.theme_btn.clicked.connect(self._toggle_theme)
        h.addWidget(self.theme_btn, 0, Qt.AlignVCenter)
        self.toggle = QPushButton("●  開始記錄", objectName="record")
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setMinimumWidth(150)
        self.toggle.clicked.connect(self.toggle_recording)
        h.addWidget(self.toggle)
        return head

    def _build_settings(self) -> QWidget:
        f, v = card()
        v.setContentsMargins(16, 12, 16, 12)
        g = QGridLayout()
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(4)
        self.source = self._combo(SOURCES)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.mic = QComboBox()
        self.speaker = QComboBox()
        self.speaker.setToolTip("Meet / Teams 的聲音從哪個裝置播出來（戴耳機就選耳機）")
        self.language = self._combo(LANGUAGES)
        self.latency = QComboBox()
        self.model = QComboBox()
        self.model.setToolTip("語音辨識模型。可以按「＋ 新增模型」從 Hugging Face 加入其他模型")
        self.model.activated.connect(self._model_chosen)
        model_row = QWidget()
        mr = QHBoxLayout(model_row)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.setSpacing(6)
        mr.addWidget(self.model, 1)
        self.add_model_btn = QPushButton("＋ 新增模型")
        self.add_model_btn.clicked.connect(self._add_model)
        mr.addWidget(self.add_model_btn)
        self.model_more = QPushButton("⋯", objectName="ghostBtn")
        self.model_more.setToolTip("模型資料夾、重新整理、刪除模型")
        self.model_more.clicked.connect(self._model_menu)
        mr.addWidget(self.model_more)
        # row 1: recognition model + language/latency; row 2: what to record and from which devices
        fields = [(0, 0, 2, "辨識模型", model_row), (0, 2, 1, "語言", self.language), (0, 3, 1, "延遲", self.latency),
                  (2, 0, 1, "音源", self.source), (2, 1, 1, "我的麥克風", self.mic),
                  (2, 2, 2, "會議聲音從哪裡播出（耳機 / 喇叭）", self.speaker)]
        for row, col, span, name, w in fields:
            lab = label(name, "fieldLabel")
            if name == "延遲":
                self.latency_label = lab
            g.addWidget(lab, row, col, 1, span)
            g.addWidget(w, row + 1, col, 1, span)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            w.setMinimumWidth(120)
        g.setVerticalSpacing(6)
        for c, s in enumerate((3, 3, 2, 2)):
            g.setColumnStretch(c, s)
        v.addLayout(g)
        row = QHBoxLayout()
        self.source_hint = label("", "hint")
        row.addWidget(self.source_hint, 1)
        self.traditional = QCheckBox("輸出繁體")
        self.traditional.setChecked(True)
        self.anti_echo = QCheckBox("外放喇叭防回音")
        self.anti_echo.setChecked(True)
        self.anti_echo.setToolTip("對方說話時自動壓低麥克風，避免喇叭的聲音被麥克風再收一次。戴耳機可關閉。")
        row.addWidget(self.anti_echo)
        row.addSpacing(10)
        row.addWidget(self.traditional)
        v.addLayout(row)
        self._refresh_devices()
        return f

    def _build_transcript(self) -> QWidget:
        f, v = card()
        v.setContentsMargins(6, 6, 6, 6)
        self.scroll = QScrollArea(objectName="transcriptArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget(objectName="transcriptInner")
        self.feed = QVBoxLayout(inner)
        self.feed.setContentsMargins(10, 8, 10, 8)
        self.feed.setSpacing(2)
        self.empty = self._empty_state()
        self.feed.addWidget(self.empty)
        self.feed.addStretch()
        self.scroll.setWidget(inner)
        self._stick = True
        self.scroll.verticalScrollBar().valueChanged.connect(self._track_stick)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._follow)
        v.addWidget(self.scroll)
        return f

    def _empty_state(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 90, 0, 0)
        v.setSpacing(8)
        v.addWidget(label("🎙", "emptyIcon"), 0, Qt.AlignHCenter)
        v.addWidget(label("按「開始記錄」，大家說的話會即時出現在這裡", "emptyTitle"), 0, Qt.AlignHCenter)
        tip = label("每位說話者會依聲音自動分成不同顏色。點名字可以改成別人，雙擊文字可以直接修改。", "hint")
        v.addWidget(tip, 0, Qt.AlignHCenter)
        return w

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setFixedWidth(330)
        v = QVBoxLayout(side)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)

        f, c = card("說話者")
        self.speaker_box = QVBoxLayout()
        self.speaker_box.setSpacing(6)
        c.addLayout(self.speaker_box)
        c.addWidget(label("改好名字按 Enter。「記住」會儲存聲紋，下次開會自動認出。", "hint", wrap=True))
        v.addWidget(f)

        f, c = card("說話者分離")
        self.detect_turns = QCheckBox("自動說話者分離（建議開啟）")
        self.detect_turns.setChecked(True)
        self.detect_turns.setToolTip("依音色判斷每一刻是誰在說話：自動判斷人數，沒有停頓、插話、短回應也能分開")
        self.detect_turns.toggled.connect(self._diar_toggled)
        c.addWidget(self.detect_turns)
        self.refine = QCheckBox("背景自動校正說話者")
        self.refine.setChecked(True)
        self.refine.setToolTip("每隔幾秒用整場會議的聲音重新判斷誰是誰，回頭改正之前的句子；會議結束時再整理整場")
        c.addWidget(self.refine)
        self.save_audio = QCheckBox("同時保存會議錄音（.wav）")
        self.save_audio.setChecked(True)
        self.save_audio.setToolTip("存在「會議紀錄」資料夾，之後可以回頭聽；約每小時 115 MB")
        c.addWidget(self.save_audio)
        # sensitivity only matters for the simple mode (one voiceprint per sentence)
        self.sens_box = QWidget()
        sb = QVBoxLayout(self.sens_box)
        sb.setContentsMargins(0, 6, 0, 0)
        sb.addWidget(label("分辨靈敏度（簡易模式）", "fieldLabel"))
        self.threshold = QSlider(Qt.Horizontal, minimum=30, maximum=85, value=55)
        self.threshold.valueChanged.connect(self._threshold_changed)
        sb.addWidget(self.threshold)
        leg = QHBoxLayout()
        leg.addWidget(label("容易合併", "hint"))
        leg.addStretch()
        self.threshold_val = label("0.55", "muted")
        leg.addWidget(self.threshold_val)
        leg.addStretch()
        leg.addWidget(label("容易分開", "hint"))
        sb.addLayout(leg)
        sb.addWidget(label("同一人被分成兩位 → 往左；不同人被當成同一位 → 往右。", "hint", wrap=True))
        c.addWidget(self.sens_box)
        self.sens_box.setVisible(False)
        v.addWidget(f)

        f, c = card("匯出")
        g = QGridLayout()
        g.setSpacing(8)
        for i, (text, kind) in enumerate([("📝 Markdown", "md"), ("📄 文字檔", "txt"), ("{ } JSON", "json")]):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=kind: self.export(k))
            g.addWidget(b, 0, i)
        b = QPushButton("📋 複製全文")
        b.clicked.connect(self.copy_all)
        g.addWidget(b, 1, 0, 1, 2)
        b = QPushButton("📂 資料夾")
        b.clicked.connect(lambda: os.startfile(TRANSCRIPT_DIR))
        g.addWidget(b, 1, 2)
        c.addLayout(g)
        c.addWidget(label("自動存檔與匯出都在程式資料夾的「會議紀錄」裡。", "hint", wrap=True))
        v.addWidget(f)

        f, c = card("已記住的聲紋")
        self.vp_box = QVBoxLayout()
        self.vp_box.setSpacing(4)
        c.addLayout(self.vp_box)
        v.addWidget(f)
        v.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(342)
        scroll.setWidget(side)
        side.setObjectName("root")
        self._refresh_voiceprints()
        return scroll

    @staticmethod
    def _combo(items) -> QComboBox:
        c = QComboBox()
        for text, value in items:
            c.addItem(text, value)
        return c

    # ================================================================== theme
    def _apply_theme(self):
        t = theme.DARK if self.dark else theme.LIGHT
        self.t = t
        self.speaker_colors = theme.SPEAKER_COLORS["dark" if self.dark else "light"]
        app = QApplication.instance()
        app.setPalette(theme.palette(t))
        app.setStyleSheet(theme.stylesheet(t))
        self.theme_btn.setText("☀" if self.dark else "☾")
        self._render_speakers()
        for seg in self.segs:
            self._render_seg(seg)

    def _toggle_theme(self):
        self.dark = not self.dark
        self._apply_theme()
        self._store_settings()

    # ================================================================== devices & settings
    def _refresh_devices(self):
        from audio import list_microphones, list_speakers

        for combo, lister, empty in ((self.mic, list_microphones, "找不到麥克風"),
                                     (self.speaker, list_speakers, "找不到播放裝置")):
            combo.clear()
            try:
                for d in lister():
                    combo.addItem(("★ " if d["default"] else "") + d["name"], d["index"])
            except Exception:
                pass
            if not combo.count():
                combo.addItem(empty, None)

    def _source_changed(self):
        mode = self.source.currentData()
        idle = not self.recording
        self.mic.setEnabled(mode in ("mic", "both") and idle)
        self.speaker.setEnabled(mode in ("system", "both") and idle)
        self.anti_echo.setVisible(mode == "both")
        self.anti_echo.setEnabled(idle)
        self.source_hint.setText(SOURCE_HINTS.get(mode, ""))

    def _restore_settings(self):
        s = dict(self.settings)
        s["language"] = str(s.get("language", "zh")).split("-")[0]
        for combo, key in ((self.source, "source"), (self.language, "language")):
            i = combo.findData(s.get(key))
            if i >= 0:
                combo.setCurrentIndex(i)
        for combo, key in ((self.mic, "mic"), (self.speaker, "speaker")):
            i = combo.findText(s.get(key, ""), Qt.MatchContains) if s.get(key) else -1
            if i >= 0:
                combo.setCurrentIndex(i)
        self.traditional.setChecked(s.get("traditional", True))
        self.anti_echo.setChecked(s.get("anti_echo", True))
        self.detect_turns.setChecked(s.get("detect_turns", True))
        self._diar_toggled(self.detect_turns.isChecked())
        self.refine.setChecked(s.get("refine", True))
        self.save_audio.setChecked(s.get("save_audio", True))
        self.threshold.setValue(int(s.get("threshold", 0.55) * 100))
        self._source_changed()

    def _store_settings(self):
        save_settings({
            "source": self.source.currentData(), "language": self.language.currentData(),
            "latency": self.latency.currentData(), "traditional": self.traditional.isChecked(),
            "threshold": self.threshold.value() / 100, "mic": self.mic.currentText().lstrip("★ "),
            "speaker": self.speaker.currentText().lstrip("★ "), "anti_echo": self.anti_echo.isChecked(),
            "dark": self.dark, "model": self.model.currentData() or self.settings.get("model"),
            "latency_by_model": self.latency_by_model, "detect_turns": self.detect_turns.isChecked(),
            "refine": self.refine.isChecked(), "save_audio": self.save_audio.isChecked(),
        })

    def _status(self, text: str):
        self.status_label.setText(text)

    # ================================================================== models
    # ================================================================== browser extension (Meet / Teams)
    def _start_extension_link(self):
        from bridge import Bridge

        self.ext_last = 0.0
        self.ext_info = {}
        self.ext_bridge = Bridge(self._extension_event, lambda: self.recording)
        if not self.ext_bridge.start():
            print("[bridge] port busy: another copy is running?", flush=True)
        self.ext_timer = QTimer(self, interval=2000, timeout=self._refresh_ext_chip)
        self.ext_timer.start()

    def _extension_event(self, ev: dict):  # HTTP thread
        if self.session and self.recording:
            self.session.platform_event(ev)
        self.bridge.event.emit({"type": "_ext", "ev": ev})

    def _refresh_ext_chip(self):
        if time.time() - self.ext_last > 10:
            self.ext_chip.hide()
            return
        ev = self.ext_info
        platform = {"meet": "Meet", "teams": "Teams"}.get(ev.get("platform"), ev.get("platform", ""))
        who = "、".join(ev.get("speaking", [])[:3])
        self.ext_chip.setText(f"🔗 {platform} 已連接" + (f" · {who}" if who else ""))
        self.ext_chip.show()

    def _load_models(self):
        """First run: fetch the voiceprint model (+ a default speech model if none). Then load the chosen one."""
        self.latency_by_model = self.settings.get("latency_by_model", {})
        self._set_busy("準備模型中…")

        def progress(text, done, total):
            self.bridge.event.emit({"type": "_download", "text": text, "done": done, "total": total})

        def work():
            try:
                import torch

                import models
                from diarize import SpeakerEncoder

                models.ensure_first_run(progress)
                self.bridge.event.emit({"type": "_download", "text": "載入聲紋模型…", "done": 0, "total": 0})
                encoder = SpeakerEncoder("cuda" if torch.cuda.is_available() else "cpu")
                from paths import MODEL_DIR
                from turns import ChangeDetector, model_path

                self.turns = ChangeDetector(model_path(MODEL_DIR))
                from diarize import ClusterEncoder, cluster_model_path

                self.cluster_encoder = ClusterEncoder(cluster_model_path())
                self.bridge.event.emit({"type": "_encoder", "encoder": encoder})
            except Exception as e:
                self.bridge.event.emit({"type": "_models_error", "message": f"{type(e).__name__}: {e}"})

        threading.Thread(target=work, daemon=True).start()

    def _set_busy(self, text: str):
        self.toggle.setEnabled(False)
        self.toggle.setText(text)
        self.model.setEnabled(False)
        self._status(text)

    def _refresh_model_list(self, select: str | None = None):
        import models

        current = select or self.model.currentData() or self.settings.get("model")
        self.model.blockSignals(True)
        self.model.clear()
        for m in models.list_models():
            self.model.addItem(m.label, m.id)
        if not self.model.count():
            self.model.addItem("（尚未安裝模型，請按「＋ 新增模型」）", None)
        i = self.model.findData(current)
        self.model.setCurrentIndex(i if i >= 0 else 0)
        self.model.blockSignals(False)

    def _model_chosen(self, _index=None):
        mid = self.model.currentData()
        if mid and getattr(self.engine, "model_id", None) != mid:
            self._switch_model(mid)

    def _switch_model(self, mid: str | None):
        import models

        m = models.get_model(mid) if mid else None
        if not m or self.recording:
            return
        old, self.engine = self.engine, None
        self._set_busy(f"載入 {m.name}…")

        def work():
            try:
                import torch

                from engines import load_engine

                if old is not None:
                    old.close()
                device = "cpu" if m.kind.startswith("sherpa") or not torch.cuda.is_available() else "cuda"
                engine = load_engine(m.kind, m.path, device)
                engine.model_id, engine.model_name = m.id, m.name
                self.bridge.event.emit({"type": "_engine", "engine": engine, "device": device})
            except Exception as e:
                self.bridge.event.emit({"type": "_models_error", "message": f"{m.name} 載入失敗：{type(e).__name__}: {e}"})

        threading.Thread(target=work, daemon=True).start()

    def _engine_ready(self, engine, device):
        self.engine = engine
        self.model.setEnabled(True)
        self.latency.blockSignals(True)
        self.latency.clear()
        for text, value in engine.latency_options():
            self.latency.addItem(text, value)
        want = self.latency_by_model.get(engine.model_id, engine.default_latency())
        i = self.latency.findData(want)
        self.latency.setCurrentIndex(i if i >= 0 else 0)
        self.latency.blockSignals(False)
        self.latency.setEnabled(self.latency.count() > 1)
        self.latency_label.setText("延遲" if engine.streaming else "文字更新頻率")
        self.toggle.setEnabled(True)
        self.toggle.setText("●  開始記錄")
        where = "GPU 加速" if device == "cuda" else "CPU"
        self._status(f"準備完成 · {engine.model_name}（{where}）· 按「開始記錄」開始")
        print(f"[models] ready: {engine.model_name} on {device}", flush=True)
        if os.environ.get("MEETINGSCRIBE_SELFTEST") and not getattr(self, "_selftest_started", False):
            self._selftest_started = True
            self._selftest(os.environ["MEETINGSCRIBE_SELFTEST"])

    def _add_model(self):
        import models
        from model_dialog import AddModelDialog

        if getattr(self, "_add_dialog", None) is None:
            self._add_dialog = AddModelDialog(self, models.ASR_DIR)
            self._add_dialog.installed.connect(self._model_installed)
        self._add_dialog.show()
        self._add_dialog.raise_()

    def _model_installed(self, m):
        self._refresh_model_list()
        self._status(f"已加入模型：{m.name}（在「辨識模型」選單中選擇即可使用）")

    def _model_menu(self):
        import models

        menu = QMenu(self)
        menu.addAction("📂  開啟模型資料夾", lambda: os.startfile(models.ASR_DIR))
        menu.addAction("🔄  重新整理模型清單", lambda: (self._refresh_model_list(), self._status("模型清單已更新")))
        mid = self.model.currentData()
        if mid and len(models.list_models()) > 1 and not self.recording:
            menu.addSeparator()
            menu.addAction("🗑  刪除目前選擇的模型", lambda: self._delete_model(mid))
        menu.exec(self.model_more.mapToGlobal(self.model_more.rect().bottomLeft()))

    def _delete_model(self, mid):
        import models

        m = models.get_model(mid)
        if not m:
            return
        r = QMessageBox.question(self, "刪除模型", f"確定要刪除「{m.name}」嗎？\n會從模型資料夾移除，之後需要可以再下載。")
        if r != QMessageBox.Yes:
            return
        if getattr(self.engine, "model_id", None) == mid:
            self.engine.close()
            self.engine = None
        models.remove_model(mid)
        self._refresh_model_list()
        self._switch_model(self.model.currentData())

    # ================================================================== recording
    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        from audio import FileFeeder, LiveCapture
        from session import MeetingSession

        mode = self.source.currentData()
        path = None
        if mode == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "選擇錄音檔", "", "音訊/影片 (*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.webm);;所有檔案 (*)")
            if not path:
                return
        self._store_settings()
        cfg = {
            "title": self.title.text(), "language": self.language.currentData(),
            "latency": self.latency.currentData(), "traditional": self.traditional.isChecked(),
            "threshold": self.threshold.value() / 100, "refine": self.refine.isChecked(),
            "save_audio": self.save_audio.isChecked(),
            "cluster_embed": self.cluster_encoder.embed if getattr(self, "cluster_encoder", None) else None,
        }
        if self.session and not self.session.stopped:
            self.session.stop(timeout=30)
        self.session = MeetingSession(self.engine, self.encoder, self.bridge.event.emit, cfg,
                                      turns=self.turns if self.detect_turns.isChecked() else None)
        try:
            if mode == "file":
                self._status("讀取錄音檔中…")
                self.capture = FileFeeder(
                    path, self._on_audio,
                    lambda t, total: self.bridge.event.emit({"type": "_progress", "t": t, "total": total}),
                    lambda: self.bridge.event.emit({"type": "_file_done"}),
                )
            else:
                self.capture = LiveCapture(mode, self.mic.currentData(), self._on_audio,
                                           speaker_index=self.speaker.currentData(),
                                           anti_echo=self.anti_echo.isChecked())
        except Exception as e:
            self.session.stop()
            QMessageBox.warning(self, "無法開始收音", f"{e}\n\n請確認麥克風 / 喇叭裝置可用。")
            return

        self._clear_transcript()
        self.capture.start()
        self.recording = True
        self.started_at = time.time()
        if mode != "file":
            self.clock.start()
        self.toggle.setText("■  停止記錄")
        repolish(self.toggle, live=True)
        repolish(self.rec, live=True)
        self.rec.setText("● 00:00:00")
        self.latency_by_model[self.engine.model_id] = self.latency.currentData()
        for w in (self.model, self.add_model_btn, self.source, self.mic, self.speaker, self.anti_echo, self.language,
                  self.latency, self.traditional, self.detect_turns, self.refine, self.save_audio):
            w.setEnabled(False)
        lat = self.engine.latency_ms(self.latency.currentData())
        self._status(f"記錄中 · 辨識延遲約 {lat} ms · 自動存檔到 {self.session.files()['md']}")

    def _on_audio(self, pcm, levels=None):
        if self.session:
            self.session.push_audio(pcm, levels)
        self.bridge.event.emit({"type": "_level", "v": float((pcm ** 2).mean() ** 0.5)})

    def stop_recording(self):
        if self.capture:
            self.capture.close()
            self.capture = None
        self.recording = False
        self.clock.stop()
        self.level.setValue(0)
        self.toggle.setText("●  開始記錄")
        repolish(self.toggle, live=False)
        repolish(self.rec, live=False)
        self.rec.setText(self.rec.text().lstrip("● "))
        for w in (self.model, self.add_model_btn, self.source, self.language, self.traditional, self.detect_turns,
                  self.save_audio):
            w.setEnabled(True)
        self.refine.setEnabled(self.detect_turns.isChecked())
        self.latency.setEnabled(self.latency.count() > 1)
        self._source_changed()
        self._status("整理最後一句中…")
        if self.session:
            threading.Thread(target=self.session.stop, daemon=True).start()

    def _tick(self):
        dot = "●" if int(time.time() * 2) % 2 == 0 else "○"
        self.rec.setText(f"{dot} {fmt(time.time() - self.started_at)}")

    # ================================================================== events
    def on_event(self, ev: dict):
        t = ev["type"]
        if t == "_download":
            if ev["total"]:
                pct = 100 * ev["done"] / ev["total"]
                self._status(f"{ev['text']}：{ev['done'] / 1e9:.2f} / {ev['total'] / 1e9:.2f} GB（{pct:.0f}%）"
                             "　第一次使用才需要下載")
                self.toggle.setText(f"下載中 {pct:.0f}%")
            else:
                self._status(ev["text"])
        elif t == "_encoder":
            self.encoder = ev["encoder"]
            self._refresh_model_list()
            self._switch_model(self.model.currentData())
        elif t == "_engine":
            self._engine_ready(ev["engine"], ev["device"])
        elif t == "_models_error":
            print("[models] failed:", ev["message"], flush=True)
            self.toggle.setText("模型載入失敗")
            self._status("模型載入失敗")
            self.model.setEnabled(True)
            QMessageBox.critical(self, "模型載入失敗",
                                 ev["message"] + "\n\n第一次使用需要網路下載模型；若是自己放入的模型，請確認檔案完整。")
        elif t == "_ext":
            self.ext_last, self.ext_info = time.time(), ev["ev"]
            self._refresh_ext_chip()
        elif t == "_level":
            self.level.setValue(min(100, int(ev["v"] * 400)))
        elif t == "_progress":
            self.rec.setText(f"{fmt(ev['t'])} / {fmt(ev['total'])}")
        elif t == "_file_done":
            if self.recording:
                self.stop_recording()
        elif t in ("partial", "final"):
            self._upsert(ev, partial=t == "partial")
        elif t == "discard":
            self._remove(ev["seg"])
        elif t == "speakers":
            self.speakers = {s["id"]: s for s in ev["speakers"]}
            self._render_speakers()
            for seg in self.segs:
                self._render_seg(seg)
        elif t == "segments":
            keep = {s["seg"] for s in ev["segments"]}
            for seg in [k for k, v in self.segs.items() if not v.get("partial") and k not in keep]:
                self._remove(seg)
            for s in ev["segments"]:
                self._upsert(s, partial=False)
            self._reorder_cards()
        elif t == "status":
            self._status(ev["message"])
        elif t == "toast":
            self._status(ev["message"])
            self._refresh_voiceprints()
        elif t == "stopped":
            self._status(f"已停止 · 紀錄已存到 {ev['files']['md']}")
        elif t == "error":
            print("[session] error:", ev["message"], flush=True)
            self._status("錯誤：" + ev["message"])

    # ================================================================== transcript
    def _color(self, sid) -> str:
        if not sid:
            return self.t["faint"]
        if sid not in self.colors:
            self.colors[sid] = len(self.colors)
        return self.speaker_colors[self.colors[sid] % len(self.speaker_colors)]

    def _name(self, sid) -> str:
        if not sid:
            return "辨識中…"
        return self.speakers.get(sid, {}).get("name", sid)

    def _clear_transcript(self):
        for c in self.cards.values():
            c.deleteLater()
        self.cards.clear()
        self.segs.clear()
        self.colors.clear()
        self.speakers.clear()
        self.empty.hide()
        self._stick = True
        self._render_speakers()

    def _upsert(self, ev: dict, partial: bool):
        seg = ev["seg"]
        prev = self.segs.get(seg)
        if prev and not prev.get("partial") and partial:
            return
        data = {**(prev or {}), **ev, "partial": partial}
        if partial and not ev.get("speaker") and prev:
            data["speaker"] = prev.get("speaker")
        self.segs[seg] = data
        if seg not in self.cards:
            c = SegmentCard(seg)
            c.speakerClicked.connect(self._reassign_menu)
            c.textEdited.connect(self._text_edited)
            self.feed.insertWidget(self.feed.count() - 1, c)
            self.cards[seg] = c
            self.empty.hide()
        self._render_seg(seg)
        if not partial:
            self._render_speakers()

    def _render_seg(self, seg):
        d, c = self.segs.get(seg), self.cards.get(seg)
        if d and c:
            sid = d.get("speaker")
            tag = "corrected" if d.get("corrected") else "guessed" if d.get("guessed") else ""
            c.update_view(self._name(sid), self._color(sid), fmt(d["start"]), d["text"], d.get("partial", False), tag)

    def _reorder_cards(self):
        """Corrections can split a sentence into new ones: keep the cards in time order."""
        order = sorted(self.cards, key=lambda k: (self.segs[k].get("partial", False), self.segs[k]["start"]))
        for c in self.cards.values():
            self.feed.removeWidget(c)
        for i, k in enumerate(order):
            self.feed.insertWidget(i + 1, self.cards[k])  # index 0 is the (hidden) empty state

    def _remove(self, seg):
        c = self.cards.pop(seg, None)
        if c:
            c.deleteLater()
        self.segs.pop(seg, None)

    def _track_stick(self, value):
        sb = self.scroll.verticalScrollBar()
        self._stick = value >= sb.maximum() - 40

    def _follow(self, _min, maximum):
        if self._stick:
            self.scroll.verticalScrollBar().setValue(maximum)

    def _reassign_menu(self, seg, pos):
        if self.segs.get(seg, {}).get("partial") or not self.speakers or not self.session:
            return
        menu = QMenu(self)
        menu.addSection("改成這位說話者")
        for sid, sp in self.speakers.items():
            act = QAction(f"●  {sp['name']}", menu)
            act.triggered.connect(lambda _=False, s=sid: self.session.control({"type": "reassign", "seg": seg, "speaker": s}))
            menu.addAction(act)
        menu.exec(pos)

    def _text_edited(self, seg, text):
        if self.session and seg in self.segs:
            self.segs[seg]["text"] = text
            self.session.control({"type": "edit", "seg": seg, "text": text})

    # ================================================================== speakers panel
    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _render_speakers(self):
        if not hasattr(self, "speaker_box"):
            return
        if any(w.hasFocus() for w in self.findChildren(QLineEdit, "spkName")):
            QTimer.singleShot(800, self._render_speakers)  # don't yank the box while someone is typing
            return
        self._clear_layout(self.speaker_box)
        counts = {}
        for s in self.segs.values():
            if not s.get("partial"):
                counts[s["speaker"]] = counts.get(s["speaker"], 0) + 1
        if not self.speakers:
            self.speaker_box.addWidget(label("開始記錄後，偵測到的說話者會出現在這裡。", "hint", wrap=True))
        for sid, sp in self.speakers.items():
            color = self._color(sid)
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 2, 0, 2)
            h.setSpacing(8)
            av = Avatar(30)
            av.set(initials(sp["name"]), color)
            h.addWidget(av)
            name = QLineEdit(sp["name"], objectName="spkName")
            name.setStyleSheet(f"color: {color};")
            name.editingFinished.connect(lambda s=sid, e=name: self._rename(s, e.text()))
            h.addWidget(name, 1)
            h.addWidget(label(f"{counts.get(sid, 0)} 句", "chip"))
            if sp["enrolled"]:
                h.addWidget(label("✓ 已記住", "okChip"))
            else:
                b = QPushButton("記住", objectName="smallBtn")
                b.setToolTip("儲存聲紋，下次開會自動認出（請先改成真實姓名）")
                b.clicked.connect(lambda _=False, s=sid: self.session and self.session.control(
                    {"type": "enroll", "speaker": s}))
                h.addWidget(b)
            more = QPushButton("⋯", objectName="ghostBtn")
            more.setToolTip("更多")
            more.clicked.connect(lambda _=False, s=sid, b=more: self._speaker_menu(s, b))
            h.addWidget(more)
            self.speaker_box.addWidget(w)

    def _speaker_menu(self, sid, button):
        if not self.session:
            return
        menu = QMenu(self)
        others = [(o, sp) for o, sp in self.speakers.items() if o != sid]
        if others:
            sub = menu.addMenu("合併到…（同一個人被分成兩位）")
            for oid, o in others:
                act = QAction(o["name"], sub)
                act.triggered.connect(lambda _=False, o=oid: self.session.control({"type": "merge", "from": sid, "into": o}))
                sub.addAction(act)
        else:
            menu.addAction("目前只有一位說話者").setEnabled(False)
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _rename(self, sid, name):
        if self.session and name.strip() and self.speakers.get(sid, {}).get("name") != name.strip():
            self.session.control({"type": "rename", "speaker": sid, "name": name})

    def _diar_toggled(self, on: bool):
        self.sens_box.setVisible(not on)
        self.refine.setEnabled(on and not self.recording)

    def _threshold_changed(self, v):
        self.threshold_val.setText(f"{v / 100:.2f}")
        if self.session:
            self.session.control({"type": "threshold", "value": v / 100})

    def _refresh_voiceprints(self):
        from diarize import forget_profile, load_profiles

        self._clear_layout(self.vp_box)
        profiles = load_profiles()
        if not profiles:
            self.vp_box.addWidget(label("尚無。開會時按說話者旁的「記住」即可建立。", "hint", wrap=True))
        for p in profiles:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            av = Avatar(24)
            av.set(initials(p["name"]), self.t["ok"] if hasattr(self, "t") else "#1f9d6b")
            h.addWidget(av)
            h.addWidget(QLabel(p["name"]), 1)
            b = QPushButton("刪除", objectName="smallBtn")
            b.clicked.connect(lambda _=False, n=p["name"]: (forget_profile(n), self._refresh_voiceprints()))
            h.addWidget(b)
            self.vp_box.addWidget(w)

    # ================================================================== export
    def _final_segments(self):
        return sorted((s for s in self.segs.values() if not s.get("partial")), key=lambda s: s["start"])

    def export_text(self, kind: str) -> str:
        segs = self._final_segments()
        title = self.title.text() or "會議紀錄"
        name = lambda s: self._name(s["speaker"])  # noqa: E731
        if kind == "json":
            return json.dumps({"title": title, "segments": [
                {k: s[k] for k in ("start", "end", "speaker", "text") if k in s} | {"name": name(s)} for s in segs
            ]}, ensure_ascii=False, indent=1)
        if kind == "md":
            people = "、".join(dict.fromkeys(name(s) for s in segs))
            body = "\n".join(f"**[{fmt(s['start'])}] {name(s)}**：{s['text']}  " for s in segs)
            return f"# {title}\n\n- 日期：{time.strftime('%Y-%m-%d %H:%M')}\n- 與會者：{people}\n\n---\n\n{body}\n"
        return f"{title}\n\n" + "\n".join(f"[{fmt(s['start'])}] {name(s)}：{s['text']}" for s in segs) + "\n"

    def export(self, kind: str):
        if not self._final_segments():
            self._status("目前沒有內容可以匯出")
            return
        safe = "".join(c for c in (self.title.text() or "會議紀錄") if c not in '\\/:*?"<>|').strip() or "會議紀錄"
        stamp = self.session.started_at.strftime("%Y%m%d_%H%M") if self.session else time.strftime("%Y%m%d_%H%M")
        path = os.path.join(TRANSCRIPT_DIR, f"{safe}_{stamp}.{kind}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.export_text(kind))
        self._status(f"已匯出：{path}")

    def copy_all(self):
        QGuiApplication.clipboard().setText(self.export_text("txt"))
        self._status("已複製到剪貼簿")

    # ================================================================== diagnostics
    def _selftest(self, wav: str):
        """Play a clip through the speakers while capturing system audio, log the result, quit."""
        import winsound

        import soundfile as sf

        dur = sf.info(wav).duration
        self.source.setCurrentIndex(self.source.findData("system"))
        self.start_recording()
        threading.Thread(target=lambda: winsound.PlaySound(wav, winsound.SND_FILENAME), daemon=True).start()

        def report():
            for s in self._final_segments():
                print("[selftest]", fmt(s["start"]), "|", self._name(s["speaker"]), "|", s["text"], flush=True)
            shot = os.environ.get("MEETINGSCRIBE_SCREENSHOT")
            if shot:
                self.grab().save(shot)
            self.recording = False
            QApplication.quit()

        QTimer.singleShot(int((dur + 3) * 1000), self.stop_recording)
        QTimer.singleShot(int((dur + 7) * 1000), report)

    # ================================================================== lifecycle
    def closeEvent(self, e):
        if self.recording:
            r = QMessageBox.question(self, "還在記錄中", "要停止記錄並關閉嗎？（目前內容已自動存檔）")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.stop_recording()
        if self.session:
            self.session.stop(timeout=20)  # don't hang the closing window; transcript is auto-saved anyway
        if getattr(self, "ext_bridge", None):
            self.ext_bridge.stop()
        if not os.environ.get("MEETINGSCRIBE_SELFTEST"):
            self._store_settings()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    icon = os.path.join(HERE, "icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    f = app.font()
    f.setFamily("Microsoft JhengHei UI")
    f.setPointSize(10)
    app.setFont(f)
    w = MainWindow()
    w.show()
    w.toggle.setFocus()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
