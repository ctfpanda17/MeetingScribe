"""Dialog: add a speech-recognition model from a Hugging Face link (or explain how to add a folder)."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QVBoxLayout,
)

SUGGESTIONS = [
    ("SenseVoice（中英日韓粵，CPU 即可，推薦）", "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", "0.24 GB"),
    ("Breeze-ASR-25（台灣華語、中英夾雜，需 GPU）", "MediaTek-Research/Breeze-ASR-25", "3.1 GB"),
    ("Paraformer 串流（中英，CPU，最低延遲）", "csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en", "0.24 GB"),
    ("Whisper large-v3-turbo（多語言，需 GPU）", "openai/whisper-large-v3-turbo", "1.6 GB"),
    ("NVIDIA nemotron 串流（多語言，需 GPU）", "nvidia/nemotron-3.5-asr-streaming-0.6b", "2.5 GB"),
]


class _Sig(QObject):
    planned = Signal(object)
    failed = Signal(str)
    progress = Signal(str, object, object)
    done = Signal(object)


class AddModelDialog(QDialog):
    """Checks the link, shows engine type + size, downloads in the background with progress."""

    installed = Signal(object)  # ModelInfo

    def __init__(self, parent, models_dir: str):
        super().__init__(parent)
        self.setWindowTitle("新增語音辨識模型")
        self.setMinimumWidth(620)
        self.sig = _Sig()
        self.sig.planned.connect(self._on_plan)
        self.sig.failed.connect(self._on_fail)
        self.sig.progress.connect(self._on_progress)
        self.sig.done.connect(self._on_done)
        self.plan = None
        self.busy = False

        v = QVBoxLayout(self)
        v.setContentsMargins(22, 20, 22, 20)
        v.setSpacing(12)
        t = QLabel("貼上 Hugging Face 模型網址")
        t.setObjectName("emptyTitle")
        v.addWidget(t)
        row = QHBoxLayout()
        self.url = QLineEdit()
        self.url.setPlaceholderText("例如 https://huggingface.co/MediaTek-Research/Breeze-ASR-25")
        self.url.returnPressed.connect(self.check)
        self.check_btn = QPushButton("檢查")
        self.check_btn.clicked.connect(self.check)
        row.addWidget(self.url, 1)
        row.addWidget(self.check_btn)
        v.addLayout(row)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setObjectName("hint")
        v.addWidget(self.info)
        self.bar = QProgressBar(objectName="level", maximum=1000, textVisible=False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        v.addWidget(self.bar)

        sub = QLabel("常用模型（點一下帶入）")
        sub.setObjectName("cardTitle")
        v.addWidget(sub)
        for name, repo, size in SUGGESTIONS:
            b = QPushButton(f"{name}　·　{size}")
            b.setStyleSheet("text-align: left; padding: 8px 12px;")
            b.clicked.connect(lambda _=False, r=repo: (self.url.setText(r), self.check()))
            v.addWidget(b)

        note = QLabel(
            "支援的格式：Whisper 系列（含微調版）、NVIDIA nemotron 串流、sherpa-onnx（Paraformer / SenseVoice / "
            "Zipformer）。只會下載辨識需要的檔案。\n\n"
            f"也可以自己把模型資料夾複製到：\n{models_dir}\n放好後按主畫面的「⋯ → 重新整理」。")
        note.setWordWrap(True)
        note.setObjectName("hint")
        note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(note)

        bottom = QHBoxLayout()
        bottom.addStretch()
        self.close_btn = QPushButton("關閉")
        self.close_btn.clicked.connect(self.reject)
        self.go = QPushButton("下載並加入")
        self.go.setObjectName("record")
        self.go.setMinimumHeight(38)
        self.go.setEnabled(False)
        self.go.clicked.connect(self.download)
        bottom.addWidget(self.close_btn)
        bottom.addWidget(self.go)
        v.addLayout(bottom)

    # ------------------------------------------------------------------------------------------
    def check(self):
        if self.busy or not self.url.text().strip():
            return
        self.go.setEnabled(False)
        self.info.setText("檢查中…")

        def work():
            try:
                import models

                self.sig.planned.emit(models.plan_download(self.url.text()))
            except Exception as e:
                self.sig.failed.emit(str(e))

        threading.Thread(target=work, daemon=True).start()

    def _on_plan(self, plan):
        from engines import KIND_LABELS

        import models

        self.plan = plan
        exists = models.get_model(plan.folder) is not None
        gpu = "CPU 即可執行" if plan.kind.startswith("sherpa") else "建議有 NVIDIA 顯示卡"
        self.info.setText(f"✓ {plan.repo}\n類型：{KIND_LABELS.get(plan.kind, plan.kind)}　·　需下載 {plan.size / 1e9:.2f} GB"
                          f"（{len(plan.files)} 個檔案）　·　{gpu}" + ("\n這個模型已經裝過了。" if exists else ""))
        self.go.setEnabled(not exists)

    def _on_fail(self, msg):
        self.busy = False
        self.check_btn.setEnabled(True)
        self.url.setEnabled(True)
        self.bar.hide()
        self.info.setText("✗ " + msg)

    def download(self):
        if not self.plan:
            return
        self.busy = True
        self.go.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.url.setEnabled(False)
        self.bar.show()
        plan = self.plan

        def work():
            try:
                import models

                m = models.install(plan, lambda t, d, n: self.sig.progress.emit(t, d, n))
                self.sig.done.emit(m)
            except Exception as e:
                self.sig.failed.emit(f"{type(e).__name__}: {e}")

        threading.Thread(target=work, daemon=True).start()
        self.close_btn.setText("在背景繼續下載")

    def _on_progress(self, text, done, total):
        if total:
            self.bar.setValue(int(1000 * done / total))
            self.info.setText(f"{text}：{done / 1e9:.2f} / {total / 1e9:.2f} GB（{100 * done / total:.0f}%）")
        else:
            self.info.setText(text)
        p = self.parent()
        if p and not self.isVisible() and total:
            p._status(f"背景下載 {self.plan.repo.split('/')[-1]}：{100 * done / total:.0f}%")

    def _on_done(self, m):
        self.busy = False
        self.bar.setValue(1000)
        self.info.setText(f"✓ 已加入：{m.name}")
        self.close_btn.setText("完成")
        self.installed.emit(m)
