"""Custom widgets: speaker avatars and chat-style transcript cards."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QTextOption
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget


def initials(name: str) -> str:
    name = (name or "").strip()
    if not name or name == "辨識中…":
        return "…"
    if name.startswith("說話者"):
        return name.replace("說話者", "").strip()[:2] or "?"
    if name[0].isascii():
        parts = name.split()
        return "".join(p[0] for p in parts[:2]).upper()
    return name[-2:] if len(name) >= 3 else name  # 王小明 -> 小明, 王經理 -> 經理


class Avatar(QWidget):
    def __init__(self, size: int = 36, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.color = QColor("#a3a9b8")
        self.text = "…"

    def set(self, text: str, color: str):
        self.text, self.color = text, QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        r = QRectF(self.rect())
        p.drawEllipse(r)
        p.setPen(QColor("white"))
        f = QFont(self.font())
        f.setBold(True)
        f.setPixelSize(int(self.height() * (0.42 if len(self.text) < 2 else 0.34)))
        p.setFont(f)
        p.drawText(r, Qt.AlignCenter, self.text)


class AutoText(QTextEdit):
    """A read-only text that looks like a label, grows with its content, and becomes editable on double-click."""

    edited = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segText")
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.document().setDocumentMargin(2)
        self.document().contentsChanged.connect(self._fit)
        self.editable = True
        self._before = ""

    def _fit(self):
        self.document().setTextWidth(self.viewport().width())
        h = int(self.document().size().height()) + 6
        if self.height() != h:
            self.setFixedHeight(h)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _set_editing(self, on: bool):
        self.setProperty("editing", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseDoubleClickEvent(self, e):
        if self.editable and self.isReadOnly():
            self._before = self.toPlainText()
            self.setReadOnly(False)
            self.setTextInteractionFlags(Qt.TextEditorInteraction)
            self._set_editing(True)
            self.setFocus()
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        if not self.isReadOnly() and e.key() in (Qt.Key_Return, Qt.Key_Enter) and not e.modifiers() & Qt.ShiftModifier:
            self.clearFocus()
            return
        if not self.isReadOnly() and e.key() == Qt.Key_Escape:
            self.setPlainText(self._before)
            self.clearFocus()
            return
        super().keyPressEvent(e)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        if not self.isReadOnly():
            self.setReadOnly(True)
            self.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._set_editing(False)
            if self.toPlainText() != self._before:
                self.edited.emit(self.toPlainText())


class SegmentCard(QFrame):
    speakerClicked = Signal(int, object)  # seg, global pos
    textEdited = Signal(int, str)

    def __init__(self, seg: int, parent=None):
        super().__init__(parent)
        self.seg = seg
        self.setObjectName("segment")
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 10, 12, 10)
        h.setSpacing(12)
        self.avatar = Avatar(38)
        h.addWidget(self.avatar, 0, Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(10)
        self.name = QPushButton()
        self.name.setObjectName("segName")
        self.name.setCursor(Qt.PointingHandCursor)
        self.name.setToolTip("點一下可以改成其他說話者")
        self.name.clicked.connect(lambda: self.speakerClicked.emit(self.seg, self.name.mapToGlobal(
            self.name.rect().bottomLeft())))
        self.time = QLabel()
        self.time.setObjectName("segTime")
        self.tag = QLabel()
        self.tag.hide()
        top.addWidget(self.name)
        top.addWidget(self.time)
        top.addWidget(self.tag)
        top.addStretch()
        v.addLayout(top)
        self.text = AutoText()
        self.text.setToolTip("雙擊可以修改文字，Enter 完成")
        self.text.edited.connect(lambda s: self.textEdited.emit(self.seg, s))
        v.addWidget(self.text)
        h.addLayout(v, 1)

    TAGS = {
        "corrected": ("tagFix", "已校正", "背景重新比對聲音後，自動改正了這一句的說話者"),
        "guessed": ("tagGuess", "待確認", "這句太短，聲音不足以判斷是誰，先歸給前一位說話者；點名字可以改"),
    }

    def update_view(self, name: str, color: str, time_text: str, text: str, partial: bool, tag: str = ""):
        self.avatar.set(initials(name) if not partial or name != "辨識中…" else "…", color)
        self.name.setText(name)
        self.name.setStyleSheet(f"color: {color};")
        self.time.setText(time_text)
        if tag in self.TAGS:
            obj, label, tip = self.TAGS[tag]
            if self.tag.objectName() != obj:
                self.tag.setObjectName(obj)
                self.tag.style().unpolish(self.tag)
                self.tag.style().polish(self.tag)
            self.tag.setText(label)
            self.tag.setToolTip(tip)
            self.tag.show()
        else:
            self.tag.hide()
        shown = text + (" ▍" if partial else "")
        if self.text.isReadOnly() and self.text.toPlainText() != shown:
            self.text.setPlainText(shown)
        if self.text.property("partial") != partial:
            self.text.setProperty("partial", partial)
            self.text.style().unpolish(self.text)
            self.text.style().polish(self.text)
        self.text.editable = not partial

    def sizeHint(self):
        return QSize(400, super().sizeHint().height())
