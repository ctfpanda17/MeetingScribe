"""Visual theme: color tokens, Qt palette and stylesheet (light + dark)."""
from __future__ import annotations

import os
import tempfile

from PySide6.QtGui import QColor, QPalette

LIGHT = {
    "bg": "#f3f5fa", "card": "#ffffff", "card2": "#f7f8fc", "border": "#e4e7ef", "text": "#1b2132",
    "muted": "#6e7689", "faint": "#a3a9b8", "accent": "#4f6bf5", "accent2": "#7b5cf0", "accent_soft": "#eef1ff",
    "danger": "#e5484d", "danger_soft": "#fdecec", "ok": "#1f9d6b", "hover": "#eef0f6",
}
DARK = {
    "bg": "#0f1218", "card": "#171b24", "card2": "#1d222d", "border": "#2a303d", "text": "#e8ebf2",
    "muted": "#9aa2b4", "faint": "#5f6778", "accent": "#7b8fff", "accent2": "#a08bff", "accent_soft": "#232a44",
    "danger": "#ff6369", "danger_soft": "#3a1c20", "ok": "#4cd29a", "hover": "#232937",
}

SPEAKER_COLORS = {
    "light": ["#4f6bf5", "#e8743b", "#1f9d6b", "#a24fd8", "#d63c6b", "#0f93ad", "#b38a00", "#5b6475"],
    "dark": ["#7b8fff", "#ff9a62", "#4cd29a", "#c58bff", "#ff7aa0", "#4fcbe3", "#e2c044", "#a3aab8"],
}


def _arrow_svg(color: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"meetingscribe_chevron_{color.strip('#')}.svg")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                    f'<path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="{color}" stroke-width="1.6" '
                    f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return path.replace("\\", "/")


def _check_svg() -> str:
    path = os.path.join(tempfile.gettempdir(), "meetingscribe_check.svg")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
                    '<path d="M3.5 8.2 6.6 11.2 12.5 4.8" fill="none" stroke="white" stroke-width="2" '
                    'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return path.replace("\\", "/")


def palette(t: dict) -> QPalette:
    p = QPalette()
    c = lambda k: QColor(t[k])  # noqa: E731
    p.setColor(QPalette.Window, c("bg"))
    p.setColor(QPalette.WindowText, c("text"))
    p.setColor(QPalette.Base, c("card"))
    p.setColor(QPalette.AlternateBase, c("card2"))
    p.setColor(QPalette.Text, c("text"))
    p.setColor(QPalette.Button, c("card"))
    p.setColor(QPalette.ButtonText, c("text"))
    p.setColor(QPalette.Highlight, c("accent"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipBase, c("card"))
    p.setColor(QPalette.ToolTipText, c("text"))
    p.setColor(QPalette.PlaceholderText, c("faint"))
    p.setColor(QPalette.Disabled, QPalette.Text, c("faint"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, c("faint"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, c("faint"))
    return p


def stylesheet(t: dict) -> str:
    arrow = _arrow_svg(t["muted"])
    check = _check_svg()
    return f"""
* {{ font-family: "Microsoft JhengHei UI", "Segoe UI"; }}
QMainWindow, #root {{ background: {t['bg']}; }}
QToolTip {{ background: {t['card']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px 8px; border-radius: 6px; }}

/* cards */
#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 14px; }}
#cardTitle {{ color: {t['muted']}; font-size: 12px; font-weight: 700; letter-spacing: 1px; }}
#hint {{ color: {t['muted']}; font-size: 12px; }}
#muted {{ color: {t['muted']}; }}

/* header */
#header {{ background: {t['card']}; border-bottom: 1px solid {t['border']}; }}
#appName {{ color: {t['muted']}; font-size: 12px; font-weight: 600; }}
#title {{ background: transparent; border: 1px solid transparent; border-radius: 8px; font-size: 20px;
          font-weight: 700; color: {t['text']}; padding: 2px 6px; }}
#title:hover {{ border-color: {t['border']}; }}
#title:focus {{ border-color: {t['accent']}; background: {t['card2']}; }}
#recPill {{ background: {t['card2']}; border: 1px solid {t['border']}; border-radius: 16px; padding: 0 14px;
            color: {t['muted']}; font-family: Consolas, "Microsoft JhengHei UI"; font-size: 14px; font-weight: 600; }}
#recPill[live="true"] {{ background: {t['danger_soft']}; border-color: {t['danger']}; color: {t['danger']}; }}

#record {{ border: none; border-radius: 22px; padding: 0 26px; min-height: 44px; font-size: 15px; font-weight: 700;
           color: white; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t['accent']}, stop:1 {t['accent2']}); }}
#record:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t['accent2']}, stop:1 {t['accent']}); }}
#record[live="true"] {{ background: {t['danger']}; }}
#record:disabled {{ background: {t['border']}; color: {t['faint']}; }}

#extChip {{ background: {t['accent_soft']}; color: {t['accent']}; border-radius: 14px; padding: 5px 12px;
            font-size: 12px; font-weight: 600; }}
#iconBtn {{ background: transparent; border: 1px solid {t['border']}; border-radius: 18px; padding: 0;
            font-size: 16px; color: {t['muted']}; }}
#iconBtn:hover {{ background: {t['hover']}; color: {t['text']}; }}

/* settings strip */
#fieldLabel {{ color: {t['muted']}; font-size: 12px; font-weight: 600; }}
QComboBox {{ background: {t['card2']}; border: 1px solid {t['border']}; border-radius: 9px; padding: 6px 30px 6px 10px;
             color: {t['text']}; min-height: 20px; }}
QComboBox:hover {{ border-color: {t['accent']}; }}
QComboBox:disabled {{ color: {t['faint']}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 8px;
             selection-background-color: {t['accent_soft']}; selection-color: {t['text']}; padding: 4px; outline: none; }}
QCheckBox {{ color: {t['text']}; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1.5px solid {t['faint']}; background: {t['card']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; image: url("{check}"); }}
QCheckBox:disabled {{ color: {t['faint']}; }}

/* level meter */
#level {{ background: {t['border']}; border: none; border-radius: 3px; }}
#level::chunk {{ border-radius: 3px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t['ok']}, stop:1 {t['accent']}); }}

/* buttons */
QPushButton {{ background: {t['card2']}; border: 1px solid {t['border']}; border-radius: 9px; padding: 7px 12px; color: {t['text']}; }}
QPushButton:hover {{ background: {t['hover']}; border-color: {t['accent']}; }}
QPushButton:pressed {{ background: {t['accent_soft']}; }}
#smallBtn {{ padding: 3px 10px; border-radius: 8px; font-size: 12px; }}
#ghostBtn {{ background: transparent; border: none; color: {t['muted']}; padding: 2px 6px; font-size: 16px; }}
#ghostBtn:hover {{ color: {t['text']}; background: {t['hover']}; border-radius: 6px; }}

/* transcript */
#transcriptArea, #transcriptInner {{ background: {t['card']}; border: none; }}
#segment {{ background: transparent; border-radius: 10px; }}
#segment:hover {{ background: {t['card2']}; }}
#segName {{ background: transparent; border: none; font-weight: 700; font-size: 14px; padding: 0; text-align: left; }}
#segName:hover {{ text-decoration: underline; }}
#segTime {{ color: {t['faint']}; font-family: Consolas; font-size: 12px; }}
#segText {{ background: transparent; border: 1px solid transparent; border-radius: 8px; color: {t['text']};
            font-size: 15px; padding: 0; selection-background-color: {t['accent_soft']}; selection-color: {t['text']}; }}
#segText[editing="true"] {{ border-color: {t['accent']}; background: {t['card2']}; padding: 4px; }}
#segText[partial="true"] {{ color: {t['muted']}; font-style: italic; }}
#tagFix {{ background: {t['accent_soft']}; color: {t['accent']}; border-radius: 8px; padding: 1px 8px; font-size: 11px; font-weight: 600; }}
#tagGuess {{ background: {t['danger_soft']}; color: {t['danger']}; border-radius: 8px; padding: 1px 8px; font-size: 11px; font-weight: 600; }}
#emptyIcon {{ font-size: 46px; }}
#emptyTitle {{ font-size: 17px; font-weight: 700; color: {t['text']}; }}

/* speakers */
#spkName {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 3px 6px; font-weight: 700; }}
#spkName:hover {{ border-color: {t['border']}; }}
#spkName:focus {{ border-color: {t['accent']}; background: {t['card2']}; }}
#chip {{ background: {t['card2']}; border: 1px solid {t['border']}; border-radius: 9px; padding: 1px 8px; color: {t['muted']}; font-size: 11px; }}
#okChip {{ background: transparent; color: {t['ok']}; font-size: 12px; font-weight: 600; }}

QSlider::groove:horizontal {{ height: 6px; background: {t['border']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 2px solid {t['accent']}; width: 14px; height: 14px; margin: -6px 0; border-radius: 9px; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t['faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
QMenu {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 18px; border-radius: 6px; color: {t['text']}; }}
QMenu::item:selected {{ background: {t['accent_soft']}; }}
QMenu::section {{ color: {t['muted']}; font-size: 11px; padding: 4px 10px; }}
QStatusBar {{ background: {t['card']}; border-top: 1px solid {t['border']}; color: {t['muted']}; }}
QStatusBar QLabel {{ color: {t['muted']}; padding: 0 8px; }}
QMessageBox {{ background: {t['card']}; }}
"""
