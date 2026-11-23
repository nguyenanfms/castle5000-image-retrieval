"""Vẽ lại các sơ đồ SVG của hệ thống: python docs/so-do-he-thong/ve_so_do.py"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

COLORS = {
    "user": ("#eef2ff", "#4f46e5"),    # người dùng, giao diện
    "local": ("#e0f2fe", "#0369a1"),   # mô hình chạy trên máy
    "gemini": ("#f3e8ff", "#7e22ce"),  # Gemini (đám mây)
    "ok": ("#dcfce7", "#15803d"),      # kết quả
    "dres": ("#ffedd5", "#c2410c"),    # DRES
    "data": ("#f1f5f9", "#475569"),    # dữ liệu
    "warn": ("#fef9c3", "#a16207"),    # dự phòng
}

# Biểu tượng vẽ trong ô 48x48, nét theo màu của thẻ.
ICONS = {
    "user": '<circle cx="24" cy="15" r="8"/><path d="M8 42c2-10 9-14 16-14s14 4 16 14"/>',
    "browser": '<rect x="5" y="9" width="38" height="30" rx="4"/><line x1="5" y1="17" x2="43" y2="17"/>'
               '<circle cx="10" cy="13" r="1.5"/><circle cx="15" cy="13" r="1.5"/>'
               '<rect x="11" y="23" width="12" height="10" rx="1"/><line x1="27" y1="25" x2="37" y2="25"/>'
               '<line x1="27" y1="31" x2="34" y2="31"/>',
    "server": '<rect x="9" y="7" width="30" height="10" rx="2"/><rect x="9" y="19" width="30" height="10" rx="2"/>'
              '<rect x="9" y="31" width="30" height="10" rx="2"/><circle cx="15" cy="12" r="1.5"/>'
              '<circle cx="15" cy="24" r="1.5"/><circle cx="15" cy="36" r="1.5"/>',
    "gemini": '<path d="M24 5c2 10 9 17 19 19-10 2-17 9-19 19-2-10-9-17-19-19 10-2 17-9 19-19z"/>',
    "search": '<circle cx="20" cy="20" r="11"/><line x1="28" y1="28" x2="41" y2="41"/>',
    "regions": '<rect x="5" y="9" width="38" height="30" rx="3"/><rect x="10" y="15" width="12" height="18" stroke-dasharray="3 2"/>'
               '<rect x="26" y="20" width="12" height="9" stroke-dasharray="3 2"/>',
    "nobody": '<circle cx="24" cy="15" r="7"/><path d="M11 40c2-8 7-11 13-11s11 3 13 11"/><line x1="7" y1="7" x2="41" y2="41"/>',
    "stack": '<rect x="13" y="7" width="28" height="20" rx="2"/><rect x="9" y="13" width="28" height="20" rx="2"/>'
             '<rect x="5" y="19" width="28" height="20" rx="2"/><path d="M9 35l7-7 5 5 4-4 6 6"/>',
    "check": '<circle cx="24" cy="24" r="17"/><path d="M15 24l6 6 12-13"/>',
    "zoom": '<circle cx="20" cy="20" r="11"/><line x1="28" y1="28" x2="41" y2="41"/>'
            '<line x1="15" y1="20" x2="25" y2="20"/><line x1="20" y1="15" x2="20" y2="25"/>',
    "tags": '<rect x="6" y="9" width="36" height="9" rx="4.5"/><rect x="6" y="21" width="27" height="9" rx="4.5"/>'
            '<rect x="6" y="33" width="31" height="9" rx="4.5"/>',
    "film": '<rect x="4" y="13" width="12" height="22" rx="2"/><rect x="18" y="11" width="12" height="26" rx="2" stroke-width="3"/>'
            '<rect x="32" y="13" width="12" height="22" rx="2"/>',
    "flag": '<line x1="12" y1="6" x2="12" y2="43"/><path d="M12 8h24l-5 8 5 8H12"/>',
    "target": '<rect x="5" y="9" width="38" height="30" rx="3"/><rect x="15" y="14" width="13" height="21" stroke-width="3"/>'
              '<circle cx="21.5" cy="18" r="2.5"/>',
    "groups": '<circle cx="14" cy="18" r="6"/><circle cx="34" cy="18" r="6"/><circle cx="24" cy="33" r="6"/>'
              '<path d="M14 24v4M34 24v4" stroke-dasharray="2 2"/>',
    "slider": '<line x1="6" y1="24" x2="42" y2="24"/><circle cx="29" cy="24" r="6"/><line x1="6" y1="34" x2="6" y2="38"/>'
              '<line x1="42" y1="34" x2="42" y2="38"/>',
    "chat": '<path d="M7 9h34v22H22l-9 8v-8H7z"/><line x1="14" y1="17" x2="34" y2="17"/><line x1="14" y1="23" x2="28" y2="23"/>',
    "plan": '<rect x="9" y="6" width="30" height="36" rx="3"/><path d="M15 16l3 3 5-6M15 28l3 3 5-6"/>'
            '<line x1="27" y1="17" x2="34" y2="17"/><line x1="27" y1="29" x2="34" y2="29"/>',
    "answer": '<path d="M7 9h34v22H22l-9 8v-8H7z"/><path d="M16 20l5 5 10-10"/>',
    "key": '<circle cx="15" cy="24" r="8"/><line x1="23" y1="24" x2="42" y2="24"/><line x1="36" y1="24" x2="36" y2="31"/>'
           '<line x1="41" y1="24" x2="41" y2="29"/>',
    "keys": '<circle cx="13" cy="17" r="6"/><line x1="19" y1="17" x2="34" y2="17"/><line x1="30" y1="17" x2="30" y2="22"/>'
            '<circle cx="17" cy="33" r="6"/><line x1="23" y1="33" x2="40" y2="33"/><line x1="36" y1="33" x2="36" y2="38"/>',
    "warning": '<path d="M24 6l19 34H5z"/><line x1="24" y1="18" x2="24" y2="29"/><circle cx="24" cy="34" r="1.5"/>',
    "save": '<rect x="7" y="7" width="34" height="34" rx="3"/><rect x="14" y="7" width="18" height="11"/>'
            '<rect x="13" y="25" width="22" height="12" rx="1"/>',
    "clock": '<circle cx="24" cy="24" r="17"/><path d="M24 13v11l7 5"/>',
    "database": '<ellipse cx="24" cy="11" rx="15" ry="5"/><path d="M9 11v26c0 3 7 5 15 5s15-2 15-5V11"/>'
                '<path d="M9 24c0 3 7 5 15 5s15-2 15-5"/>',
}

W_CARD, H_CARD = 150, 118


def card(x, y, kind, icon, title, sub="", w=W_CARD, h=H_CARD):
    fill, stroke = COLORS[kind]
    lines = [
        f'<g transform="translate({x},{y})">',
        f'<rect width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<g transform="translate({w / 2 - 22},10) scale(0.92)" fill="none" stroke="{stroke}" stroke-width="2.4" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICONS[icon]}</g>',
        f'<text x="{w / 2}" y="{h - 34}" text-anchor="middle" class="t">{title}</text>',
    ]
    if sub:
        lines.append(f'<text x="{w / 2}" y="{h - 15}" text-anchor="middle" class="s">{sub}</text>')
    lines.append("</g>")
    return "\n".join(lines)


def arrow(x1, y1, x2, y2, label="", color="#64748b", dashed=False, curve=None, label_dy=-8):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    if curve is None:
        path = f"M{x1},{y1} L{x2},{y2}"
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + label_dy
    else:
        cx, cy = curve
        path = f"M{x1},{y1} Q{cx},{cy} {x2},{y2}"
        lx, ly = (x1 + 2 * cx + x2) / 4, (y1 + 2 * cy + y2) / 4 + label_dy
    out = f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#arrow)"/>'
    if label:
        out += f'<text x="{lx}" y="{ly}" text-anchor="middle" class="a">{label}</text>'
    return out


def elbow(points, label="", color="#64748b", dashed=False, label_at=None):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    out = f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#arrow)"/>'
    if label and label_at:
        out += f'<text x="{label_at[0]}" y="{label_at[1]}" text-anchor="middle" class="a">{label}</text>'
    return out


def badge(x, y, text, kind="data"):
    fill, stroke = COLORS[kind]
    w = 14 + 7.2 * len(text)
    return (f'<g transform="translate({x - w / 2},{y})"><rect width="{w}" height="24" rx="12" fill="#fff" '
            f'stroke="{stroke}" stroke-width="1.3"/><text x="{w / 2}" y="16" text-anchor="middle" class="b" '
            f'fill="{stroke}">{text}</text></g>')


def lane(x, y, w, h, label, kind):
    fill, stroke = COLORS[kind]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" fill-opacity="0.35" '
            f'stroke="{stroke}" stroke-dasharray="5 6" stroke-width="1.2"/>'
            f'<text x="{x + 16}" y="{y + 22}" class="l" fill="{stroke}">{label}</text>')


def legend(y):
    items = [("user", "Người dùng / giao diện"), ("local", "Mô hình trên máy"), ("gemini", "Gemini"),
             ("ok", "Kết quả"), ("dres", "DRES")]
    out, x = [], 40
    for kind, text in items:
        fill, stroke = COLORS[kind]
        out.append(f'<rect x="{x}" y="{y}" width="16" height="16" rx="4" fill="{fill}" stroke="{stroke}"/>'
                   f'<text x="{x + 22}" y="{y + 13}" class="s" text-anchor="start">{text}</text>')
        x += 40 + 8 * len(text)
    return "\n".join(out)


def svg(width, height, title, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{title}">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker>
<style>
text {{ font-family: "Segoe UI", Roboto, Arial, sans-serif; }}
.h {{ font-size: 24px; font-weight: 700; fill: #0f172a; }}
.t {{ font-size: 14.5px; font-weight: 650; fill: #0f172a; }}
.s {{ font-size: 12px; fill: #475569; }}
.a {{ font-size: 12px; fill: #334155; font-weight: 600; }}
.b {{ font-size: 12px; font-weight: 650; }}
.l {{ font-size: 13px; font-weight: 700; }}
</style>
</defs>
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="40" y="46" class="h">{title}</text>
{body}
</svg>
'''


def overview():
    b = []
    y = 150
    b.append(card(40, y, "user", "user", "Người dùng", "đội thi"))
    b.append(card(240, y, "user", "browser", "Trang web", "/search-v2/"))
    b.append(card(440, y, "data", "server", "Máy chủ", "FastAPI"))
    b.append(arrow(190, y + 59, 238, y + 59))
    b.append(arrow(390, y + 59, 438, y + 59))
    # ba chức năng
    b.append(card(680, 80, "local", "search", "Tìm ảnh", "sơ đồ 2"))
    b.append(card(680, 230, "local", "chat", "Hỏi đáp", "sơ đồ 3"))
    b.append(card(680, 380, "dres", "flag", "Nộp bài", "ảnh · đáp án QA"))
    for ty in (139, 289, 439):
        b.append(arrow(590, y + 59, 678, ty, curve=(630, ty)))
    b.append(card(920, 380, "dres", "flag", "DRES", "KIS tự chấm · QA chấm tay", w=190))
    b.append(arrow(830, 439, 918, 439))
    b.append(card(920, 150, "gemini", "gemini", "Gemini", "đám mây"))
    b.append(arrow(830, 139, 918, 195, color="#7e22ce", dashed=True))
    b.append(arrow(830, 289, 918, 222, color="#7e22ce", dashed=True))
    b.append(card(440, 380, "data", "database", "Dữ liệu", "ảnh · chỉ mục · metadata"))
    b.append(arrow(515, 378, 515, y + 120, color="#475569", dashed=True))
    b.append(card(200, 380, "gemini", "tags", "Metadata", "Gemini, tạo trước 1 lần"))
    b.append(arrow(350, 439, 438, 439, color="#7e22ce", dashed=True))
    b.append(legend(540))
    return svg(1150, 580, "1 · Tổng quan hệ thống", "\n".join(b))


def search_flow():
    b = []
    b.append(lane(25, 70, 1080, 175, "Tìm (≈ 0,3 giây)", "local"))
    b.append(lane(25, 275, 1080, 175, "Kiểm tra có cơ sở (≈ 5–8 giây, 2–3 lượt Gemini, chạy sau khi kết quả đã hiện)", "gemini"))
    y1, y2 = 105, 310
    xs = [45, 255, 465, 675, 885]
    row1 = [("user", "user", "Câu hỏi", "Việt / Anh"),
            ("gemini", "gemini", "Hiểu câu", "cảnh · cụm · điều kiện"),
            ("local", "search", "SigLIP 2", "+ metadata · theo sự kiện"),
            ("local", "regions", "Vùng SigLIP 2", "từng vùng"),
            ("local", "nobody", "Lọc điều kiện", "vd. không có người")]
    for x, (k, i, t, s) in zip(xs, row1):
        b.append(card(x, y1, k, i, t, s, w=170))
    for a, c in zip(xs, xs[1:]):
        b.append(arrow(a + 170, y1 + 59, c - 2, y1 + 59))
    row2 = [("gemini", "tags", "Xếp theo metadata", "50 → 4 ảnh, chỉ chữ"),
            ("gemini", "check", "Gemini kiểm", "từng yêu cầu"),
            ("gemini", "zoom", "Phóng to", "xác nhận chủ thể"),
            ("ok", "tags", "Gắn nhãn", "Khớp · Một phần · Không"),
            ("user", "film", "Khung lân cận", "← → trước khi nộp")]
    for x, (k, i, t, s) in zip(xs, row2):
        b.append(card(x, y2, k, i, t, s, w=170))
    for a, c in zip(xs, xs[1:]):
        b.append(arrow(a + 170, y2 + 59, c - 2, y2 + 59))
    b.append(elbow([(970, y1 + 118), (970, 260), (35, 260), (35, y2 + 59), (43, y2 + 59)]))
    b.append(card(885, 480, "dres", "flag", "Nộp DRES", "1 ảnh đúng là đủ", w=170))
    b.append(arrow(970, y2 + 118, 970, 478))
    b.append(badge(690, 527, "Ảnh khớp lên đầu · ảnh không khớp xuống cuối", "ok"))
    b.append(legend(620))
    return svg(1130, 660, "2 · Tìm ảnh", "\n".join(b))


def qa_flow():
    b = []
    y = 100
    b.append(card(40, y, "user", "chat", "Câu hỏi", ""))
    b.append(card(230, y, "gemini", "plan", "Lập kế hoạch", "loại câu · câu tìm ảnh"))
    b.append(card(420, y, "local", "search", "KIS tìm ảnh", "dùng lại Tìm ảnh · 50 ảnh"))
    b.append(arrow(190, y + 59, 228, y + 59))
    b.append(arrow(380, y + 59, 418, y + 59))
    b.append(badge(470, 74, "xếp theo metadata (chỉ chữ)", "gemini"))
    # đọc ảnh theo đợt, song song
    b.append(lane(610, 70, 330, 300, "Đọc ảnh theo đợt (2 ảnh song song)", "local"))
    b.append(card(630, 105, "local", "target", "DINO dò", "khoanh đối tượng", w=140))
    b.append(card(780, 105, "gemini", "gemini", "Gemini đọc", "ảnh + ghi chú metadata", w=140))
    b.append(arrow(570, y + 59, 628, y + 59))
    b.append(arrow(770, 150, 778, 150))
    b.append('<g transform="translate(775,300)"><path d="M0,-42 L70,0 L0,42 L-70,0 Z" fill="#fff7ed" '
             'stroke="#c2410c" stroke-width="1.5"/><text x="0" y="-3" text-anchor="middle" class="t">Đủ bằng</text>'
             '<text x="0" y="15" text-anchor="middle" class="t">chứng?</text></g>')
    b.append(arrow(850, 223, 800, 260))
    b.append(arrow(705, 300, 690, 225, curve=(660, 285)))
    b.append('<text x="650" y="300" text-anchor="middle" class="a">chưa</text>')
    b.append(card(975, 240, "gemini", "check", "Chọn cảnh", "so ảnh khớp · tổng hợp", w=180))
    b.append(arrow(845, 300, 973, 300, label="đủ → dừng sớm"))
    b.append(card(975, 430, "ok", "answer", "Trả lời", "đáp án ngắn · đối tượng", w=180))
    b.append(arrow(1065, 358, 1065, 428))
    # nhánh đếm người khác nhau trong cả buổi
    b.append(lane(40, 410, 880, 180, "Câu đếm người khác nhau trong cả buổi (chạy nền, lưu kết quả)", "ok"))
    b.append(card(60, 445, "data", "tags", "Chọn khung", "metadata: khung đông + khung rõ", w=190))
    b.append(card(275, 445, "gemini", "groups", "Gemini đối chiếu", "từng khung với khung mốc", w=190))
    b.append(card(490, 445, "local", "stack", "re-ID + thuộc tính", "xét trùng, chạy trên máy", w=190))
    b.append(card(705, 445, "user", "regions", "Gộp / nghi trùng", "hiện trên ô đối tượng", w=190))
    b.append(arrow(250, 504, 273, 504))
    b.append(arrow(465, 504, 488, 504))
    b.append(arrow(680, 504, 703, 504))
    b.append(arrow(895, 504, 973, 489))
    b.append(elbow([(345, 218), (345, 392), (28, 392), (28, 504), (58, 504)], color="#15803d", dashed=True))
    b.append(badge(475, 600, "Đếm đông nhất cùng lúc: đọc 16 khung đông nhất theo metadata", "data"))
    b.append(legend(650))
    return svg(1180, 690, "3 · Hỏi đáp (QA)", "\n".join(b))


def resilience():
    b = []
    b.append(lane(25, 75, 1050, 190, "Hết quota / quá tải", "warn"))
    b.append(card(50, 110, "gemini", "key", "Key chính", "QA_API_KEY"))
    b.append(card(330, 110, "warn", "warning", "429 / 503", "hết quota · quá tải"))
    b.append(card(610, 110, "gemini", "keys", "Key dự phòng", "BACKUP, BACKUP1…9"))
    b.append(card(890, 110, "ok", "check", "Gọi lại", "cùng model, key kế tiếp"))
    b.append(arrow(200, 169, 328, 169))
    b.append(arrow(480, 169, 608, 169))
    b.append(arrow(760, 169, 888, 169))
    b.append(badge(545, 236, "xoay vòng key · giới hạn phút: nghỉ theo Google · hết quota ngày: nghỉ 1 giờ", "warn"))
    b.append(lane(25, 315, 1050, 190, "Gemini lỗi giữa chừng", "warn"))
    b.append(card(50, 350, "local", "stack", "Đã đọc N ảnh", ""))
    b.append(card(330, 350, "warn", "warning", "Lỗi lượt sau", ""))
    b.append(card(610, 350, "local", "save", "Giữ kết quả", "không bỏ ảnh đã đọc"))
    b.append(card(890, 350, "ok", "answer", "Trả lời", "+ ghi chú giới hạn"))
    b.append(arrow(200, 409, 328, 409))
    b.append(arrow(480, 409, 608, 409))
    b.append(arrow(760, 409, 888, 409))
    b.append(card(50, 540, "data", "clock", "180 giây", "giới hạn mỗi câu QA", w=200))
    b.append(card(290, 540, "data", "groups", "2 câu QA", "chạy cùng lúc", w=200))
    b.append(card(530, 540, "data", "target", "DINO", "chỉ dùng cho QA", w=200))
    b.append(card(770, 540, "data", "groups", "Danh sách người", "chạy nền ≤ 15 phút, lưu lại", w=230))
    return svg(1100, 690, "4 · Dự phòng và giới hạn", "\n".join(b))


DIAGRAMS = {
    "1-tong-quan.svg": overview,
    "2-tim-anh.svg": search_flow,
    "3-hoi-dap.svg": qa_flow,
    "4-du-phong.svg": resilience,
}

if __name__ == "__main__":
    for name, draw in DIAGRAMS.items():
        (OUT / name).write_text(draw(), encoding="utf-8")
        print("wrote", OUT / name)
