import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & BẢO MẬT
# ==============================================================================
try:
    import cv2
    from deepface import DeepFace

    AI_BACKEND = "deepface"
    SIMILARITY_THRESHOLD = 0.35
except Exception:
    AI_BACKEND = "numpy_hog"
    SIMILARITY_THRESHOLD = 0.24

DB_PATH = "luxstay.db"
# Khóa bí mật mã hóa vector sinh trắc học (trong thực tế lưu ở biến môi trường)
MASTER_SECRET = os.environ.get("LUXSTAY_SECRET", "LuxStay-AI-2026-Biometric-Key").encode("utf-8")

# Tài khoản demo Quầy Lễ Tân (lưu dạng băm SHA-256)
STAFF_ACCOUNTS = {
    "admin": {"hash": hashlib.sha256(b"admin123").hexdigest(), "role": "Quản lý (Admin)"},
    "letan": {"hash": hashlib.sha256(b"123456").hexdigest(), "role": "Lễ tân (Reception)"},
}
# Mã Master Buồng phòng (Housekeeping)
HOUSEKEEPING_PIN_HASH = hashlib.sha256(b"999999").hexdigest()

st.set_page_config(
    page_title="LuxStay AI v2.0 - Khóa Khuôn Mặt & PMS",
    page_icon="🏨",
    layout="wide",
)


# ==============================================================================
# 1. MÃ HÓA VECTOR KHUÔN MẶT & BĂM MẬT KHẨU
# ==============================================================================
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encrypt_embedding(vec: list) -> str:
    """Mã hóa vector khuôn mặt trước khi lưu vào SQLite (PBKDF2 + Stream XOR + HMAC)."""
    raw = json.dumps(vec).encode("utf-8")
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", MASTER_SECRET, salt, 10000, dklen=32)

    keystream = bytearray()
    counter = 0
    while len(keystream) < len(raw):
        keystream.extend(hashlib.sha256(key + counter.to_bytes(4, "big")).digest())
        counter += 1

    cipher = bytes(a ^ b for a, b in zip(raw, keystream[: len(raw)]))
    sig = hmac.new(key, salt + cipher, hashlib.sha256).digest()
    return base64.b64encode(salt + sig + cipher).decode("ascii")


def decrypt_embedding(token: str) -> list:
    """Giải mã và xác thực toàn vẹn vector khuôn mặt từ SQLite."""
    blob = base64.b64decode(token.encode("ascii"))
    salt, sig, cipher = blob[:16], blob[16:48], blob[48:]
    key = hashlib.pbkdf2_hmac("sha256", MASTER_SECRET, salt, 10000, dklen=32)
    expected_sig = hmac.new(key, salt + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("Dữ liệu sinh trắc học bị thay đổi trái phép!")

    keystream = bytearray()
    counter = 0
    while len(keystream) < len(cipher):
        keystream.extend(hashlib.sha256(key + counter.to_bytes(4, "big")).digest())
        counter += 1

    raw = bytes(a ^ b for a, b in zip(cipher, keystream[: len(cipher)]))
    return json.loads(raw.decode("utf-8"))


# ==============================================================================
# 2. CƠ SỞ DỮ LIỆU SQLITE (WAL MODE - HỖ TRỢ ĐA THIẾT BỊ ĐỒNG THỜI)
# ==============================================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS rooms (
            room_id TEXT PRIMARY KEY,
            room_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'vacant',
            dnd INTEGER NOT NULL DEFAULT 0,
            cleaning INTEGER NOT NULL DEFAULT 0,
            backup_pin_hash TEXT,
            checkout_at TEXT
        );

        CREATE TABLE IF NOT EXISTS guests (
            guest_id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT NOT NULL,
            title TEXT NOT NULL,
            name TEXT NOT NULL,
            tier TEXT NOT NULL,
            booking_id TEXT NOT NULL,
            checkin_at TEXT NOT NULL,
            checkout_at TEXT NOT NULL,
            notes TEXT,
            embedding_enc TEXT NOT NULL,
            FOREIGN KEY (room_id) REFERENCES rooms(room_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS access_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            room_id TEXT NOT NULL,
            actor TEXT NOT NULL,
            method TEXT NOT NULL,
            direction TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT
        );
        """
    )

    cur.execute("SELECT COUNT(*) FROM rooms")
    if cur.fetchone()[0] == 0:
        default_rooms = [
            ("102", "Deluxe King"),
            ("204", "Executive Suite"),
            ("305", "Ocean Panorama"),
            ("501", "Presidential Suite"),
        ]
        cur.executemany("INSERT INTO rooms (room_id, room_type) VALUES (?, ?)", default_rooms)
    conn.commit()
    conn.close()


def add_log(room_id, actor, method, direction, status, note):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO access_logs (timestamp, room_id, actor, method, direction, status, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), room_id, actor, method, direction, status, note),
    )
    conn.commit()
    conn.close()


def auto_expire_checkouts():
    """Tự động thu hồi quyền mở cửa cho các phòng đã quá giờ Check-out."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn = get_conn()
    expired = conn.execute(
        "SELECT room_id, checkout_at FROM rooms WHERE status = 'occupied' AND checkout_at IS NOT NULL AND checkout_at < ?",
        (now_str,),
    ).fetchall()

    for row in expired:
        r_id = row["room_id"]
        conn.execute("DELETE FROM guests WHERE room_id = ?", (r_id,))
        conn.execute(
            "UPDATE rooms SET status = 'vacant', dnd = 0, cleaning = 0, backup_pin_hash = NULL, checkout_at = NULL WHERE room_id = ?",
            (r_id,),
        )
        conn.execute(
            """
            INSERT INTO access_logs (timestamp, room_id, actor, method, direction, status, note)
            VALUES (?, ?, 'Hệ thống tự động', 'Auto-Expire', 'Hết hạn lưu trú', 'Đã thu hồi quyền', ?)
            """,
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), r_id, f"Quá giờ check-out ({row['checkout_at']})"),
        )
    conn.commit()
    conn.close()


# ==============================================================================
# 3. TIỀN XỬ LÝ ÁNH SÁNG, ANTI-SPOOFING (LIVENESS) & TRÍCH XUẤT KHUÔN MẶT
# ==============================================================================
@st.cache_resource
def warmup_model():
    if AI_BACKEND == "deepface":
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            DeepFace.represent(dummy, model_name="Facenet512", enforce_detection=False)
        except Exception:
            pass
    return AI_BACKEND


def normalize_lighting(pil_img: Image.Image) -> Image.Image:
    """Tự động hiệu chỉnh Gamma & cân bằng tương phản khi hành lang tối hoặc ngược sáng."""
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    mean_brightness = float(np.mean(arr))

    if mean_brightness < 0.35:
        gamma = 0.6  # Làm sáng vùng tối
        arr = np.power(np.clip(arr, 1e-5, 1.0), gamma)
    elif mean_brightness > 0.75:
        gamma = 1.4  # Giảm chói ngược sáng
        arr = np.power(np.clip(arr, 1e-5, 1.0), gamma)

    enhanced = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return ImageOps.autocontrast(enhanced, cutoff=1)


def check_liveness(img1_file, img2_file) -> tuple[bool, str]:
    """
    Kiểm tra thực thể sống (Anti-Spoofing) qua 2 khung hình thử thách:
    - Chặn ảnh tĩnh / ảnh in / màn hình đứng yên (độ chênh lệch pixel quá thấp).
    - Chặn việc tráo 2 ảnh hoàn toàn khác nhau (độ lệch quá lớn).
    """
    p1 = ImageOps.grayscale(Image.open(img1_file).convert("RGB")).resize((96, 96))
    p2 = ImageOps.grayscale(Image.open(img2_file).convert("RGB")).resize((96, 96))

    a1 = np.asarray(p1, dtype=np.float32) / 255.0
    a2 = np.asarray(p2, dtype=np.float32) / 255.0

    motion_score = float(np.mean(np.abs(a1 - a2)))

    if motion_score < 0.018:
        return False, f"Phát hiện ảnh tĩnh / nghi ngờ giả mạo (biến thiên chuyển động = {motion_score:.4f} < 0.018)."
    if motion_score > 0.38:
        return False, f"Khung hình thay đổi bất thường (biến thiên = {motion_score:.4f}). Vui lòng giữ mặt trong khung."
    return True, f"Xác thực thực thể sống đạt chuẩn (biến thiên = {motion_score:.4f})."


def extract_face_embedding(uploaded_or_cam_file) -> list:
    pil_img = normalize_lighting(Image.open(uploaded_or_cam_file).convert("RGB"))

    if AI_BACKEND == "deepface":
        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        results = DeepFace.represent(
            img_path=bgr_img,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
        )
        return results[0]["embedding"]

    # Chế độ NumPy HOG cải tiến (chuẩn hóa ánh sáng + kết hợp đặc trưng cạnh)
    w, h = pil_img.size
    face_crop = pil_img.crop((int(w * 0.22), int(h * 0.15), int(w * 0.78), int(h * 0.85)))
    gray = ImageOps.equalize(ImageOps.grayscale(face_crop)).resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    gx = np.diff(arr, axis=1, append=arr[:, -1:])
    gy = np.diff(arr, axis=0, append=arr[-1:, :])
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    hog_features = []
    for r in range(0, 64, 8):
        for c in range(0, 64, 8):
            hist, _ = np.histogram(
                ang[r : r + 8, c : c + 8].ravel(),
                bins=8,
                range=(0, 180),
                weights=mag[r : r + 8, c : c + 8].ravel(),
            )
            hog_features.append(hist)

    thumb = np.asarray(gray.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255.0
    vec = np.concatenate([np.concatenate(hog_features), thumb])
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


def cosine_distance(vec1, vec2) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ==============================================================================
# KHỞI CHẠY ỨNG DỤNG
# ==============================================================================
init_db()
auto_expire_checkouts()
warmup_model()

st.title("🏨 LuxStay AI v2.0 — Khóa Khuôn Mặt & Quản Lý Khách Sạn")
st.caption(
    f"🗄️ CSDL: SQLite WAL (Mã hóa sinh trắc học) | 🤖 Engine: "
    f"{'DeepFace Facenet512' if AI_BACKEND == 'deepface' else 'NumPy HOG + Auto-Lighting (Windows Safe Mode)'}"
)

if "door_mode" not in st.session_state:
    st.session_state.door_mode = "idle"  # idle | face | pin | staff
if "scan_direction" not in st.session_state:
    st.session_state.scan_direction = "Vào phòng"
if "last_auth_result" not in st.session_state:
    st.session_state.last_auth_result = None
if "pms_user" not in st.session_state:
    st.session_state.pms_user = None
if "last_generated_pin" not in st.session_state:
    st.session_state.last_generated_pin = None

tab_door, tab_pms = st.tabs(["🚪 Thiết Bị Cửa Phòng (Smart Door)", "🛎️ Quầy Lễ Tân (PMS Bảo Mật)"])

# ==============================================================================
# TAB 1: THIẾT BỊ CỬA PHÒNG
# ==============================================================================
with tab_door:
    conn = get_conn()
    rooms_list = conn.execute("SELECT * FROM rooms ORDER BY room_id").fetchall()
    conn.close()

    col_ctrl, col_screen = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.subheader("Cấu hình thiết bị cửa")
        room_map = {r["room_id"]: r for r in rooms_list}
        selected_room = st.selectbox(
            "Chọn số phòng gắn thiết bị:",
            options=list(room_map.keys()),
            format_func=lambda r: f"Phòng {r} ({room_map[r]['room_type']})",
        )
        r_info = room_map[selected_room]

        conn = get_conn()
        guest_count = conn.execute("SELECT COUNT(*) FROM guests WHERE room_id = ?", (selected_room,)).fetchone()[0]
        conn.close()

        if r_info["status"] == "occupied":
            st.success(f"🟢 **Đang có {guest_count} khách lưu trú**\n\nHết hạn lúc: `{r_info['checkout_at']}`")
        else:
            st.info("⚪ **Phòng trống**")

        dnd_state = st.toggle("🔕 Không làm phiền (Do Not Disturb)", value=bool(r_info["dnd"]))
        clean_state = st.toggle("🧹 Yêu cầu dọn phòng (Make Up Room)", value=bool(r_info["cleaning"]))
        if int(dnd_state) != r_info["dnd"] or int(clean_state) != r_info["cleaning"]:
            conn = get_conn()
            conn.execute(
                "UPDATE rooms SET dnd = ?, cleaning = ? WHERE room_id = ?",
                (int(dnd_state), int(clean_state), selected_room),
            )
            conn.commit()
            conn.close()
            st.rerun()

    with col_screen:
        st.subheader(f"Màn Hình Cảm Ứng Cửa Phòng {selected_room}")

        if st.session_state.door_mode == "idle":
            st.markdown(
                """
                <div style="padding:20px; border-radius:12px; background:#0f172a; color:#f8fafc; border:1px solid #d4af37;">
                    <h3 style="color:#d4af37; margin-top:0;">🔒 CỬA ĐANG KHÓA • CAMERA ĐANG TẮT</h3>
                    <p style="color:#cbd5e1; margin-bottom:0;">Chọn phương thức xác thực bên dưới (Hỗ trợ Khuôn mặt có chống giả mạo, Mã PIN dự phòng hoặc Thẻ Buồng phòng).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔓 Quét Khuôn Mặt (Vào Phòng)", use_container_width=True, type="primary"):
                    st.session_state.scan_direction = "Vào phòng"
                    st.session_state.door_mode = "face"
                    st.session_state.last_auth_result = None
                    st.rerun()
            with c2:
                if st.button("🚶 Quét Khuôn Mặt (Ra Phòng)", use_container_width=True):
                    st.session_state.scan_direction = "Ra khỏi phòng"
                    st.session_state.door_mode = "face"
                    st.session_state.last_auth_result = None
                    st.rerun()

            c3, c4 = st.columns(2)
            with c3:
                if st.button("🔢 Mở Bằng Mã PIN Dự Phòng", use_container_width=True):
                    st.session_state.door_mode = "pin"
                    st.session_state.last_auth_result = None
                    st.rerun()
            with c4:
                if st.button("🧹 Truy Cập Nhân Viên Buồng Phòng", use_container_width=True):
                    st.session_state.door_mode = "staff"
                    st.session_state.last_auth_result = None
                    st.rerun()

        # --- CHẾ ĐỘ 1: QUÉT KHUÔN MẶT + ANTI-SPOOFING ---
        elif st.session_state.door_mode == "face":
            st.warning(f"📷 Xác thực khuôn mặt ({st.session_state.scan_direction}) — Chế độ chống giả mạo (Liveness)")
            enable_liveness = st.checkbox("🛡️ Bật kiểm tra thực thể sống 2 bước (Anti-Spoofing)", value=True)
            door_src = st.radio("Nguồn ảnh:", ["Dùng Camera cửa", "Tải ảnh kiểm thử"], horizontal=True)

            if door_src == "Dùng Camera cửa":
                img1 = st.camera_input("Bước 1: Nhìn thẳng vào camera và chụp")
                img2 = st.camera_input("Bước 2 (Chống giả mạo): Quay nhẹ đầu / chớp mắt rồi chụp") if enable_liveness else None
            else:
                img1 = st.file_uploader("Bước 1: Tải ảnh chính diện", type=["jpg", "jpeg", "png"], key="d_img1")
                img2 = (
                    st.file_uploader("Bước 2 (Chống giả mạo): Tải ảnh có cử động nhẹ", type=["jpg", "jpeg", "png"], key="d_img2")
                    if enable_liveness
                    else None
                )

            col_b1, col_b2 = st.columns(2)
            verify_clicked = col_b1.button("✅ Xác Thực Mở Khóa", type="primary", use_container_width=True)
            if col_b2.button("❌ Hủy & Đóng Camera", use_container_width=True):
                st.session_state.door_mode = "idle"
                st.rerun()

            if verify_clicked:
                if img1 is None or (enable_liveness and img2 is None):
                    st.error("Vui lòng cung cấp đủ ảnh theo yêu cầu.")
                else:
                    with st.spinner("Đang kiểm tra thực thể sống & đối chiếu sinh trắc học..."):
                        if enable_liveness:
                            live_ok, live_msg = check_liveness(img1, img2)
                            if not live_ok:
                                st.session_state.last_auth_result = {"type": "spoof", "msg": live_msg}
                                add_log(selected_room, "Nghi vấn giả mạo", "Face + Liveness", st.session_state.scan_direction, "Cảnh báo giả mạo", live_msg)
                                st.session_state.door_mode = "idle"
                                st.rerun()

                        try:
                            live_emb = extract_face_embedding(img1)
                            conn = get_conn()
                            all_guests = conn.execute("SELECT * FROM guests").fetchall()
                            conn.close()

                            best_guest = None
                            best_dist = 999.0
                            for g in all_guests:
                                stored_emb = decrypt_embedding(g["embedding_enc"])
                                dist = cosine_distance(live_emb, stored_emb)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_guest = g

                            if best_guest is not None and best_dist <= SIMILARITY_THRESHOLD:
                                if best_guest["room_id"] == selected_room:
                                    st.session_state.last_auth_result = {
                                        "type": "granted",
                                        "guest": dict(best_guest),
                                        "direction": st.session_state.scan_direction,
                                        "method": "Khuôn mặt (Liveness OK)" if enable_liveness else "Khuôn mặt",
                                    }
                                    add_log(
                                        selected_room,
                                        best_guest["name"],
                                        "Face AI",
                                        st.session_state.scan_direction,
                                        "Thành công",
                                        f"Sai lệch: {best_dist:.3f}",
                                    )
                                else:
                                    # Bảo mật riêng tư: KHÔNG tiết lộ tên hay số phòng thật trên màn hình cửa
                                    st.session_state.last_auth_result = {"type": "wrong_room"}
                                    add_log(
                                        selected_room,
                                        best_guest["name"],
                                        "Face AI",
                                        st.session_state.scan_direction,
                                        "Từ chối (Nhầm phòng)",
                                        f"Khách thuộc P.{best_guest['room_id']} (đã ẩn trên màn hình cửa)",
                                    )
                            else:
                                st.session_state.last_auth_result = {"type": "denied"}
                                add_log(
                                    selected_room,
                                    "Người lạ",
                                    "Face AI",
                                    st.session_state.scan_direction,
                                    "Từ chối",
                                    f"Sai lệch: {best_dist:.3f}",
                                )

                            st.session_state.door_mode = "idle"
                            st.rerun()
                        except ValueError:
                            st.error("Không nhận diện được khuôn mặt rõ nét. Vui lòng thử lại hoặc dùng Mã PIN dự phòng.")

        # --- CHẾ ĐỘ 2: MÃ PIN DỰ PHÒNG ---
        elif st.session_state.door_mode == "pin":
            st.info("🔢 Phương thức dự phòng: Nhập mã PIN 6 số được cấp khi Check-in")
            pin_input = st.text_input("Nhập mã PIN phòng:", type="password", max_chars=6)
            c1, c2 = st.columns(2)
            if c1.button("🔓 Mở Khóa Bằng PIN", type="primary", use_container_width=True):
                if r_info["status"] == "occupied" and r_info["backup_pin_hash"] == hash_text(pin_input):
                    st.session_state.last_auth_result = {"type": "pin_ok"}
                    add_log(selected_room, f"Khách P.{selected_room}", "PIN Dự phòng", "Vào phòng", "Thành công", "Xác thực bằng mã PIN")
                else:
                    st.session_state.last_auth_result = {"type": "denied"}
                    add_log(selected_room, "Không xác định", "PIN Dự phòng", "Vào phòng", "Từ chối", "Sai mã PIN hoặc phòng trống")
                st.session_state.door_mode = "idle"
                st.rerun()
            if c2.button("❌ Quay Lại", use_container_width=True):
                st.session_state.door_mode = "idle"
                st.rerun()

        # --- CHẾ ĐỘ 3: NHÂN VIÊN BUỒNG PHÒNG (MASTER ACCESS) ---
        elif st.session_state.door_mode == "staff":
            st.info("🧹 Xác thực Nhân viên Buồng phòng (Mã Master demo: `999999`)")
            staff_pin = st.text_input("Nhập mã Master Buồng phòng:", type="password", max_chars=6)
            c1, c2 = st.columns(2)
            if c1.button("🔓 Mở Cửa Dọn Phòng", type="primary", use_container_width=True):
                if hash_text(staff_pin) == HOUSEKEEPING_PIN_HASH:
                    if r_info["dnd"]:
                        st.session_state.last_auth_result = {"type": "dnd_block"}
                        add_log(selected_room, "Buồng phòng", "Master PIN", "Dọn phòng", "Bị chặn (DND)", "Phòng đang bật Không làm phiền")
                    else:
                        st.session_state.last_auth_result = {"type": "staff_ok"}
                        add_log(selected_room, "Buồng phòng", "Master PIN", "Dọn phòng", "Thành công", "Mở cửa phục vụ buồng phòng")
                else:
                    st.session_state.last_auth_result = {"type": "denied"}
                    add_log(selected_room, "Giả mạo Staff", "Master PIN", "Dọn phòng", "Từ chối", "Sai mã Master")
                st.session_state.door_mode = "idle"
                st.rerun()
            if c2.button("❌ Quay Lại", use_container_width=True):
                st.session_state.door_mode = "idle"
                st.rerun()

        # --- HIỂN THỊ KẾT QUẢ XÁC THỰC ---
        res = st.session_state.last_auth_result
        if res:
            st.divider()
            if res["type"] == "granted":
                g = res["guest"]
                st.success(f"✅ MỞ KHÓA THÀNH CÔNG ({res['method']}) — {res['direction'].upper()}")
                st.markdown(
                    f"- **Khách lưu trú:** {g['title']} **{g['name']}** (`{g['tier']}`)\n"
                    f"- **Hạn lưu trú:** đến `{g['checkout_at']}` | **Ghi chú:** {g['notes'] or 'Không có'}"
                )
            elif res["type"] == "pin_ok":
                st.success(f"✅ MỞ KHÓA THÀNH CÔNG BẰNG MÃ PIN DỰ PHÒNG — PHÒNG {selected_room}")
            elif res["type"] == "staff_ok":
                st.success(f"✅ MỞ KHÓA THÀNH CÔNG — CHẾ ĐỘ NHÂN VIÊN BUỒNG PHÒNG (P.{selected_room})")
            elif res["type"] == "dnd_block":
                st.warning("🔕 **KHÔNG THỂ MỞ CỬA:** Phòng hiện đang bật chế độ **Không làm phiền (Do Not Disturb)**.")
            elif res["type"] == "spoof":
                st.error(f"🚨 **CẢNH BÁO GIẢ MẠO (ANTI-SPOOFING):** {res['msg']}")
            elif res["type"] == "wrong_room":
                st.warning("⚠️ **KHÔNG CÓ QUYỀN TRUY CẬP PHÒNG NÀY:** Vui lòng kiểm tra lại số phòng của quý khách (thông tin cá nhân được bảo mật).")
            elif res["type"] == "denied":
                st.error("🚨 **TỪ CHỐI TRUY CẬP:** Xác thực không hợp lệ. Vui lòng thử lại hoặc liên hệ Quầy Lễ Tân.")

# ==============================================================================
# TAB 2: QUẦY LỄ TÂN (PMS CÓ PHÂN QUYỀN)
# ==============================================================================
with tab_pms:
    if st.session_state.pms_user is None:
        st.subheader("🔐 Đăng Nhập Hệ Thống Quầy Lễ Tân (PMS)")
        st.caption("Tài khoản demo: `admin` / `admin123` hoặc `letan` / `123456`")
        with st.form("pms_login"):
            u = st.text_input("Tên đăng nhập:")
            p = st.text_input("Mật khẩu:", type="password")
            if st.form_submit_button("Đăng nhập", type="primary"):
                if u in STAFF_ACCOUNTS and STAFF_ACCOUNTS[u]["hash"] == hash_text(p):
                    st.session_state.pms_user = {"username": u, "role": STAFF_ACCOUNTS[u]["role"]}
                    st.rerun()
                else:
                    st.error("Sai tên đăng nhập hoặc mật khẩu.")
    else:
        top_c1, top_c2 = st.columns([4, 1])
        top_c1.info(f"👤 Đang đăng nhập: **{st.session_state.pms_user['username']}** ({st.session_state.pms_user['role']})")
        if top_c2.button("Đăng xuất", use_container_width=True):
            st.session_state.pms_user = None
            st.rerun()

        if st.session_state.last_generated_pin:
            st.success(f"🔑 **Mã PIN dự phòng vừa cấp cho {st.session_state.last_generated_pin}** (Hãy gửi mã này cho khách hàng).")

        st.subheader("1. Sơ Đồ Phòng & Danh Sách Khách Lưu Trú")
        conn = get_conn()
        rooms_data = conn.execute("SELECT * FROM rooms ORDER BY room_id").fetchall()
        room_cols = st.columns(len(rooms_data))

        for idx, r in enumerate(rooms_data):
            r_id = r["room_id"]
            g_rows = conn.execute("SELECT name, tier FROM guests WHERE room_id = ?", (r_id,)).fetchall()
            with room_cols[idx]:
                if r["status"] == "occupied":
                    names_str = ", ".join([g["name"] for g in g_rows])
                    st.success(f"**Phòng {r_id}** ({r['room_type']})\n\n👥 {len(g_rows)} khách: {names_str}\n\n⏰ Hạn: `{r['checkout_at']}`")
                    if st.button(f"Check-out P.{r_id}", key=f"co_{r_id}", use_container_width=True):
                        conn.execute("DELETE FROM guests WHERE room_id = ?", (r_id,))
                        conn.execute(
                            "UPDATE rooms SET status='vacant', dnd=0, cleaning=0, backup_pin_hash=NULL, checkout_at=NULL WHERE room_id=?",
                            (r_id,),
                        )
                        conn.commit()
                        add_log(r_id, names_str, "PMS", "Check-out", "Hoàn tất", "Đã xóa toàn bộ vector khuôn mặt & mã PIN")
                        st.session_state.last_generated_pin = None
                        st.rerun()
                else:
                    st.info(f"**Phòng {r_id}** ({r['room_type']})\n\n⚪ Phòng trống\n\nSẵn sàng đón khách")
        conn.close()

        st.divider()
        col_form, col_logs = st.columns([1, 1.3], gap="large")

        with col_form:
            st.subheader("2. Check-in / Thêm Khách Vào Phòng (Tối đa 4 khách)")
            photo_source = st.radio(
                "Nguồn ảnh khuôn mặt:",
                ["Tải ảnh chân dung", "Chụp trực tiếp tại quầy"],
                horizontal=True,
                key="pms_photo_src",
            )
            with st.form("checkin_form"):
                reg_room = st.selectbox("Chọn phòng (có thể thêm nhiều khách cùng phòng):", [r["room_id"] for r in rooms_data])
                reg_title = st.selectbox("Danh xưng:", ["Ông", "Bà", "Anh", "Chị"])
                reg_name = st.text_input("Họ và tên khách hàng:", placeholder="Nguyễn Minh Quân")
                reg_tier = st.selectbox("Hạng thành viên:", ["Diamond VIP", "Gold Member", "Standard"])
                reg_booking = st.text_input("Mã Booking:", value=f"BK-{datetime.now().strftime('%H%M%S')}")

                c1, c2 = st.columns(2)
                reg_checkout_date = c1.date_input("Ngày trả phòng:", value=datetime.now().date() + timedelta(days=1))
                reg_checkout_time = c2.time_input("Giờ trả phòng:", value=time(12, 0))
                reg_notes = st.text_area("Ghi chú phục vụ:", placeholder="Thêm gối, dị ứng hải sản...")

                target_img = (
                    st.camera_input("Chụp ảnh tại quầy")
                    if photo_source == "Chụp trực tiếp tại quầy"
                    else st.file_uploader("Chọn tệp ảnh chân dung (JPG/PNG):", type=["jpg", "jpeg", "png"])
                )

                if st.form_submit_button("✅ Lưu Khách & Cấp Quyền Mở Cửa", type="primary"):
                    if not reg_name.strip() or target_img is None:
                        st.error("Vui lòng nhập họ tên và cung cấp ảnh khuôn mặt.")
                    else:
                        conn = get_conn()
                        cur_count = conn.execute("SELECT COUNT(*) FROM guests WHERE room_id = ?", (reg_room,)).fetchone()[0]
                        if cur_count >= 4:
                            conn.close()
                            st.error(f"Phòng {reg_room} đã đạt giới hạn tối đa 4 khách.")
                        else:
                            with st.spinner("Đang chuẩn hóa ánh sáng, trích xuất & mã hóa vector..."):
                                try:
                                    emb = extract_face_embedding(target_img)
                                    emb_enc = encrypt_embedding(emb)
                                    checkout_str = f"{reg_checkout_date} {reg_checkout_time.strftime('%H:%M')}"

                                    room_row = conn.execute("SELECT status FROM rooms WHERE room_id = ?", (reg_room,)).fetchone()
                                    if room_row["status"] == "vacant":
                                        new_pin = f"{secrets.randbelow(900000) + 100000}"
                                        conn.execute(
                                            "UPDATE rooms SET status='occupied', backup_pin_hash=?, checkout_at=? WHERE room_id=?",
                                            (hash_text(new_pin), checkout_str, reg_room),
                                        )
                                        st.session_state.last_generated_pin = f"Phòng {reg_room}: `{new_pin}`"

                                    conn.execute(
                                        """
                                        INSERT INTO guests (room_id, title, name, tier, booking_id, checkin_at, checkout_at, notes, embedding_enc)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            reg_room,
                                            reg_title,
                                            reg_name.strip(),
                                            reg_tier,
                                            reg_booking,
                                            datetime.now().strftime("%Y-%m-%d %H:%M"),
                                            checkout_str,
                                            reg_notes.strip(),
                                            emb_enc,
                                        ),
                                    )
                                    conn.commit()
                                    conn.close()
                                    add_log(reg_room, reg_name.strip(), "PMS", "Check-in", "Đã cấp quyền", f"Hạn: {checkout_str}")
                                    st.rerun()
                                except ValueError:
                                    conn.close()
                                    st.error("Không tìm thấy khuôn mặt rõ nét trong ảnh.")

        with col_logs:
            st.subheader("3. Nhật Ký Ra / Vào & Cảnh Báo An Ninh")
            conn = get_conn()
            logs = conn.execute(
                "SELECT timestamp AS 'Thời gian', room_id AS 'Phòng', actor AS 'Đối tượng', method AS 'Phương thức', direction AS 'Hành động', status AS 'Kết quả', note AS 'Ghi chú' FROM access_logs ORDER BY log_id DESC LIMIT 100"
            ).fetchall()
            conn.close()

            if logs:
                df_logs = pd.DataFrame([dict(row) for row in logs])
                st.dataframe(df_logs, use_container_width=True, height=420)
                st.download_button(
                    "📥 Xuất Nhật Ký (CSV Excel)",
                    data=df_logs.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"LuxStay_Logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Chưa có lịch sử truy cập nào.")
