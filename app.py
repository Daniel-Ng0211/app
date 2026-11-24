import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Tự động chọn bộ nhận diện tương thích với Windows Smart App Control
try:
    import cv2
    from deepface import DeepFace
    AI_BACKEND = "deepface"
    SIMILARITY_THRESHOLD = 0.35
except Exception:
    AI_BACKEND = "numpy_hog"
    SIMILARITY_THRESHOLD = 0.22

st.set_page_config(
    page_title="LuxStay AI - Khóa Khuôn Mặt Khách Sạn",
    page_icon="🏨",
    layout="wide",
)

DB_FILE = "hotel_db.json"


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    default_data = {
        "rooms": {
            "102": {"type": "Deluxe King", "status": "vacant", "dnd": False, "cleaning": False, "guest": None},
            "204": {"type": "Executive Suite", "status": "vacant", "dnd": False, "cleaning": False, "guest": None},
            "305": {"type": "Ocean Panorama", "status": "vacant", "dnd": False, "cleaning": False, "guest": None},
            "501": {"type": "Presidential Suite", "status": "vacant", "dnd": False, "cleaning": False, "guest": None},
        },
        "logs": [],
    }
    save_db(default_data)
    return default_data


def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@st.cache_resource
def warmup_model():
    if AI_BACKEND == "deepface":
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            DeepFace.represent(dummy, model_name="Facenet512", enforce_detection=False)
        except Exception:
            pass
    return AI_BACKEND


def extract_face_embedding(uploaded_or_cam_file):
    pil_img = Image.open(uploaded_or_cam_file).convert("RGB")

    if AI_BACKEND == "deepface":
        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        results = DeepFace.represent(
            img_path=bgr_img,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
        )
        return results[0]["embedding"]

    w, h = pil_img.size
    left, top = int(w * 0.22), int(h * 0.15)
    right, bottom = int(w * 0.78), int(h * 0.85)
    face_crop = pil_img.crop((left, top, right, bottom))

    gray = ImageOps.equalize(ImageOps.grayscale(face_crop)).resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    gx = np.diff(arr, axis=1, append=arr[:, -1:])
    gy = np.diff(arr, axis=0, append=arr[-1:, :])
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    hog_features = []
    for r in range(0, 64, 8):
        for c in range(0, 64, 8):
            cell_mag = mag[r : r + 8, c : c + 8].ravel()
            cell_ang = ang[r : r + 8, c : c + 8].ravel()
            hist, _ = np.histogram(cell_ang, bins=8, range=(0, 180), weights=cell_mag)
            hog_features.append(hist)

    thumb = np.asarray(gray.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255.0
    vec = np.concatenate([np.concatenate(hog_features), thumb])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def cosine_distance(vec1, vec2):
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def add_log(db, room_id, guest_name, direction, status, note):
    db["logs"].insert(
        0,
        {
            "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Phòng": room_id,
            "Khách hàng": guest_name,
            "Hành động": direction,
            "Kết quả": status,
            "Ghi chú": note,
        },
    )
    save_db(db)


st.title("🏨 LuxStay AI — Hệ Thống Khóa Khuôn Mặt & Quản Lý Phòng")
warmup_model()
db = load_db()

if "door_camera_open" not in st.session_state:
    st.session_state.door_camera_open = False
if "scan_direction" not in st.session_state:
    st.session_state.scan_direction = "Vào phòng"
if "last_auth_result" not in st.session_state:
    st.session_state.last_auth_result = None

tab_door, tab_pms = st.tabs(["🚪 Thiết Bị Cửa Phòng (Smart Door Terminal)", "🛎️ Quầy Lễ Tân (PMS)"])

# ==============================================================================
# TAB 1: THIẾT BỊ CỬA PHÒNG
# ==============================================================================
with tab_door:
    col_ctrl, col_screen = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.subheader("Cấu hình thiết bị cửa")
        selected_room = st.selectbox(
            "Chọn số phòng gắn thiết bị:",
            options=list(db["rooms"].keys()),
            format_func=lambda r: f"Phòng {r} ({db['rooms'][r]['type']})",
        )
        room_info = db["rooms"][selected_room]

        st.info(
            f"**Trạng thái phòng {selected_room}:** "
            f"{'🟢 Đang có khách lưu trú' if room_info['status'] == 'occupied' else '⚪ Phòng trống'}"
        )

        dnd_state = st.toggle("🔕 Không làm phiền (Do Not Disturb)", value=room_info["dnd"])
        clean_state = st.toggle("🧹 Yêu cầu dọn phòng (Make Up Room)", value=room_info["cleaning"])
        if dnd_state != room_info["dnd"] or clean_state != room_info["cleaning"]:
            db["rooms"][selected_room]["dnd"] = dnd_state
            db["rooms"][selected_room]["cleaning"] = clean_state
            save_db(db)
            st.rerun()

    with col_screen:
        st.subheader(f"Màn Hình Cảm Ứng Cửa Phòng {selected_room}")

        if not st.session_state.door_camera_open:
            st.markdown(
                """
                <div style="padding:24px; border-radius:12px; background:#0f172a; color:#f8fafc; border:1px solid #d4af37;">
                    <h3 style="color:#d4af37; margin-top:0;">🔒 CỬA ĐANG KHÓA • CAMERA ĐANG TẮT</h3>
                    <p style="color:#cbd5e1;">Hệ thống chỉ kích hoạt cảm biến khuôn mặt khi quý khách nhấn nút xác thực bên dưới.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔓 Quét Khuôn Mặt Vào Phòng", use_container_width=True, type="primary"):
                    st.session_state.scan_direction = "Vào phòng"
                    st.session_state.door_camera_open = True
                    st.session_state.last_auth_result = None
                    st.rerun()
            with btn_col2:
                if st.button("🚶 Xác Nhận Ra Khỏi Phòng", use_container_width=True):
                    st.session_state.scan_direction = "Ra khỏi phòng"
                    st.session_state.door_camera_open = True
                    st.session_state.last_auth_result = None
                    st.rerun()
        else:
            st.warning(f"📷 Đang mở cảm biến xác thực: **{st.session_state.scan_direction}**")
            door_source = st.radio(
                "Chọn cách quét:", ["Dùng Camera cửa", "Tải ảnh kiểm thử"], horizontal=True, key="door_src"
            )
            cam_photo = (
                st.camera_input("Đưa khuôn mặt vào khung hình và nhấn Take Photo")
                if door_source == "Dùng Camera cửa"
                else st.file_uploader("Chọn ảnh khuôn mặt để quét thử:", type=["jpg", "jpeg", "png"], key="door_up")
            )

            if st.button("❌ Hủy & Đóng Cảm Biến"):
                st.session_state.door_camera_open = False
                st.rerun()

            if cam_photo is not None:
                with st.spinner("Đang đối chiếu sinh trắc học..."):
                    try:
                        live_emb = extract_face_embedding(cam_photo)

                        best_room = None
                        best_dist = 999.0
                        for r_id, r_data in db["rooms"].items():
                            if r_data["status"] == "occupied" and r_data["guest"]:
                                dist = cosine_distance(live_emb, r_data["guest"]["embedding"])
                                if dist < best_dist:
                                    best_dist = dist
                                    best_room = r_id

                        if best_dist <= SIMILARITY_THRESHOLD:
                            matched_guest = db["rooms"][best_room]["guest"]
                            if best_room == selected_room:
                                st.session_state.last_auth_result = {
                                    "type": "granted",
                                    "guest": matched_guest,
                                    "distance": best_dist,
                                    "direction": st.session_state.scan_direction,
                                }
                                add_log(
                                    db,
                                    selected_room,
                                    matched_guest["name"],
                                    st.session_state.scan_direction,
                                    "Thành công",
                                    f"Độ sai lệch: {best_dist:.3f}",
                                )
                            else:
                                st.session_state.last_auth_result = {
                                    "type": "wrong_room",
                                    "guest": matched_guest,
                                    "actual_room": best_room,
                                }
                                add_log(
                                    db,
                                    selected_room,
                                    matched_guest["name"],
                                    st.session_state.scan_direction,
                                    "Từ chối (Nhầm phòng)",
                                    f"Khách thuộc phòng {best_room}",
                                )
                        else:
                            st.session_state.last_auth_result = {"type": "denied"}
                            add_log(
                                db,
                                selected_room,
                                "Người lạ",
                                st.session_state.scan_direction,
                                "Cảnh báo từ chối",
                                f"Sai lệch: {best_dist:.3f}",
                            )

                        st.session_state.door_camera_open = False
                        st.rerun()
                    except ValueError:
                        st.error("Không tìm thấy khuôn mặt rõ nét trong ảnh. Vui lòng thử lại.")

        res = st.session_state.last_auth_result
        if res:
            st.divider()
            if res["type"] == "granted":
                g = res["guest"]
                st.success(f"✅ MỞ KHÓA THÀNH CÔNG — {res['direction'].upper()}")
                st.markdown(
                    f"""
                    - **Khách lưu trú:** {g['title']} **{g['name']}**
                    - **Hạng thẻ:** `{g['tier']}` | **Mã đặt phòng:** `{g['booking_id']}`
                    - **Thời gian lưu trú:** {g['checkin']} → {g['checkout']}
                    - **Ghi chú VIP:** {g['notes'] or 'Không có'}
                    """
                )
            elif res["type"] == "wrong_room":
                g = res["guest"]
                st.warning(
                    f"⚠️ **NHẦM PHÒNG:** Xin chào {g['title']} {g['name']}, "
                    f"quý khách được đăng ký tại **Phòng {res['actual_room']}**, không phải Phòng {selected_room}."
                )
            elif res["type"] == "denied":
                st.error("🚨 **TỪ CHỐI TRUY CẬP:** Khuôn mặt chưa được đăng ký cho phòng này. Vui lòng liên hệ Quầy Lễ Tân.")

# ==============================================================================
# TAB 2: QUẦY LỄ TÂN (PMS)
# ==============================================================================
with tab_pms:
    st.subheader("1. Sơ Đồ Trạng Thái Phòng")
    room_cols = st.columns(len(db["rooms"]))
    for idx, (r_id, r_data) in enumerate(db["rooms"].items()):
        with room_cols[idx]:
            if r_data["status"] == "occupied":
                g = r_data["guest"]
                st.success(f"**Phòng {r_id}** ({r_data['type']})\n\n👤 {g['name']}\n\n🏅 {g['tier']}")
                if st.button(f"Check-out P.{r_id}", key=f"co_{r_id}", use_container_width=True):
                    add_log(db, r_id, g["name"], "Check-out Lễ tân", "Hoàn tất", "Xóa dữ liệu khuôn mặt khỏi khóa")
                    db["rooms"][r_id]["status"] = "vacant"
                    db["rooms"][r_id]["guest"] = None
                    db["rooms"][r_id]["dnd"] = False
                    db["rooms"][r_id]["cleaning"] = False
                    save_db(db)
                    st.rerun()
            else:
                st.info(f"**Phòng {r_id}** ({r_data['type']})\n\n⚪ Phòng trống\n\nSẵn sàng đón khách")

    st.divider()
    col_form, col_logs = st.columns([1, 1.3], gap="large")

    with col_form:
        st.subheader("2. Check-in & Đăng Ký Khuôn Mặt")
        vacant_rooms = [r for r, d in db["rooms"].items() if d["status"] == "vacant"]

        if not vacant_rooms:
            st.warning("Tất cả các phòng hiện đang có khách.")
        else:
            photo_source = st.radio(
                "Nguồn ảnh khuôn mặt:",
                ["Tải ảnh chân dung", "Chụp trực tiếp tại quầy"],
                horizontal=True,
            )
            with st.form("checkin_form"):
                reg_room = st.selectbox("Chọn phòng trống:", vacant_rooms)
                reg_title = st.selectbox("Danh xưng:", ["Ông", "Bà", "Anh", "Chị"])
                reg_name = st.text_input("Họ và tên khách hàng:", placeholder="Nguyễn Minh Quân")
                reg_tier = st.selectbox("Hạng thành viên:", ["Diamond VIP", "Gold Member", "Standard"])
                reg_booking = st.text_input("Mã Booking:", value=f"BK-{datetime.now().strftime('%H%M%S')}")
                c1, c2 = st.columns(2)
                reg_checkin = c1.date_input("Ngày nhận phòng")
                reg_checkout = c2.date_input("Ngày trả phòng")
                reg_notes = st.text_area("Ghi chú phục vụ:", placeholder="Dị ứng hải sản, cần thêm gối...")

                target_img = None
                if photo_source == "Chụp trực tiếp tại quầy":
                    target_img = st.camera_input("Chụp ảnh tại quầy")
                else:
                    target_img = st.file_uploader("Chọn tệp ảnh chân dung (JPG/PNG):", type=["jpg", "jpeg", "png"])

                submitted = st.form_submit_button("✅ Hoàn Tất Check-in & Cấp Quyền Mở Cửa", type="primary")
                if submitted:
                    if not reg_name.strip() or target_img is None:
                        st.error("Vui lòng nhập họ tên và cung cấp ảnh khuôn mặt khách hàng.")
                    else:
                        with st.spinner("Đang trích xuất đặc trưng khuôn mặt..."):
                            try:
                                emb = extract_face_embedding(target_img)
                                db["rooms"][reg_room]["status"] = "occupied"
                                db["rooms"][reg_room]["guest"] = {
                                    "title": reg_title,
                                    "name": reg_name.strip(),
                                    "tier": reg_tier,
                                    "booking_id": reg_booking,
                                    "checkin": str(reg_checkin),
                                    "checkout": str(reg_checkout),
                                    "notes": reg_notes.strip(),
                                    "embedding": emb,
                                }
                                add_log(db, reg_room, reg_name.strip(), "Check-in Lễ tân", "Đã cấp quyền", f"Hạng {reg_tier}")
                                st.success(f"Đã đăng ký khuôn mặt cho {reg_name} tại Phòng {reg_room}!")
                                st.rerun()
                            except ValueError:
                                st.error("Không nhận diện được khuôn mặt trong ảnh. Vui lòng chọn ảnh rõ mặt hơn.")

    with col_logs:
        st.subheader("3. Nhật Ký Ra / Vào Phòng")
        if db["logs"]:
            df_logs = pd.DataFrame(db["logs"])
            st.dataframe(df_logs, use_container_width=True, height=400)

            csv_bytes = df_logs.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Xuất Báo Cáo Nhật Ký (CSV Excel)",
                data=csv_bytes,
                file_name=f"LuxStay_Access_Logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Chưa có lịch sử ra/vào nào được ghi nhận.")
