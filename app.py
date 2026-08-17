"""
Aplikasi Deteksi Penyakit Kulit — ResNet-50 & EfficientNetB0 (8 Kelas)
Skripsi Sistem Informasi

Jalankan dengan:
    py -m streamlit run app.py
"""

import time
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Preprocessing tiap arsitektur dibungkus di dalam model (layer Lambda) sejak training,
# jadi harus didaftarkan ulang di sini dengan nama & package yang sama persis
# supaya file .h5 bisa dimuat (tanpa ini load_model akan gagal).
@tf.keras.utils.register_keras_serializable(package="skin_disease")
def resnet_preprocess(x):
    return tf.keras.applications.resnet50.preprocess_input(x)


@tf.keras.utils.register_keras_serializable(package="skin_disease")
def efficientnet_preprocess(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)


# =========================================================================
# KONFIGURASI — cukup ubah bagian ini, tidak perlu edit kode di bawah
# =========================================================================
APP_NAME    = "DermaScan"                                          # <── ganti nama website di sini
APP_TAGLINE = "Skrining awal penyakit kulit berbasis kecerdasan buatan"
APP_ICON    = "🩺"                                                 # ikon tab browser

# Dua arsitektur yang bisa dipilih pengguna langsung dari halaman utama.
MODEL_OPTIONS = {
    "ResNet-50": "resnet50_skin_disease_fixx.h5",
    "EfficientNetB0": "efficientnetb0_skin_disease_fixx.h5",
}
IMAGE_SIZE = (224, 224)

# ⚠️ PENTING: urutan kelas HARUS sama dengan CLASS_NAMES di notebook training,
# yaitu urutan alfabetis label folder (perilaku standar flow_from_directory Keras):
# Eksim, Impetigo, Jerawat, Kurap, Kutil, KutuAir, Panu, Skabies.
# "warna" dipakai untuk aksen bar chart & kartu hasil per penyakit.
CLASSES = [
    {
        "nama": "Eksim",
        "warna": "#B7472A",
        "deskripsi": "Peradangan kulit kronis (dermatitis atopik) yang menyebabkan kulit kering, gatal, dan meradang berulang.",
    },
    {
        "nama": "Impetigo",
        "warna": "#C4693A",
        "deskripsi": "Infeksi bakteri menular yang menyebabkan luka lepuh berkerak.",
    },
    {
        "nama": "Jerawat",
        "warna": "#A8447A",
        "deskripsi": "Peradangan pada folikel rambut dan kelenjar minyak kulit.",
    },
    {
        "nama": "Kurap",
        "warna": "#6E5AA8",
        "deskripsi": "Infeksi jamur yang membentuk ruam melingkar bersisik.",
    },
    {
        "nama": "Kutil",
        "warna": "#6F6A5E",
        "deskripsi": "Pertumbuhan kulit akibat infeksi virus HPV.",
    },
    {
        "nama": "Kutu Air",
        "warna": "#2E6E8E",
        "deskripsi": "Infeksi jamur pada kaki yang menyebabkan kulit mengelupas.",
    },
    {
        "nama": "Panu",
        "warna": "#3E8E63",
        "deskripsi": "Infeksi jamur Malassezia yang menyebabkan bercak putih atau kecokelatan pada kulit.",
    },
    {
        "nama": "Skabies",
        "warna": "#A3341F",
        "deskripsi": "Infeksi kulit menular akibat tungau Sarcoptes scabiei yang menyebabkan gatal hebat, terutama pada malam hari.",
    },
]

PRIMARY    = "#1F6F5C"
PRIMARY_D  = "#164F41"
ACCENT     = "#C4693A"
ACCENT_SOFT= "#F3E4DA"
INK        = "#211F1B"
MUTED      = "#6F6A5E"
LINE       = "#E4DECF"
PAPER      = "#F7F4EC"
CARD_BG    = "#FFFFFF"
STONE_BG   = "#F1EDE2"

CLASS_NAMES = [c["nama"] for c in CLASSES]

# Ambang batas keyakinan (persen, 0–100). Bila keyakinan tertinggi model berada
# di BAWAH nilai ini, hasil dianggap "Tidak Terklasifikasi" — artinya gambar
# kemungkinan bukan salah satu dari 8 penyakit yang dikenali, bukan gambar kulit,
# atau kualitasnya kurang baik. Naikkan angkanya agar lebih ketat (lebih sering
# "tidak terklasifikasi"), turunkan agar lebih longgar.
UNKNOWN_THRESHOLD = 60.0

# "Kelas" semu untuk hasil yang tidak dapat dipastikan. Nilai keyakinan sengaja
# TIDAK ditampilkan ke pengguna untuk kelas ini — lihat render_results/render_comparison.
UNKNOWN = {
    "nama": "Tidak Terklasifikasi",
    "warna": "#6F6A5E",
    "deskripsi": (
        "Gambar tidak dapat dikenali dengan cukup yakin sebagai salah satu dari 8 "
        "penyakit kulit yang didukung. Kemungkinan gambar kurang jelas, bukan gambar "
        "kulit, atau kondisinya berada di luar cakupan model."
    ),
}

# =========================================================================
# KONFIGURASI HALAMAN
# =========================================================================
st.set_page_config(
    page_title=f"{APP_NAME} — Deteksi Penyakit Kulit",
    page_icon=APP_ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================================
# CUSTOM CSS
# =========================================================================
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');

        .stApp {{ background-color: {PAPER}; }}
        .block-container {{ max-width: 800px; padding-top: 1.4rem; padding-bottom: 3rem; }}
        html, body, [class*="css"] {{ color: {INK}; }}

        h1, h2, h3, .result-disease, .hero h1 {{
            font-family: 'Fraunces', Georgia, serif;
        }}

        /* ---------- Hero header ---------- */
        .hero {{
            position: relative;
            background: {PRIMARY_D};
            border-radius: 10px;
            padding: 2.2rem 1.7rem 1.8rem 1.7rem;
            margin-bottom: 1.1rem;
            border-bottom: 4px solid {ACCENT};
        }}
        .hero-top {{ display: flex; align-items: center; gap: 0.9rem; }}
        .hero-badge {{
            flex-shrink: 0; width: 46px; height: 46px; border-radius: 50%;
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
            display: flex; align-items: center; justify-content: center;
        }}
        .hero h1 {{
            color: #FFFFFF; font-weight: 600; letter-spacing: -0.3px;
            margin: 0; font-size: 2.05rem; line-height: 1.1;
        }}
        .hero p {{
            color: rgba(255,255,255,0.78); font-size: 0.95rem; margin: 0.35rem 0 0 0;
        }}

        /* ---------- Stat cards di bawah hero ---------- */
        .stat-row {{ display: flex; gap: 0.7rem; margin-bottom: 1.5rem; }}
        .stat-card {{
            flex: 1; text-align: center;
            background: {CARD_BG};
            border: 1px solid {LINE};
            border-radius: 10px;
            padding: 0.75rem 0.5rem;
        }}
        .stat-card .num {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.2rem; font-weight: 600; color: {PRIMARY_D};
        }}
        .stat-card .lbl {{
            font-size: 0.7rem; color: {MUTED}; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.6px; margin-top: 0.1rem;
        }}

        /* ---------- Card umum ---------- */
        .card {{
            background-color: {CARD_BG};
            border-radius: 10px;
            padding: 1.3rem 1.4rem;
            border: 1px solid {LINE};
            margin-bottom: 1.1rem;
        }}
        .card-title {{
            font-weight: 700; color: {INK};
            font-size: 0.78rem; margin-bottom: 0.7rem;
            text-transform: uppercase; letter-spacing: 0.7px;
        }}

        /* ---------- Uploader & kamera ---------- */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {STONE_BG};
            border: 1.5px dashed {PRIMARY};
            border-radius: 10px;
        }}

        /* ---------- Hasil ---------- */
        .result-card {{
            background: {CARD_BG};
            border: 1px solid {LINE};
            border-top: 3px solid {PRIMARY};
            border-radius: 10px;
            padding: 1.3rem 1.4rem;
        }}
        .result-eyebrow {{
            font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
            color: {MUTED}; text-transform: uppercase; margin-bottom: 0.5rem;
        }}
        .result-disease {{
            font-size: 1.7rem; font-weight: 600;
            margin-bottom: 0.55rem; line-height: 1.2;
        }}
        .result-description {{
            color: {INK}; opacity: 0.75; font-size: 0.9rem; margin: 0.7rem 0 0.5rem 0;
            line-height: 1.5;
        }}
        .confidence-badge {{
            display: inline-block; padding: 0.3rem 0.85rem; border-radius: 6px;
            font-weight: 700; font-size: 0.85rem;
        }}
        .confidence-high   {{ background: #E4EFE5; color: #2C6E3F; }}
        .confidence-medium {{ background: #F6ECD9; color: #92600E; }}
        .confidence-low    {{ background: #F5E1DB; color: #A3341F; }}

        /* ---------- Hasil "Tidak Terklasifikasi" ---------- */
        .unknown-card {{
            background: {CARD_BG};
            border: 1px solid {LINE};
            border-top: 3px solid {MUTED};
            border-radius: 10px;
            padding: 1.3rem 1.4rem;
        }}
        .unknown-note {{
            background: {STONE_BG}; border: 1px solid {LINE}; border-radius: 8px;
            padding: 0.75rem 0.95rem; font-size: 0.83rem; color: {MUTED}; margin-top: 0.8rem;
            line-height: 1.5;
        }}

        /* ---------- Perbandingan model ---------- */
        .time-badge {{
            display: inline-block; margin-top: 0.65rem; padding: 0.28rem 0.8rem;
            border-radius: 6px; font-weight: 600; font-size: 0.78rem;
            background: {STONE_BG}; color: {MUTED};
        }}
        .compare-badge {{
            display: inline-block; margin: 0.5rem 0.4rem 0 0; padding: 0.22rem 0.65rem;
            border-radius: 6px; font-weight: 700; font-size: 0.72rem;
            text-transform: uppercase; letter-spacing: 0.4px;
        }}
        .compare-badge-fast {{ background: #E3ECF3; color: #2A5C82; }}
        .compare-badge-conf {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
        .compare-summary {{
            font-size: 0.9rem; color: {INK}; margin-bottom: 0.5rem; line-height: 1.55;
        }}

        /* ---------- Bar chart custom ---------- */
        .bar-row  {{ display: flex; align-items: center; margin-bottom: 0.55rem; gap: 0.65rem; }}
        .bar-name {{
            width: 140px; min-width: 140px; font-size: 0.82rem;
            color: {INK}; font-weight: 500; text-align: right;
        }}
        .bar-track {{
            flex: 1; background: {STONE_BG}; border-radius: 5px; height: 15px; overflow: hidden;
        }}
        .bar-fill {{
            height: 100%; border-radius: 5px;
            transition: width 0.6s ease;
        }}
        .bar-pct {{
            width: 52px; min-width: 52px; font-size: 0.8rem;
            color: {INK}; font-weight: 600;
        }}

        /* ---------- Riwayat ---------- */
        .history-item {{
            display: flex; align-items: center; gap: 0.4rem;
            background: {CARD_BG}; border: 1px solid {LINE};
            border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
        }}
        .history-item .h-name {{ font-weight: 600; color: {INK}; font-size: 0.9rem; }}
        .history-item .h-meta {{ font-size: 0.76rem; color: {MUTED}; }}
        .history-badge {{
            margin-left: auto; padding: 0.2rem 0.65rem; border-radius: 6px;
            font-size: 0.76rem; font-weight: 700; white-space: nowrap;
        }}

        /* ---------- Disclaimer & footer ---------- */
        .disclaimer {{
            display: flex; gap: 0.7rem; align-items: flex-start;
            background-color: {STONE_BG}; border: 1px solid {LINE};
            border-radius: 10px; padding: 1rem 1.15rem; font-size: 0.83rem;
            color: {INK}; opacity: 0.92; margin-top: 2rem;
        }}
        .disclaimer-mark {{
            flex-shrink: 0; width: 20px; height: 20px; border-radius: 50%;
            background: {ACCENT}; color: #FFF; font-weight: 800; font-size: 0.72rem;
            display: flex; align-items: center; justify-content: center;
        }}
        .footer-credit {{
            text-align: center; font-size: 0.76rem; color: {MUTED};
            margin-top: 1.2rem;
        }}

        /* ---------- Tombol ---------- */
        .stButton > button {{
            border-radius: 8px; font-weight: 600;
        }}

        section[data-testid="stSidebar"] {{ display: none; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# HEADER + STAT
# =========================================================================
HERO_ICON_SVG = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 21C12 21 4 15.5 4 9.5C4 6.46 6.46 4 9.5 4C11.04 4 12.5 4.72 13.5 5.9C14.24 5.03 15.29 4.38 16.5 4.13" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
    <path d="M18 4C19.6569 4 21 5.34315 21 7C21 8.65685 19.6569 10 18 10C16.3431 10 15 8.65685 15 7C15 5.34315 16.3431 4 18 4Z" stroke="white" stroke-width="1.6"/>
    <path d="M4 14C6.5 14 6.5 16.5 9 16.5C11.5 16.5 11.5 14 14 14" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-top">
            <div class="hero-badge">{HERO_ICON_SVG}</div>
            <div>
                <h1>{APP_NAME}</h1>
                <p>{APP_TAGLINE}</p>
            </div>
        </div>
    </div>
    <div class="stat-row">
        <div class="stat-card"><div class="num">8</div><div class="lbl">Kelas Penyakit</div></div>
        <div class="stat-card"><div class="num">2</div><div class="lbl">Arsitektur AI</div></div>
        <div class="stat-card"><div class="num">224×224</div><div class="lbl">Resolusi Input</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card-title">Pilih Arsitektur Model</div>', unsafe_allow_html=True)
arsitektur = st.selectbox(
    "Pilih arsitektur model AI",
    list(MODEL_OPTIONS.keys()),
    label_visibility="collapsed",
)
st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)


# =========================================================================
# MODEL & PREPROCESSING
# =========================================================================
@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Model sudah membungkus preprocessing ResNet50 di dalam dirinya (layer Lambda),
    # jadi di sini cukup diresize dan dikirim sebagai piksel mentah [0, 255].
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.array(image).astype(np.float32)
    array = np.expand_dims(array, axis=0)
    return array


def confidence_class(pct: float) -> str:
    if pct >= 80:
        return "confidence-high"
    if pct >= 50:
        return "confidence-medium"
    return "confidence-low"


def confidence_colors(pct: float):
    if pct >= 80:
        return "#E4EFE5", "#2C6E3F"
    if pct >= 50:
        return "#F6ECD9", "#92600E"
    return "#F5E1DB", "#A3341F"


def predict_with_timing(model_path: str, img_array: np.ndarray):
    """Prediksi + ukur waktu klasifikasi murni (tidak termasuk waktu load model)."""
    model = get_model(model_path)
    t0 = time.perf_counter()
    preds = model.predict(img_array, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return preds * 100.0, elapsed_ms


def effective_result(probabilities: np.ndarray):
    """Tentukan hasil tampilan berdasarkan ambang keyakinan.

    Mengembalikan tuple:
        display   -> dict kelas yang ditampilkan (UNKNOWN bila di bawah ambang)
        conf      -> keyakinan tertinggi (persen)
        unknown   -> True bila di bawah ambang (Tidak Terklasifikasi)
        candidate -> dict kelas kandidat terdekat (top-1 asli model)
    """
    top_idx = int(np.argmax(probabilities))
    conf = float(probabilities[top_idx])
    candidate = CLASSES[top_idx]
    if conf < UNKNOWN_THRESHOLD:
        return UNKNOWN, conf, True, candidate
    return candidate, conf, False, candidate


def bar_chart_html(probabilities: np.ndarray) -> str:
    order = np.argsort(probabilities)[::-1]
    bars_html = ""
    for rank, i in enumerate(order):
        pct = float(probabilities[i])
        warna = CLASSES[i]["warna"]
        opacity = "1" if rank == 0 else "0.4"
        bars_html += (
            f'<div class="bar-row">'
            f'<div class="bar-name">{CLASSES[i]["nama"]}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{max(pct,1.5):.1f}%; background:{warna}; opacity:{opacity};"></div></div>'
            f'<div class="bar-pct">{pct:.1f}%</div>'
            f"</div>"
        )
    return bars_html


# =========================================================================
# SESSION STATE
# =========================================================================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: nama, conf, waktu, sumber


def reset_app():
    st.session_state.uploader_key += 1


def render_results(image: Image.Image, sumber: str):
    """Prediksi + tampilkan hasil untuk satu gambar."""
    with st.spinner(f"Model {arsitektur} sedang menganalisis gambar..."):
        try:
            probabilities, elapsed_ms = predict_with_timing(
                MODEL_OPTIONS[arsitektur], preprocess_image(image)
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            return

    display, top_confidence, unknown, candidate = effective_result(probabilities)

    # simpan ke riwayat (hindari duplikat saat rerun)
    entry = {
        "nama": display["nama"],
        "conf": top_confidence,
        "waktu": datetime.now().strftime("%H:%M:%S"),
        "sumber": sumber,
        "model": arsitektur,
        "unknown": unknown,
    }
    if not st.session_state.history or st.session_state.history[-1] != entry:
        st.session_state.history.append(entry)

    col_left, col_right = st.columns([1, 1.15])

    with col_left:
        st.image(image, width="stretch", caption="Gambar yang dianalisis")
        st.button("Periksa Gambar Lain", on_click=reset_app, width="stretch", type="primary")

    with col_right:
        if unknown:
            st.markdown(
                f"""
                <div class="unknown-card">
                    <div class="result-eyebrow">Hasil Deteksi &middot; {arsitektur}</div>
                    <div class="result-disease" style="color:{UNKNOWN['warna']};">{UNKNOWN['nama']}</div>
                    <div class="result-description">{UNKNOWN['deskripsi']}</div>
                    <span class="time-badge">Waktu klasifikasi: {elapsed_ms:.1f} ms</span>
                    <div class="unknown-note">
                        Kandidat terdekat: <b>{candidate['nama']}</b>, namun tingkat keyakinan model
                        belum mencukupi untuk dipastikan. Coba unggah foto yang lebih jelas, fokus,
                        dan dengan pencahayaan yang cukup.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-eyebrow">Hasil Deteksi &middot; {arsitektur}</div>
                    <div class="result-disease" style="color:{display['warna']};">{display["nama"]}</div>
                    <span class="confidence-badge {confidence_class(top_confidence)}">
                        Keyakinan: {top_confidence:.1f}%
                    </span>
                    <div class="result-description">{display["deskripsi"]}</div>
                    <span class="time-badge">Waktu klasifikasi: {elapsed_ms:.1f} ms</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------- Bar chart custom: hanya ditampilkan bila hasil terklasifikasi ----------
    if not unknown:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Distribusi Keyakinan Semua Kelas</div>
                {bar_chart_html(probabilities)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Unduh ringkasan hasil ----------
    if unknown:
        baris_hasil = (
            f"Hasil               : {UNKNOWN['nama']}\n"
            f"Kandidat terdekat   : {candidate['nama']}\n"
            f"Waktu klasifikasi   : {elapsed_ms:.1f} ms\n"
            f"Keterangan          : {UNKNOWN['deskripsi']}\n\n"
            "Catatan: tingkat keyakinan tidak ditampilkan karena gambar tidak\n"
            "berhasil diklasifikasikan dengan cukup yakin.\n"
        )
    else:
        baris_hasil = (
            f"Hasil       : {display['nama']}\n"
            f"Keyakinan   : {top_confidence:.1f}%\n"
            f"Waktu klasifikasi : {elapsed_ms:.1f} ms\n"
            f"Deskripsi   : {display['deskripsi']}\n\n"
            "Distribusi semua kelas:\n"
            + "\n".join(
                f"  - {CLASSES[i]['nama']:<20} {float(probabilities[i]):5.1f}%"
                for i in np.argsort(probabilities)[::-1]
            )
            + "\n"
        )
    ringkasan = (
        f"{APP_NAME} — Ringkasan Hasil Deteksi\n"
        f"Waktu       : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
        f"Model AI    : {arsitektur}\n"
        + baris_hasil
        + "\nCatatan: Hasil ini adalah skrining awal berbasis AI, bukan diagnosis medis.\n"
    )
    st.download_button(
        "Unduh Ringkasan Hasil (.txt)",
        data=ringkasan,
        file_name=f"hasil_{APP_NAME.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        width="stretch",
    )


def render_comparison(image: Image.Image):
    """Jalankan ResNet-50 & EfficientNetB0 sekaligus lalu bandingkan hasil dan kecepatannya."""
    names = list(MODEL_OPTIONS.keys())
    results = {}
    with st.spinner("Menjalankan ResNet-50 dan EfficientNetB0 sekaligus..."):
        try:
            img_array = preprocess_image(image)
            for name in names:
                probs, elapsed_ms = predict_with_timing(MODEL_OPTIONS[name], img_array)
                display, conf, unknown, candidate = effective_result(probs)
                results[name] = {
                    "top": display,
                    "conf": conf,
                    "time_ms": elapsed_ms,
                    "probs": probs,
                    "unknown": unknown,
                    "candidate": candidate,
                }
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            return

    st.image(image, width="stretch", caption="Gambar yang dibandingkan")
    st.button(
        "Bandingkan Gambar Lain", on_click=reset_app, width="stretch",
        type="primary", key="reset_compare",
    )

    fastest = min(names, key=lambda n: results[n]["time_ms"])
    slowest = max(names, key=lambda n: results[n]["time_ms"])
    confident_names = [n for n in names if not results[n]["unknown"]]
    most_confident = max(confident_names, key=lambda n: results[n]["conf"]) if confident_names else None

    cols = st.columns(2)
    for col, name in zip(cols, names):
        r = results[name]
        badges = ""
        if name == fastest:
            badges += '<span class="compare-badge compare-badge-fast">Tercepat</span>'
        if name == most_confident:
            badges += '<span class="compare-badge compare-badge-conf">Paling Yakin</span>'
        card_cls = "unknown-card" if r["unknown"] else "result-card"
        if r["unknown"]:
            meta_html = (
                f'<div style="font-size:0.78rem; color:{MUTED}; margin-top:0.5rem;">'
                f'Kandidat terdekat: {r["candidate"]["nama"]}</div>'
            )
        else:
            meta_html = (
                f'<span class="confidence-badge {confidence_class(r["conf"])}">'
                f'Keyakinan: {r["conf"]:.1f}%</span>'
            )
        # NB: setiap baris HTML di bawah harus tidak kosong — st.markdown menutup
        # blok HTML mentah di baris kosong pertama, sehingga baris sesudahnya
        # (mis. <br> / <span>) malah dirender sebagai teks/code literal.
        with col:
            st.markdown(
                f"""<div class="{card_cls}">
                    <div class="result-eyebrow">{name}</div>
                    <div class="result-disease" style="color:{r['top']['warna']}; font-size:1.3rem;">{r['top']['nama']}</div>
                    {meta_html}
                    <div style="margin-top:0.65rem;"><span class="time-badge">Waktu klasifikasi: {r['time_ms']:.1f} ms</span></div>
                    <div>{badges}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    sepakat = results[names[0]]["top"]["nama"] == results[names[1]]["top"]["nama"]
    r0, r1 = results[names[0]], results[names[1]]
    if r0["unknown"] and r1["unknown"]:
        kesimpulan = (
            "Kedua model sama-sama <b>tidak dapat memastikan</b> hasilnya — status "
            "<b>Tidak Terklasifikasi</b>. Kemungkinan gambar di luar 8 kelas yang dikenali "
            "atau kualitasnya kurang baik."
        )
    elif sepakat:
        kesimpulan = f"Kedua model sepakat: <b>{r0['top']['nama']}</b>."
    elif r0["unknown"] or r1["unknown"]:
        unknown_name, known_name = (names[0], names[1]) if r0["unknown"] else (names[1], names[0])
        known_r = results[known_name]
        kesimpulan = (
            f"Kedua model berbeda pendapat — <b>{unknown_name}</b> tidak dapat memastikan "
            f"hasilnya, sementara <b>{known_name}</b> mendeteksi <b>{known_r['top']['nama']}</b> "
            f"({known_r['conf']:.1f}%)."
        )
    else:
        kesimpulan = (
            f"Kedua model berbeda pendapat — <b>{names[0]}</b>: "
            f"{r0['top']['nama']} ({r0['conf']:.1f}%), "
            f"<b>{names[1]}</b>: {r1['top']['nama']} "
            f"({r1['conf']:.1f}%)."
        )
    time_gap_ms = results[slowest]["time_ms"] - results[fastest]["time_ms"]
    ratio = results[slowest]["time_ms"] / max(results[fastest]["time_ms"], 1e-6)
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Ringkasan Perbandingan</div>
            <div class="compare-summary">{kesimpulan}</div>
            <div class="compare-summary">
                <b>{fastest}</b> lebih cepat {ratio:.1f}× dibanding <b>{slowest}</b>
                pada gambar ini (selisih {time_gap_ms:.1f} ms).
            </div>
            <div style="font-size:0.78rem; color:{MUTED};">
                Catatan: percobaan pertama tiap model bisa terasa lebih lambat karena
                TensorFlow baru membangun graph komputasinya; jalankan sekali lagi untuk
                waktu yang lebih representatif.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Distribusi tiap model, berdampingan (hanya bila terklasifikasi) ----------
    cols2 = st.columns(2)
    for col, name in zip(cols2, names):
        r = results[name]
        with col:
            if r["unknown"]:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">Distribusi {name}</div>
                        <div style="font-size:0.82rem; color:{MUTED};">
                            Tidak ditampilkan — hasil tidak terklasifikasi.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">Distribusi {name}</div>
                        {bar_chart_html(r["probs"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---------- Unduh ringkasan perbandingan ----------
    ringkasan = (
        f"{APP_NAME} — Ringkasan Perbandingan Model\n"
        f"Waktu       : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n"
    )
    for name in names:
        r = results[name]
        if r["unknown"]:
            baris = (
                f"  Hasil               : {UNKNOWN['nama']}\n"
                f"  Kandidat terdekat   : {r['candidate']['nama']}\n"
                f"  Waktu klasifikasi   : {r['time_ms']:.1f} ms\n"
                "  Catatan             : tingkat keyakinan tidak ditampilkan karena\n"
                "                        gambar tidak berhasil diklasifikasikan.\n"
            )
        else:
            order = np.argsort(r["probs"])[::-1]
            baris = (
                f"  Hasil             : {r['top']['nama']}\n"
                f"  Keyakinan         : {r['conf']:.1f}%\n"
                f"  Waktu klasifikasi : {r['time_ms']:.1f} ms\n"
                "  Distribusi semua kelas:\n"
                + "\n".join(
                    f"    - {CLASSES[i]['nama']:<20} {float(r['probs'][i]):5.1f}%" for i in order
                )
                + "\n"
            )
        ringkasan += f"[{name}]\n" + baris + "\n"
    ringkasan += "Catatan: Hasil ini adalah skrining awal berbasis AI, bukan diagnosis medis.\n"
    st.download_button(
        "Unduh Ringkasan Perbandingan (.txt)",
        data=ringkasan,
        file_name=f"perbandingan_{APP_NAME.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        width="stretch",
    )


# =========================================================================
# TAB: UPLOAD | BANDINGKAN | RIWAYAT
# =========================================================================
tab_upload, tab_bandingkan, tab_riwayat = st.tabs(
    ["Unggah Foto", "Bandingkan Model", "Riwayat Sesi"]
)

with tab_upload:
    uploaded_file = st.file_uploader(
        "Seret dan lepas gambar di sini, atau klik untuk memilih file",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div style="font-size:0.8rem; color:{MUTED};">Gunakan foto yang jelas, fokus pada area kulit '
        "yang bermasalah, dengan pencahayaan cukup. Format: JPG, JPEG, PNG</div>",
        unsafe_allow_html=True,
    )
    if uploaded_file is not None:
        render_results(Image.open(uploaded_file), sumber="Upload")

with tab_bandingkan:
    st.markdown(
        f'<div style="font-size:0.85rem; color:{MUTED}; margin-bottom:0.6rem;">'
        "Unggah satu gambar untuk dijalankan pada <b>ResNet-50</b> dan <b>EfficientNetB0</b> "
        "sekaligus, lengkap dengan perbandingan hasil dan waktu klasifikasinya.</div>",
        unsafe_allow_html=True,
    )
    compare_file = st.file_uploader(
        "Seret dan lepas gambar di sini, atau klik untuk memilih file",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_compare_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    if compare_file is not None:
        render_comparison(Image.open(compare_file))

with tab_riwayat:
    if not st.session_state.history:
        st.markdown(
            f'<div class="card" style="text-align:center; color:{MUTED};">'
            "Belum ada riwayat. Hasil deteksi selama sesi ini akan muncul di sini.</div>",
            unsafe_allow_html=True,
        )
    else:
        items_html = ""
        for h in reversed(st.session_state.history):
            if h.get("unknown"):
                bg, fg, badge_text = STONE_BG, MUTED, "Tidak diketahui"
            else:
                bg, fg = confidence_colors(h["conf"])
                badge_text = f'{h["conf"]:.1f}%'
            items_html += (
                f'<div class="history-item">'
                f'<div><div class="h-name">{h["nama"]}</div>'
                f'<div class="h-meta">{h["waktu"]} • via {h["sumber"]} • {h.get("model", "")}</div></div>'
                f'<div class="history-badge" style="background:{bg}; color:{fg};">{badge_text}</div>'
                f"</div>"
            )
        st.markdown(items_html, unsafe_allow_html=True)
        if st.button("Hapus Riwayat", width="stretch"):
            st.session_state.history = []
            st.rerun()

# =========================================================================
# INFO 8 KELAS (edukasi)
# =========================================================================
with st.expander("Penyakit kulit apa saja yang dapat dideteksi?"):
    for c in CLASSES:
        st.markdown(
            f'<div style="border-left:3px solid {c["warna"]}; padding:0.35rem 0.8rem; '
            f'margin-bottom:0.45rem; background:{CARD_BG}; border-radius:0 6px 6px 0;">'
            f'<b>{c["nama"]}</b> — {c["deskripsi"]}</div>',
            unsafe_allow_html=True,
        )

with st.expander("Tentang model AI ini"):
    st.markdown(
        """
        Website ini membandingkan dua arsitektur *transfer learning* dari ImageNet, dan
        kamu bisa memilih salah satunya untuk mendeteksi gambar di atas:

        | | ResNet-50 | EfficientNetB0 |
        |---|---|---|
        | Jumlah parameter | ± 24,8 juta | ± 4,8 juta |
        | Keunggulan | Akurasi lebih tinggi & val loss lebih stabil | Lebih ringan, lebih cepat dilatih & inferensi |

        - **Jumlah kelas:** 8 penyakit kulit umum di Indonesia
        - **Input:** foto lesi kulit, diproses ke ukuran 224×224 piksel
        - **Output:** probabilitas untuk setiap kelas (softmax)

        Kedua model dilatih pada dataset gambar penyakit kulit yang sama, yang telah
        melalui tahap pembersihan (filter kualitas, deduplikasi, dan seleksi relevansi).
        """
    )

# =========================================================================
# DISCLAIMER & FOOTER
# =========================================================================
st.markdown(
    f"""
    <div class="disclaimer">
        <div class="disclaimer-mark">!</div>
        <div><strong>Disclaimer:</strong> {APP_NAME} adalah alat bantu skrining awal berbasis
        kecerdasan buatan dan <strong>bukan pengganti diagnosis medis profesional</strong>.
        Hasil prediksi tidak menjamin keakuratan 100%. Silakan konsultasikan kondisi kulit
        Anda dengan dokter atau tenaga medis untuk diagnosis dan penanganan yang tepat.</div>
    </div>
    <div class="footer-credit">© 2026 {APP_NAME} — Skripsi Sistem Informasi • Perbandingan ResNet-50 & EfficientNetB0</div>
    """,
    unsafe_allow_html=True,
)
