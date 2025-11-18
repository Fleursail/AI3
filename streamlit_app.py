# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1plY1yS4oApifKpJ8DWPQvYrEtuS8xHeM")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

     labels[0]: {
       "texts": ["용서받지 못한 자", "바람의 검술", "이부 동생"],
       "images": ["data:image/jpeg;base64,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", "https://i.ytimg.com/vi/5QH8tlJBY74/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCAZK0RA1CioM04P330lS363sdPQA"],
       "videos": ["https://youtu.be/7xmgRLTjxIw?si=uUQTp3M3C1rCmJrW"]
     },
    labels[1]: {
       "texts": ["잊히지 못한 자", "이부 형", "바람의 검술"],
       "images": ["https://i.namu.wiki/i/eohH0OCjU9CYMY3HIwCIO-1AD0lOcmxqI_VYFeD-nLXJknPqql3oAVgGwV_AIIosu-hcNc4t617WX_qRSiUpjQ.webp", "https://prodigits.co.uk/pthumbs/wallpapers/2022/p2/misc/47/ybe8b640.jpg"],
       "videos": ["https://youtu.be/7xmgRLTjxIw?si=uUQTp3M3C1rCmJrW"]
     },
    labels[2]: {
       "texts": ["여명의 성위", "대가리 꽃밭", "쿨쿨 헤롱헤롱 방울"],
       "images": ["https://i.namu.wiki/i/osL_I9dSDCK4xbfgVvUr8Lzg6i28RThhfS3woTkfYzzW-hMSZE2HQt8CjzQlN0EbD5Vzj01rtPZYvCJG477dRQ.webp", "https://i.namu.wiki/i/EJzaHxsMnOJIufW7SjeMWE5cOjdQgIScUJzEa_ABje72OdXz20abAlTRFtPyatFBrLMZg6gbXdhNsRArbuIx8g.webp"],
       "videos": ["https://youtu.be/BDi29cwLY_I?si=2tAM6ea0zhHMUIcI"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
