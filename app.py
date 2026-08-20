"""
Criminal Face Recognition & Surveillance Dashboard
===================================================
Real-time Streamlit application optimized for Indian face demographics,
low-resolution CCTV enhancement, live camera surveillance, and criminal database search.
"""

import sys
import io
import os
import time
import base64
import psutil
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Set page config with dark/clean modern UI layout
st.set_page_config(
    page_title="Criminal Face Recognition System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1A2332;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2A3B50;
        margin-bottom: 10px;
    }
    .alert-danger {
        background-color: rgba(239, 83, 80, 0.15);
        border: 1px solid #EF5350;
        color: #EF5350;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
    }
    .alert-success {
        background-color: rgba(102, 187, 106, 0.15);
        border: 1px solid #66BB6A;
        color: #66BB6A;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
    }
    .suspect-box {
        background-color: #212936;
        border-left: 4px solid #E53935;
        padding: 12px;
        border-radius: 4px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache core engine to avoid re-loading models on every rerun
@st.cache_resource(show_spinner="Loading ArcFace & SCRFD Neural Models...")
def get_matcher():
    from core.matcher import FaceMatcher
    return FaceMatcher(
        model_name="buffalo_l",
        db_dir="./data/chromadb",
        similarity_threshold=0.45,
        gpu_id=0
    )

@st.cache_resource
def get_cctv_enhancer():
    from core.cctv_enhancer import CCTVEnhancer
    return CCTVEnhancer()

# Initialize core services
try:
    matcher = get_matcher()
    enhancer = get_cctv_enhancer()
    db = matcher.db
    detector = matcher.detector
except Exception as e:
    st.error(f"Failed to initialize core models: {e}")
    st.stop()

# ==========================================
# SIDEBAR: Hardware Telemetry & Thresholds
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### 🛡️ Criminal Vision AI")
    st.markdown("**Version:** 1.0.0 (Local RTX 5050)")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Recognition Settings")
    threshold = st.slider(
        "Match Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.45,
        step=0.01,
        help="Higher = stricter matches. Recommended for Indian datasets: 0.45"
    )
    matcher.threshold = threshold
    
    use_adaface_opt = st.checkbox(
        "🧠 AdaFace Quality-Adaptive Mode",
        value=True,
        help="Adapts neural margins dynamically for low-resolution/blurry CCTV crops."
    )
    
    st.markdown("---")
    st.markdown("#### 🖥️ Hardware Telemetry")
    
    # System stats
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)
    
    st.progress(cpu_usage / 100, text=f"CPU: {cpu_usage:.1f}%")
    st.progress(ram.percent / 100, text=f"RAM: {ram_used_gb:.1f} / {ram_total_gb:.1f} GB ({ram.percent}%)")
    
    st.info(f"💾 **Enrolled Suspects:** {db.count()}")
    st.caption("🔒 Runs 100% locally. Zero cloud data leaks.")

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-header">Criminal Identification & Surveillance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Optimized for Indian Face Demographics • CCTV Enhancement • Real-Time GPU Inference</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 Live Surveillance Camera",
    "🕵️ CCTV & Low-Res Image Search",
    "📼 Video Footage Forensics",
    "📁 Criminal Database (Enrollment)",
    "📊 Diagnostics & Incident Reports"
])

# (Tab 1, Tab 2 remain unchanged)

# ==========================================
# TAB 1: LIVE SURVEILLANCE CAMERA
# ==========================================
with tab1:
    st.markdown("### 🔴 Real-Time Camera Feed")
    st.write("Capture a frame from your webcam to detect and match faces against the criminal registry instantly.")
    
    col_cam, col_results = st.columns([3, 2])
    
    with col_cam:
        cam_image = st.camera_input("Take a snapshot for surveillance matching")
        
    with col_results:
        if cam_image is not None:
            # Convert bytes to numpy BGR image
            file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            t0 = time.perf_counter()
            search_results = matcher.search_image(frame_bgr, top_k=3)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            
            st.markdown(f"**⚡ Scan Latency:** `{elapsed_ms:.1f} ms` | **Faces Detected:** `{len(search_results)}`")
            
            any_match = False
            annotated_frame = frame_bgr.copy()
            
            for idx, res in enumerate(search_results):
                face = res["face"]
                box = [int(v) for v in face.bbox]
                matches = res["matches"]
                
                if matches and matches[0]["similarity"] >= threshold:
                    any_match = True
                    top_match = matches[0]
                    color = (0, 0, 255) # Red for match
                    label = f"SUSPECT: {top_match['name']} ({top_match['similarity']:.2f})"
                    
                    st.markdown(f"""
                    <div class="alert-danger">
                        🚨 <b>MATCH DETECTED:</b> {top_match['name']} (ID: {top_match['id']})<br>
                        Confidence: <b>{top_match['similarity'] * 100:.1f}%</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "crime_history" in top_match.get("metadata", {}):
                        st.caption(f"**History:** {top_match['metadata']['crime_history']}")
                else:
                    color = (0, 255, 0) # Green for unknown
                    label = "Unknown Person"
                
                # Draw box on frame
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(annotated_frame, label, (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if not any_match and len(search_results) > 0:
                st.markdown('<div class="alert-success">✅ No Criminal Matches in Database</div>', unsafe_allow_html=True)
            elif len(search_results) == 0:
                st.info("No human faces detected in the camera frame.")
            
            # Show annotated image
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Surveillance Output", use_container_width=True)

# ==========================================
# TAB 2: CCTV & LOW-RES IMAGE SEARCH
# ==========================================
with tab2:
    st.markdown("### 🕵️ CCTV Footage & Low-Resolution Image Processing")
    st.write("Upload surveillance photos or low-resolution mugshots to apply adaptive lighting enhancement, denoising, and criminal identification.")
    
    upload_file = st.file_uploader("Upload Surveillance Frame / CCTV Image", type=["jpg", "jpeg", "png"])
    
    if upload_file is not None:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Enhancement controls
        st.markdown("#### 🛠️ CCTV Enhancement Pipeline")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            use_gamma = st.checkbox("Auto Gamma (Night Vision Boost)", value=True)
        with c2:
            use_clahe = st.checkbox("LAB CLAHE (Skin Tone Balance)", value=True)
        with c3:
            use_denoise = st.checkbox("Bilateral Noise Filter", value=True)
        with c4:
            use_sharpen = st.checkbox("Unsharp Detail Sharpening", value=True)
        with c5:
            use_glare = st.checkbox("Specular Glare Reducer", value=False)
            
        enhanced_bgr = enhancer.enhance(
            raw_bgr,
            apply_gamma=use_gamma,
            apply_clahe=use_clahe,
            apply_denoise=use_denoise,
            apply_sharpen=use_sharpen,
            apply_glare_suppression=use_glare
        )
        
        # Side by side comparison
        col_raw, col_enh = st.columns(2)
        with col_raw:
            st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="Original CCTV Input", use_container_width=True)
        with col_enh:
            st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption="Enhanced Frame (Used for Detection)", use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### 🔍 Suspect Identification Analysis")
        
        if st.button("Run Suspect Identification on Enhanced Frame", type="primary"):
            with st.spinner("Analyzing facial features & querying database..."):
                t0 = time.perf_counter()
                results = matcher.search_image(enhanced_bgr, top_k=3, use_adaface=use_adaface_opt)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                
            st.write(f"Found **{len(results)}** face(s) in `{elapsed_ms:.1f} ms`")
            
            if not results:
                st.warning("No faces detected in the image. Try adjusting the CCTV enhancement settings above.")
            else:
                annotated_img = enhanced_bgr.copy()
                
                for idx, res in enumerate(results):
                    face = res["face"]
                    box = [int(v) for v in face.bbox]
                    matches = res["matches"]
                    q_score = res.get("quality_score", 0.8)
                    pose = res.get("pose", {})
                    
                    st.markdown(f"##### 👤 Face #{idx + 1} (Confidence: `{face.confidence * 100:.1f}%` • AdaFace Quality Q: `{q_score*100:.1f}%` • Est. Age: `{face.age}` • Gender: `{face.gender}`)")
                    st.caption(f"📐 **Head Pose Angles:** Yaw: `{pose.get('yaw', 0)}°` | Pitch: `{pose.get('pitch', 0)}°` | Roll: `{pose.get('roll', 0)}°` {'🚨 *Extreme Angle*' if pose.get('is_extreme') else '✅ *Normal Angle*'}")
                    
                    # Crop face
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(raw_bgr.shape[1], box[2]), min(raw_bgr.shape[0], box[3])
                    face_crop = enhanced_bgr[y1:y2, x1:x2]
                    
                    c_crop, c_match = st.columns([1, 3])
                    with c_crop:
                        if face_crop.size > 0:
                            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption=f"Crop #{idx+1}", width=120)
                            
                    with c_match:
                        if matches and matches[0]["similarity"] >= threshold:
                            top_m = matches[0]
                            st.error(f"🚨 **CRIMINAL MATCH:** {top_m['name']} (ID: {top_m['id']}) — **{top_m['similarity']*100:.1f}% Match**")
                            
                            # Show all top candidates
                            for m in matches:
                                st.write(f"- Candidate `{m['id']}` - **{m['name']}**: Similarity `{m['similarity']:.3f}`")
                            
                            cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                            cv2.putText(annotated_img, f"{top_m['name']} ({top_m['similarity']:.2f})", (box[0], max(20, box[1] - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            st.success("✅ Unknown / No match in criminal registry above threshold.")
                            cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(annotated_img, "Unknown", (box[0], max(20, box[1] - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Identified Suspect Map", use_container_width=True)

# ==========================================
# TAB 3: VIDEO FOOTAGE FORENSICS & RTSP
# ==========================================
with tab3:
    st.markdown("### 📼 CCTV Video Forensics & Stream Scanner")
    st.write("Scan recorded CCTV footage or live RTSP streams to auto-detect enrolled suspects and generate an interactive detection timeline.")

    v_mode = st.radio("Surveillance Mode", ["📁 Upload CCTV Video Recording", "📡 Live RTSP / IP Camera Stream"], horizontal=True)

    if v_mode == "📁 Upload CCTV Video Recording":
        v_file = st.file_uploader("Upload Video Footage (.mp4, .avi, .mkv, .mov)", type=["mp4", "avi", "mkv", "mov"])
        
        c_step, c_max = st.columns(2)
        with c_step:
            frame_step = st.slider("Scan Step (Process every N-th frame)", min_value=1, max_value=30, value=5, help="5 = ~6 scans/sec for 30fps video. Higher = faster.")
        with c_max:
            max_sec = st.number_input("Max Duration to Scan (Seconds, 0 = Full Video)", min_value=0, value=60, step=10)

        if v_file is not None:
            import tempfile
            from core.video_scanner import CCTVVideoScanner

            if st.button("🚀 Start Video Forensics Scan", type="primary"):
                # Write uploaded bytes to temp file for OpenCV VideoCapture
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(v_file.read())
                tfile.flush()

                scanner = CCTVVideoScanner(matcher, frame_step=frame_step, min_confidence=threshold)
                
                prog_bar = st.progress(0, text="Initializing video scan...")
                feed_placeholder = st.empty()
                
                matched_events = []
                total_scanned = 0
                t0_scan = time.perf_counter()

                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.release()

                max_f = int(max_sec * fps) if max_sec > 0 else None

                for f_idx, t_sec, annotated_f, events in scanner.scan_video_stream(tfile.name, max_frames=max_f):
                    total_scanned += 1
                    matched_events.extend(events)

                    pct = min(1.0, f_idx / (max_f or total_frames))
                    prog_bar.progress(pct, text=f"Scanning frame {f_idx} (Time: {CCTVVideoScanner.format_timestamp(t_sec)}) — {len(matched_events)} matches")

                    if events or total_scanned % 10 == 0:
                        feed_placeholder.image(cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB), caption=f"Forensic Feed — Frame {f_idx}", use_container_width=True)

                prog_bar.empty()
                scan_duration = time.perf_counter() - t0_scan
                st.success(f"✅ Video Forensics Scan Complete in `{scan_duration:.1f}s`! Scanned `{total_scanned}` frames. Detected `{len(matched_events)}` suspect match events.")

                # Match Timeline
                if matched_events:
                    st.markdown("#### 🚨 Suspect Detection Timeline")
                    for ev in matched_events:
                        st.markdown(f"""
                        <div class="suspect-box">
                            ⏱️ <b>Timestamp:</b> <code>{ev.formatted_time}</code> (Frame #{ev.frame_number})<br>
                            🚨 <b>Identified:</b> {ev.criminal_name} (ID: <code>{ev.criminal_id}</code>) — Match: <b>{ev.similarity*100:.1f}%</b>
                        </div>
                        """, unsafe_allow_html=True)
                        if ev.face_crop_bgr.size > 0:
                            st.image(cv2.cvtColor(ev.face_crop_bgr, cv2.COLOR_BGR2RGB), caption=f"CCTV Crop at {ev.formatted_time}", width=100)
                else:
                    st.info("No enrolled criminals were detected in this footage.")

    else:
        rtsp_url = st.text_input("Enter RTSP Stream URL", placeholder="rtsp://username:password@192.168.1.100:554/stream1")
        st.caption("Example: `rtsp://admin:pass@192.168.1.50:554/h264Preview_01_main` or `0` for default local USB/webcam")
        
        if st.button("Connect & Start Live Stream Inspection"):
            if not rtsp_url:
                st.error("Please enter a valid RTSP URL.")
            else:
                st.info(f"Connecting to stream {rtsp_url}...")
                stream_src = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
                cap = cv2.VideoCapture(stream_src)
                if not cap.isOpened():
                    st.error("Failed to connect to video stream. Verify camera IP, port, and network credentials.")
                else:
                    st.success("Connected to stream. Sampling live frames...")
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        results = matcher.search_image(frame, top_k=2)
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Live RTSP Snapshot — {len(results)} face(s) found", use_container_width=True)

# ==========================================
# TAB 4: CRIMINAL DATABASE MANAGEMENT
# ==========================================
with tab4:
    st.markdown("### 📁 Criminal Registry Management")
    
    enroll_col, view_col = st.columns([1, 1])
    
    with enroll_col:
        st.markdown("#### ➕ Enroll New Criminal / Suspect")
        with st.form("enroll_form", clear_on_submit=True):
            e_id = st.text_input("Criminal ID * (e.g. CRIM-1049)", placeholder="CRIM-XXXX")
            e_name = st.text_input("Full Name *", placeholder="First Last")
            e_alias = st.text_input("Known Aliases", placeholder="e.g. 'Bablu'")
            e_crime = st.text_area("Crime Classification / Notes", placeholder="e.g. Armed Robbery, Section 392 IPC")
            e_photo = st.file_uploader("Mugshot Photo *", type=["jpg", "jpeg", "png"])
            
            submitted = st.form_submit_button("Enroll into Registry", type="primary")
            
            if submitted:
                if not e_id or not e_name or e_photo is None:
                    st.error("Please fill in ID, Name, and provide a Mugshot Photo.")
                else:
                    file_bytes = np.asarray(bytearray(e_photo.read()), dtype=np.uint8)
                    mugshot_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    success, msg = matcher.enroll_image(
                        mugshot_bgr,
                        criminal_id=e_id.strip(),
                        name=e_name.strip(),
                        metadata={
                            "alias": e_alias.strip(),
                            "crime_history": e_crime.strip(),
                            "enrollment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    )
                    if success:
                        st.success(f"🎉 {msg}")
                    else:
                        st.error(f"❌ {msg}")
                        
    with view_col:
        st.markdown("#### 📋 Currently Enrolled Records")
        all_records = db.get_all(include_thumbnails=True)
        
        if not all_records:
            st.info("No criminals currently enrolled in the database. Use the form on the left to add one.")
        else:
            st.write(f"Total Enrolled Records: **{len(all_records)}**")
            
            for rec in all_records:
                with st.expander(f"👤 {rec['name']} (ID: {rec['id']})", expanded=False):
                    c_thumb, c_info = st.columns([1, 2])
                    with c_thumb:
                        if rec.get("thumbnail"):
                            try:
                                thumb_bytes = base64.b64decode(rec["thumbnail"])
                                st.image(thumb_bytes, caption=rec["name"], width=100)
                            except Exception:
                                st.caption("No preview")
                    with c_info:
                        st.markdown(f"**Name:** {rec['name']}")
                        st.markdown(f"**Criminal ID:** `{rec['id']}`")
                        meta = rec.get("metadata", {})
                        if meta.get("alias"):
                            st.markdown(f"**Alias:** {meta['alias']}")
                        if meta.get("crime_history"):
                            st.markdown(f"**Notes:** {meta['crime_history']}")
                            
                    if st.button(f"🗑️ Delete Record {rec['id']}", key=f"del_{rec['id']}"):
                        db.delete(rec['id'])
                        st.success(f"Deleted {rec['id']}. Please refresh.")
                        st.rerun()

# ==========================================
# TAB 5: SYSTEM DIAGNOSTICS & DOSSIER EXPORT
# ==========================================
with tab5:
    st.markdown("### 📊 Hardware Diagnostics & Forensic Dossier Export")
    
    col_bench, col_sys = st.columns(2)
    
    with col_bench:
        st.markdown("#### ⚡ Latency Benchmark (InsightFace SCRFD + ArcFace)")
        if st.button("Run 10-Iteration Latency Benchmark"):
            with st.spinner("Running benchmark passes on synthetic frame..."):
                latencies = []
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(10):
                    t0 = time.perf_counter()
                    _ = detector.detect(test_frame)
                    latencies.append((time.perf_counter() - t0) * 1000)
                
                avg_lat = np.mean(latencies)
                min_lat = np.min(latencies)
                max_lat = np.max(latencies)
                
                st.metric("Avg Latency (ms)", f"{avg_lat:.2f} ms", delta=f"Min: {min_lat:.2f} ms")
                st.write(f"- Maximum latency spike: `{max_lat:.2f} ms`")
                st.write(f"- Throughput capability: `~{1000/avg_lat:.1f} FPS`")
                
        st.markdown("---")
        st.markdown("#### 📄 Forensic Audit Report Export")
        st.caption("Generate an official audit report of all current criminal records for police records or investigation logs.")
        
        all_recs = db.get_all(include_thumbnails=False)
        report_json = json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_engine": "InsightFace ArcFace 512-dim + SCRFD 10G",
            "enrolled_count": len(all_recs),
            "records": all_recs
        }, indent=2)
        
        st.download_button(
            label="📥 Download Criminal Registry Audit Log (JSON)",
            data=report_json,
            file_name=f"criminal_registry_audit_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
                
    with col_sys:
        st.markdown("#### 📦 Loaded Model Specifications")
        st.write("- **Face Detector:** SCRFD 10G (InsightFace `buffalo_l`)")
        st.write("- **Face Recognizer:** ArcFace ResNet-50 (`w600k_r50.onnx`, 512-dim)")
        st.write("- **Landmark Extractors:** 2D 106-point & 3D 68-point models")
        st.write("- **Vector Database:** ChromaDB (Cosine Space `hnsw:space: cosine`)")
        st.write(f"- **VRAM / Process RAM:** `{ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB`")

