import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from transformers import pipeline
from io import StringIO

st.set_page_config(
    page_title="YouTube Summarizer",
    page_icon="📺",
    layout="wide"
)

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

try:
    with st.spinner("Loading AI Model (This might take a minute)..."):
        summarizer = load_summarizer()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def extract_video_id(url: str) -> str:
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    video_ids = qs.get('v')
    if not video_ids:
        raise ValueError("Invalid YouTube URL: Could not find video ID.")
    return video_ids[0]

def fetch_transcript(video_url: str) -> str:
    video_id = extract_video_id(video_url)
    try:
        api = YouTubeTranscriptApi()
        transcript_snippets = api.fetch(
            video_id=video_id,
            languages=['en', 'en-US', 'en-GB']
        )
        parts = []
        for snippet in transcript_snippets:
            if hasattr(snippet, "text"):
                parts.append(snippet.text)
            elif isinstance(snippet, dict) and "text" in snippet:
                parts.append(snippet["text"])
        text = " ".join(parts)
        if not text.strip():
            raise Exception("Transcript is empty or could not be parsed.")
        return text
    except Exception as e:
        raise Exception(f"Could not retrieve transcript. Reason: {str(e)}")

def safe_summarize(text, style="normal", chunk_size=2000):
    if len(text) < 200:
        return "The transcript is too short to summarize properly."
    if style == "short":
        max_len = 80
        min_len = 20
    elif style == "detailed":
        max_len = 220
        min_len = 70
    else:
        max_len = 150
        min_len = 40
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    progress_bar = st.progress(0)
    for idx, chunk in enumerate(chunks):
        try:
            progress = (idx + 1) / len(chunks)
            progress_bar.progress(progress)
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk {idx}: {e}")
            continue
    progress_bar.empty()
    return " ".join(summaries)

def extract_bullets_from_summary(summary_text: str, max_points: int = 8):
    parts = [p.strip(" •-") for p in summary_text.split(".") if p.strip()]
    bullets = []
    for p in parts:
        if len(bullets) >= max_points:
            break
        bullets.append(p)
    return bullets

def generate_open_questions(summary_text: str, max_q: int = 6):
    sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
    questions = []
    for i, s in enumerate(sentences[:max_q]):
        questions.append(f"Q{i+1}: Explain in your own words: \"{s}\"")
    return questions

def generate_fill_in_blank(summary_text: str, max_q: int = 6):
    sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
    blanks = []
    q_idx = 1
    for s in sentences:
        if q_idx > max_q:
            break
        words = s.split()
        if len(words) < 5:
            continue
        idx = len(words) // 2
        removed = words[idx].strip(",.")
        if len(removed) <= 3:
            continue
        words[idx] = "____"
        question = " ".join(words)
        blanks.append((f"Q{q_idx}: {question}", removed))
        q_idx += 1
    return blanks

st.markdown(
    """
    <style>
        .main {
            background: radial-gradient(circle at top left, #020617 0, #020617 35%, #020617 55%, #020617 70%, #020617 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2.6rem;
            padding-bottom: 2.6rem;
            max-width: 1200px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #020617 40%, #020617 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.25);
        }
        .app-shell {
            position: relative;
            border-radius: 26px;
            padding: 1.4rem 1.6rem;
            background: radial-gradient(circle at top left, rgba(236,72,153,0.16), transparent 45%),
                        radial-gradient(circle at top right, rgba(56,189,248,0.12), transparent 55%),
                        linear-gradient(135deg, rgba(15,23,42,0.97), rgba(15,23,42,0.96));
            border: 1px solid rgba(148, 163, 184, 0.40);
            box-shadow:
                0 24px 70px rgba(0,0,0,0.85),
                0 0 0 1px rgba(148,163,184,0.18);
            overflow: hidden;
        }
        .app-shell:before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: radial-gradient(circle at 10% -10%, rgba(236,72,153,0.12), transparent 55%),
                        radial-gradient(circle at 110% 0%, rgba(129,140,248,0.12), transparent 55%);
            opacity: 0.9;
        }
        .app-shell-inner {
            position: relative;
            z-index: 2;
        }
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            margin-bottom: 1.4rem;
        }
        .app-title-block {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .app-logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 0%, #f97316, transparent 55%),
                        radial-gradient(circle at 120% 0%, #ec4899, transparent 55%),
                        #020617;
            box-shadow: 0 0 0 1px rgba(248,250,252,0.1), 0 18px 35px rgba(0,0,0,0.6);
            font-size: 1.4rem;
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }
        .app-subtitle {
            font-size: 0.9rem;
            opacity: 0.85;
        }
        .app-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.35rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            background: rgba(15,23,42,0.85);
            border: 1px solid rgba(148,163,184,0.6);
        }
        .pill-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34,197,94,0.2);
        }
        .pill-soft {
            background: rgba(30,64,175,0.4);
            border-color: rgba(129,140,248,0.8);
        }
        .pill-soft2 {
            background: rgba(180,83,9,0.4);
            border-color: rgba(248,250,252,0.15);
        }
        .pill-soft3 {
            background: rgba(14,116,144,0.4);
            border-color: rgba(45,212,191,0.7);
        }
        .app-header-right {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.25rem;
            font-size: 0.8rem;
        }
        .metric-row {
            display: flex;
            gap: 0.5rem;
        }
        .metric-chip {
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.5);
            display: inline-flex;
            gap: 0.3rem;
            align-items: center;
            font-size: 0.75rem;
        }
        .metric-chip-label {
            opacity: 0.7;
        }
        .metric-chip-value {
            font-weight: 600;
        }
        .glass-card {
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.93));
            border-radius: 18px;
            padding: 1.5rem 1.4rem 1.3rem 1.4rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.75);
            position: relative;
            overflow: hidden;
        }
        .glass-card:before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: radial-gradient(circle at top left, rgba(148,163,184,0.18), transparent 55%);
            opacity: 0.6;
        }
        .glass-card-inner {
            position: relative;
            z-index: 2;
        }
        .section-title {
            font-size: 0.98rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .section-caption {
            font-size: 0.8rem;
            opacity: 0.75;
            margin-bottom: 0.8rem;
        }
        .stTextInput>div>div>input {
            background-color: #020617 !important;
            border-radius: 9999px !important;
            border: 1px solid #4b5563 !important;
            padding: 0.6rem 1.1rem !important;
            font-size: 0.9rem !important;
            color: #e5e7eb !important;
        }
        .stTextInput>div>div>input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 1px #6366f1 !important;
        }
        .stTextArea textarea {
            background-color: #020617 !important;
            border-radius: 14px !important;
            border: 1px solid #4b5563 !important;
            font-size: 0.84rem !important;
            color: #e5e7eb !important;
        }
        .btn-primary button {
            width: 100%;
            border-radius: 9999px;
            border: none;
            padding: 0.7rem 1.3rem;
            font-weight: 600;
            font-size: 0.93rem;
            background: linear-gradient(90deg, #f97316, #ec4899, #8b5cf6);
            box-shadow: 0 12px 30px rgba(0,0,0,0.75);
            transition: transform 120ms ease-out, box-shadow 120ms ease-out, filter 120ms ease-out;
        }
        .btn-primary button:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.9);
        }
        .btn-primary button:active {
            transform: translateY(0px) scale(0.99);
            box-shadow: 0 10px 24px rgba(0,0,0,0.8);
        }
        .tags-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.4rem;
        }
        .tag {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.26rem 0.65rem;
            border-radius: 999px;
            font-size: 0.74rem;
            background: rgba(31, 41, 55, 0.85);
            border: 1px solid rgba(75, 85, 99, 0.9);
        }
        .tag-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: #22c55e;
        }
        .video-thumb {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.35);
            margin-bottom: 0.65rem;
        }
        .video-thumb img {
            display: block;
        }
        .video-meta {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            font-size: 0.8rem;
        }
        .video-meta-label {
            opacity: 0.7;
        }
        .video-meta-value {
            font-family: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.78rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.4);
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }
        .two-pane {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 0.8rem;
            margin-top: 0.4rem;
        }
        .pane-title {
            font-size: 0.86rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        .helper-text {
            font-size: 0.76rem;
            opacity: 0.7;
            margin-top: 0.2rem;
        }
        .hint-banner {
            margin-top: 0.7rem;
            font-size: 0.78rem;
            padding: 0.55rem 0.7rem;
            border-radius: 12px;
            background: rgba(15,23,42,0.95);
            border: 1px dashed rgba(148,163,184,0.7);
        }
        .hint-accent {
            color: #f97316;
            font-weight: 600;
        }
        .footer-note {
            margin-top: 1.1rem;
            font-size: 0.76rem;
            opacity: 0.7;
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
        }
        .footer-note span {
            display: inline-flex;
            gap: 0.25rem;
            align-items: center;
        }
        .footer-dot {
            width: 5px;
            height: 5px;
            border-radius: 999px;
            background: #22c55e;
        }
        .takeaways-box {
            margin-top: 0.75rem;
            padding: 0.75rem 0.8rem;
            border-radius: 14px;
            background: rgba(15,23,42,0.97);
            border: 1px solid rgba(148,163,184,0.65);
            font-size: 0.8rem;
        }
        .takeaways-title {
            font-weight: 600;
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
        }
        .takeaways-list {
            margin: 0;
            padding-left: 1.1rem;
        }
        .takeaways-list li {
            margin-bottom: 0.15rem;
        }
        .advanced-panel {
            margin-top: 0.85rem;
            font-size: 0.78rem;
            padding: 0.55rem 0.7rem;
            border-radius: 12px;
            background: rgba(15,23,42,0.96);
            border: 1px solid rgba(75,85,99,0.9);
        }
        .advanced-label {
            font-weight: 600;
            font-size: 0.8rem;
            margin-bottom: 0.3rem;
        }
        .study-box {
            margin-top: 0.75rem;
            padding: 0.75rem 0.9rem;
            border-radius: 14px;
            background: rgba(15,23,42,0.97);
            border: 1px solid rgba(55,65,81,0.9);
            font-size: 0.8rem;
        }
        .study-title {
            font-weight: 600;
            margin-bottom: 0.35rem;
            font-size: 0.84rem;
        }
        .study-list {
            margin: 0;
            padding-left: 1.1rem;
        }
        .study-list li {
            margin-bottom: 0.18rem;
        }
        .answer-details {
            font-size: 0.78rem;
            opacity: 0.9;
        }
        details summary {
            cursor: pointer;
        }
        @media (max-width: 900px) {
            .app-header {
                flex-direction: column;
                align-items: flex-start;
            }
            .app-header-right {
                align-items: flex-start;
            }
        }
        @media (max-width: 768px) {
            .two-pane {
                grid-template-columns: minmax(0, 1fr);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-shell">
        <div class="app-shell-inner">
            <div class="app-header">
                <div style="display:flex; gap:0.9rem; align-items:center;">
                    <div class="app-logo">📺</div>
                    <div class="app-title-block">
                        <div class="app-title">YouTube AI Summarizer</div>
                        <div class="app-subtitle">
                            Turn long YouTube videos into short, structured summaries in seconds.
                        </div>
                        <div class="app-badges">
                            <div class="pill">
                                <span class="pill-dot"></span>
                                Live transcript + AI summary
                            </div>
                            <div class="pill pill-soft">
                                BART large CNN
                            </div>
                            <div class="pill pill-soft2">
                                English content
                            </div>
                            <div class="pill pill-soft3">
                                No upload needed
                            </div>
                        </div>
                    </div>
                </div>
                <div class="app-header-right">
                    <div class="metric-row">
                        <div class="metric-chip">
                            <span class="metric-chip-label">Mode</span>
                            <span class="metric-chip-value">Summarize + Study</span>
                        </div>
                        <div class="metric-chip">
                            <span class="metric-chip-label">Length</span>
                            <span class="metric-chip-value">Adaptive</span>
                        </div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-chip">
                            <span class="metric-chip-label">Status</span>
                            <span class="metric-chip-value">Model ready ✅</span>
                        </div>
                    </div>
                </div>
            </div>
    """,
    unsafe_allow_html=True,
)

top_col1, top_col2 = st.columns([3, 2], gap="large")

with top_col2:
    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-inner">
                <div class="section-title">1. Paste a YouTube link</div>
                <div class="section-caption">
                    We will fetch the transcript directly from YouTube (if available) and generate a summary.
                </div>
        """,
        unsafe_allow_html=True,
    )

    video_url = st.text_input(
        " ",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="advanced-panel">
            <div class="advanced-label">Summary style</div>
        """,
        unsafe_allow_html=True,
    )

    summary_style = st.radio(
        "",
        options=["Short", "Normal", "Detailed"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    run_button = st.button("Fetch transcript & generate summary", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
                <div class="helper-text">
                    Works best with public videos that have English transcripts enabled.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_col1:
    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-inner">
                <div class="section-title">What does this app do?</div>
                <div class="section-caption">
                    Paste any YouTube URL and get:
                </div>
                <div class="tags-row">
                    <span class="tag"><span class="tag-dot"></span>Full transcript text</span>
                    <span class="tag">Clean AI-generated summary</span>
                    <span class="tag">Key bullet-point takeaways</span>
                    <span class="tag">Study mode questions</span>
                </div>
                <div class="hint-banner">
                    <span class="hint-accent">Tip</span> · Use <b>Study mode</b> for lectures and tutorials to test yourself with questions generated from the video.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

download_buffer = None
summary_text_global = ""
full_text_global = ""

if run_button:
    if not video_url.strip():
        st.warning("⚠️ Please enter a valid YouTube URL first.")
    else:
        video_id = None
        try:
            video_id = extract_video_id(video_url)
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)
        content_col_left, content_col_right = st.columns([1.6, 2.4], gap="large")

        with content_col_left:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-inner">
                        <div class="section-title">Video details</div>
                        <div class="section-caption">Basic information based on the URL you provided.</div>
                """,
                unsafe_allow_html=True,
            )
            if video_id:
                thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                st.markdown(
                    f"""
                    <div class="video-thumb">
                        <img src="{thumb_url}" width="100%" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="video-meta">
                        <div class="video-meta-label">Video ID</div>
                        <div class="video-meta-value">
                            <span>{video_id}</span>
                        </div>
                        <div class="helper-text">
                            Thumbnail and transcript are loaded directly from YouTube.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Could not parse video ID from URL.")
            st.markdown(
                """
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with content_col_right:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-inner">
                """,
                unsafe_allow_html=True,
            )
            try:
                with st.spinner("🎧 Fetching transcript from YouTube..."):
                    full_text = fetch_transcript(video_url)
                    full_text_global = full_text

                st.markdown(
                    """
                    <div class="section-title">Transcript, summary & study mode</div>
                    <div class="section-caption">
                        Switch between a clean AI summary and an interactive study mode with questions.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                tab_summary, tab_study = st.tabs(["Summary", "Study mode"])

                mode_internal = "normal"
                if summary_style == "Short":
                    mode_internal = "short"
                elif summary_style == "Detailed":
                    mode_internal = "detailed"

                with st.spinner("🤖 Summarizing, please wait..."):
                    summary_text = safe_summarize(full_text, style=mode_internal)
                    summary_text_global = summary_text

                bullets = extract_bullets_from_summary(summary_text)
                open_questions = generate_open_questions(summary_text)
                blanks = generate_fill_in_blank(summary_text)

                with tab_summary:
                    st.markdown(
                        """
                        <div class="two-pane">
                        """,
                        unsafe_allow_html=True,
                    )

                    col1, col2 = st.columns(2, gap="medium")

                    with col1:
                        st.markdown('<div class="pane-title">Original transcript</div>', unsafe_allow_html=True)
                        st.text_area(
                            "",
                            full_text,
                            height=320,
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            '<div class="helper-text">Raw text fetched from the YouTube transcript API.</div>',
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown('<div class="pane-title">AI summary</div>', unsafe_allow_html=True)
                        st.text_area(
                            "",
                            summary_text,
                            height=320,
                            label_visibility="collapsed",
                        )
                        if bullets:
                            st.markdown(
                                '<div class="takeaways-box"><div class="takeaways-title">Key takeaways</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown("<ul class='takeaways-list'>", unsafe_allow_html=True)
                            for b in bullets:
                                st.markdown(f"<li>{b}</li>", unsafe_allow_html=True)
                            st.markdown("</ul></div>", unsafe_allow_html=True)
                        st.markdown(
                            '<div class="helper-text">Short, dense summary generated by the BART model, plus distilled key points.</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        """
                        </div>
                        <div class="footer-note">
                            <span><span class="footer-dot"></span> Your text is processed locally in this session.</span>
                            <span>You can download the summary below for your notes or knowledge base.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with tab_study:
                    st.markdown(
                        """
                        <div class="study-box">
                            <div class="study-title">How to use study mode</div>
                            <div class="helper-text">
                                Read the key concepts, then answer the open questions and fill in the blanks without looking at the transcript. 
                                Check the answers after you think about them.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("### Key concepts")
                    if bullets:
                        st.markdown("<ul class='study-list'>", unsafe_allow_html=True)
                        for b in bullets:
                            st.markdown(f"<li>{b}</li>", unsafe_allow_html=True)
                        st.markdown("</ul>", unsafe_allow_html=True)
                    else:
                        st.write("No key concepts could be extracted from this summary.")

                    st.markdown("### Open questions")
                    if open_questions:
                        for q in open_questions:
                            st.markdown(f"- {q}")
                    else:
                        st.write("No questions available for this summary.")

                    st.markdown("### Fill in the blanks")
                    if blanks:
                        for q, ans in blanks:
                            st.markdown(
                                f"""
                                <div class="study-box">
                                    <div class="study-title">{q}</div>
                                    <div class="answer-details">
                                        <details>
                                            <summary>Show answer</summary>
                                            <span>{ans}</span>
                                        </details>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.write("Could not generate fill-in-the-blank questions from this summary.")

                combined_export = StringIO()
                combined_export.write("YouTube AI Summary + Study Notes\n")
                combined_export.write("================================\n\n")
                combined_export.write("Summary:\n\n")
                combined_export.write(summary_text_global + "\n\n")
                if bullets:
                    combined_export.write("Key takeaways:\n\n")
                    for i, b in enumerate(bullets, start=1):
                        combined_export.write(f"{i}. {b}\n")
                    combined_export.write("\n")
                if open_questions:
                    combined_export.write("Open questions:\n\n")
                    for q in open_questions:
                        combined_export.write(f"- {q}\n")
                    combined_export.write("\n")
                if blanks:
                    combined_export.write("Fill in the blanks:\n\n")
                    for i, (q, ans) in enumerate(blanks, start=1):
                        combined_export.write(f"{i}. {q}\n")
                        combined_export.write(f"   Answer: {ans}\n")
                    combined_export.write("\n")
                combined_export.write("Full transcript:\n\n")
                combined_export.write(full_text_global)
                download_buffer = combined_export.getvalue()

                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Download summary + study notes as TXT",
                    data=download_buffer,
                    file_name="youtube_summary_study_mode.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Error: {e}")

            st.markdown(
                """
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    placeholder_col1, placeholder_col2 = st.columns([2, 3], gap="large")
    with placeholder_col2:
        st.info(
            "Paste a YouTube link above and click **Fetch transcript & generate summary** "
            "to see the transcript, AI summary, and study mode questions here."
        )

st.markdown(
    """
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
