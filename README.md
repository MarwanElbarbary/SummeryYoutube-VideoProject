# 📺 YouTube AI Summarizer

Turn long YouTube videos into clean, structured summaries and study notes using AI – directly from the URL, with no file uploads.

This app fetches the transcript of a YouTube video (if available), summarizes it using a powerful Transformer model, and generates:
- A readable AI summary
- Key bullet-point takeaways
- Open questions for self-testing
- Fill-in-the-blank questions for active recall
- A combined TXT export (summary + questions + full transcript)

Built with **Streamlit** and **Hugging Face Transformers**.

---

## 🚀 Features

- 🔗 **Paste any YouTube URL**  
  Just drop a link like `https://www.youtube.com/watch?v=...` and the app will:
  - Extract the video ID
  - Request the transcript via `youtube-transcript-api`
  - Clean and join the transcript into a single text block

- 🤖 **AI summarization (BART Large CNN)**  
  Uses the `facebook/bart-large-cnn` model from `transformers` to generate:
  - Short, normal, or detailed summaries
  - Automatic chunking for long transcripts
  - Progress bar while summarizing

- 🧠 **Study mode**  
  From the summary, the app builds:
  - Key concept bullet points
  - Open “explain in your own words” questions
  - Fill-in-the-blank questions (with hidden answers)

- 🗂 **Export to TXT**  
  Download a single `.txt` file containing:
  - Summary
  - Key takeaways
  - Open questions
  - Fill-in-the-blank questions (with answers)
  - Full transcript

- 🎨 **Custom UI / UX**  
  - Dark, glass-morphism inspired layout
  - Two-pane view: original transcript vs AI summary
  - Clean, modern pills, chips, and highlighted boxes
  - Responsive layout for different screen sizes

---

## 🧩 Tech Stack

- [Streamlit](https://streamlit.io/) – UI and app framework  
- [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) – Fetching YouTube transcripts  
- [Transformers](https://huggingface.co/docs/transformers/index) – Summarization pipeline (`facebook/bart-large-cnn`)  
- [PyTorch](https://pytorch.org/) – Backend for the Transformer model  
- Python standard libraries:
  - `urllib.parse` for URL parsing
  - `io.StringIO` for building the TXT export

---
