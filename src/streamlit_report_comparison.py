import streamlit as st
import json
from collections import defaultdict
from pathlib import Path
import difflib
import io

st.set_page_config(page_title="Compare JSONL Reports", layout="wide")

st.title("Compare reports side-by-side (JSONL)")

st.markdown(
    """
Upload one or more JSONL files (newline-delimited JSON). Each line should contain keys
`befund_schluessel` and `befund_text_plain`.

The app groups lines by `befund_schluessel` and displays two different `befund_text_plain`
values for the same key side-by-side.
"""
)

# ------ Inputs: uploader or local directory
input_mode = st.radio("Load JSONL from:", ("Upload files", "Local folder"))

uploaded_files = []
if input_mode == "Upload files":
    uploaded_files = st.file_uploader("Upload JSONL files", type=["jsonl", "txt", "json"], accept_multiple_files=True)
else:
    folder = st.text_input("Local folder path (server-side)", value="./")
    if st.button("Load from folder"):
        p = Path(folder)
        if not p.exists():
            st.error("Folder does not exist on server. Make sure you run Streamlit server where the folder is reachable.")
        else:
            for f in p.glob("**/*.jsonl"):
                uploaded_files.append(open(f, "rb"))

# ------ Helper to parse jsonl from a file-like

def parse_jsonl_file(file_like):
    """Yield dicts from a file-like object (bytes or text)."""
    # make sure we can iterate lines as text
    if hasattr(file_like, "read"):
        raw = file_like.read()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                raw = raw.decode("latin-1")
    else:
        raw = file_like
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception as e:
            # skip malformed lines but warn later
            continue

# ------ Load and group

if uploaded_files:
    with st.spinner("Parsing files..."):
        groups = defaultdict(list)  # befund_schluessel -> list of befund_text_plain
        malformed_count = 0
        for f in uploaded_files:
            try:
                for obj in parse_jsonl_file(f):
                    if not isinstance(obj, dict):
                        malformed_count += 1
                        continue
                    key = obj.get("befund_schluessel")
                    txt = obj.get("befund_text_plain")
                    if key is None:
                        malformed_count += 1
                        continue
                    # normalize to str
                    if txt is None:
                        txt = ""
                    groups[str(key)].append(str(txt))
            except Exception as e:
                st.warning(f"Failed to read a file: {e}")

        st.success(f"Loaded data for {len(groups)} unique befund_schluessel")
        if malformed_count:
            st.info(f"Skipped {malformed_count} malformed lines")

    # create a simple index
    index = []
    for key, texts in groups.items():
        unique_texts = []
        for t in texts:
            if t not in unique_texts:
                unique_texts.append(t)
        index.append({"key": key, "count": len(texts), "unique_count": len(unique_texts)})

    import pandas as pd
    df = pd.DataFrame(index)

    st.sidebar.header("Filter / navigation")
    search = st.sidebar.text_input("Search key or preview text")
    min_unique = st.sidebar.number_input("Min unique texts", min_value=1, value=2)
    show_only_multiple = st.sidebar.checkbox("Show only keys with ≥ 2 different texts", value=True)

    # apply filters
    shown = df
    if show_only_multiple:
        shown = shown[shown["unique_count"] >= min_unique]
    if search:
        mask = shown["key"].str.contains(search, case=False, na=False)
        # also search in preview text
        if not mask.any():
            # fallback: search through groups (could be slower)
            keys = [k for k, vals in groups.items() if any(search.lower() in (v or "").lower() for v in vals)]
            mask = shown["key"].isin(keys)
        shown = shown[mask]

    st.sidebar.markdown(f"**Matching keys:** {len(shown)}")

    # show table peek
    st.subheader("Index of keys")
    st.dataframe(shown.sort_values(["unique_count", "count"], ascending=False).reset_index(drop=True))

    # selection
    st.markdown("---")
    st.subheader("Inspect a key side-by-side")
    key_choice = st.selectbox("Select befund_schluessel", options=sorted(shown["key"].tolist()))

    # pick two different texts for this key
    selected_texts = groups.get(key_choice, [])
    # produce unique in order
    unique_texts = []
    for t in selected_texts:
        if t not in unique_texts:
            unique_texts.append(t)

    if not unique_texts:
        st.warning("No texts found for this key")
    else:
        left = unique_texts[0]
        right = unique_texts[1] if len(unique_texts) > 1 else ""

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Left:** {key_choice} — version 1")
            st.text_area("left_text", value=left, height=400)
        with col2:
            st.markdown(f"**Right:** {key_choice} — version 2")
            st.text_area("right_text", value=right, height=400)

        # show a simple inline diff
        if right:
            st.subheader("Inline diff (context)")
            diff = difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm="")
            diff_text = "\n".join(diff)
            st.code(diff_text)

        # allow quick export of the pair to a small JSON
        if st.button("Export this pair as JSON" ):
            out = {"befund_schluessel": key_choice, "text_a": left, "text_b": right}
            st.download_button("Download JSON", data=json.dumps(out, ensure_ascii=False, indent=2), file_name=f"pair_{key_choice}.json", mime="application/json")

else:
    st.info("Upload JSONL files or choose a local folder to begin.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ — shows the first two unique befund_text_plain values per befund_schluessel")
