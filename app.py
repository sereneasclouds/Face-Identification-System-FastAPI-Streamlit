import streamlit as st
import requests
from PIL import Image
import io
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Face Identification System", page_icon="🧑‍💻", layout="wide")
st.title("🧑‍💻 Face Photo Identification Matching")
st.markdown("Upload an image to identify known faces using the FastAPI backend.")

# Sidebar — controls
with st.sidebar:
    st.header(" Settings")
    st.markdown("**API Status**")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ API Online — {health['known_faces']} faces loaded")
    except:
        st.error(" API Offline. Run: `uvicorn api:app --reload`")

    st.divider()

    if st.button(" Reload Known Faces"):
        try:
            res = requests.post(f"{API_URL}/reload-faces").json()
            st.success(res["message"])
            st.write(res["names"])
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.markdown("**Known Faces in DB**")
    try:
        faces = requests.get(f"{API_URL}/known-faces", timeout=3).json()
        if faces["names"]:
            for name in faces["names"]:
                st.markdown(f"- `{name}`")
        else:
            st.info("No reference faces loaded.\nAdd `.jpg` images to `known_faces/` folder and reload.")
    except:
        pass

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader(" Identification Results")

    if uploaded_file and st.button("🚀 Identify Face", type="primary"):
        with st.spinner("Analyzing..."):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

            try:
                response = requests.post(f"{API_URL}/identify", files=files, timeout=30)
                data = response.json()

                st.markdown(f"**Faces Detected:** {data.get('faces_detected', 0)}")
                st.divider()

                results = data.get("results", [])
                if not results:
                    st.warning(data.get("message", "No face found in image."))
                else:
                    for r in results:
                        with st.container():
                            st.markdown(f"#### Face #{r['face_index']}")
                            if r["match_found"]:
                                st.success(f"✅ Match Found: **{r['name']}**")
                                conf = r.get("confidence", 0)
                                st.metric("Confidence", f"{conf}%")
                                st.progress(int(conf))
                                st.caption(f"Distance score: {r.get('distance', 'N/A')}")
                            else:
                                st.error(f"❌ No Match — Identity Unknown")
                                if r.get("confidence"):
                                    st.caption(f"Closest match confidence: {r['confidence']}%")
                            st.divider()

                # Raw JSON expander
                with st.expander(" Raw API Response"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running.")
            except Exception as e:
                st.error(f"Error: {e}")
    elif not uploaded_file:
        st.info("Upload an image on the left to get started.")