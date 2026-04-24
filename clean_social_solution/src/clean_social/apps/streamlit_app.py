from __future__ import annotations

import httpx
import streamlit as st


def api_is_healthy(api_url: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{api_url}/health")
            response.raise_for_status()
        return True, "API is online"
    except Exception as exc:
        return False, f"API is offline: {exc}"


def render_app() -> None:
    st.set_page_config(page_title="CleanSocial Sentiment Demo", page_icon="💬", layout="centered")
    st.title("CleanSocial Sentiment Analysis")
    st.caption("Deployment demo: Streamlit client calling FastAPI /predict")

    api_url = st.text_input("API base URL", value="http://127.0.0.1:8000")
    healthy, health_message = api_is_healthy(api_url)
    if healthy:
        st.success(health_message)
    else:
        st.warning(health_message)

    text = st.text_area("Enter text", height=140, placeholder="I love this product")

    if st.button("Predict sentiment", disabled=not healthy):
        if not text.strip():
            st.error("Please enter text before prediction.")
        else:
            try:
                with httpx.Client(timeout=20.0) as client:
                    response = client.post(f"{api_url}/predict", json={"text": text})
                    response.raise_for_status()
                    payload = response.json()

                st.success(f"Sentiment: {payload['sentiment']}")
                st.metric("Confidence", f"{payload['confidence']:.2%}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    render_app()
