import streamlit as st
import requests

st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🔍 Quora Duplicate Question Detector")
st.write("Enter two questions to check whether they have the same meaning.")

st.divider()

question1 = st.text_area(
    "Question 1",
    placeholder="Example: How can I learn Python?",
    height=120
)

question2 = st.text_area(
    "Question 2",
    placeholder="Example: What is the best way to learn Python?",
    height=120
)

if st.button("🔎 Check Duplicate", type="primary", use_container_width=True):

    if not question1.strip() or not question2.strip():
        st.warning("Please enter both questions.")
        st.stop()

    payload = {
        "question1": question1,
        "question2": question2
    }

    try:
        with st.spinner("Checking questions..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()

            st.divider()

            prediction = result.get("prediction")
            probability = result.get("probability", 0)

            if result.get("is_duplicate") == 1:
                st.success("✅ The questions are likely DUPLICATE.")
            else:
                st.info("❌ The questions are likely NOT DUPLICATE.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", prediction)

            with col2:
                st.metric("Probability", f"{probability:.2%}")

            st.progress(probability)

        elif response.status_code == 422:
            st.error("Invalid input. Please enter valid questions.")

        else:
            st.error(f"API error: {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. "
            "Please start the FastAPI server first."
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

with st.sidebar:
    st.header("About")
    st.write(
        "This application uses a trained neural network to "
        "classify whether two Quora questions are duplicates."
    )

    st.divider()
    st.caption("Backend: FastAPI")
    st.caption("Frontend: Streamlit")