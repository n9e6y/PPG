import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import html

# page configuration
st.set_page_config(
    page_title="Persian Poetry Generator",
    page_icon="✍️",
    layout="wide"
)

# Initialize session state for model and tokenizer if not already present
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'generated_poem' not in st.session_state:
    st.session_state.generated_poem = ""
if 'raw_poem_for_copy' not in st.session_state:
    st.session_state.raw_poem_for_copy = ""

# App title and description
st.title("Persian Poetry Generator")
st.markdown("Generate Persian poetry in different classical styles using a fine-tuned GPT-2 model.")

# Ensure the model files are in this directory relative to app.py
MODEL_PATH = "./gpt2-farsi-poetry"


@st.cache_resource
def load_model_from_local():
    try:
        st.info(f"Loading model from {MODEL_PATH}...")
        # Check if model path exists , ensure the path is correct
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.eval()
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        st.session_state.device = device
        return model, tokenizer, True
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
        st.error("Please ensure the model files are present in the specified directory.")
        return None, None, False


# Auto-load model
if not st.session_state.model_loaded:
    with st.spinner("Loading model... This may take a moment."):
        model, tokenizer, success = load_model_from_local()
        if success:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model. Please check the path and model files.")



def generate_poem(
        model,
        tokenizer,
        device,
        prompt="<START>",
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.95
):
    """
    Generate Persian poems from a fine-tuned GPT-2 model with improved cleaning.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device) # Encode input

    # Generate text
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<END>") if "<END>" in tokenizer.get_vocab() else tokenizer.eos_token_id
        )

    text = tokenizer.decode(output_sequences[0], skip_special_tokens=False) # Decode the text


    # --- Cleaning Steps ---
    text = text.replace("<START>", "")
    text = text.replace("<END>", "")
    text = text.replace("<PAD>", "")
    text = text.replace("<|endoftext|>", "")

    text = text.replace("<LINE_BREAK>", "\n")

    text = re.sub(r'<STYLE:[A-Z]+>', '', text)

    text = re.sub(r'<[^>]*>', '', text)

    text = re.sub(r'line[a-zA-Z]*[<>]?', '', text)

    persian_pattern = re.compile(r'[^\u0600-\u06FF\s\d.,!?؛،؟«»:؛()[\]{}،\-—+\n]')
    text = persian_pattern.sub('', text)

    # Normalize whitespace: replace multiple spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = text.strip()

    raw_text_for_copy = text
    safe_text_for_display = html.escape(text)

    return safe_text_for_display, raw_text_for_copy



def format_hemistich(poem_text):
    """ Function to format poem in Hemistich style """

    lines = poem_text.strip().split('\n')
    formatted_lines = []

    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            line1 = lines[i].strip()
            line2 = lines[i+1].strip()
            if line1 or line2:
                formatted_lines.append(f"{line1}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{line2}")
        elif lines[i].strip():
            formatted_lines.append(lines[i].strip())

    return '<br>'.join(formatted_lines)


st.header("Generate Poetry")

style_descriptions = {
    "ROBAEE": "رباعی (Ruba'i)",
    "MASNAVI": "مثنوی (Mathnawi)",
    "GHAZAL": "غزل (Ghazal)"
}
style_options = list(style_descriptions.keys())
style_display = [f"{k}: {v}" for k, v in style_descriptions.items()]
style_index = st.selectbox(
    "Poetry Style:",
    range(len(style_options)),
    format_func=lambda i: style_display[i]
)
style = style_options[style_index]

# Recommended settings
style_settings = {
    "ROBAEE": {"max_length": 80, "temperature": 0.7, "top_k": 70, "top_p": 0.9},
    "MASNAVI": {"max_length": 160, "temperature": 0.8, "top_k": 60, "top_p": 0.92},
    "GHAZAL": {"max_length": 120, "temperature": 0.75, "top_k": 65, "top_p": 0.95}
}
recommended_settings = style_settings[style]

# Advanced options
with st.expander("Advanced Options", expanded=False):
    st.info(f"Recommended settings for {style} are pre-selected.")
    max_length = st.slider("Maximum Length:", min_value=30, max_value=200, value=recommended_settings["max_length"])
    temperature = st.slider("Temperature:", min_value=0.1, max_value=1.5, value=recommended_settings["temperature"], step=0.05)
    top_k = st.slider("Top-k:", min_value=10, max_value=100, value=recommended_settings["top_k"])
    top_p = st.slider("Top-p:", min_value=0.1, max_value=1.0, value=recommended_settings["top_p"], step=0.05)

# Generate button
if st.button("Generate Poetry"):
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please wait or check the console for errors.")
    else:
        try:
            with st.spinner("Generating poetry..."):
                prompt = f"<START><STYLE:{style}>"
                # Generate poem using the updated function
                safe_poem_display, raw_poem_copy = generate_poem(
                    model=st.session_state.model,
                    tokenizer=st.session_state.tokenizer,
                    device=st.session_state.device,
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # Format the *raw* poem for hemistich style, result is HTML-safe
                formatted_poem_html = format_hemistich(raw_poem_copy)
                st.session_state.generated_poem = formatted_poem_html
                st.session_state.raw_poem_for_copy = raw_poem_copy

        except Exception as e:
            st.error(f"Error generating poetry: {str(e)}")
            st.exception(e)

# Display the poem if available
if st.session_state.generated_poem:
    st.markdown("""
    <style>
    .rtl-poem-container {
        direction: rtl;
        text-align: right; /* Use right alignment for typical poetry */
        font-family: 'Tahoma', 'Arial', sans-serif; /* Keep fonts */
        font-size: 18px;
        line-height: 2.2; /* Adjust line height */
        background-color: #333333; /* Keep background color */
        color: #ffffff; /* Keep text color */
        padding: 25px; /* Adjust padding */
        border-radius: 8px;
        border: 1px solid #444444;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        white-space: normal; /* Allow wrapping */
    }
    .rtl-poem-container br { /* Ensure <br> works as expected */
        display: block;
        margin-bottom: 0.5em; /* Add space between lines */
        content: "";
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader(f"Generated {style} Poem:")
    # Display the HTML-formatted poem using markdown
    st.markdown(f'<div class="rtl-poem-container">{st.session_state.generated_poem}</div>', unsafe_allow_html=True)

    st.text_area("Copy raw poem text:", st.session_state.raw_poem_for_copy, height=200)


# Sidebar content
st.sidebar.header("About")
st.sidebar.info("""
This app generates Persian poetry using a fine-tuned GPT-2 model.
The model was trained on classic Persian poetry in various styles.

## Poetry Styles:
- **Robaee (رباعی)**: Ruba'i
- **Masnavi (مثنوی)**: Mathnawi
- **Ghazal (غزل)**: Ghazal

Experiment with different styles and settings!
""")
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Model**: Local version based on this [repository](https://github.com/n9e6y/PPG)")
st.sidebar.markdown("Fine-tuned GPT-2 for Persian poetry.")
st.sidebar.header("Technical Info")
device_info = st.session_state.get('device', 'CPU (default)')
st.sidebar.markdown(f"**Running on**: {device_info}")
if str(device_info).startswith("cuda"):
    st.sidebar.markdown(f"**GPU**: {torch.cuda.get_device_name(0)}")
