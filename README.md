# Persian Poetry Generator

## Overview

This application generates Persian poetry in three classical styles (Robaee, Masnavi, and Ghazal) using a fine-tuned GPT-2 model. The model has been trained on a diverse collection of Persian poetry to capture the essence and structure of these traditional formats.

### Try It Live

**Try the live demo**: [try it!](https://huggingface.co/spaces/n9e6y/PPG)

## Features

- **Multiple Poetry Styles**: Generate poetry in Robaee (quatrains), Masnavi (rhyming couplets), or Ghazal (lyrical poems) styles
- **Customizable Generation**: Fine-tune parameters like temperature, length, and sampling methods
- **Multiple Output Options**: Generate up to 5 poems at once
- **Easy-to-Use Interface**: Simple Streamlit interface with both basic and advanced options
- **Model Source Options**: Load the model from Hugging Face Hub or a local path

## Poetry Styles

- **Robaee (رباعی)**: Four-line quatrains with an AABA rhyme scheme.
- **Masnavi (مثنوی)**: A series of rhyming couplets, with each line sharing the same meter.
- **Ghazal (غزل)**: A form with the same rhyme repeated throughout, often exploring themes of love and spirituality

## Installation

### Setup

1. Clone this repository:
   ```bash
   gh repo clone n9e6y/PPG
   cd PPG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Using the Application

1. **Select Poetry Style**:
   - Choose from Robaee, Masnavi, or Ghazal styles
   - Each style comes with pre-configured recommended settings

2. **Set Generation Parameters**:
   - Basic: Choose the number of poems to generate
   - Advanced: Customize length, temperature, and sampling parameters

3. **Generate**:
   - Click "Generate Poetry" to create your poems

## Advanced Parameters

Under "Advanced Options," you can customize:

- **Maximum Length**: Controls how long the poem will be
- **Temperature**: Higher values (>1.0) make output more random, lower values (<1.0) make it more deterministic
- **Top-k**: Limits token selection to the k most likely tokens
- **Top-p (Nucleus Sampling)**: Dynamically selects tokens from the smallest possible set whose cumulative probability exceeds p

## Model Details

The model is a fine-tuned version of [GPT-2](https://huggingface.co/HooshvareLab/gpt2-fa) trained specifically on [Persian poetry collections](https://github.com/ganjoor/desktop/releases/tag/v2.81). The model was trained to understand the patterns, rhythms, and structures of traditional Persian poetry forms.

- **Model Name**: persian-poetry-gpt2
- **Hosted on**: [Hugging Face Spaces](https://huggingface.co/spaces/n9e6y/PPG)
- **Model** : [Hugging Face Hub](https://huggingface.co/n9e6y/persian-poetry-gpt2)
- **Base Architecture**: GPT-2

## Local Development

If you want to develop or modify the application:

1. Fork this repository
2. Make your changes
3. Test locally with `streamlit run app.py`
4. Submit a pull request

## Customizing Style-Specific Settings

To modify the recommended settings for each poetry style, edit the `style_settings` dictionary in the `app.py` file:

```python
style_settings = {
    "ROBAEE": {
        "max_length": 80,    # Modify these values
        "temperature": 0.7,  # to suit your
        "top_k": 70,         # preferences
        "top_p": 0.9,
    },
    # ...other styles...
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

- This project utilizes the [Transformers](https://github.com/huggingface/transformers) library from Hugging Face
- The poetry generation model was trained on [collections of classical Persian poetry](https://github.com/ganjoor/desktop/releases/tag/v2.81)
- Thanks to the [Streamlit](https://streamlit.io/) team for their excellent framework

## Contact

For questions, issues, or feedback, please [open an issue](https://github.com/n9e6y/PPG/issues) on this repository.