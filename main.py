
import streamlit as st
from better_profanity import Profanity
from diffusers import EulerAncestralDiscreteScheduler as EAD
from diffusers import StableDiffusionPipeline as sdp
from diffusers import PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
st.set_page_config(
    page_title="Imagify",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_theme = """
        [theme]
        primaryColor="#19454c"
        backgroundColor="#316371"
        secondaryBackgroundColor="#131743"
        textColor="#ffffff"
        font="serif"
    """

    # Apply the custom theme
st.write(f"<style>{custom_theme}</style>", unsafe_allow_html=True)
def has_profanity(text):
    return Profanity().contains_profanity(text)

def filter_text(text):
    while has_profanity(text):
        text = st.text_input("Please provide an alternative prompt:")
    return text

def images(prompt, selected_scheduler):
    model = "dreamlike-art/dreamlike-photoreal-2.0"
    st.write("// image gen model begin.. model = ", model)

    # Use the selected scheduler
    if selected_scheduler == "PNDMScheduler":
        scheduler = PNDMScheduler.from_pretrained(model, subfolder="scheduler")
    elif selected_scheduler == "DDIMScheduler":
        scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
    elif selected_scheduler == "LMSDiscreteScheduler":
        scheduler = LMSDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    elif selected_scheduler == "EulerDiscreteScheduler":
        scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    elif selected_scheduler == "DPMSolverMultistepScheduler":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler")
    else:
        # Default to PNDM scheduler if no valid option is selected
        scheduler = PNDMScheduler.from_pretrained(model, subfolder="scheduler")

    pipe = sdp.from_pretrained(
        model,
        scheduler=scheduler
    )

    device = "cuda"

    pipe = pipe.to(device)
    prompt = prompt
    st.write("// generating images for:", prompt)
    num_images = 1
    filtered_input = filter_text(prompt)
    images = pipe(
        filtered_input,
        height=512,
        width=512,
        num_inference_steps=70,
        guidance_scale=10,
        num_images_per_prompt=num_images
    ).images
    st.image(images)

def main():
    
    image_path = "/imagify.jpg"
    st.image(image_path, use_column_width=True)
    st.title("IMAGIFY")
    # Add text after heading
    st.header("Pixelating thoughts with AI")
    # Get user input for the prompt
    prompt = st.text_input("Enter your prompt here:")

    # Add a dropdown for selecting the scheduler
    selected_scheduler = st.selectbox("Select Scheduler", ["PNDMScheduler", "DDIMScheduler", "LMSDiscreteScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler"])

    # Filter text to remove profanity
    filtered_input = filter_text(prompt)

    # Display the filtered input
    # st.write("Filtered Input:", filtered_input)

    if st.button("Generate images") and filtered_input:
        images(filtered_input, selected_scheduler)

if __name__ == "__main__":
    main()
