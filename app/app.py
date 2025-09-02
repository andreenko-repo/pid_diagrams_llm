# app.py

import gradio as gr
from PIL import Image
import os

import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# For some reason it doesn't find GOOGLE_API_KEY automatically
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Constants
VIEWPORT_W = 900
VIEWPORT_H = 600

PID_LIST = {
    "NextEra Energy - ML101620329, Page 3": "../output_images/ML101620329_page_3.png",
    "NextEra Energy - ML101620329, Page 4": "../output_images/ML101620329_page_4.png",
    "NextEra Energy - ML101620329, Page 5": "../output_images/ML101620329_page_5.png",
}
PID_CONTEXT = ""  # P&ID in JSON format; Load with P&ID image

# Expert model. Gemini 2.5 Flash also works
EXPERT_MODEL = genai.GenerativeModel(
    "models/gemini-2.5-pro",
)


def _ensure_image(img):
    if img is None:
        raise gr.Error("Error during image processing.")
    return img


def crop_and_resize(
    img: Image.Image,
    zoom: float,
    pan_x: float,
    pan_y: float,
    viewport_w: int = VIEWPORT_W,
    viewport_h: int = VIEWPORT_H,
) -> Image.Image:

    img = _ensure_image(img)

    w, h = img.size
    # Determine crop window size based on zoom
    crop_w = max(1, int(w / max(zoom, 1.0)))
    crop_h = max(1, int(h / max(zoom, 1.0)))

    # Clamp pan to valid top-left so the crop stays inside the image
    max_left = max(0, w - crop_w)
    max_top = max(0, h - crop_h)
    left = int(round(max_left * min(max(pan_x, 0.0), 1.0)))
    top = int(round(max_top * min(max(pan_y, 0.0), 1.0)))
    right = left + crop_w
    bottom = top + crop_h

    cropped = img.crop((left, top, right, bottom))

    return cropped.resize((viewport_w, viewport_h), Image.LANCZOS)


def on_image_upload(img: Image.Image):
    if img is None:
        return None, None, 1.0, 0.5, 0.5, "No image loaded."

    display = crop_and_resize(img, zoom=1.0, pan_x=0.5, pan_y=0.5)
    meta_text = get_image_meta(img)

    return display, img, 1.0, 0.5, 0.5, meta_text


def load_from_predefined_pid(selection):
    if not selection:
        return None, None, 1.0, 0.5, 0.5, "No image loaded."

    image_path = PID_LIST.get(selection)
    json_path = image_path.replace(".png", ".json")

    if not image_path:
        raise gr.Error(f"No path defined for selection: {selection}")

    if not os.path.exists(image_path):
        raise gr.Error(f"Image file not found at '{image_path}'.")

    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        raise gr.Error(f"Failed to load image: {e}")

    try:
        with open(json_path, "r") as f:
            global PID_CONTEXT
            PID_CONTEXT = f.read()

    except Exception as e:
        raise gr.Error(f"Failed to load JSON file {json_path}: {e}")

    return on_image_upload(pil_image)


def update_view(img_state: Image.Image, zoom: float, pan_x: float, pan_y: float):
    if img_state is None:
        raise gr.Error("Please select an image first.")
    display = crop_and_resize(img_state, zoom, pan_x, pan_y)
    return display


def reset_view(img_state: Image.Image):
    if img_state is None:
        raise gr.Error("Please select an image first.")
    display = crop_and_resize(img_state, zoom=1.0, pan_x=0.5, pan_y=0.5)
    return display, 1.0, 0.5, 0.5


def get_image_meta(img_state: Image.Image):
    if img_state is None:
        return "No image selected."
    w, h = img_state.size
    return f"Loaded image: {w}×{h}px"


# ===== Chatbot (stub) =====
def chatbot_respond(message, chat_history):
    global PID_CONTEXT
    global EXPERT_MODEL

    # Probably can be optimized.
    prompt = [
        "You are an experienced industrial process engineer with nuclear power plant P&ID expertise.",
        "Your purpose is to support and consult engineers working on systems at the NextEra Energy Seabrook nuclear power plant.",
        f"CONTEXT (P&ID converted to JSON format):\n{str(PID_CONTEXT)}",
        f"REQUEST:\n{message}",
        """TASKS:
        1) Interpret the P&ID context: identify equipment, instruments, lines, valves, control loops, interlocks, spec breaks, and system boundaries; note data gaps/ambiguities.
        2) Use the P&ID to address the request with practical, plant-relevant guidance (design review, troubleshooting, reliability, MOC, operability, ALARA). If details are missing, explain whats missing and provide best-practice, plant-agnostic guidance.
        3) For any needed calculations (e.g., ΔP, Cv, velocities, ranges), provide only inputs, formula(s), and final results—no step-by-step reasoning.
        4) Keep assumptions minimal and explicit; flag any speculation; quantify uncertainty where possible.
        5) Provide actionable next steps and highlight risks with mitigations.""",
        """OUTPUT FORMAT (Markdown)
        - Summary (2-4 sentences): Direct answer.
        - Data referenced: Bullet list of key tags/lines/loops (e.g., P-120A/B, FV-314, LT-205,).
        - Analysis (concise): Key engineering points; short calculations (inputs, formula, results only); constraints; uncertainties.
        - Assumptions: Minimal, explicit list.
        - Risks & mitigations: Credible failure modes; safeguards.
        - Recommended next steps: 3-6 concrete actions.
        - Missing information: Exact items needed from other docs (PFDs, datasheets, control narratives, setpoint lists, spec breaks).""",
        """STYLE
        - Be precise and practical; use standard engineering units with clear labels.
        - Prefer bullets/tables over long prose; keep it as short as possible while complete.
        
        ERROR HANDLING
        - If the P&ID context is absent/invalid, state this and provide only high-level, non-plant-specific guidance.
        
        COMPLIANCE
        - Do not reveal chain-of-thought; provide conclusions and final computed values only.
        - When uncertain, state bounds and give a conservative recommendation.""",
        "Now produce the answer using the structure above.",
    ]

    expert_response = EXPERT_MODEL.generate_content(prompt)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": expert_response.text})

    return "", chat_history


with gr.Blocks(title="P&ID Question Answering", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        "## P&ID Question Answering\nSelect a predefined P&ID from the sidebar."
    )

    with gr.Row():
        with gr.Sidebar():
            gr.Markdown("### Load P&ID")
            pid_selector = gr.Dropdown(
                label="Select an Example P&ID",
                choices=list(PID_LIST.keys()),
                value=None,
            )

        # ===== LEFT PANE =====
        with gr.Column(scale=3):
            gr.Markdown("### Explore Drawing")
            img_display = gr.Image(
                label="P&ID Drawing",
                interactive=False,
                height=VIEWPORT_H / 2,
                show_download_button=False,
                show_fullscreen_button=True,
            )

            # Controls
            with gr.Row():
                zoom = gr.Slider(
                    1.0,
                    6.0,
                    value=1.0,
                    step=0.1,
                    label="Zoom (1.0 = fit whole image)",
                )
            with gr.Row():
                pan_x = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="Pan X (left ↔ right)"
                )
                pan_y = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="Pan Y (top ↕ bottom)"
                )
            with gr.Row():
                btn_reset = gr.Button("Reset View", size="sm")
                meta = gr.Markdown(
                    "", elem_classes=["text-sm", "text-muted-foreground"]
                )

            # Keep original image in state
            img_state = gr.State(value=None)

        # ===== RIGHT PANE =====
        with gr.Column(scale=5):
            gr.Markdown("### Q&A Chatbot")
            chatbot = gr.Chatbot(height=500, type="messages", label="P&ID QA")
            with gr.Row():
                user_msg = gr.Textbox(
                    label="Ask a question about the drawing",
                    placeholder="Example: 'What is the tag of this control valve near the compressor?'",
                    lines=2,
                )
            with gr.Row():
                send_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

    # Events
    pid_selector.change(
        fn=load_from_predefined_pid,
        inputs=[pid_selector],
        outputs=[img_display, img_state, zoom, pan_x, pan_y, meta],
    )

    zoom.change(
        fn=update_view,
        inputs=[img_state, zoom, pan_x, pan_y],
        outputs=[img_display],
        show_progress=False,
    )
    pan_x.change(
        fn=update_view,
        inputs=[img_state, zoom, pan_x, pan_y],
        outputs=[img_display],
        show_progress=False,
    )
    pan_y.change(
        fn=update_view,
        inputs=[img_state, zoom, pan_x, pan_y],
        outputs=[img_display],
        show_progress=False,
    )

    btn_reset.click(
        fn=reset_view,
        inputs=[img_state],
        outputs=[img_display, zoom, pan_x, pan_y],
    )

    # Show simple metadata
    btn_reset.click(fn=get_image_meta, inputs=[img_state], outputs=[meta])
    zoom.change(fn=get_image_meta, inputs=[img_state], outputs=[meta])
    pan_x.change(fn=get_image_meta, inputs=[img_state], outputs=[meta])
    pan_y.change(fn=get_image_meta, inputs=[img_state], outputs=[meta])

    # Chatbot actions
    send_btn.click(
        fn=chatbot_respond,
        inputs=[user_msg, chatbot],
        outputs=[user_msg, chatbot],
    )
    user_msg.submit(
        fn=chatbot_respond,
        inputs=[user_msg, chatbot],
        outputs=[user_msg, chatbot],
    )
    clear_btn.click(lambda: [], None, chatbot)
    clear_btn.click(lambda: "", None, user_msg)

if __name__ == "__main__":
    demo.launch()
