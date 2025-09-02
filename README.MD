# P&ID Question Answering with Gemini LLM

This project is a proof-of-concept application demonstrating the use of a Large Language Model (in this case, Google's Gemini 2.5 Pro) to perform question-answering on Piping and Instrumentation Diagrams (P&IDs).

The application provides an interface to load a P&ID drawing, explore it with zoom and pan controls, and ask complex questions in a natural language chat.

![Screenshot of the App](https://github.com/andreenko-repo/pid_diagrams_llm/blob/main/screenshots/image_1.png)

## The Challenge: Making LLMs "Understand" P&IDs

Piping and Instrumentation Diagrams (P&IDs) are essential documents for any industrial plant. They are dense technical drawings that display crucial information about process lines, connections, field instrumentation, control philosophies, and much more.

Getting an LLM to "understand" a P&ID is not straightforward. The core challenges are:

1.  **Symbol Recognition**: P&IDs are visual and symbolic. An LLM needs to first identify all the different symbols for pumps, valves, instruments, etc. This can be approached with custom Computer Vision models (like CNNs) or pre-trained object detection models (like YOLO).

2.  **Connection Tracing**: Simply knowing what equipment is on the diagram isn't enough. The connections, process flows, and control loops between components are often more important than the components themselves. An LLM must understand this connectivity to reason about the system. To solve the connectivity problem, the industry is moving towards structured data formats. An open standard called **DEXPI** (Data Exchange in the Process Industry) aims to create a standardized, machine-readable format for P&IDs. [3, 5] This allows for interoperability between different engineering tools and systems. [3, 4] Other approaches, like representing the P&ID as a graph, can also be effective.

## My Approach: Structured JSON

For this project, I used a simple, structured JSON format to represent the P&ID data. Each key piece of equipment, instrument, or line from the drawing is an object in the JSON file, with its properties and connections explicitly defined.

This approach has a few key benefits for a proof-of-concept:
*   **Simplicity**: JSON is easy to create and read.
*   **LLM-Friendly**: Modern LLMs can parse and understand structured JSON with very high fidelity.
*   **Effectiveness**: It provides the LLM with the explicit connectivity information that it can't get from the image alone.

While this manual JSON creation process isn't scalable for an entire plant's worth of documents, it is a perfect candidate for a demonstrator application like this one.

## Data Source

The P&ID drawings used in this demo are from the **NextEra Energy Seabrook Station**, a nuclear power plant located in Seabrook, NH. [7, 8] These documents are openly accessible to the public through the U.S. Nuclear Regulatory Commission's (NRC) website at [www.nrc.gov](https://www.nrc.gov/).

## How It Works

The application integrates a P&ID viewer with a chatbot interface powered by the Gemini 2.5 Pro model.

1.  **Select P&ID**: The user selects a predefined P&ID from a dropdown.
2.  **Load Data**: The app loads both the P&ID image file and its corresponding JSON data file. The JSON data is stored in a global context variable.
3.  **Explore**: The user can visually explore the drawing using zoom and pan controls.
4.  **Ask a Question**: The user types a question into the chatbot.
5.  **Generate Prompt**: The application constructs a detailed prompt for the LLM. This prompt includes:
    *   A persona (`"You are an experienced industrial process engineer..."`).
    *   The full P&ID data from the loaded JSON file as context.
    *   The user's specific question.
    *   A structured set of tasks and formatting guidelines for the response.
6.  **Get Response**: The prompt is sent to the Gemini API, which analyzes the question in the context of the provided P&ID data and generates a detailed, structured response.
7.  **Display Answer**: The LLM's response is displayed in the chat window.

## Getting Started

### Prerequisites

*   Python 3.8+
*   A Google API Key with the Gemini API enabled.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andreenko-repo/pid_diagrams_llm.git
    cd pid_diagrams_llm
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file, create one with the contents: `gradio`, `pillow`, `google-generativeai`, `python-dotenv`)*

3.  **Set up your environment variables:**
    *   Create a file named `.env` in the root of the project directory.
    *   Add your Google API key to this file:
      ```
      GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      ```

4.  **Verify Project Structure:**
    The application expects the P&ID images and JSON files to be in a specific location. Ensure your project structure looks like this:
    ```
    .
    ├── app/
    │   └── app.py
    ├── output_images/
    │   ├── ML101620329_page_3.png
    │   ├── ML101620329_page_3.json
    │   └── ... (other images and json files)
    ├── .env
    └── requirements.txt
    ```

### Running the Application

1.  Navigate to the `app` directory:
    ```bash
    cd app
    ```
2.  Run the Gradio app:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to the local URL provided (usually `http://127.0.0.1:7860`).

## How to Use the App

1.  **Load a Drawing**: Use the "Select an Example P&ID" dropdown in the left sidebar to choose a drawing. The image and its data will load automatically.
2.  **Navigate the Drawing**: Use the "Zoom" and "Pan" sliders below the image to inspect different areas of the P&ID. Click the "Reset View" button to return to the default overview.
3.  **Ask a Question**: In the "Q&A Chatbot" pane on the right, type a question about the loaded diagram into the text box. For example:
    *   *"What is the tag of this control valve near the compressor?"*
    *   *"Describe the function of the components in the loop for LT-205."*
    *   *"What are the risks associated with the failure of pump P-120A?"*
4.  **Submit**: Press Enter or click the "Ask" button.
5.  **Review the Answer**: The chatbot will process your request and provide a structured answer based on the P&ID data.
6.  **Clear the Chat**: Use the "Clear" button to reset the chat history.
