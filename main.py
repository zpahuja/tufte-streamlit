import os
import pandas as pd
import streamlit as st
from tufte import Orchestrator

os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Tufte: Automatic Generation of Data Visualizations using LLMs",
    page_icon="📊",
)

st.write("# Tufte: Automatic Generation of Visualizations using LLMS 📊")

# Initialize session state variables
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'goals' not in st.session_state:
    st.session_state.goals = None
if 'selected_goal_index' not in st.session_state:
    st.session_state.selected_goal_index = 0
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None
if 'orc' not in st.session_state:
    st.session_state.orc = None

openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_key:
    st.sidebar.write("## Setup")
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")

st.markdown(
    """
    Tufte is an AI data analyst that uses LLMs to automatically generate data visualizations and goals from datasets.
    You can upload your own dataset or use one of the sample datasets provided. Tufte generates a summary of the dataset
    and then uses the LLM to generate goals and visualizations based on the summary. The tool also allows you to add your
    own goals and visualizations
""")

# Select a dataset
if openai_key:
    st.sidebar.write("### Choose a dataset")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Choose a dataset',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

        if uploaded_file is not None:
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            st.session_state.selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})
    else:
        st.session_state.selected_dataset = datasets[[dataset["label"] for dataset in datasets].index(selected_dataset_label)]["url"]

    if not st.session_state.get('selected_dataset'):
        st.info("To continue, select a dataset from the sidebar on the left or upload your own.")

# Generate data summary and goals
if openai_key and st.session_state.get('selected_dataset'):
    os.environ["OPENAI_API_KEY"] = openai_key
    
    # Create a new Orchestrator instance only if it doesn't exist or if the dataset has changed
    if st.session_state.orc is None or st.session_state.get('last_dataset') != st.session_state.selected_dataset:
        st.session_state.orc = Orchestrator()
        st.session_state.last_dataset = st.session_state.selected_dataset
        # Reset summary and goals when dataset changes
        st.session_state.summary = None
        st.session_state.goals = None

    # Only generate summary and goals if they don't exist
    if st.session_state.summary is None:
        st.session_state.summary = st.session_state.orc.summarize(st.session_state.selected_dataset, enrich=True)
    
    if st.session_state.goals is None:
        st.session_state.goals = st.session_state.orc.explore_goals(st.session_state.summary, n_goals=10)

    st.write("## Summary")
    if "description" in st.session_state.summary:
        st.write(st.session_state.summary["description"])

    if "fields" in st.session_state.summary:
        st.write(st.session_state.summary["fields"])

    # Display goals
    st.sidebar.write("### Goal Exploration")

    num_goals = st.sidebar.slider(
        "Number of goals to display",
        min_value=1,
        max_value=min(10, len(st.session_state.goals)),
        value=min(4, len(st.session_state.goals)))

    own_goal = st.sidebar.checkbox("Add Your Own Goal")

    displayed_goals = st.session_state.goals[:num_goals]
    
    if own_goal:
        user_goal = st.sidebar.text_input("Describe Your Goal")
        if user_goal:
            new_goal = {"question": user_goal, "visualization": user_goal}
            displayed_goals.append(new_goal)

    st.write(f"## Goals ({len(displayed_goals)})")

    goal_questions = [goal["question"] for goal in displayed_goals]
    selected_goal = st.selectbox('Select a goal', options=goal_questions, index=st.session_state.selected_goal_index)
    st.session_state.selected_goal_index = goal_questions.index(selected_goal)

    st.write(displayed_goals[st.session_state.selected_goal_index]["question"])

    # Generate visualizations
    st.sidebar.write("## Visualization Library")
    visualization_libraries = ["seaborn", "matplotlib", "plotly", "ggplot"]

    selected_library = st.sidebar.selectbox(
        'Choose a visualization library',
        options=visualization_libraries,
        index=0
    )

    st.write("## Visualizations")

    # Only generate visualizations if they don't exist or if the goal or library has changed
    if (st.session_state.visualizations is None or 
        st.session_state.get('last_goal_index') != st.session_state.selected_goal_index or
        st.session_state.get('last_library') != selected_library):
        
        st.session_state.visualizations = st.session_state.orc.visualize(
            summary=st.session_state.summary,
            goal=displayed_goals[st.session_state.selected_goal_index],
            library=selected_library)
        
        st.session_state.last_goal_index = st.session_state.selected_goal_index
        st.session_state.last_library = selected_library

    if st.session_state.visualizations:
        selected_viz = st.session_state.visualizations[0]

        if selected_viz.raster:
            from PIL import Image
            import io
            import base64

            imgdata = base64.b64decode(selected_viz.raster)
            img = Image.open(io.BytesIO(imgdata))
            st.image(img, use_column_width=True)

        st.write("### Visualization Code")
        st.code(selected_viz.code)