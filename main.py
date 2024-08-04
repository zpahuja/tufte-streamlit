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

openai_key = os.getenv("OPENAI_API_KEY")

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
    selected_dataset = None

    st.sidebar.write("### Choose a dataset")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
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

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})
    else:
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info("To continue, select a dataset from the sidebar on the left or upload your own.")


# Generate data summary
if openai_key and selected_dataset:
    os.environ["OPENAI_API_KEY"] = openai_key
    orc = Orchestrator()

    st.write("## Summary")
    summary = orc.summarize(selected_dataset, enrich=True)

    if "description" in summary:
        st.write(summary["description"])

    if "fields" in summary:
        st.write(summary["fields"])

    # Generate goals
    if summary:
        st.sidebar.write("### Goal Exploration")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox("Add Your Own Goal")

        goals = orc.explore_goals(summary, n_goals=num_goals)
        st.write(f"## Goals ({len(goals)})")

        default_goal = goals[0]["question"]
        goal_questions = [goal["question"] for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:
                new_goal = {"question": user_goal, "visualization": user_goal}
                goals.append(new_goal)
                goal_questions.append(new_goal["question"])

        selected_goal = st.selectbox('Select a goal', options=goal_questions, index=0)

        selected_goal_index = goal_questions.index(selected_goal)
        st.write(goals[selected_goal_index]["question"])

        selected_goal_object = goals[selected_goal_index]

        # Generate visualizations
        if selected_goal_object:
            st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly", "ggplot"]

            selected_library = st.sidebar.selectbox(
                'Choose a visualization library',
                options=visualization_libraries,
                index=0
            )

            st.write("## Visualizations")

            visualizations = orc.visualize(
                summary=summary,
                goal=selected_goal_object,
                library=selected_library)

            selected_viz = visualizations[0]

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                st.image(img, use_column_width=True)

            st.write("### Visualization Code")
            st.code(selected_viz.code)
