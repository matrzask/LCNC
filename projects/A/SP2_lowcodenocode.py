import re
import requests
import time
import pandas as pd
import streamlit as st
import random
import math
import numpy as np
from graphviz import Digraph

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct-v0.3"
EMBED_MODEL = "text-embedding-bge-small-en-v1.5"

# session state variables
if "output" not in st.session_state:
    st.session_state.output = None
if "fragments" not in st.session_state:
    st.session_state.fragments = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "original_output" not in st.session_state:
    st.session_state.original_output = None
if "stabilized" not in st.session_state:
    st.session_state.stabilized = None
if "mapping" not in st.session_state:
    st.session_state.mapping = None
if "refined_prompt" not in st.session_state:
    st.session_state.refined_prompt = None
if "refined_output" not in st.session_state:
    st.session_state.refined_output = None
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = "I need a task management app that works offline and sends reminders."
if "basic_time" not in st.session_state:
    st.session_state.basic_time = 0
if "full_time" not in st.session_state:
    st.session_state.full_time = 0

###############################################################################################################
# LLM call
def call_llm(prompt, role="user", temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": f"You are a low-code app generator and explainer. {prompt}"
            }
        ],
        "temperature": temperature,
        "max_tokens": 512
    }
    try:
        r = requests.post(LM_STUDIO_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error contacting LLM: {e}]\n{r.text if 'r' in locals() else ''}"

def generate_app(prompt):
    gen_prompt = f"""
    You are a system that generates a FORMAL, STRUCTURED application architecture
    from a user prompt.
    
    ========================
    OUTPUT FORMAT (STRICT – DO NOT DEVIATE):
    ========================
    
    ENTITIES:
    - EntityName:
        attributes:
          - attribute_name : type
    
    RELATIONSHIPS:
    - EntityA -> EntityB : relationship_type
    
    OPERATIONS:
    - OperationName:
        description: short, precise action
    
    DATA_STORAGE:
    - StorageType (e.g. local, database, file):
        stored_entities: [EntityName]
    
    ========================
    GENERATION RULES (VERY IMPORTANT):
    ========================
    - If no fragment supports an element, DO NOT include it.
    - DO NOT infer features.
    - DO NOT add best practices.
    - DO NOT add optional or "nice-to-have" elements.
    - DO NOT explain anything outside the structure.
    
    ========================
    USER PROMPT:
    ========================
    \"\"\"{prompt}\"\"\"
    """
    return call_llm(gen_prompt)


###############################################################################################################
# embedding
def get_embedding(text: str):
    try:
        r = requests.post("http://127.0.0.1:1234/v1/embeddings", json={
            "model": EMBED_MODEL,
            "input": text
        })
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception as e:
        print("Embedding error:", e)
        return None

def cosine_distance(a, b):
    if not a or not b:
        return 1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return 1 - (dot / (na * nb))

###############################################################################################################
# SHAP
def shap_like_attribution(prompt, fragments, samples=5):
    baseline_output = generate_app(prompt)
    scores = {}

    progress = st.progress(0)
    step = 1 / max(len(fragments), 1)

    for i, frag in enumerate(fragments):

        contributions = []
        other_frags = [f for f in fragments if f != frag]

        for _ in range(samples):
            subset_size = random.randint(0, len(other_frags))
            subset = random.sample(other_frags, k=subset_size)

            prompt_without = " ".join(subset)
            prompt_with = " ".join(subset + [frag])

            out_without = generate_app(prompt_without) if subset else ""
            out_with = generate_app(prompt_with)

            # compare
            emb_with = get_embedding(out_with)
            emb_without = get_embedding(out_without)
            contribution = cosine_distance(emb_with, emb_without)
            contributions.append(contribution)

        # average
        scores[frag] = round(sum(contributions) / len(contributions), 3)
        progress.progress(min((i+1)*step, 1.0))

    return scores, baseline_output

#SHAP stabilizing
def stabilized_shap_attribution(prompt, fragments, samples=5, runs=3):
    all_scores = []

    #repeating shap 3 times
    for _ in range(runs):
        scores, _ = shap_like_attribution(prompt, fragments, samples=samples)
        all_scores.append(scores)

    #calculating mean and std
    result = {}
    for frag in fragments:
        values = [run[frag] for run in all_scores if frag in run]
        result[frag] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values
        }

    return result


###############################################################################################################
# natural language explanation
def explain(app_summary, fragments, scores):
    explanation_prompt = f"""
    You are an assistant that explains how automatically generated low-code apps behave.
    You must write short, human-centered explanations of design choices — like what a user
    would see in a tool-tip or help message. Avoid academic or analytic language.

    Given this prototype summary:
    {app_summary}
    
    And these feature fragments with importance scores:
    {list(zip(fragments, scores))}
    
    Write 2–3 concise sentences (max 60 words total) explaining how these features influence
    the app's behavior. Focus on *what users experience* — not model details or "scores".
    
    Now produce the natural language explanation:
    """
    result = call_llm(explanation_prompt)
    return result.strip()

###############################################################################################################
# feature mapping
def map_fragments_to_features(app_output, fragments):
    # extract candidate elements
    elements = []

    for line in app_output.splitlines():
        line = line.strip()
        if line.startswith("-"):
            name = line.lstrip("- ").rstrip(":")
            elements.append(name)

    if not elements:
        return [{"error": "No architectural elements detected"}]

    frag_emb = [get_embedding(f) for f in fragments]
    elem_emb = [get_embedding(e) for e in elements]

    frag_arr = np.array(frag_emb)
    elem_arr = np.array(elem_emb)

    sims = frag_arr @ elem_arr.T / (
            np.linalg.norm(frag_arr, axis=1)[:, None] *
            np.linalg.norm(elem_arr, axis=1)[None, :]
    )

    mapping = []
    for i, frag in enumerate(fragments):
        top_idx = sims[i].argsort()[::-1][:3]
        related = [elements[j] for j in top_idx]
        mapping.append({
            "fragment": frag,
            "features": related
        })

    return mapping


###############################################################################################################
# prompt contribution graph
def create_structured_workflow(fragments, mapping, scores):
    # visual preparation to blend in streamlit nicely
    dot = Digraph(
        "Explainable LCNC Workflow",
        graph_attr={"rankdir": "TB",
                    "splines": "ortho",
                    "nodesep": "0.7",
                    "ranksep": "1.0",
                    "bgcolor": "#0e1117"},
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "style": "rounded,filled",
            "color": "#ffffff",
            "fontcolor": "#ffffff",
            "fillcolor": "#111827"},
        edge_attr={
            "color": "#9ca3af",
            "fontcolor": "#ffffff",
            "fontsize": "10"}
    )
    # wrapper
    def wrap_text(text, width=25):
        words = text.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) <= width:
                line += w + " "
            else:
                lines.append(line.strip())
                line = w + " "
        lines.append(line.strip())
        return "\n".join(lines)

    dot.node("U", "User\nPrompt", shape="ellipse")
    dot.node("C", "Generated\nApp", shape="ellipse")
    dot.edge("U", "C")

    # listing element underneath all the attributes
    for i, frag in enumerate(fragments):
        if i >= len(mapping) or "features" not in mapping[i]:
            continue

        f_id = f"F{i}"
        dot.node(f_id, wrap_text(frag), shape="box")
        shap_value = scores.get(frag, 0.0)
        dot.edge("C", f_id, label=f"+{shap_value:.2f}",
                 fontsize="20", fontcolor="#ffffff",color=shap_color(shap_value))

        with dot.subgraph(name=f"cluster_{i}") as c:
            c.attr(style="rounded,filled",
                   label=wrap_text(frag),
                   color="#111827",
                   fontcolor="#ffffff",
                   fontsize ="15",
                   penwidth="2.5")

            prev = f_id
            for j, feat in enumerate(mapping[i]["features"]):
                feat_id = f"{f_id}_{j}"
                c.node(feat_id, wrap_text(feat), shape="box")
                c.edge(prev, feat_id, arrowhead="none")
                prev = feat_id

    return dot

# to visualize score strength better
def shap_color(score):
    score = max(0.0, min(1.0, score))

    r1, g1, b1 = (200, 150, 150)
    r2, g2, b2 = (255, 10, 10)

    r = int(r1 + (r2 - r1) * score)
    g = int(g1 + (g2 - g1) * score)
    b = int(b1 + (b2 - b1) * score)

    return f"#{r:02x}{g:02x}{b:02x}"

###############################################################################################################
# splitting prompt for attributes
def extract_prompt_fragments(prompt):
    helper_prompt = f"""
    Extract distinct requirement fragments from the user prompt BELOW.

    STRICT RULES:
    - DO NOT translate the text.
    - DO NOT paraphrase.
    - DO NOT reorder.
    - DO NOT add or remove words.
    - DO NOT merge ideas.
    - Only split the user's text into meaningful requirement fragments.
    - Output ONLY the fragments, each on a new line.
    - NO bullets, NO numbers, NO explanations.
    
    User prompt:
    \"\"\"{prompt}\"\"\"
    """

    result = call_llm(helper_prompt)
    fragments = [re.sub(r"^[-•\d. ]+", "", line.strip()) for line in result.split('\n') if line.strip()]
    return fragments

###############################################################################################################
# consistency test
def consistency_test(prompt, fragments, scores):
    # find most important fragment
    top_frag = max(scores, key=scores.get)

    # generate prompt without said fragment
    reduced_fragments = [f for f in fragments if f != top_frag]
    reduced_prompt = " ".join(reduced_fragments)

    original_output = generate_app(prompt)
    reduced_output = generate_app(reduced_prompt)

    return top_frag, original_output, reduced_output

###############################################################################################################
# UI
text_window_height = 450

st.title("Low-Code/No-Code Explainability")
st.markdown("( Explainable LLM-Augmented Low/No-Code Prototyping )")

# User input - use session state to persist
user_prompt = st.text_input(
    "How can I help u?",
    value=st.session_state.user_prompt
)

# Update session state when input changes
if user_prompt != st.session_state.user_prompt:
    st.session_state.user_prompt = user_prompt

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("Run")
with col2:
    samples = st.segmented_control('Samples', options=[2, 5, 10], default=2)

if run_clicked:
    # Reset refined outputs when running new analysis
    st.session_state.refined_prompt = None
    st.session_state.refined_output = None

    status = st.empty()
    status.info("Generating...")

    # full time measuring
    start_full = time.perf_counter()

    fragments = extract_prompt_fragments(user_prompt)
    st.session_state.fragments = fragments

    # stabilized SHAP attribution
    stabilized = stabilized_shap_attribution(
        user_prompt,
        fragments,
        samples=samples,
        runs=3
    )
    scores = {f: stabilized[f]["mean"] for f in stabilized}

    st.session_state.stabilized = stabilized
    st.session_state.scores = scores

    # basic time measuring
    start_basic = time.perf_counter()
    app_output = generate_app(user_prompt)
    basic_time = time.perf_counter() - start_basic

    # save for before/after
    st.session_state.original_output = app_output
    st.session_state.output = app_output
    st.session_state.basic_time = basic_time

    status.empty()

    mapping = map_fragments_to_features(app_output, fragments)
    st.session_state.mapping = mapping

    end_full = time.perf_counter()
    full_time = end_full - start_full
    st.session_state.full_time = full_time

    # Mark pipeline as done
    st.session_state.pipeline_done = True

# Display main results (always show if pipeline has run)
if st.session_state.pipeline_done and st.session_state.output:
    st.subheader("Generated Prototype App")
    st.text_area("App Output", st.session_state.output, height=text_window_height, key="main_output")

    st.subheader("Extracted Attributes")
    for frag in st.session_state.fragments:
        st.markdown(f"- **{frag}**")

    st.subheader("Attribution Scores")
    st.caption("We use heuristic SHAP-inspired attribution method")
    df_scores = pd.DataFrame({
        "Attribute": list(st.session_state.scores.keys()),
        "SHAP Score": list(st.session_state.scores.values())
    })
    st.bar_chart(df_scores.set_index("Attribute"))

    st.subheader("Attribution Stability")
    df_stable = pd.DataFrame([
        {
            "Fragment": frag,
            "Mean SHAP": st.session_state.stabilized[frag]["mean"],
            "Std Dev": st.session_state.stabilized[frag]["std"]
        }
        for frag in st.session_state.stabilized
    ])
    st.dataframe(df_stable)

    with st.expander("SHAP Stability Distribution", expanded=False):
        st.caption("Shows variance across multiple SHAP runs for each fragment")
        for frag in st.session_state.stabilized:
            st.write(f"**{frag}**")
            st.line_chart(st.session_state.stabilized[frag]["values"])

    #st.subheader("Fragment to Feature Mapping")
    #if isinstance(st.session_state.mapping, list):
    #    st.json(st.session_state.mapping)
    #else:
    #    st.warning("Showing recovered problematic JSON")
    #    st.write(st.session_state.mapping)

    st.subheader("Prompt Contribution Graph")
    dot = create_structured_workflow(
        st.session_state.fragments,
        st.session_state.mapping,
        st.session_state.scores
    )
    st.graphviz_chart(dot)

    st.subheader("Natural Language Explanation")
    explanation = explain(
        st.session_state.output,
        st.session_state.fragments,
        st.session_state.scores
    )
    st.write(explanation)

    # basic and full time
    st.subheader("Explainability Cost")
    st.metric("Basic app generation", f"{st.session_state.basic_time:.2f} s")
    st.metric("Full explainable pipeline", f"{st.session_state.full_time:.2f} s")
    st.metric("Overhead comparison", f"{st.session_state.full_time - st.session_state.basic_time:.2f} s ({((st.session_state.full_time - st.session_state.basic_time) / st.session_state.basic_time * 100):.1f}%)")


# Show these sections only if pipeline has been run
if st.session_state.pipeline_done and st.session_state.output:

    st.divider()

    # Consistency Test Section
    st.subheader("Attribution Consistency Test")
    st.caption("Want to see how removing the highest-scored fragment changes the output ?")

    if st.button("Compare"):
        with st.spinner("Running consistency test..."):
            removed, original, reduced = consistency_test(
                st.session_state.user_prompt,
                st.session_state.fragments,
                st.session_state.scores
            )

            st.success(f"Removed fragment with highest SHAP score: **{removed}**")

            colA, colB = st.columns(2)

            with colA:
                st.write("**Original Prototype**")
                st.text_area("", original, height=text_window_height, key="consistency_original")

            with colB:
                st.write("**After Removing Top Fragment**")
                st.text_area("", reduced, height=text_window_height, key="consistency_reduced")

    st.divider()

    # Refinement Loop Section
    st.subheader("Refinement Loop")
    st.caption("Do you need any suggestions how to refine prompt ?")

    if st.button("Generate Refinement Suggestion"):
        with st.spinner("Analyzing attribution and generating suggestions..."):
            suggestion_prompt = f"""
            You are an assistant helping the user improve their prompt based on SHAP attribution analysis.
        
            Original prompt: {st.session_state.user_prompt}
            
            SHAP attribution scores:
            {st.session_state.scores}
        
            Task:
            Suggest how the user could modify the prompt to achieve better or more precise
            functionality in the generated low-code application. Be direct, practical,
            and provide concrete examples of an improved prompt.
        
            Respond with:
            1. A brief analysis (1-2 sentences)
            2. A complete refined prompt that the user can use directly
            
            Format your response as:
            ANALYSIS: [your analysis]
            REFINED_PROMPT: [the improved prompt]
            """

            refined_response = call_llm(suggestion_prompt)
            st.session_state.refined_prompt = refined_response

            st.markdown("**Suggestion:**")
            st.text_area("Refinement suggestion", value=refined_response, height=text_window_height, key="refined_suggestion")

    # Show refined prompt generation if suggestion exists
    if st.session_state.refined_prompt:
        if st.button("Generate Refined Prototype"):
            with st.spinner("Generating refined prototype..."):
                # Extract the refined prompt from the response
                refined_text = st.session_state.refined_prompt
                if "REFINED_PROMPT:" in refined_text:
                    refined_prompt = refined_text.split("REFINED_PROMPT:")[1].strip()
                else:
                    refined_prompt = refined_text

                refined_output = generate_app(refined_prompt)
                st.session_state.refined_output = refined_output

                st.subheader("Before vs After Explainability")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Original Prototype**")
                    st.text_area("", st.session_state.original_output, height=text_window_height, key="before_refine")

                with col2:
                    st.write("**Refined Prototype**")
                    st.text_area("", refined_output, height=text_window_height, key="after_refine")

                st.success("Refinement loop completed, compare the outputs above.")

    st.divider()
    st.caption("dziękuję za uwagę !")