"""
Experiment 2: Web UI for interactive prompt refinement
"""
import base64
import io
import json
import threading
import uuid
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import plotly
import plotly.graph_objects as go
import matplotlib

from visualize import plot_attribution_heatmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lcnc import (
    extract_prompt_fragments,
    shap_sampling_attribution,
    generate_architecture,
    extract_feature_vector,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

ANALYSIS_JOBS = {}
ANALYSIS_JOBS_LOCK = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_prompt():
    """
    Analyze a user prompt:
    1. Extract fragments
    2. Generate architecture
    3. Extract features
    4. Compute SHAP-inspired attributions
    5. Return visualizations and data
    """
    data = request.json
    prompt = data.get("prompt", "").strip()
    samples = int(data.get("samples", 5))

    if samples < 0:
        samples = 0
    elif samples > 50:
        samples = 50

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    job_id = str(uuid.uuid4())

    with ANALYSIS_JOBS_LOCK:
        ANALYSIS_JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=run_analysis_job,
        args=(job_id, prompt, samples),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    with ANALYSIS_JOBS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)

    if job is None:
        return jsonify({"error": "Unknown job id"}), 404

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job["error"],
    }

    if job["status"] == "done" and job["result"] is not None:
        response["result"] = job["result"]

    return jsonify(response)


def update_job(job_id, **updates):
    with ANALYSIS_JOBS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)


def run_analysis_job(job_id, prompt, samples):
    try:
        update_job(job_id, status="running", progress=3, message="Extracting fragments")

        fragments = extract_prompt_fragments(prompt)
        if not fragments:
            raise ValueError("No fragments extracted")

        update_job(job_id, progress=15, message="Generating architecture")
        arch = generate_architecture(prompt)
        if arch is None:
            raise ValueError("Could not parse architecture")

        feature_vector = extract_feature_vector(arch)

        attributions = {frag: {} for frag in fragments}
        heatmap_image = None

        if samples > 0:
            total_steps = max(1, len(fragments) * samples)

            def progress_callback(completed, total, fragment, fragment_index, fragment_total, sample_index, sample_total):
                percent = 15 + int((completed / total_steps) * 70)
                message = f"Analyzing fragment {fragment_index}/{fragment_total}: sample {sample_index}/{sample_total}"
                update_job(
                    job_id,
                    progress=min(90, percent),
                    message=message,
                )

            update_job(job_id, progress=20, message="Computing attributions")
            attributions = shap_sampling_attribution(
                fragments,
                generate_architecture,
                samples=samples,
                progress=False,
                progress_callback=progress_callback,
            )

            heatmap_image = create_attribution_heatmap(attributions)
        else:
            update_job(job_id, progress=90, message="Analysis disabled, preparing summary")
            heatmap_image = create_empty_heatmap_image()

        feature_chart_fig = create_feature_chart(feature_vector)
        feature_chart_json = json.loads(plotly.io.to_json(feature_chart_fig))

        result = {
            "prompt": prompt,
            "samples": samples,
            "fragments": fragments,
            "feature_vector": feature_vector,
            "architecture": arch,
            "attributions": attributions,
            "heatmap_image": heatmap_image,
            "feature_chart": feature_chart_json,
        }

        update_job(job_id, status="done", progress=100, message="Done", result=result)

    except Exception as e:
        error_msg = str(e)
        if "Failed to establish a new connection" in error_msg or "refused" in error_msg.lower():
            error_msg = "❌ Ollama is not running! Please start Ollama with 'ollama serve' in another terminal."
        elif "Connection refused" in error_msg:
            error_msg = "❌ Connection refused. Is Ollama running on localhost:11434?"

        update_job(job_id, status="error", progress=100, message="Error", error=error_msg)


def create_attribution_heatmap(attr_df):
    """Create a Matplotlib heatmap image encoded as base64."""
    fig = plot_attribution_heatmap(attr_df)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def create_empty_heatmap_image(message="No attribution data available"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="#555")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def create_feature_chart(feature_vector):
    """Create a bar chart showing extracted features."""
    # Filter to only binary features (0/1 indicators)
    binary_features = {
        k: v
        for k, v in feature_vector.items()
        if isinstance(v, int) and (k.startswith("has_") or k.startswith("uses_"))
    }
    
    features = list(binary_features.keys())
    values = list(binary_features.values())
    colors = ["green" if v == 1 else "lightgray" for v in values]

    fig = go.Figure(
        data=go.Bar(
            x=features,
            y=values,
            marker=dict(color=colors),
            hovertemplate="<b>%{x}</b><br>Present: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Extracted Features & Architecture",
        xaxis_title="Feature",
        yaxis_title="Present (1) / Absent (0)",
        height=300,
        yaxis=dict(range=[0, 1.2]),
        autosize=True,
        margin=dict(b=150),
        xaxis=dict(tickangle=-45),
    )

    return fig


@app.route("/api/history/save", methods=["POST"])
def save_history():
    """Save prompt history (JSON array) to a file on the server."""
    data = request.json or {}
    history = data.get("history")
    filename = data.get("filename", "prompt_history.json")

    if not isinstance(history, list):
        return jsonify({"error": "Invalid history payload, expected an array."}), 400

    # Basic safety: do not allow path traversal
    filename = filename.replace("..", "")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return jsonify({"status": "ok", "path": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
