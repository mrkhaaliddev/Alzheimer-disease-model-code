"""
alzheimer_api.py – Flask API for Alzheimer MRI staging
Works with Groq's native-vision Llama-4 models.
"""
import os, base64, json, cv2, numpy as np, torch
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from groq import Groq                                # <─ NEW
from alzheimer_classifier_training import AlzheimerClassifier

# ────────────────────────────── CONFIG ──────────────────────────────
UPLOAD_FOLDER        = "uploads"
ALLOWED_EXTENSIONS   = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}
MAX_UPLOAD_MB        = 16
MODEL_CKPT           = "models/Resnet_50_alzheimer_classifier_90%_.pth"
GROQ_MODEL           = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY         = "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"       # keep secrets out of source
# ────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER,
                  MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

groq_client = Groq(api_key=GROQ_API_KEY)
classifier, model = None, None                        # loaded lazily

# ──────────────────────────── UTILITIES ─────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_classifier():
    """lazy-load the PyTorch Alzheimer model once."""
    global classifier, model
    if model is not None:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = AlzheimerClassifier(device=device)

    ckpt = torch.load(MODEL_CKPT, map_location=device)
    arch = (ckpt.get("model_architecture") or "resnet50").lower()
    classifier.build_model("resnet50" if "resnet" in arch else arch)
    classifier.model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    )
    classifier.model.eval()
    model = classifier.model

def tensor_from_cv2(img_bgr: np.ndarray):
    t = classifier.get_val_transforms()(image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return t["image"].unsqueeze(0)

# ────────────────────── GROQ BASE64 VALIDATION ──────────────────────
def validate_mri_with_groq(img_b64: str) -> None:
    """
    Raises RuntimeError with a descriptive message if the image is NOT
    a brain MRI suitable for Alzheimer staging. Otherwise returns None.
    """
    if len(img_b64) * 3 / 4 > 4 * 1024 * 1024:                    # > 4 MB raw
        raise RuntimeError("Image is larger than Groq's 4 MB base64 limit")

    prompt_text = (
        "You are a medical-imaging expert. Classify the image strictly as one of "
        'VALID_MRI, NOT_MRI, NOT_BRAIN_MRI, NOT_ALZHEIMER_MRI. Return JSON: {"result": "<choice>"}.'
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ],
        }],
        temperature=0.1,
        max_completion_tokens=50,                       # <- correct param
        response_format={"type": "json_object"},        # <- JSON mode guarantees parsable output
    )

    result = json.loads(response.choices[0].message.content)["result"]
    ERRORS = {
        "NOT_MRI":           "Not an MRI scan. Please upload a brain MRI.",
        "NOT_BRAIN_MRI":     "The MRI is not of the brain.",
        "NOT_ALZHEIMER_MRI": "Brain MRI found, but unsuitable for Alzheimer staging.",
    }
    if result != "VALID_MRI":
        raise RuntimeError(ERRORS.get(result, "Image validation failed."))

# ──────────────────────────── ENDPOINTS ─────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- ingest file or base64 --------
        if "file" in request.files:
            f = request.files["file"]
            if f.filename == "" or not allowed_file(f.filename):
                return jsonify(error="Invalid file."), 400
            img_bytes = f.read()
        elif request.json and request.json.get("image_base64"):
            b64 = request.json["image_base64"].split(",")[-1]
            img_bytes = base64.b64decode(b64)
        else:
            return jsonify(error="No image provided."), 400

        img_b64 = base64.b64encode(img_bytes).decode()   # for Groq
        validate_mri_with_groq(img_b64)                  # ← will raise on bad input

        # -------- run CNN inference --------
        load_classifier()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is None:
            return jsonify(error="Unable to decode image."), 400

        input_tensor = tensor_from_cv2(cv_img).to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = torch.argmax(probs).item()

        classes = ["No Impairment", "Very Mild Impairment", "Mild Impairment", "Moderate Impairment"]
        response = dict(
            predicted_class=classes[idx],
            confidence_percent=round(float(probs[idx]) * 100, 2),
            class_probabilities={c: round(float(p) * 100, 2) for c, p in zip(classes, probs)},
            model_info=dict(architecture="ResNet50", classes=len(classes), image_size="224×224"),
            validation_passed=True,
        )
        return jsonify(response)
    except RuntimeError as re:
        return jsonify(error=str(re), validation_failed=True), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {e}"), 500

@app.route("/model-info")
def model_info():
    load_classifier()
    return jsonify(model_loaded=True, device=str(next(model.parameters()).device))

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    print("Alzheimer Classifier API running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
