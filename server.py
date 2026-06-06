from flask import Flask, request, jsonify, send_from_directory
import os, pickle, json, io, base64
import pandas as pd
import numpy as np
import requests as http_requests
import lime
import lime.lime_tabular
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
# GROQ CONFIG
# ═══════════════════════════════════════════════════════════
GROK_API_KEY = "GROQ_API_KEY"
GROK_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROK_MODEL   = "llama-3.3-70b-versatile"

if GROK_API_KEY:
    print("  ✅ Groq client   : ready (Groq API — Llama 3.3 70B)")
else:
    print("  ⚠️  Groq client   : API key not set — /explain disabled")

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════
# CORS
# ═══════════════════════════════════════════════════════════
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin",  "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ═══════════════════════════════════════════════════════════
# LOAD ML PIPELINE
# ═══════════════════════════════════════════════════════════
print("=" * 55)
print("  CareerScan AI — Starting Server")
print("=" * 55)

try:
    with open('ai_impact_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    MODEL        = pipeline['model']
    SCALER       = pipeline['scaler']
    LE           = pipeline['label_encoder']
    COL_ENC      = pipeline['col_encoders']
    FEATURE_COLS = pipeline['feature_columns']
    MODEL_NAME   = pipeline['best_model_name']

    # Real X_train data — LIME ke liye (dummy nahi)
    X_TRAIN_DATA = pipeline.get('X_train', np.zeros((200, len(FEATURE_COLS))))

    ENCODERS: dict[str, dict] = {
        col: dict(zip(enc.classes_, enc.transform(enc.classes_).tolist()))
        for col, enc in COL_ENC.items()
    }

    print(f"  ✅ Model loaded  : {MODEL_NAME}")
    print(f"  ✅ Classes       : {list(LE.classes_)}")
    print(f"  ✅ Features      : {len(FEATURE_COLS)}")
    for col, mapping in ENCODERS.items():
        print(f"     {col}: {mapping}")

except FileNotFoundError:
    print("  ❌ ai_impact_pipeline.pkl NOT FOUND — run train.py first.")
    MODEL = SCALER = LE = COL_ENC = None
    ENCODERS = {}
    FEATURE_COLS = []
    MODEL_NAME = "none"
    X_TRAIN_DATA = None

# ═══════════════════════════════════════════════════════════
# Industry salary-outlook lookup
# ═══════════════════════════════════════════════════════════
INDUSTRY_SALARY = {
    'Marketing': 6.08, 'Manufacturing': 4.70, 'IT': 6.23,
    'Education': 6.06, 'Healthcare': 7.59, 'Finance': 5.23, 'Retail': 6.63,
}

RISK_DIRECTION = {
    'Automation_Risk': 1, 'AI_Adoption_Level': 1,
    'Years_Experience': -1, 'Education_Level': -1, 'Job_Satisfaction': -1,
}

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def encode_field(field: str, raw_value: str):
    mapping = ENCODERS.get(field)
    if mapping is None:
        raise KeyError(f"No encoder found for field '{field}'")
    cleaned_value = str(raw_value).strip().title()
    if cleaned_value not in mapping:
        raise ValueError(
            f"Unknown value '{raw_value}' for field '{field}'. "
            f"Valid options: {list(mapping.keys())}"
        )
    return mapping[cleaned_value]


# ═══════════════════════════════════════════════════════════
# LIME HELPER
# ═══════════════════════════════════════════════════════════
def get_lime_features(X, prediction):
    """
    LIME explanation — real X_train use karta hai dummy nahi.
    discretize_continuous=False taake feature names exact match hon.
    """
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data         = X_TRAIN_DATA,
        feature_names         = FEATURE_COLS,
        class_names           = list(LE.classes_),
        mode                  = 'classification',
        discretize_continuous = False,  # ZAROORI — exact feature names chahiye
        random_state          = 42,
    )

    X_inp_lime = SCALER.transform(X) if MODEL_NAME == 'Logistic Regression' else X.values

    lime_exp = lime_explainer.explain_instance(
        data_row     = X_inp_lime[0],
        predict_fn   = MODEL.predict_proba,
        num_features = len(FEATURE_COLS),
        top_labels   = 1,
    )

    pred_class_idx = list(LE.classes_).index(prediction)
    lime_weights   = dict(lime_exp.as_list(label=pred_class_idx))
    print("LIME weights:", lime_weights)

    # Normalize to 10–95 range
    all_abs   = [abs(v) for v in lime_weights.values()]
    max_abs   = max(all_abs) if all_abs and max(all_abs) > 0 else 1.0
    min_abs   = min(all_abs) if all_abs else 0.0
    range_abs = (max_abs - min_abs) if max_abs != min_abs else 1.0

    features_impact = []
    for feat in FEATURE_COLS:
        weight  = lime_weights.get(feat, 0.0)
        abs_val = abs(weight)

        normalized  = ((abs_val - min_abs) / range_abs) * 85 + 10
        val         = max(10, min(95, int(normalized)))
        direction   = RISK_DIRECTION.get(feat, 0)
        is_positive = weight > 0 if direction == 0 else direction > 0

        features_impact.append({
            'name':     feat.replace('_', ' ').title(),
            'value':    val,
            'positive': is_positive,
            'lime':     round(float(weight), 4),
        })

    features_impact.sort(key=lambda x: x['value'], reverse=True)
    return features_impact

# ═══════════════════════════════════════════════════════════
# SHAP HELPER
# ═══════════════════════════════════════════════════════════
def get_shap_values(X, prediction):
    """
    SHAP explanation — TreeExplainer for tree models,
    LinearExplainer fallback for Logistic Regression.
    Returns (sv, features_impact)
    """
    pred_class_idx = list(LE.classes_).index(prediction)

    try:
        if MODEL_NAME == 'Logistic Regression':
            X_scaled = SCALER.transform(X)
            background_scaled = X_TRAIN_DATA[:100] if X_TRAIN_DATA is not None else X_scaled
            if X_TRAIN_DATA is not None and hasattr(SCALER, 'mean_'):
                background_scaled = SCALER.transform(pd.DataFrame(X_TRAIN_DATA, columns=FEATURE_COLS))[:100]

            explainer = shap.LinearExplainer(MODEL, background_scaled)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer = shap.TreeExplainer(MODEL)
            shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[pred_class_idx][0]
        else:
            sv = shap_values[0]

    except Exception as e:
        print(f"Internal SHAP math fallback error: {e}")
        sv = np.zeros(len(FEATURE_COLS))

    sv = np.array(sv).flatten()

    # ── FIX: Calculate maximum scale using ONLY the visible features ──
    visible_svs = [abs(float(sv_val)) for f, sv_val in zip(FEATURE_COLS, sv) if f not in ['Age', 'Gender']]
    max_visible_abs = max(visible_svs) if visible_svs and max(visible_svs) > 0 else 1.0

    features_impact = []
    for feat, shap_val in zip(FEATURE_COLS, sv):
        # SKIP Age and Gender entirely so they never get sent to the UI
        if feat in ['Age', 'Gender']:
            continue

        norm = abs(float(shap_val)) / max_visible_abs
        val = max(5, min(95, int(norm * 95)))

        features_impact.append({
            'name':     feat.replace('_', ' ').title(),
            'value':    val,
            'positive': float(shap_val) > 0,
            'shap':     round(float(shap_val), 4),
        })

    features_impact.sort(key=lambda x: x['value'], reverse=True)
    return sv, features_impact
# ═══════════════════════════════════════════════════════════
# SHAP PNG HELPER
# ═══════════════════════════════════════════════════════════
def generate_shap_png(feature_names, shap_vals, prediction):
    """SHAP waterfall bar chart — base64 PNG (Filtered to hide demographics)."""
    # ── FIX: Filter out Age and Gender BEFORE building the chart pairs ──
    filtered_pairs = [
        (f, v) for f, v in zip(feature_names, shap_vals) 
        if f not in ['Age', 'Gender']
    ]
    
    pairs  = sorted(filtered_pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
    names  = [p[0].replace('_', ' ') for p in pairs]
    values = [float(p[1]) for p in pairs]
    colors = ['#f87171' if v > 0 else '#38bdf8' for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0c1022')
    ax.set_facecolor('#0c1022')

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                   edgecolor='none', height=0.6)

    for bar, val in zip(bars, values[::-1]):
        x = bar.get_width()
        ax.text(x + (0.001 if x >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center',
                ha='left' if x >= 0 else 'right',
                color='#f1f5f9', fontsize=8)

    ax.axvline(0, color=(1, 1, 1, 0.2), linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP Value  (+ = increases risk,  − = reduces risk)',
                  color='#94a3b8', fontsize=9)
    ax.set_title(f'SHAP Feature Impact  —  Prediction: {prediction}',
                 color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.grid(axis='x', color='#1e293b', linewidth=0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
# ═══════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':   'ok',
        'model':    MODEL_NAME if MODEL else 'not_loaded',
        'classes':  list(LE.classes_) if LE else [],
        'features': len(FEATURE_COLS),
        'encoders': {col: list(m.keys()) for col, m in ENCODERS.items()},
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    # ── Encode ──────────────────────────────────────────────
    try:
        encoded = {
            'Age':                    int(data['age']),
            'Years_Experience':       int(data['experience']),
            'Salary_Before_AI':       int(data['salary']),
            'Work_Hours_Per_Week':    int(data['hours']),
            'Job_Satisfaction':       int(data['satisfaction']),
            'Productivity_Change_%':  int(data['productivity']),
            'Gender':                 encode_field('Gender',              data['gender']),
            'Education_Level':        encode_field('Education_Level',     data['education']),
            'Industry':               encode_field('Industry',            data['industry']),
            'Job_Role':               encode_field('Job_Role',            data['role']),
            'AI_Adoption_Level':      encode_field('AI_Adoption_Level',   data['ai_adoption']),
            'Automation_Risk':        encode_field('Automation_Risk',     data['automation_risk']),
            'Upskilling_Required':    encode_field('Upskilling_Required', data['upskilling']),
            'Remote_Work':            encode_field('Remote_Work',         data['remote']),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    # ── Build feature DataFrame ──────────────────────────────
    try:
        X = pd.DataFrame([encoded])[FEATURE_COLS]
    except KeyError as e:
        return jsonify({"error": f"Feature mismatch: {e}"}), 500

    # ── Predict ──────────────────────────────────────────────
    X_inp    = SCALER.transform(X) if MODEL_NAME == 'Logistic Regression' else X
    proba    = MODEL.predict_proba(X_inp)[0]
    pred_idx = MODEL.predict(X_inp)[0]

    prediction = LE.inverse_transform([pred_idx])[0]
    confidence = float(np.max(proba))
    probs      = {cls: float(p) for cls, p in zip(LE.classes_, proba)}

    # ── Risk score ───────────────────────────────────────────
    replaced_prob = probs.get('Replaced', 0)
    modified_prob = probs.get('Modified', 0)
    risk_score    = min(100, int(replaced_prob * 100 * 1.5 + modified_prob * 100 * 0.5))

    # ── LIME Feature Importance ──────────────────────────────
    try:
        lime_features = get_lime_features(X, prediction)
    except Exception as e:
        print(f"LIME error: {e}")
        lime_features = [
            {'name': f.replace('_', ' ').title(), 'value': 50,
             'positive': RISK_DIRECTION.get(f, 0) > 0, 'lime': 0.0}
            for f in FEATURE_COLS
        ]

    # ── SHAP Values ──────────────────────────────────────────
    try:
        sv, shap_features = get_shap_values(X, prediction)
        shap_png = generate_shap_png(FEATURE_COLS, sv, prediction)
        shap_json = [{'name': f.replace('_', ' ').title(), 'shap': round(float(v), 4)}
                     for f, v in zip(FEATURE_COLS, sv)]
    except Exception as e:
        print(f"SHAP error: {e}")
        shap_features = lime_features  # fallback to LIME
        shap_png      = ""
        shap_json     = []

    # ── Salary outlook ───────────────────────────────────────
    salary_change = INDUSTRY_SALARY.get(data['industry'], 5.0)
    if prediction == 'Replaced':
        salary_change = -15.0
    elif prediction == 'Modified':
        salary_change *= 0.5

    return jsonify({
        'prediction':      prediction,
        'confidence':      round(confidence * 100, 1),
        'risk_score':      risk_score,
        'risk_level':      ('High'   if risk_score >= 60 else
                            'Medium' if risk_score >= 35 else 'Low'),
        'probabilities':   {k: round(v * 100, 1) for k, v in probs.items()},
        'features':        lime_features[:8],   # LIME — frontend bar chart ke liye
        'shap_features':   shap_features[:8],   # SHAP — alag display ke liye
        'salary_outlook':  round(salary_change, 1),
        'salary_direction':('up'   if salary_change > 3  else
                            'down' if salary_change < 0  else 'neutral'),
        'job_status':      prediction,
        'model_used':      MODEL_NAME,
        'shap_png':        shap_png,    # base64 PNG
        'shap_values':     [item for item in shap_json if item['name'] not in ['Age', 'Gender']],
    })


# ═══════════════════════════════════════════════════════════
# GROK — AI EXPLANATION HELPER
# ═══════════════════════════════════════════════════════════
def build_explain_prompt(user: dict, pred: dict) -> str:
    risk_factors    = [f['name'] for f in pred['features'] if f['positive']][:3]
    protect_factors = [f['name'] for f in pred['features'] if not f['positive']][:3]
    salary_sign     = '+' if pred['salary_outlook'] > 0 else ''

    return f"""You are an expert career advisor specialising in AI's impact on the workforce.
A professional has just received a machine-learning career-risk prediction. Your job is to
explain the result in plain English and provide highly specific, actionable next steps.

=== PROFESSIONAL PROFILE ===
Age            : {user.get('age')}
Gender         : {user.get('gender')}
Education      : {user.get('education')}
Industry       : {user.get('industry')}
Job Role       : {user.get('role')}
Experience     : {user.get('experience')} years
Annual Salary  : ${int(user.get('salary', 0)):,}
Hours/Week     : {user.get('hours')}
Job Satisfaction: {user.get('satisfaction')}/10
Productivity Delta : {user.get('productivity')}%
AI Adoption    : {user.get('ai_adoption')}
Automation Risk: {user.get('automation_risk')}
Upskilling Req : {user.get('upskilling')}
Remote Work    : {user.get('remote')}

=== ML MODEL RESULT ===
Outcome        : {pred['prediction']}  ({pred['confidence']}% confidence)
Risk Score     : {pred['risk_score']}/100  ({pred['risk_level']} risk)
Salary Outlook : {salary_sign}{pred['salary_outlook']}%  ({pred['salary_direction']})
Top risk factors    : {', '.join(risk_factors) or 'none'}
Protective factors  : {', '.join(protect_factors) or 'none'}

=== YOUR TASK ===
Return ONLY a valid JSON object — no markdown, no backticks, no extra text.

{{
  "explanation": "<3-4 sentences. Be direct, empathetic and specific to THIS person's role and industry. Reference actual numbers from their profile. Explain what drives the prediction and what it means day-to-day.>",
  "suggestions": [
    {{
      "icon": "<single relevant emoji>",
      "title": "<action title, max 6 words>",
      "description": "<2 concrete sentences tailored to the {user.get('role')} role in {user.get('industry')}. Be specific — name tools, certifications, or strategies.>",
      "tag": "<one of: urgent | growth | skill | mindset>"
    }}
  ]
}}

Provide exactly 4 suggestions ordered by priority (most urgent first).
Make every suggestion role-specific — avoid generic advice."""


def call_grok(prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": GROK_MODEL,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a concise, expert career advisor. "
                    "You always respond with valid JSON only — no markdown, no prose outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens":  1200,
    }
    resp  = http_requests.post(GROK_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    raw   = resp.json()["choices"][0]["message"]["content"].strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


# ═══════════════════════════════════════════════════════════
# /explain   POST
# ═══════════════════════════════════════════════════════════
@app.route('/explain', methods=['POST', 'OPTIONS'])
def explain():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    if not GROK_API_KEY:
        return jsonify({"error": "Grok API not configured."}), 503

    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    # ── Encode ──────────────────────────────────────────────
    try:
        encoded = {
            'Age':                    int(data['age']),
            'Years_Experience':       int(data['experience']),
            'Salary_Before_AI':       int(data['salary']),
            'Work_Hours_Per_Week':    int(data['hours']),
            'Job_Satisfaction':       int(data['satisfaction']),
            'Productivity_Change_%':  int(data['productivity']),
            'Gender':                 encode_field('Gender',              data['gender']),
            'Education_Level':        encode_field('Education_Level',     data['education']),
            'Industry':               encode_field('Industry',            data['industry']),
            'Job_Role':               encode_field('Job_Role',            data['role']),
            'AI_Adoption_Level':      encode_field('AI_Adoption_Level',   data['ai_adoption']),
            'Automation_Risk':        encode_field('Automation_Risk',     data['automation_risk']),
            'Upskilling_Required':    encode_field('Upskilling_Required', data['upskilling']),
            'Remote_Work':            encode_field('Remote_Work',         data['remote']),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    X      = pd.DataFrame([encoded])[FEATURE_COLS]
    X_inp  = SCALER.transform(X) if MODEL_NAME == 'Logistic Regression' else X
    proba  = MODEL.predict_proba(X_inp)[0]
    pred_i = MODEL.predict(X_inp)[0]

    prediction = LE.inverse_transform([pred_i])[0]
    probs      = {cls: float(p) for cls, p in zip(LE.classes_, proba)}

    replaced_prob = probs.get('Replaced', 0)
    modified_prob = probs.get('Modified', 0)
    risk_score    = min(100, int(replaced_prob * 100 * 1.5 + modified_prob * 100 * 0.5))

    # ── LIME Feature Importance ──────────────────────────────
    try:
        lime_features = get_lime_features(X, prediction)
    except Exception as e:
        print(f"LIME error in /explain: {e}")
        lime_features = [
            {'name': f.replace('_', ' ').title(), 'value': 50,
             'positive': RISK_DIRECTION.get(f, 0) > 0, 'lime': 0.0}
            for f in FEATURE_COLS
        ]

    # ── SHAP Values ──────────────────────────────────────────
    try:
        sv, shap_features = get_shap_values(X, prediction)
    except Exception as e:
        print(f"SHAP error in /explain: {e}")
        shap_features = lime_features  # fallback

    salary_change = INDUSTRY_SALARY.get(data['industry'], 5.0)
    if prediction == 'Replaced':
        salary_change = -15.0
    elif prediction == 'Modified':
        salary_change *= 0.5

    pred_summary = {
        'prediction':       prediction,
        'confidence':       round(float(np.max(proba)) * 100, 1),
        'risk_score':       risk_score,
        'risk_level':       ('High'   if risk_score >= 60 else
                             'Medium' if risk_score >= 35 else 'Low'),
        'probabilities':    {k: round(v * 100, 1) for k, v in probs.items()},
        'features':         lime_features[:8],
        'shap_features':    shap_features[:8],
        'salary_outlook':   round(salary_change, 1),
        'salary_direction': ('up'   if salary_change > 3  else
                             'down' if salary_change < 0  else 'neutral'),
    }

    # ── Call Grok ────────────────────────────────────────────
    try:
        prompt   = build_explain_prompt(data, pred_summary)
        grok_out = call_grok(prompt)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Grok returned invalid JSON: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Grok API error: {e}"}), 502

    return jsonify({
        **pred_summary,
        "explanation": grok_out.get("explanation", ""),
        "suggestions": grok_out.get("suggestions", []),
        "model_used":  MODEL_NAME,
        "ai_provider": "xAI Grok",
    })


# ═══════════════════════════════════════════════════════════
# SERVE FRONTEND
# ═══════════════════════════════════════════════════════════
@app.route('/')
def serve_index():
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    return "<h1>index.html not found</h1>", 404


@app.route('/<path:filename>')
def serve_static(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return "File not found", 404


# ═══════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print()
    print("=" * 55)
    print("  🚀 CareerScan AI")
    print("=" * 55)
    print(f"  Model    : {'✅ ' + MODEL_NAME if MODEL else '❌ not loaded'}")
    print(f"  Grok     : {'✅ ready' if GROK_API_KEY else '⚠️  GROK_API_KEY not set'}")
    print("  Frontend : http://localhost:5000")
    print("  Health   : http://localhost:5000/health")
    print("  Predict  : http://localhost:5000/predict  [POST]")
    print("  Explain  : http://localhost:5000/explain  [POST]  ← Grok AI")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=True)
