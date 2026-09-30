"""
Vehicle Health Monitoring — Local Testing System
------------------------------------------------
Flask backend for:
    - XGBoost vehicle failure prediction
    - Feature engineering
    - CVHS health scoring
    - SHAP explainability
    - Batch CSV prediction
    - Model information

Run:
    python app.py

Then open:
    http://localhost:5000
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pickle
import io
import traceback

import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file
)

# SHAP is optional for the application itself.
# If it is installed, the app will provide explanations.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)


# ============================================================================
# BASE DIRECTORY
# ============================================================================

BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# PICKLE LOADER
# ============================================================================

def _load(name):
    """
    Load a pickle artifact from the same directory as app.py.
    """

    path = os.path.join(BASE, name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nMissing artifact: {name}\n"
            f"Expected location:\n{path}\n\n"
            f"Copy {name} next to app.py."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

MODEL = _load("best_model.pkl")

SCALER = _load("scaler.pkl")

FEATURES = _load("feature_names.pkl")

# IMPORTANT:
# This file contains the SAME normalization constants used during training.
#
# engineer_params.pkl:
# {
#     "rpm_max": 4670.0,
#     "temp_max": 135.0
# }
#
# DO NOT calculate these from the incoming request.
ENGINEER_PARAMS = _load("engineer_params.pkl")


# ============================================================================
# ENGINEERING PARAMETERS
# ============================================================================

RPM_MAX = float(ENGINEER_PARAMS["rpm_max"])

TEMP_MAX = float(ENGINEER_PARAMS["temp_max"])


# ============================================================================
# OPTIMAL THRESHOLD
# ============================================================================

try:

    RS = _load("research_summary.pkl")

    OPTIMAL_THR = float(
        RS["gap4_optimal_thr"]
    )

except Exception:

    # Current research pipeline produced 0.01
    OPTIMAL_THR = 0.01


# ============================================================================
# PRINT MODEL INFORMATION
# ============================================================================

print("\n" + "=" * 70)

print("VEHICLE HEALTH MONITORING SYSTEM")

print("=" * 70)

print(
    f"✓ Model loaded       : "
    f"{type(MODEL).__name__}"
)

print(
    f"✓ Number of trees    : "
    f"{getattr(MODEL, 'n_estimators', 'N/A')}"
)

print(
    f"✓ Scaler loaded      : "
    f"{type(SCALER).__name__}"
)

print(
    f"✓ Features loaded    : "
    f"{len(FEATURES)}"
)

print(
    f"✓ RPM reference max  : "
    f"{RPM_MAX}"
)

print(
    f"✓ Temp reference max : "
    f"{TEMP_MAX}"
)

print(
    f"✓ Optimal threshold  : "
    f"{OPTIMAL_THR}"
)

print(
    f"✓ SHAP available     : "
    f"{SHAP_AVAILABLE}"
)

print("=" * 70 + "\n")


# ============================================================================
# DOMAIN KNOWLEDGE
# ============================================================================

SAFE_RANGES = {

    "Engine_Temperature_C":
        (60, 100),

    "RPM":
        (600, 3000),

    "Oil_Pressure_psi":
        (25, 80),

    "Vibration_mm_s":
        (0, 3.0),

    "Battery_Voltage_V":
        (11.8, 14.8),

    "Coolant_Temperature_C":
        (50, 95),

    "Fuel_Consumption_L_100km":
        (6, 12),

    "Vehicle_Speed_kmh":
        (0, 130),

    "Operating_Hours":
        (0, 8000),
}


# ============================================================================
# CVHS WEIGHTS
# ============================================================================

WEIGHTS = {

    "Engine_Temperature_C":
        0.18,

    "RPM":
        0.08,

    "Oil_Pressure_psi":
        0.12,

    "Vibration_mm_s":
        0.15,

    "Battery_Voltage_V":
        0.10,

    "Coolant_Temperature_C":
        0.13,

    "Fuel_Consumption_L_100km":
        0.09,

    "Vehicle_Speed_kmh":
        0.05,

    "Operating_Hours":
        0.10,
}


# ============================================================================
# MAINTENANCE ACTIONS
# ============================================================================

ACTION_MAP = {

    "Engine_Temperature_C": {

        "high": (
            100,
            "Engine overheating — check cooling system, "
            "thermostat, and water pump"
        ),

        "low": (
            65,
            "Engine too cold — check thermostat "
            "for stuck-open condition"
        ),
    },

    "Oil_Pressure_psi": {

        "low": (
            25,
            "CRITICAL: Low oil pressure — inspect "
            "oil pump and oil level immediately"
        ),

        "high": (
            80,
            "Excessive oil pressure — inspect "
            "pressure relief valve"
        ),
    },

    "Vibration_mm_s": {

        "high": (
            3.0,
            "High vibration — inspect engine mounts, "
            "driveshaft, and wheel balance"
        ),
    },

    "Battery_Voltage_V": {

        "low": (
            11.8,
            "Low battery voltage — test alternator "
            "output and battery health"
        ),

        "high": (
            14.8,
            "Battery overcharging — inspect voltage regulator"
        ),
    },

    "Coolant_Temperature_C": {

        "high": (
            95,
            "Coolant temperature elevated — check coolant "
            "level, radiator, and hose integrity"
        ),

        "low": (
            50,
            "Coolant temperature too low — thermostat "
            "may be stuck open"
        ),
    },

    "RPM": {

        "high": (
            3000,
            "High RPM — inspect throttle control "
            "and idle system"
        ),
    },

    "Operating_Hours": {

        "high": (
            8000,
            "High operating hours — schedule "
            "comprehensive service"
        ),
    },

    "Fuel_Consumption_L_100km": {

        "high": (
            12,
            "Elevated fuel consumption — inspect "
            "air filter, injectors, and O2 sensors"
        ),
    },
}


# ============================================================================
# RAW FEATURES
# ============================================================================

RAW_FEATURES = [

    "Engine_Temperature_C",

    "RPM",

    "Oil_Pressure_psi",

    "Vibration_mm_s",

    "Battery_Voltage_V",

    "Coolant_Temperature_C",

    "Fuel_Consumption_L_100km",

    "Vehicle_Speed_kmh",

    "Operating_Hours",
]


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(d: pd.DataFrame) -> pd.DataFrame:

    """
    EXACT feature engineering used by the trained model.

    IMPORTANT:
    Engine_Load_Proxy must use the SAME reference maxima
    used during training.

    Training:
        rpm_max  = 4670
        temp_max = 135

    We therefore NEVER use:
        d['RPM'].max()
        d['Engine_Temperature_C'].max()

    on the incoming request.
    """

    d = d.copy()

    # ------------------------------------------------------------------------
    # 1. Thermal Stress
    # ------------------------------------------------------------------------

    d["Thermal_Stress"] = (
        d["Engine_Temperature_C"]
        -
        d["Coolant_Temperature_C"]
    )


    # ------------------------------------------------------------------------
    # 2. Engine Load Proxy
    #
    # TRAINING FORM:
    #
    # (RPM / dataset_RPM_max)
    # *
    # (Engine_Temperature / dataset_temp_max)
    #
    # Saved training parameters:
    # RPM_MAX  = 4670
    # TEMP_MAX = 135
    # ------------------------------------------------------------------------

    d["Engine_Load_Proxy"] = (

        d["RPM"] / RPM_MAX

    ) * (

        d["Engine_Temperature_C"] / TEMP_MAX

    )


    # ------------------------------------------------------------------------
    # 3. Oil Pressure per RPM
    # ------------------------------------------------------------------------

    d["Oil_Press_per_RPM"] = (

        d["Oil_Pressure_psi"]

        /

        (d["RPM"] / 1000 + 0.001)

    )


    # ------------------------------------------------------------------------
    # 4. Vibration per RPM
    # ------------------------------------------------------------------------

    d["Vibration_per_RPM"] = (

        d["Vibration_mm_s"]

        /

        (d["RPM"] / 1000 + 0.001)

    )


    # ------------------------------------------------------------------------
    # 5. Age × Temperature
    # ------------------------------------------------------------------------

    d["Age_Temp_Interaction"] = (

        d["Operating_Hours"]

        *
        
        d["Engine_Temperature_C"]

        /

        1000

    )


    # ------------------------------------------------------------------------
    # 6. Fuel / Speed Ratio
    # ------------------------------------------------------------------------

    d["Fuel_Speed_Ratio"] = (

        d["Fuel_Consumption_L_100km"]

        /

        (d["Vehicle_Speed_kmh"] + 1)

    )


    # ------------------------------------------------------------------------
    # 7. Thermal Ratio
    # ------------------------------------------------------------------------

    d["Thermal_Ratio"] = (

        d["Engine_Temperature_C"]

        /

        (d["Coolant_Temperature_C"] + 1)

    )


    return d


# ============================================================================
# CVHS
# ============================================================================

def compute_cvhs(row: dict) -> float:

    """
    Composite Vehicle Health Score.

    This is a DOMAIN-KNOWLEDGE score.
    It is independent of the XGBoost prediction.
    """

    scores = []

    total_weight = sum(
        WEIGHTS.values()
    )

    for feat, (lo, hi) in SAFE_RANGES.items():

        val = float(
            row.get(
                feat,
                (lo + hi) / 2
            )
        )

        mid = (
            lo + hi
        ) / 2.0

        half = (
            hi - lo
        ) / 2.0

        health = float(
            np.clip(
                1.0
                -
                abs(val - mid) / half,
                0.0,
                1.0
            )
        )

        scores.append(
            health * WEIGHTS[feat]
        )

    return round(
        sum(scores) / total_weight,
        4
    )


# ============================================================================
# CVHS TIER
# ============================================================================

def cvhs_tier(cvhs: float) -> str:

    if cvhs >= 0.80:
        return "Healthy"

    if cvhs >= 0.60:
        return "Monitor"

    if cvhs >= 0.40:
        return "At Risk"

    return "Critical"


# ============================================================================
# MAINTENANCE ACTIONS
# ============================================================================

def get_actions(row: dict) -> list:

    actions = []

    for feat, checks in ACTION_MAP.items():

        val = float(
            row.get(feat, 0)
        )

        for severity, (
            threshold,
            message
        ) in checks.items():

            if (
                severity == "high"
                and val > threshold
            ):

                actions.append(message)

            elif (
                severity == "low"
                and val < threshold
            ):

                actions.append(message)

    if not actions:

        return [
            "No specific threshold violations detected"
        ]

    return actions


# ============================================================================
# SHAP EXPLANATION
# ============================================================================

def get_shap_explanation(
    X_scaled: np.ndarray,
    X_engineered: pd.DataFrame
):
    """
    Calculate SHAP contributions for one prediction.

    Returns the largest contributors to the model prediction.
    """

    if not SHAP_AVAILABLE:

        return []


    try:

        # ------------------------------------------------------------
        # Create TreeExplainer
        # ------------------------------------------------------------

        explainer = shap.TreeExplainer(
            MODEL
        )


        # ------------------------------------------------------------
        # Calculate SHAP values
        # ------------------------------------------------------------

        shap_values = explainer.shap_values(
            X_scaled
        )


        # ------------------------------------------------------------
        # Handle different SHAP return formats
        # ------------------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            # Binary classification:
            # usually class 1 is failure
            shap_row = np.asarray(
                shap_values[-1]
            )[0]

        else:

            shap_array = np.asarray(
                shap_values
            )

            # Some SHAP versions return:
            # (samples, features)
            #
            # Others can return:
            # (samples, features, classes)

            if shap_array.ndim == 3:

                shap_row = shap_array[
                    0,
                    :,
                    -1
                ]

            else:

                shap_row = shap_array[0]


        # ------------------------------------------------------------
        # Sort by absolute impact
        # ------------------------------------------------------------

        order = np.argsort(
            np.abs(shap_row)
        )[::-1]


        explanations = []


        for idx in order[:7]:

            feature = FEATURES[idx]

            shap_value = float(
                shap_row[idx]
            )


            # Original engineered feature value
            if feature in X_engineered.columns:

                original_value = float(
                    X_engineered.iloc[0][feature]
                )

            else:

                original_value = None


            direction = (
                "increases_failure_risk"
                if shap_value > 0
                else
                "decreases_failure_risk"
            )


            explanations.append({

                "feature":
                    feature,

                "value":
                    original_value,

                "shap_value":
                    round(
                        shap_value,
                        6
                    ),

                "direction":
                    direction,

            })


        return explanations


    except Exception as e:

        print(
            "\n⚠ SHAP calculation failed:"
        )

        print(
            str(e)
        )

        return []


# ============================================================================
# PRINT DEBUG INFORMATION
# ============================================================================

def print_debug_prediction(
    raw_row,
    df_engineered,
    X_scaled,
    probability,
    shap_explanations
):

    print("\n")
    print("=" * 80)
    print("MODEL DEBUG — SINGLE VEHICLE")
    print("=" * 80)


    # ------------------------------------------------------------------------
    # Raw input
    # ------------------------------------------------------------------------

    print("\nRAW INPUT:")

    for feature in RAW_FEATURES:

        print(
            f"  {feature:35s}: "
            f"{raw_row[feature]}"
        )


    # ------------------------------------------------------------------------
    # Engineered features
    # ------------------------------------------------------------------------

    print("\nENGINEERED FEATURES:")

    engineered_features = [
        "Thermal_Stress",
        "Engine_Load_Proxy",
        "Oil_Press_per_RPM",
        "Vibration_per_RPM",
        "Age_Temp_Interaction",
        "Fuel_Speed_Ratio",
        "Thermal_Ratio",
    ]


    for feature in engineered_features:

        value = float(
            df_engineered.iloc[0][feature]
        )

        print(
            f"  {feature:35s}: "
            f"{value:.6f}"
        )


    # ------------------------------------------------------------------------
    # Model probability
    # ------------------------------------------------------------------------

    print("\nMODEL OUTPUT:")

    print(
        f"  Failure probability : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"  Threshold 0.01      : "
        f"{int(probability >= 0.01)}"
    )

    print(
        f"  Threshold 0.50      : "
        f"{int(probability >= 0.50)}"
    )


    # ------------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------------

    print("\nTOP SHAP CONTRIBUTORS:")

    if not shap_explanations:

        print(
            "  SHAP information unavailable."
        )

    else:

        for rank, item in enumerate(
            shap_explanations,
            start=1
        ):

            arrow = (
                "↑"
                if item["shap_value"] > 0
                else
                "↓"
            )

            print(
                f"  #{rank} "
                f"{item['feature']:30s} "
                f"value={str(item['value']):>10s} "
                f"SHAP={item['shap_value']:+.6f} "
                f"{arrow}"
            )


    print("=" * 80)
    print()


# ============================================================================
# SINGLE PREDICTION
# ============================================================================

def predict_single(
    raw_row: dict
) -> dict:

    """
    Complete prediction pipeline for one vehicle.
    """

    # ------------------------------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------------------------------

    df = pd.DataFrame(
        [raw_row]
    )[
        RAW_FEATURES
    ]


    # ------------------------------------------------------------------------
    # Engineer features
    # ------------------------------------------------------------------------

    df_eng = engineer_features(
        df
    )


    # ------------------------------------------------------------------------
    # Ensure correct feature order
    # ------------------------------------------------------------------------

    missing_features = [
        f
        for f in FEATURES
        if f not in df_eng.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing model features: "
            +
            str(missing_features)
        )


    X = df_eng[
        FEATURES
    ]


    # ------------------------------------------------------------------------
    # Scale exactly like training
    # ------------------------------------------------------------------------

    X_sc = SCALER.transform(
        X
    )


    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------

    prob = float(
        MODEL.predict_proba(
            X_sc
        )[0, 1]
    )


    # ------------------------------------------------------------------------
    # Threshold predictions
    # ------------------------------------------------------------------------

    pred_opt = int(
        prob >= OPTIMAL_THR
    )

    pred_def = int(
        prob >= 0.50
    )


    # ------------------------------------------------------------------------
    # CVHS
    # ------------------------------------------------------------------------

    cvhs = compute_cvhs(
        raw_row
    )

    tier = cvhs_tier(
        cvhs
    )


    # ------------------------------------------------------------------------
    # Maintenance actions
    # ------------------------------------------------------------------------

    actions = get_actions(
        raw_row
    )


    # ------------------------------------------------------------------------
    # Risk level
    # ------------------------------------------------------------------------

    if prob > 0.70:

        risk = "Critical"

    elif prob > 0.50:

        risk = "High"

    elif prob > 0.30:

        risk = "Medium"

    else:

        risk = "Low"


    # ------------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------------

    shap_explanations = (
        get_shap_explanation(
            X_sc,
            df_eng
        )
    )


    # ------------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------------

    print_debug_prediction(
        raw_row=raw_row,
        df_engineered=df_eng,
        X_scaled=X_sc,
        probability=prob,
        shap_explanations=shap_explanations
    )


    # ------------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------------

    return {

        "failure_probability":
            round(
                prob * 100,
                2
            ),

        "prediction_optimal":
            pred_opt,

        "prediction_default":
            pred_def,

        "optimal_threshold":
            OPTIMAL_THR,

        "risk_level":
            risk,

        "cvhs":
            cvhs,

        "cvhs_tier":
            tier,

        "recommended_actions":
            actions,

        "shap_explanations":
            shap_explanations,

    }


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_batch(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Batch prediction.

    IMPORTANT:
    Feature engineering uses the SAME fixed training parameters
    as single-row prediction.

    Therefore batch predictions no longer depend on:
        df['RPM'].max()
        df['Engine_Temperature_C'].max()

    from the uploaded CSV.
    """

    # ------------------------------------------------------------------------
    # Verify columns
    # ------------------------------------------------------------------------

    missing = [
        c
        for c in RAW_FEATURES
        if c not in df.columns
    ]


    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )


    # ------------------------------------------------------------------------
    # Keep raw features
    # ------------------------------------------------------------------------

    df_clean = df[
        RAW_FEATURES
    ].copy()


    # ------------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------------

    df_eng = engineer_features(
        df_clean
    )


    # ------------------------------------------------------------------------
    # Model features
    # ------------------------------------------------------------------------

    X = df_eng[
        FEATURES
    ]


    # ------------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------------

    X_sc = SCALER.transform(
        X
    )


    # ------------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------------

    probs = MODEL.predict_proba(
        X_sc
    )[:, 1]


    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------

    preds = (
        probs >= OPTIMAL_THR
    ).astype(int)


    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    df_out = df.copy()


    df_out[
        "Failure_Probability_%"
    ] = (
        probs * 100
    ).round(2)


    df_out[
        "Prediction"
    ] = preds


    # ------------------------------------------------------------------------
    # CVHS
    # ------------------------------------------------------------------------

    df_out[
        "CVHS"
    ] = [

        compute_cvhs(row)

        for _, row
        in df_clean.iterrows()

    ]


    df_out[
        "CVHS_Tier"
    ] = df_out[
        "CVHS"
    ].apply(
        cvhs_tier
    )


    # ------------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------------

    df_out[
        "Risk_Level"
    ] = pd.cut(

        probs,

        bins=[
            -0.001,
            0.30,
            0.50,
            0.70,
            1.001
        ],

        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ]

    )


    return df_out


# ============================================================================
# HOME PAGE
# ============================================================================

@app.route("/")
def index():

    return render_template(

        "index.html",

        features=RAW_FEATURES,

        safe_ranges=SAFE_RANGES,

        optimal_thr=OPTIMAL_THR

    )


# ============================================================================
# SINGLE PREDICTION API
# ============================================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    try:

        data = request.get_json(
            force=True
        )


        # ------------------------------------------------------------
        # Validate input
        # ------------------------------------------------------------

        if not data:

            return jsonify({

                "error":
                    "No JSON input received"

            }), 400


        # ------------------------------------------------------------
        # Convert values
        # ------------------------------------------------------------

        raw = {

            f:
            float(data[f])

            for f in RAW_FEATURES

        }


        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        result = predict_single(
            raw
        )


        # ------------------------------------------------------------
        # Include input
        # ------------------------------------------------------------

        result["input"] = raw


        return jsonify(
            result
        )


    except KeyError as e:

        return jsonify({

            "error":
                f"Missing field: {e}"

        }), 400


    except ValueError as e:

        return jsonify({

            "error":
                f"Invalid value: {e}"

        }), 400


    except Exception as e:

        print("\nERROR IN /api/predict")
        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================================
# BATCH PREDICTION API
# ============================================================================

@app.route(
    "/api/batch",
    methods=["POST"]
)
def api_batch():

    try:

        if "file" not in request.files:

            return jsonify({

                "error":
                    "No file uploaded"

            }), 400


        file = request.files[
            "file"
        ]


        # ------------------------------------------------------------
        # Read CSV
        # ------------------------------------------------------------

        try:

            df = pd.read_csv(
                file
            )

        except Exception as e:

            return jsonify({

                "error":
                    f"Could not parse CSV: {e}"

            }), 400


        # ------------------------------------------------------------
        # Validate columns
        # ------------------------------------------------------------

        missing = [

            c

            for c in RAW_FEATURES

            if c not in df.columns

        ]


        if missing:

            return jsonify({

                "error":
                    f"Missing columns: {missing}"

            }), 400


        # ------------------------------------------------------------
        # Predict
        # ------------------------------------------------------------

        df_out = predict_batch(
            df
        )


        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------

        n_total = len(
            df_out
        )


        n_fail = int(
            (
                df_out["Prediction"]
                == 1
            ).sum()
        )


        n_crit = int(
            (
                df_out["Risk_Level"]
                == "Critical"
            ).sum()
        )


        failure_rate = (

            round(
                n_fail
                /
                n_total
                *
                100,
                2
            )

            if n_total > 0

            else 0

        )


        # ------------------------------------------------------------
        # Preview
        # ------------------------------------------------------------

        preview = (

            df_out
            .head(200)
            .to_dict(
                orient="records"
            )

        )


        return jsonify({

            "total_vehicles":
                n_total,

            "predicted_failures":
                n_fail,

            "critical_risk":
                n_crit,

            "failure_rate_%":
                failure_rate,

            "preview":
                preview,

        })


    except Exception as e:

        print("\nERROR IN /api/batch")
        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================================
# BATCH DOWNLOAD API
# ============================================================================

@app.route(
    "/api/batch/download",
    methods=["POST"]
)
def api_batch_download():

    try:

        if "file" not in request.files:

            return jsonify({

                "error":
                    "No file uploaded"

            }), 400


        file = request.files[
            "file"
        ]


        # ------------------------------------------------------------
        # Read CSV
        # ------------------------------------------------------------

        df = pd.read_csv(
            file
        )


        # ------------------------------------------------------------
        # Predict
        # ------------------------------------------------------------

        df_out = predict_batch(
            df
        )


        # ------------------------------------------------------------
        # Convert to CSV
        # ------------------------------------------------------------

        buf = io.StringIO()

        df_out.to_csv(
            buf,
            index=False
        )

        buf.seek(0)


        # ------------------------------------------------------------
        # Download
        # ------------------------------------------------------------

        return send_file(

            io.BytesIO(
                buf.getvalue().encode()
            ),

            mimetype="text/csv",

            as_attachment=True,

            download_name=
                "vehicle_predictions.csv"

        )


    except Exception as e:

        print(
            "\nERROR IN /api/batch/download"
        )

        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================================
# MODEL INFORMATION API
# ============================================================================

@app.route(
    "/api/model_info"
)
def api_model_info():

    return jsonify({

        "model_type":
            type(MODEL).__name__,

        "n_estimators":
            getattr(
                MODEL,
                "n_estimators",
                None
            ),

        "features":
            list(FEATURES),

        "n_features":
            len(FEATURES),

        "optimal_threshold":
            OPTIMAL_THR,

        "default_threshold":
            0.50,

        "scaler":
            type(SCALER).__name__,

        "engineer_params":
            {

                "rpm_max":
                    RPM_MAX,

                "temp_max":
                    TEMP_MAX

            },

        "shap_available":
            SHAP_AVAILABLE,

    })


# ============================================================================
# OPTIONAL DEBUG API
# ============================================================================

@app.route(
    "/api/debug_prediction",
    methods=["POST"]
)
def api_debug_prediction():

    """
    Debug endpoint.

    Returns:
        - raw input
        - engineered features
        - scaled values
        - probability
        - SHAP contributors
    """

    try:

        data = request.get_json(
            force=True
        )


        raw = {

            f:
            float(data[f])

            for f in RAW_FEATURES

        }


        # ------------------------------------------------------------
        # Engineer
        # ------------------------------------------------------------

        df = pd.DataFrame(
            [raw]
        )[
            RAW_FEATURES
        ]


        df_eng = engineer_features(
            df
        )


        X = df_eng[
            FEATURES
        ]


        X_sc = SCALER.transform(
            X
        )


        probability = float(

            MODEL
            .predict_proba(
                X_sc
            )[0, 1]

        )


        # ------------------------------------------------------------
        # SHAP
        # ------------------------------------------------------------

        shap_explanations = (
            get_shap_explanation(
                X_sc,
                df_eng
            )
        )


        # ------------------------------------------------------------
        # Engineered values
        # ------------------------------------------------------------

        engineered = {}


        for feature in [

            "Thermal_Stress",

            "Engine_Load_Proxy",

            "Oil_Press_per_RPM",

            "Vibration_per_RPM",

            "Age_Temp_Interaction",

            "Fuel_Speed_Ratio",

            "Thermal_Ratio",

        ]:

            engineered[
                feature
            ] = float(
                df_eng.iloc[0][feature]
            )


        # ------------------------------------------------------------
        # Scaled values
        # ------------------------------------------------------------

        scaled = {

            feature:
            float(X_sc[0][i])

            for i, feature
            in enumerate(FEATURES)

        }


        return jsonify({

            "raw_input":
                raw,

            "engineered_features":
                engineered,

            "scaled_features":
                scaled,

            "failure_probability":
                round(
                    probability * 100,
                    4
                ),

            "prediction_0.01":
                int(
                    probability >= 0.01
                ),

            "prediction_0.50":
                int(
                    probability >= 0.50
                ),

            "shap_explanations":
                shap_explanations,

        })


    except Exception as e:

        print(
            "\nERROR IN /api/debug_prediction"
        )

        traceback.print_exc()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================================
# START SERVER
# ============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "Vehicle Health Monitoring — Local Test System"
    )

    print(
        "Open: http://localhost:5000"
    )

    print("=" * 60 + "\n")


    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )