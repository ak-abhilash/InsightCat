from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import io

app = FastAPI()

# Enable CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can limit to localhost:5173 for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_json(data):
    """Remove non-serializable values (like NaN, inf)"""
    return json.loads(json.dumps(data, default=str))

def analyze_dataframe(df):
    insights = []
    column_summaries = []

    for col in df.columns:
        col_data = df[col]
        summary = {"name": col}

        if pd.api.types.is_numeric_dtype(col_data):
            summary["type"] = "numeric"
            summary["insights"] = []

            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            null_count = col_data.isnull().sum()
            skewness = col_data.skew()

            insight_parts = []
            insight_parts.append(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std Dev: {std_val:.2f}")
            insight_parts.append(f"Nulls: {null_count}")
            if skewness > 1:
                insight_parts.append("Highly right-skewed")
            elif skewness < -1:
                insight_parts.append("Highly left-skewed")
            else:
                insight_parts.append("Roughly symmetrical")

            summary["insights"].append(". ".join(insight_parts))
            summary["suggested_charts"] = ["histogram", "boxplot"]

        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            summary["type"] = "categorical"
            summary["insights"] = []

            top_vals = col_data.value_counts(dropna=True).head(3).to_dict()
            total_unique = col_data.nunique()
            null_count = col_data.isnull().sum()

            if top_vals:
                dominant_val = max(top_vals.items(), key=lambda x: x[1])
            else:
                dominant_val = ("None", 0)

            insight_parts = []
            insight_parts.append(f"Top categories: {list(top_vals.keys())}")
            insight_parts.append(f"Most common: '{dominant_val[0]}' ({dominant_val[1]} occurrences)")
            insight_parts.append(f"Unique categories: {total_unique}")
            insight_parts.append(f"Nulls: {null_count}")

            summary["insights"].append(". ".join(insight_parts))
            summary["suggested_charts"] = ["bar", "pie"]

        else:
            summary["type"] = "other"
            summary["insights"] = [f"Unsupported column type: {col_data.dtype}"]
            summary["suggested_charts"] = []

        column_summaries.append(summary)

    # Global insights
    global_insights = []
    missing_report = df.isnull().mean() * 100
    high_missing = missing_report[missing_report > 30]
    if not high_missing.empty:
        global_insights.append(f"{len(high_missing)} columns have >30% missing values: {list(high_missing.index)}")

    if "target" in df.columns:
        target_type = df["target"].dtype
        if target_type == object or df["target"].nunique() <= 10:
            global_insights.append("Target appears suitable for classification.")
        elif pd.api.types.is_numeric_dtype(df["target"]):
            global_insights.append("Target appears suitable for regression.")

    return clean_json({
        "columns": column_summaries,
        "global_insights": global_insights
    })

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        result = analyze_dataframe(df)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})