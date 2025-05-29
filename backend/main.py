from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import json
import uvicorn

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(title="InsightCat API", description="Data Analysis and Visualization API")

origins = [
    "https://insight-cat.vercel.app",
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "InsightCat API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_key_configured": bool(OPENROUTER_API_KEY)}


def get_data_overview(df):
    """Generate a comprehensive data overview"""
    try:
        overview = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_info": []
        }
        
        for col in df.columns:
            try:
                col_info = {
                    "name": str(col),
                    "type": "text",
                    "non_null_count": int(df[col].notna().sum()),
                    "null_count": int(df[col].isna().sum()),
                    "unique_count": 0
                }
                
                # Try to get unique count safely
                try:
                    col_info["unique_count"] = int(df[col].nunique())
                except:
                    col_info["unique_count"] = 0
                
                # Determine column type
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    col_info["type"] = "numeric"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_info["type"] = "datetime"
                else:
                    # Check if it's numeric but stored as object
                    try:
                        numeric_version = pd.to_numeric(df[col], errors='coerce')
                        if numeric_version.notna().sum() > len(df) * 0.5:
                            col_info["type"] = "numeric"
                        else:
                            col_info["type"] = "categorical"
                    except:
                        col_info["type"] = "categorical"
                
                overview["column_info"].append(col_info)
                
            except Exception as e:
                # If anything fails for this column, add basic info
                overview["column_info"].append({
                    "name": str(col),
                    "type": "text",
                    "non_null_count": 0,
                    "null_count": len(df),
                    "unique_count": 0
                })
        
        return overview
        
    except Exception as e:
        # Return minimal overview if everything fails
        return {
            "total_rows": len(df) if df is not None else 0,
            "total_columns": len(df.columns) if df is not None else 0,
            "column_info": []
        }


def get_data_quality(df):
    """Generate data quality assessment that matches frontend expectations"""
    try:
        total_cells = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        missing_percentage = round((total_missing / total_cells) * 100, 1) if total_cells > 0 else 0
        
        # Get duplicate rows
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = round((duplicate_rows / len(df)) * 100, 1) if len(df) > 0 else 0
        
        # Calculate quality score
        quality_score = 100
        quality_score -= min(missing_percentage * 2, 40)  # Penalize missing values
        quality_score -= min(duplicate_percentage * 3, 30)  # Penalize duplicates
        quality_score = max(0, round(quality_score))
        
        # Determine status
        if quality_score >= 90:
            status = 'excellent'
        elif quality_score >= 70:
            status = 'good'
        elif quality_score >= 50:
            status = 'needs_attention'
        else:
            status = 'poor'
        
        # Get columns with missing values
        columns_with_missing = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                columns_with_missing[col] = int(missing_count)
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": int(total_missing),
            "missing_percentage": missing_percentage,
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": duplicate_percentage,
            "quality_score": quality_score,
            "status": status,
            "columns_with_missing": columns_with_missing
        }
        
    except Exception as e:
        print(f"Error calculating data quality: {e}")
        return {
            "total_rows": len(df) if df is not None else 0,
            "total_columns": len(df.columns) if df is not None else 0,
            "missing_values": 0,
            "missing_percentage": 0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0,
            "quality_score": 0,
            "status": "unknown",
            "columns_with_missing": {}
        }


def analyze_column_relevance(df: pd.DataFrame, col: str, col_type: str):
    try:
        non_null_count = df[col].count()
        total_count = len(df)
        null_percentage = (total_count - non_null_count) / total_count * 100
        
        if null_percentage > 70:
            return False, None, f"Too many missing values ({null_percentage:.1f}%)"
        
        if col_type == 'numeric':
            numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
            unique_count = len(numeric_data.unique())
            
            if unique_count < 3:
                return False, None, f"Not enough variation ({unique_count} unique values)"
            
            if numeric_data.std() == 0:
                return False, None, "All values are identical"
            
            if unique_count > 20:
                return True, 'histogram', f"Good distribution with {unique_count} unique values"
            else:
                return True, 'bar_numeric', f"Discrete numeric data with {unique_count} values"
                
        elif col_type == 'categorical':
            try:
                value_counts = df[col].value_counts()
            except (TypeError, ValueError):
                string_col = df[col].astype(str)
                value_counts = string_col.value_counts()
            
            unique_count = len(value_counts)
            
            if unique_count > 50:
                return False, None, f"Too many categories ({unique_count}), likely IDs or free text"
            
            if unique_count < 2:
                return False, None, "Only one category"
            
            if unique_count / non_null_count > 0.8:
                return False, None, "Most values are unique, likely identifiers"
            
            top_category_pct = value_counts.iloc[0] / non_null_count * 100
            if top_category_pct > 95:
                return False, None, f"One category dominates ({top_category_pct:.1f}%)"
            
            return True, 'bar_categorical', f"Good categorical distribution with {unique_count} categories"
            
    except Exception as e:
        print(f"Error analyzing column {col}: {e}")
        return False, None, f"Analysis error: {str(e)}"
    
    return False, None, "Unknown column type"


def generate_smart_charts(df: pd.DataFrame, max_charts: int = 6):
    charts = []
    chart_candidates = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in numeric_cols:
        should_viz, chart_type, reason = analyze_column_relevance(df, col, 'numeric')
        if should_viz:
            non_null_count = df[col].count()
            unique_ratio = len(df[col].dropna().unique()) / non_null_count if non_null_count > 0 else 0
            priority = non_null_count * unique_ratio
            
            chart_candidates.append({
                'column': col,
                'type': chart_type,
                'priority': priority,
                'reason': reason,
                'data_type': 'numeric'
            })
    
    for col in categorical_cols:
        should_viz, chart_type, reason = analyze_column_relevance(df, col, 'categorical')
        if should_viz:
            try:
                value_counts = df[col].value_counts()
            except (TypeError, ValueError):
                value_counts = df[col].astype(str).value_counts()
            
            proportions = [count/len(df) for count in value_counts]
            entropy = -sum([p * np.log2(p) for p in proportions if p > 0])
            priority = entropy * len(value_counts)
            
            chart_candidates.append({
                'column': col,
                'type': chart_type,
                'priority': priority,
                'reason': reason,
                'data_type': 'categorical'
            })
    
    chart_candidates.sort(key=lambda x: x['priority'], reverse=True)
    selected_charts = chart_candidates[:max_charts]
    
    print(f"Selected {len(selected_charts)} charts out of {len(chart_candidates)} candidates")
    
    for chart_info in selected_charts:
        try:
            col = chart_info['column']
            chart_type = chart_info['type']
            
            if chart_type == 'histogram':
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                plt.figure(figsize=(8, 5))
                bins = min(30, max(10, len(numeric_data.unique())//2))
                sns.histplot(numeric_data, bins=bins, kde=True, color="skyblue", alpha=0.7)
                plt.title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                plt.xlabel(col)
                plt.ylabel("Frequency")
                
            elif chart_type == 'bar_numeric':
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                value_counts = numeric_data.value_counts().sort_index()
                plt.figure(figsize=(8, 5))
                plt.bar(value_counts.index.astype(str), value_counts.values, color="lightcoral", alpha=0.8)
                plt.title(f"Count by {col}", fontsize=14, fontweight='bold')
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                
            elif chart_type == 'bar_categorical':
                try:
                    value_counts = df[col].value_counts().nlargest(15)
                except (TypeError, ValueError):
                    value_counts = df[col].astype(str).value_counts().nlargest(15)
                
                plt.figure(figsize=(10, 6))
                labels = [str(label)[:25] + '...' if len(str(label)) > 25 else str(label) 
                         for label in value_counts.index]
                
                bars = plt.barh(range(len(labels)), value_counts.values, color="mediumseagreen", alpha=0.8)
                plt.yticks(range(len(labels)), labels)
                plt.xlabel("Count")
                plt.title(f"Top Categories in {col}", fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width + max(value_counts.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center', fontsize=9)
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
            plt.close()
            
            charts.append({
                "title": f"{chart_info['type'].replace('_', ' ').title()}: {col}",
                "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
                "insight": chart_info['reason']
            })
            
        except Exception as e:
            print(f"Error creating chart for {col}: {e}")
            plt.close()
            continue
    
    return charts


def call_llm_insights_from_prompt(prompt: str):
    if not OPENROUTER_API_KEY:
        print("Warning: AI_API_KEY not configured")
        return None
        
    try:
        client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful, business-friendly data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return None


def read_uploaded_file(file: UploadFile):
    if not file.filename:
        raise ValueError("No filename provided")
    
    file_extension = file.filename.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            try:
                df = pd.read_csv(file.file, encoding='utf-8')
            except UnicodeDecodeError:
                file.file.seek(0)
                df = pd.read_csv(file.file, encoding='latin-1')
        
        elif file_extension in ['xlsx', 'xls']:
            file_content = file.file.read()
            file.file.seek(0)
            excel_buffer = io.BytesIO(file_content)
            
            try:
                df = pd.read_excel(excel_buffer, engine='openpyxl')
            except Exception as e:
                excel_buffer.seek(0)
                try:
                    df = pd.read_excel(excel_buffer, engine='xlrd')
                except Exception:
                    excel_buffer.seek(0)
                    df = pd.read_excel(excel_buffer)
            
        elif file_extension == 'json':
            content = file.file.read()
            file.file.seek(0)
            
            if isinstance(content, bytes):
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    content_str = content.decode('latin-1')
            else:
                content_str = content
            
            content_str = content_str.strip()
            
            try:
                json_data = json.loads(content_str)
            except json.JSONDecodeError as e:
                try:
                    json_objects = []
                    for line in content_str.split('\n'):
                        line = line.strip()
                        if line:
                            json_objects.append(json.loads(line))
                    json_data = json_objects
                except json.JSONDecodeError:
                    try:
                        cleaned_content = content_str.rstrip(',').rstrip()
                        if not cleaned_content.endswith('}') and not cleaned_content.endswith(']'):
                            if cleaned_content.startswith('['):
                                cleaned_content += ']'
                            elif cleaned_content.startswith('{'):
                                cleaned_content += '}'
                        json_data = json.loads(cleaned_content)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON format. Please ensure your JSON file is properly formatted. Error: {str(e)}")
            
            try:
                if isinstance(json_data, list):
                    if len(json_data) == 0:
                        raise ValueError("JSON file contains an empty array")
                    
                    if isinstance(json_data[0], dict):
                        df = pd.DataFrame(json_data)
                    else:
                        df = pd.DataFrame(json_data, columns=['value'])
                
                elif isinstance(json_data, dict):
                    try:
                        if any(isinstance(v, list) for v in json_data.values()):
                            df = pd.DataFrame(json_data)
                        else:
                            df = pd.DataFrame([json_data])
                    except ValueError:
                        df = pd.json_normalize(json_data)
                
                else:
                    df = pd.DataFrame([json_data], columns=['value'])
                    
            except Exception as conversion_error:
                raise ValueError(f"Unable to convert JSON to tabular format: {str(conversion_error)}")
        
        else:
            raise ValueError(f"Unsupported file format: .{file_extension}. Please upload CSV, Excel (.xlsx/.xls), or JSON files.")
    
    except Exception as e:
        if "Unsupported file format" in str(e) or "Invalid JSON format" in str(e) or "Unable to" in str(e):
            raise e
        else:
            raise ValueError(f"Error reading {file_extension.upper()} file: {str(e)}")
    
    return df


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        try:
            df = read_uploaded_file(file)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"File parsing failed: {str(e)}"})

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "Uploaded file contains no data."})

        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "No valid data found after cleaning."})

        for col in df.columns:
            sample_values = df[col].dropna().head(5)
            if len(sample_values) > 0:
                first_val = sample_values.iloc[0]
                if isinstance(first_val, (dict, list)):
                    df[col] = df[col].astype(str)

        # Generate data overview and quality
        overview = get_data_overview(df)
        data_quality = get_data_quality(df)

        df_sample = df.iloc[:5, :20]
        
        try:
            dtypes_md = df.dtypes.reset_index()
            dtypes_md.columns = ["Column", "Type"]
            dtypes_md_str = dtypes_md.to_markdown(index=False)
        except Exception as e:
            print(f"Error creating dtypes summary: {e}")
            dtypes_md_str = "Could not generate column types summary"

        prompt = f"""
You're a professional data analyst. A user has uploaded this dataset sample:

{df_sample.to_markdown(index=False)}

Dataset columns and types:

{dtypes_md_str}

Dataset info:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}

Please write **5 cool, casual, human-friendly insights** about the data. Follow these exact formatting rules:

- Each insight must start with the emoji 🔍 on its own line.
- The next line starts with 📊 then a short catchy title, a hyphen, and a brief observation (one sentence).
- Then on a new line, write: "- Why it matters:" followed by a short sentence.
- Then on a new line, write: "- Suggested action:" followed by a short sentence.
- Put a blank line between insights (two newlines total).
- Use simple, non-technical language. Make it sound friendly and helpful.
- Do NOT write insights as paragraphs or multiple insights on one line.
"""

        insights_raw = call_llm_insights_from_prompt(prompt)
        charts = generate_smart_charts(df, max_charts=6)

        insight_blocks = []
        if insights_raw:
            for line in insights_raw.strip().split("\n"):
                if line.startswith(("📊", "🤔", "💡", "📈", "📉")):
                    insight_blocks.append(line)
                elif insight_blocks:
                    insight_blocks[-1] += "\n" + line
                else:
                    insight_blocks.append(line)

        if not insight_blocks:
            insight_blocks = ["No insights generated. Please check the dataset or AI key."]

        if not charts:
            charts = [{"title": "No charts generated", "image": ""}]

        return {
            "insights": insight_blocks,
            "charts": charts,
            "overview": overview,
            "data_quality": data_quality,  # Added this line
            "file_info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "file_type": file.filename.split('.')[-1].upper()
            }
        }

    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        reload=False
    )