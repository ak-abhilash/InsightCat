from fastapi import FastAPI, File, UploadFile, Request
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
import warnings
from typing import Optional, Dict, Any, List, Tuple
import gc
import psutil
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

warnings.filterwarnings('ignore')

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(title="InsightCat API", description="Data Analysis and Visualization API")

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Add rate limit exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.on_event("startup")
async def verify_api_key():
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY is not set. AI insights will not work.")
    else:
        print("OpenRouter API Key loaded.")

# CORS setup for web deployment
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

# Memory management limits to prevent server overload
MAX_ROWS = 50000
MAX_COLS = 100
SAMPLE_SIZE = 1000

@app.get("/")
async def root():
    return {"message": "InsightCat API is running!", "status": "healthy"}

@app.middleware("http")
async def add_memory_guard(request: Request, call_next):
    mem = psutil.virtual_memory()
    if mem.available < 500 * 1024 * 1024:  # 500 MB threshold
        return JSONResponse(status_code=503, content={"error": "System memory too low for processing"})
    response = await call_next(request)
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_key_configured": bool(OPENROUTER_API_KEY)}


def safe_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to convert string columns to numeric if they contain numeric data.
    Handles common formatting like commas, dollar signs, percentages.
    """
    converted_df = df.copy()
    
    for col in converted_df.columns:
        try:
            if pd.api.types.is_numeric_dtype(converted_df[col]):
                continue
                
            sample = converted_df[col].dropna().head(100)
            if len(sample) > 0:
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                        numeric_count += 1
                    except:
                        pass

                if numeric_count / len(sample) > 0.7:
                    try:
                        cleaned = converted_df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                        converted_df[col] = pd.to_numeric(cleaned, errors='ignore')
                    except:
                        pass
                        
        except Exception as e:
            print(f"Warning: Could not process column {col}: {e}")
            continue
    
    return converted_df

def safe_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to convert string columns to numeric if they contain numeric data.
    Handles common formatting like commas, dollar signs, percentages.
    """
    converted_df = df.copy()
    
    for col in converted_df.columns:
        try:
            if pd.api.types.is_numeric_dtype(converted_df[col]):
                continue
                
            sample = converted_df[col].dropna().head(100)
            if len(sample) > 0:
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                        numeric_count += 1
                    except:
                        pass

                if numeric_count / len(sample) > 0.7:
                    try:
                        cleaned = converted_df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                        converted_df[col] = pd.to_numeric(cleaned, errors='ignore')
                    except:
                        pass
                        
        except Exception as e:
            print(f"Warning: Could not process column {col}: {e}")
            continue
    
    return converted_df


def get_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates basic statistics about the dataset structure.
    Returns column info with types, null counts, and unique values.
    """
    try:
        total_missing = 0
        try:
            total_missing = int(df.isnull().sum().sum())
        except Exception as e:
            print(f"Warning: Could not calculate total missing values: {e}")

        overview = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "total_missing_values": total_missing,
            "missing_percentage": round((total_missing / (len(df) * len(df.columns))) * 100, 1) if len(df) > 0 and len(df.columns) > 0 else 0,
            "column_info": []
        }
        
        for col in df.columns:
            try:
                total_count = len(df)
                non_null_count = int(df[col].notna().sum())
                null_count = total_count - non_null_count
                
                col_info = {
                    "name": str(col)[:100],
                    "type": "text",
                    "non_null_count": non_null_count,
                    "null_count": null_count,
                    "unique_count": 0
                }
                
                # Calculate unique count with sampling for large datasets
                try:
                    if non_null_count > 0:
                        if non_null_count > 10000:
                            sample = df[col].dropna().sample(n=min(10000, non_null_count), random_state=42)
                            estimated_unique = len(sample.unique())
                            col_info["unique_count"] = min(estimated_unique, non_null_count)
                        else:
                            col_info["unique_count"] = int(df[col].nunique())
                except Exception as e:
                    print(f"Warning: Could not calculate unique count for {col}: {e}")
                    col_info["unique_count"] = 0
                
                # Determine column type with heuristics
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_info["type"] = "numeric"
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        col_info["type"] = "datetime"
                    else:
                        # Check if values might be numeric by sampling
                        sample_values = df[col].dropna().head(50)
                        if len(sample_values) > 0:
                            numeric_count = 0
                            for val in sample_values:
                                try:
                                    float(str(val))
                                    numeric_count += 1
                                except:
                                    pass
                            
                            if numeric_count / len(sample_values) > 0.8:
                                col_info["type"] = "numeric"
                            elif col_info["unique_count"] < non_null_count * 0.5:
                                col_info["type"] = "categorical"
                            else:
                                col_info["type"] = "text"
                        else:
                            col_info["type"] = "text"
                except Exception as e:
                    print(f"Warning: Type detection failed for {col}: {e}")
                    col_info["type"] = "text"
                
                overview["column_info"].append(col_info)
                
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                # Add fallback entry for problematic columns
                overview["column_info"].append({
                    "name": str(col)[:100],
                    "type": "text",
                    "non_null_count": 0,
                    "null_count": len(df),
                    "unique_count": 0
                })
        
        return overview
        
    except Exception as e:
        print(f"Error in get_data_overview: {e}")
        return {
            "total_rows": len(df) if df is not None and not df.empty else 0,
            "total_columns": len(df.columns) if df is not None and not df.empty else 0,
            "column_info": []
        }


def get_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates data quality metrics including missing values, duplicates,
    and overall quality score (0-100).
    """
    try:
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        
        if total_cells == 0:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "missing_values": 0,
                "missing_percentage": 0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0,
                "quality_score": 0,
                "status": "no_data",
                "columns_with_missing": {}
            }
        
        # Calculate missing values
        try:
            total_missing = int(df.isnull().sum().sum())
            missing_percentage = round((total_missing / total_cells) * 100, 1)
        except Exception as e:
            print(f"Warning: Could not calculate missing values: {e}")
            total_missing = 0
            missing_percentage = 0
        
        # Calculate duplicates with sampling for large datasets
        try:
            if total_rows > 10000:
                sample_df = df.sample(n=min(5000, total_rows), random_state=42)
                sample_duplicates = sample_df.duplicated().sum()
                duplicate_rows = int((sample_duplicates / len(sample_df)) * total_rows)
            else:
                duplicate_rows = int(df.duplicated().sum())
            
            duplicate_percentage = round((duplicate_rows / total_rows) * 100, 1) if total_rows > 0 else 0
        except Exception as e:
            print(f"Warning: Could not calculate duplicates: {e}")
            duplicate_rows = 0
            duplicate_percentage = 0
        
        # Calculate quality score (0-100)
        quality_score = 100
        quality_score -= min(missing_percentage * 2, 40)  
        quality_score -= min(duplicate_percentage * 3, 30)
        quality_score = max(0, round(quality_score))
        
        # Status based on quality score
        if quality_score >= 90:
            status = 'excellent'
        elif quality_score >= 70:
            status = 'good'
        elif quality_score >= 50:
            status = 'needs_attention'
        else:
            status = 'poor'
        
        # Identify columns with missing values
        columns_with_missing = {}
        try:
            for col in df.columns:
                try:
                    missing_count = int(df[col].isnull().sum())
                    if missing_count > 0:
                        columns_with_missing[str(col)[:100]] = missing_count
                except Exception as e:
                    print(f"Warning: Could not check missing values for column {col}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not process columns for missing values: {e}")
        
        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "missing_values": total_missing,
            "missing_percentage": missing_percentage,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "quality_score": quality_score,
            "status": status,
            "columns_with_missing": columns_with_missing
        }
        
    except Exception as e:
        print(f"Error calculating data quality: {e}")
        return {
            "total_rows": len(df) if df is not None and not df.empty else 0,
            "total_columns": len(df.columns) if df is not None and not df.empty else 0,
            "missing_values": 0,
            "missing_percentage": 0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0,
            "quality_score": 0,
            "status": "error",
            "columns_with_missing": {}
        }


def analyze_column_relevance(df: pd.DataFrame, col: str, col_type: str) -> Tuple[bool, Optional[str], str]:
    """
    Determines if a column is suitable for visualization.
    Returns (should_visualize, chart_type, reason).
    """
    try:
        non_null_count = df[col].count()
        total_count = len(df)
        
        if non_null_count == 0:
            return False, None, "Column is empty"
        
        null_percentage = (total_count - non_null_count) / total_count * 100
        
        if null_percentage > 80:
            return False, None, f"Too many missing values ({null_percentage:.1f}%)"
        
        if col_type == 'numeric':
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(numeric_data) < 3:
                    return False, None, "Not enough numeric values"
                
                unique_count = len(numeric_data.unique())
                
                if unique_count < 2:
                    return False, None, f"Not enough variation ({unique_count} unique values)"
                
                # Check for zero variance
                try:
                    if numeric_data.std() == 0:
                        return False, None, "All values are identical"
                except:
                    pass
                
                # Choose chart type based on unique value count
                if unique_count > 50:
                    return True, 'histogram', f"Continuous distribution ({unique_count} unique values)"
                else:
                    return True, 'bar_numeric', f"Discrete numeric ({unique_count} values)"
                    
            except Exception as e:
                print(f"Error analyzing numeric column {col}: {e}")
                return False, None, f"Numeric analysis error"
                
        elif col_type == 'categorical':
            try:
                sample_data = df[col].dropna()
                if len(sample_data) > 5000:
                    sample_data = sample_data.sample(n=5000, random_state=42)
                
                try:
                    value_counts = sample_data.value_counts()
                except (TypeError, ValueError, AttributeError):
                    # Fallback to string conversion if value_counts fails
                    string_col = sample_data.astype(str)
                    value_counts = string_col.value_counts()
                
                unique_count = len(value_counts)
                
                if unique_count > 50:
                    return False, None, f"Too many categories ({unique_count})"
                
                if unique_count < 2:
                    return False, None, "Only one category"
                
                # Skip columns that are mostly unique (likely IDs)
                if unique_count / len(sample_data) > 0.9:
                    return False, None, "Mostly unique values (likely IDs)"
                
                # Skip if one category heavily dominates
                top_category_pct = value_counts.iloc[0] / len(sample_data) * 100
                if top_category_pct > 98:
                    return False, None, f"One category dominates ({top_category_pct:.1f}%)"
                
                return True, 'bar_categorical', f"Good categorical distribution ({unique_count} categories)"
                
            except Exception as e:
                print(f"Error analyzing categorical column {col}: {e}")
                return False, None, f"Categorical analysis error"
            
    except Exception as e:
        print(f"Error analyzing column {col}: {e}")
        return False, None, f"Analysis error: {str(e)}"
    
    return False, None, "Unknown column type"


def generate_smart_charts(df: pd.DataFrame, max_charts: int = 6) -> List[Dict[str, str]]:
    """
    Automatically generates the most relevant charts based on data analysis.
    Prioritizes columns with high information content and good distribution.
    """
    charts = []
    chart_candidates = []
    
    try:
        # Use sample for large datasets to improve performance
        working_df = df
        if len(df) > SAMPLE_SIZE:
            working_df = df.sample(n=SAMPLE_SIZE, random_state=42)
        
        # Categorize columns by type
        numeric_cols = []
        categorical_cols = []
        
        for col in working_df.columns:
            try:
                if pd.api.types.is_numeric_dtype(working_df[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except Exception as e:
                print(f"Warning: Could not determine type for column {col}: {e}")
                categorical_cols.append(col)
        
        # Analyze numeric columns for visualization potential
        for col in numeric_cols[:20]: 
            try:
                should_viz, chart_type, reason = analyze_column_relevance(working_df, col, 'numeric')
                if should_viz:
                    non_null_count = working_df[col].count()
                    unique_ratio = len(working_df[col].dropna().unique()) / max(non_null_count, 1)
                    priority = non_null_count * unique_ratio
                    
                    chart_candidates.append({
                        'column': col,
                        'type': chart_type,
                        'priority': priority,
                        'reason': reason,
                        'data_type': 'numeric'
                    })
            except Exception as e:
                print(f"Error analyzing numeric column {col}: {e}")
                continue
        
        # Analyze categorical columns
        for col in categorical_cols[:20]:
            try:
                should_viz, chart_type, reason = analyze_column_relevance(working_df, col, 'categorical')
                if should_viz:
                    try:
                        sample_col = working_df[col].dropna()
                        if len(sample_col) > 1000:
                            sample_col = sample_col.sample(n=1000, random_state=42)
                        
                        try:
                            value_counts = sample_col.value_counts()
                        except (TypeError, ValueError):
                            value_counts = sample_col.astype(str).value_counts()
                        
                        if len(value_counts) > 0:
                            # Calculate entropy as priority metric (higher entropy = more interesting distribution)
                            proportions = [count/len(sample_col) for count in value_counts]
                            entropy = -sum([p * np.log2(p) for p in proportions if p > 0])
                            priority = entropy * len(value_counts)
                            
                            chart_candidates.append({
                                'column': col,
                                'type': chart_type,
                                'priority': priority,
                                'reason': reason,
                                'data_type': 'categorical'
                            })
                    except Exception as e:
                        print(f"Error calculating priority for {col}: {e}")
                        continue
            except Exception as e:
                print(f"Error analyzing categorical column {col}: {e}")
                continue
        
        # Sort by priority and select top candidates
        chart_candidates.sort(key=lambda x: x['priority'], reverse=True)
        selected_charts = chart_candidates[:max_charts]
        
        print(f"Selected {len(selected_charts)} charts out of {len(chart_candidates)} candidates")
        
        # Generate actual chart images
        for chart_info in selected_charts:
            try:
                col = chart_info['column']
                chart_type = chart_info['type']
                
                plt.style.use('default')
                plt.figure(figsize=(10, 6))
                
                if chart_type == 'histogram':
                    try:
                        numeric_data = pd.to_numeric(working_df[col], errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            bins = min(30, max(5, len(numeric_data.unique())//2))
                            plt.hist(numeric_data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                            plt.title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                            plt.xlabel(col)
                            plt.ylabel("Frequency")
                            plt.grid(True, alpha=0.3)
                    except Exception as e:
                        print(f"Error creating histogram for {col}: {e}")
                        plt.close()
                        continue
                        
                elif chart_type == 'bar_numeric':
                    try:
                        numeric_data = pd.to_numeric(working_df[col], errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            value_counts = numeric_data.value_counts().sort_index().head(20)
                            plt.bar(range(len(value_counts)), value_counts.values, 
                                   color="lightcoral", alpha=0.8)
                            plt.title(f"Count by {col}", fontsize=14, fontweight='bold')
                            plt.xlabel(col)
                            plt.ylabel("Count")
                            plt.xticks(range(len(value_counts)), 
                                     [str(x) for x in value_counts.index], rotation=45)
                            plt.grid(True, alpha=0.3)
                    except Exception as e:
                        print(f"Error creating bar chart for {col}: {e}")
                        plt.close()
                        continue
                        
                elif chart_type == 'bar_categorical':
                    try:
                        sample_col = working_df[col].dropna()
                        if len(sample_col) > 1000:
                            sample_col = sample_col.sample(n=1000, random_state=42)
                        
                        try:
                            value_counts = sample_col.value_counts().head(15)
                        except (TypeError, ValueError):
                            value_counts = sample_col.astype(str).value_counts().head(15)
                        
                        if len(value_counts) > 0:
                            plt.figure(figsize=(max(8, len(value_counts) * 0.5), 6))
                            
                            # Truncate long labels
                            labels = [str(label)[:30] + '...' if len(str(label)) > 30 else str(label) 
                                     for label in value_counts.index]
                            
                            bars = plt.barh(range(len(labels)), value_counts.values, 
                                          color="mediumseagreen", alpha=0.8)
                            plt.yticks(range(len(labels)), labels)
                            plt.xlabel("Count")
                            plt.title(f"Top Categories in {col}", fontsize=14, fontweight='bold')
                            plt.gca().invert_yaxis()
                            plt.grid(True, alpha=0.3, axis='x')
                            
                            # Add value labels on bars
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                plt.text(width + max(value_counts.values) * 0.01, 
                                        bar.get_y() + bar.get_height()/2, 
                                        f'{int(width)}', ha='left', va='center', fontsize=9)
                    except Exception as e:
                        print(f"Error creating categorical bar chart for {col}: {e}")
                        plt.close()
                        continue
                
                # Save chart as base64 encoded image
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=100, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                charts.append({
                    "title": f"{chart_info['type'].replace('_', ' ').title()}: {col}",
                    "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
                    "insight": chart_info['reason']
                })
                
            except Exception as e:
                print(f"Error creating chart for {col}: {e}")
                try:
                    plt.close()
                except:
                    pass
                continue
        
        # Clean up memory
        gc.collect()
        
    except Exception as e:
        print(f"Error in generate_smart_charts: {e}")
    
    return charts


def call_llm_insights_from_prompt(prompt: str) -> Optional[str]:
    """
    Calls OpenRouter API to generate AI insights about the dataset.
    Returns None if API key not configured or request fails.
    """
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not configured")
        return None
        
    try:
        client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful, business-friendly data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return None


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """
    Reads uploaded files (CSV, Excel, JSON) with multiple encoding fallbacks.
    Applies memory limits and data type optimization.
    """
    if not file.filename:
        raise ValueError("No filename provided")
    
    file_extension = file.filename.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            # Try multiple encodings for CSV files
            try:
                df = pd.read_csv(file.file, encoding='utf-8', low_memory=False, 
               on_bad_lines='skip', keep_default_na=True, na_values=[''])
            except UnicodeDecodeError:
                file.file.seek(0)
                try:
                    df = pd.read_csv(file.file, encoding='latin-1', low_memory=False,
                                   on_bad_lines='skip', keep_default_na=True, na_values=[''])
                except Exception:
                    file.file.seek(0)
                    df = pd.read_csv(file.file, encoding='cp1252', low_memory=False,
                                   on_bad_lines='skip', keep_default_na=True, na_values=[''])
        
        elif file_extension in ['xlsx', 'xls']:
            file_content = file.file.read()
            file.file.seek(0)
            excel_buffer = io.BytesIO(file_content)

            # Try reading with different engines
            try:
                df = pd.read_excel(excel_buffer, engine='openpyxl', sheet_name=0)
            except Exception as e1:
                excel_buffer.seek(0)
                try:
                    df = pd.read_excel(excel_buffer, engine='xlrd', sheet_name=0, nrows=MAX_ROWS)
                except Exception as e2:
                    excel_buffer.seek(0)
                    try:
                        df = pd.read_excel(excel_buffer, sheet_name=0, nrows=MAX_ROWS)
                    except Exception as e3:
                        raise ValueError(f"Unable to read Excel file: {e3 or e2 or e1}")
            
        elif file_extension == 'json':
            content = file.file.read()
            file.file.seek(0)
            
            # Handle different text encodings
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
            except json.JSONDecodeError:
                # Try line-by-line JSON (JSONL format)
                try:
                    json_objects = []
                    for line in content_str.split('\n'):
                        line = line.strip()
                        if line:
                            json_objects.append(json.loads(line))
                    json_data = json_objects
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format. Please ensure your JSON file is properly formatted.")
            
            # Convert JSON to DataFrame with different structure handling
            try:
                if isinstance(json_data, list):
                    if len(json_data) == 0:
                        raise ValueError("JSON file contains an empty array")
                    
                    if len(json_data) > MAX_ROWS:
                        json_data = json_data[:MAX_ROWS]
                    
                    if isinstance(json_data[0], dict):
                        df = pd.DataFrame(json_data)
                    else:
                        df = pd.DataFrame(json_data, columns=['value'])
                
                elif isinstance(json_data, dict):
                    try:
                        df = pd.DataFrame(json_data)
                        if len(df) > MAX_ROWS:
                            df = df.head(MAX_ROWS)
                    except ValueError:
                        # Flatten nested JSON structure
                        df = pd.json_normalize(json_data)
                        if len(df) > MAX_ROWS:
                            df = df.head(MAX_ROWS)
                
                else:
                    df = pd.DataFrame([json_data], columns=['value'])
                    
            except Exception as conversion_error:
                raise ValueError(f"Unable to convert JSON to tabular format: {str(conversion_error)}")
        
        else:
            raise ValueError(f"Unsupported file format: .{file_extension}. Please upload CSV, Excel (.xlsx/.xls), or JSON files.")
    
        if df is None or df.empty:
            raise ValueError("Uploaded file contains no data or couldn't be read properly.")

        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.empty:
            raise ValueError("No valid data found after cleaning.")

        if len(df.columns) > MAX_COLS:
            df = df.iloc[:, :MAX_COLS]
            
    except Exception as e:
        if "Unsupported file format" in str(e) or "Invalid JSON format" in str(e) or "Unable to" in str(e):
            raise e
        else:
            raise ValueError(f"Error reading {file_extension.upper()} file: {str(e)}")
    
    # Apply column limit
    if len(df.columns) > MAX_COLS:
        df = df.iloc[:, :MAX_COLS]
    
    return df


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Main endpoint for file upload and analysis.
    Returns insights, charts, data overview, and quality metrics.
    """
    try:
        # Read and validate uploaded file
        try:
            df = read_uploaded_file(file)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"File parsing failed: {str(e)}"})

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "Uploaded file contains no data."})

        # Store original shape before any processing
        original_shape = df.shape

        # Apply row limit AFTER getting original shape
        was_truncated = len(df) > MAX_ROWS
        if was_truncated:
            df = df.head(MAX_ROWS)

        # Clean empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "No valid data found after cleaning."})

        print(f"Data shape: {original_shape} -> {df.shape}")

        # Handle complex data types (convert dicts/lists to strings)
        for col in df.columns:
            try:
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, (dict, list)):
                        df[col] = df[col].astype(str)
            except Exception as e:
                print(f"Warning: Could not process column {col}: {e}")
                continue

        # Attempt automatic type conversion
        try:
            df = safe_convert_types(df)
        except Exception as e:
            print(f"Warning: Type conversion failed: {e}")

        # Generate analysis components with fallbacks
        try:
            overview = get_data_overview(df)
        except Exception as e:
            print(f"Error generating overview: {e}")
            overview = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_info": []
            }

        try:
            data_quality = get_data_quality(df)
        except Exception as e:
            print(f"Error generating data quality: {e}")
            data_quality = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": 0,
                "missing_percentage": 0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0,
                "quality_score": 50,
                "status": "unknown",
                "columns_with_missing": {}
            }

        # Generate sample for insights
        try:
            df_sample = df.sample(n=min(5, len(df)), random_state=42).iloc[:, :15]
            
            # Create dtypes summary safely
            try:
                dtypes_info = []
                for col in df.columns[:20]:  
                    try:
                        col_type = str(df[col].dtype)
                        dtypes_info.append({"Column": str(col)[:50], "Type": col_type})
                    except Exception as e:
                        print(f"Warning: Could not get dtype for {col}: {e}")
                        dtypes_info.append({"Column": str(col)[:50], "Type": "unknown"})
                
                dtypes_df = pd.DataFrame(dtypes_info)
                dtypes_md_str = dtypes_df.to_markdown(index=False)
            except Exception as e:
                print(f"Error creating dtypes summary: {e}")
                dtypes_md_str = "Could not generate column types summary"

            # Create insights prompt
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
        except Exception as e:
            print(f"Error generating insights: {e}")
            insights_raw = None

        # Generate charts
        try:
            charts = generate_smart_charts(df, max_charts=6)
        except Exception as e:
            print(f"Error generating charts: {e}")
            charts = []

        # Process insights
        insight_blocks = []
        if insights_raw:
            try:
                for line in insights_raw.strip().split("\n"):
                    if line.startswith(("📊", "🤔", "💡", "📈", "📉", "🔍")):
                        insight_blocks.append(line)
                    elif insight_blocks:
                        insight_blocks[-1] += "\n" + line
                    else:
                        insight_blocks.append(line)
            except Exception as e:
                print(f"Error processing insights: {e}")
                insight_blocks = ["📊 Data Analysis Complete - Your dataset has been successfully processed and analyzed."]

        if not insight_blocks:
            insight_blocks = [
                "📊 Data Upload Successful - Your dataset has been processed and is ready for analysis.",
                "🔍 Data Structure - The dataset contains structured information that can be analyzed.",
                "📈 Ready for Analysis - Charts and visualizations have been generated from your data.",
                "💡 Insights Available - Key patterns and trends have been identified in your dataset.",
                "🎯 Next Steps - Use the generated charts and quality metrics to understand your data better."
            ]

        if not charts:
            charts = [{
                "title": "Data Processing Complete",
                "image": "",
                "insight": "Charts could not be generated, but your data has been successfully analyzed."
            }]

        # Force cleanup
        try:
            del df
            gc.collect()
        except:
            pass

        return {
            "insights": insight_blocks,
            "charts": charts,
            "overview": overview,
            "data_quality": {
                **data_quality,
                "truncated_to_50k": was_truncated
            },
            "file_info": {
                "filename": file.filename,
                "rows": original_shape[0],  
                "columns": original_shape[1],
                "file_type": file.filename.split('.')[-1].upper()
            }
        }

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Force cleanup on error
        try:
            gc.collect()
        except:
            pass
        
        return JSONResponse(
            status_code=500, 
            content={
                "error": f"Server error during processing. Please try with a smaller dataset or different format.",
                "details": str(e)[:200] 
            }
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        reload=False,
        workers=1,  # Single worker to manage memory better
        timeout_keep_alive=30
    )