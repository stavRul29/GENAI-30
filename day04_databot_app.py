import io
import os
import re
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODEL  = "llama-3.3-70b-versatile"

MAX_HISTORY_TURNS = 20
MAX_INPUT_LENGTH  = 3000

# ── Guardrails ─────────────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|your)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|your)?\s*instructions?",
    r"override\s+(your\s+)?(instructions?|rules?|guidelines?)",
    r"bypass\s+(your\s+)?(instructions?|rules?|guidelines?|restrictions?)",
    r"(disable|remove|turn\s+off)\s+(the\s+)?(guardrails?|restrictions?|rules?)",
    r"developer\s+mode", r"admin\s+(mode|access|override)",
    r"jailbreak", r"DAN\s+mode", r"do\s+anything\s+now",
]

HARD_OFF_TOPIC_PATTERNS = [
    r"\bwrite\s+(me\s+)?(an?\s+)?(email|letter|poem|song|essay|story)\b",
    r"\b(joke|funny|humour|humor)\b",
    r"\b(recipe|cook(?!ie)|restaurant)\b",
    r"\b(politics|election|vote)\b",
    r"\b(sports?\s+score|cricket\s+score|ipl\s+score)\b",
]


def scan_input(text: str) -> tuple[bool, str]:
    text_lower = text.lower().strip()
    if len(text) > MAX_INPUT_LENGTH:
        return False, "too_long"
    for p in INJECTION_PATTERNS:
        if re.search(p, text_lower, re.IGNORECASE):
            return False, "injection"
    for p in HARD_OFF_TOPIC_PATTERNS:
        if re.search(p, text_lower, re.IGNORECASE):
            return False, "off_topic"
    return True, "clean"


# ── Prompts ────────────────────────────────────────────────────────────────────

SQL_BUILDER_PROMPT = """You are a SQL expert helping a non-technical business user generate SQL queries.

Table schema:
{schema}

The user will describe what they want to find out. Generate a SQL query that answers their question.

Respond using EXACTLY this format — nothing else:

```sql
-- query here
```

Approach: [One sentence: what the query computes and the key logic used]

RULES:
- Use only the table and columns defined in the schema above
- Write clean, readable SQL with proper formatting and aliases where helpful
- If a column name is ambiguous or not in the schema, use the closest match and note it
- For follow-up questions, build on the prior query shown in conversation history
- No extra explanation beyond the one Approach line"""

EXECUTION_PROMPT = """You are a data analyst. You have a pandas DataFrame named `df`.

Columns available: {columns}

Dataset info:
{dataset_context}

For the user's question, respond using EXACTLY this format — nothing else:

```python
result = df[...]  # pandas code that computes the answer
```

Approach: [One sentence: what was computed and the key logic used]

STRICT RULES:
- Use only `df` and `pd` (already in scope — do not import anything)
- Store the final answer in a variable named `result`
- `result` must be a scalar, pandas Series, or DataFrame
- No print(), no plotting, no imports
- For follow-up questions, use context from prior conversation
- If the question truly cannot be computed, write result = None and explain in Approach"""

SUMMARISER_PROMPT = """You are a senior BI analyst presenting to a C-suite audience.

Given a dataset, produce exactly 3 executive bullets.

FORMAT — use this exactly:
- [INSIGHT]: One clear finding in plain English
  [METRIC]: The specific number that proves it
  [ACTION]: One immediately actionable recommendation

RULES: Most important insight first. Specific numbers always. No intro or conclusion. No hedging."""


# ── File loading & stats ───────────────────────────────────────────────────────

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext == ".tsv":
        return pd.read_csv(uploaded_file, sep="\t")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(uploaded_file)
    elif ext == ".json":
        return pd.read_json(uploaded_file)
    elif ext == ".parquet":
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def compute_stats(df: pd.DataFrame) -> dict:
    stats: dict = {
        "total_rows":    len(df),
        "total_columns": len(df.columns),
        "columns":       list(df.columns),
        "dtypes":        {col: str(df[col].dtype) for col in df.columns},
        "numeric":       {},
        "categorical":   {},
    }
    for col in df.select_dtypes(include="number").columns:
        stats["numeric"][col] = {
            "min":  round(float(df[col].min()), 2),
            "max":  round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
        }
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        vc = df[col].value_counts()
        stats["categorical"][col] = {
            "unique_count": int(df[col].nunique()),
            "top_values":   vc.head(5).to_dict(),
        }
    return stats


def build_dataset_context(filename: str, df: pd.DataFrame, stats: dict) -> str:
    return (
        f"FILE: {filename}  |  TABLE: {Path(filename).stem}\n"
        f"ROWS: {stats['total_rows']}  |  COLUMNS: {stats['total_columns']}\n"
        f"COLUMNS: {', '.join(stats['columns'])}\n\n"
        f"NUMERIC STATS:\n{json.dumps(stats['numeric'], indent=2)}\n\n"
        f"CATEGORICAL (top 5):\n{json.dumps(stats['categorical'], indent=2)}\n\n"
        f"SAMPLE ROWS:\n{df.head(3).to_string(index=False)}"
    )


def build_schema_from_df(table_name: str, df: pd.DataFrame) -> str:
    lines = [f"Table: {table_name}", "Columns:"]
    for col in df.columns:
        lines.append(f"  - {col} ({df[col].dtype})")
    if not df.empty:
        lines.append(f"\nSample values (first 3 rows):\n{df.head(3).to_string(index=False)}")
    return "\n".join(lines)


def build_schema_from_manual(table_name: str, columns_text: str) -> str:
    lines = [f"Table: {table_name}", "Columns:"]
    for col in columns_text.strip().splitlines():
        col = col.strip().strip(",")
        if col:
            lines.append(f"  - {col}")
    return "\n".join(lines)


def build_summary_context(filename: str, stats: dict) -> str:
    lines = [
        f"File: {filename}",
        f"Rows: {stats['total_rows']}  |  Columns: {stats['total_columns']}",
        f"Columns: {', '.join(stats['columns'])}",
        "", "Numeric stats (mean | min | max):",
    ]
    for col, s in stats["numeric"].items():
        lines.append(f"  {col}: mean={s['mean']}, min={s['min']}, max={s['max']}")
    lines.append("") ; lines.append("Categorical (top 3 values):")
    for col, s in stats["categorical"].items():
        top = list(s["top_values"].items())[:3]
        lines.append(f"  {col} ({s['unique_count']} unique): {', '.join(f'{k}={v}' for k,v in top)}")
    return "\n".join(lines)


# ── Query Builder engine ───────────────────────────────────────────────────────

def extract_sql(text: str) -> str | None:
    match = re.search(r"```sql\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_approach(text: str) -> str:
    match = re.search(r"Approach:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def generate_sql(history: list, schema: str) -> dict:
    trimmed = [{"role": m["role"], "content": m["content"]}
               for m in history[-(MAX_HISTORY_TURNS * 2):]]
    system  = SQL_BUILDER_PROMPT.format(schema=schema)
    resp    = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + trimmed,
        temperature=0,
    )
    raw      = resp.choices[0].message.content
    sql      = extract_sql(raw)
    approach = extract_approach(raw)
    return {"sql": sql, "approach": approach, "raw": raw}


# ── File Analyst engine ────────────────────────────────────────────────────────

def extract_code(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def execute_code(code: str, df: pd.DataFrame) -> tuple:
    namespace = {"df": df.copy(), "pd": pd}
    try:
        exec(code, namespace)  # noqa: S102
        return namespace.get("result"), None
    except Exception as e:
        return None, str(e)


def run_analysis(history: list, df: pd.DataFrame, dataset_context: str) -> dict:
    trimmed = [{"role": m["role"], "content": m["content"]}
               for m in history[-(MAX_HISTORY_TURNS * 2):]]
    system  = EXECUTION_PROMPT.format(
        columns=", ".join(df.columns.tolist()),
        dataset_context=dataset_context,
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + trimmed,
        temperature=0,
    )
    raw      = resp.choices[0].message.content
    code     = extract_code(raw)
    approach = extract_approach(raw)

    if code:
        result, error = execute_code(code, df)
        if error:
            return {"type": "error",  "approach": approach, "error": error,
                    "text": f"**Approach:** {approach}\n\n*Error: {error}*"}
        return {"type": "result", "approach": approach, "result": result,
                "text": f"**Approach:** {approach}"}
    return {"type": "text", "text": raw}


def stream_summary(filename: str, stats: dict):
    context = build_summary_context(filename, stats)
    for chunk in client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARISER_PROMPT},
            {"role": "user",   "content": f"Analyse this dataset.\n\n{context}"},
        ],
        stream=True,
    ):
        yield chunk.choices[0].delta.content or ""


# ── Session state ──────────────────────────────────────────────────────────────

for key, default in {
    "mode":             None,       # None = not yet chosen
    # Query Builder
    "qb_schema":        None,
    "qb_table_name":    "",
    "qb_history":       [],
    # File Analyst
    "fa_df":            None,
    "fa_filename":      None,
    "fa_stats":         None,
    "fa_context":       "",
    "fa_history":       [],
    "fa_summary":       None,
    "fa_summary_requested": False,
    "fa_choice_made":   False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DataBot", page_icon="📊", layout="wide")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 DataBot")
    st.divider()

    if st.session_state.mode is not None:
        mode = st.radio(
            "Mode",
            ["🔍 Query Builder", "📊 File Analyst"],
            index=0 if st.session_state.mode == "query_builder" else 1,
            label_visibility="collapsed",
        )
        new_mode = "query_builder" if mode == "🔍 Query Builder" else "file_analyst"
        if new_mode != st.session_state.mode:
            st.session_state.mode = new_mode
            st.rerun()
    st.divider()

    # ── Query Builder sidebar ──
    if st.session_state.mode == "query_builder":
        st.caption("**Query Builder**  \nDescribe your data and ask questions. Get ready-to-run SQL.")
        st.divider()

        schema_source = st.radio("Schema source", ["Upload sample CSV", "Enter manually"], label_visibility="collapsed")

        if schema_source == "Upload sample CSV":
            schema_file = st.file_uploader(
                "Upload a sample of your table (CSV)",
                type=["csv"],
                key="qb_upload",
            )
            table_name = st.text_input("Table name (as it is in your DB)", placeholder="e.g. employees")
            if schema_file and table_name:
                sample_df = pd.read_csv(schema_file, nrows=5)
                schema    = build_schema_from_df(table_name, sample_df)
                if st.button("✅ Set Schema", use_container_width=True):
                    st.session_state.qb_schema     = schema
                    st.session_state.qb_table_name = table_name
                    st.session_state.qb_history    = []
                    st.rerun()

        else:
            table_name   = st.text_input("Table name", placeholder="e.g. employees")
            columns_text = st.text_area(
                "Column names (one per line or comma-separated)",
                placeholder="employee_id\nname\ndepartment\nsalary\nhire_date",
                height=150,
            )
            if table_name and columns_text:
                if st.button("✅ Set Schema", use_container_width=True):
                    st.session_state.qb_schema     = build_schema_from_manual(table_name, columns_text)
                    st.session_state.qb_table_name = table_name
                    st.session_state.qb_history    = []
                    st.rerun()

        if st.session_state.qb_schema:
            st.divider()
            st.caption(f"Table: **{st.session_state.qb_table_name}**")
            with st.expander("Schema"):
                st.code(st.session_state.qb_schema, language=None)
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.qb_history = []
                st.rerun()

    # ── File Analyst sidebar ──
    else:
        st.caption("**File Analyst**  \nUpload a data file. Get real computed answers.")
        st.divider()

        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
            label_visibility="collapsed",
            key="fa_upload",
        )
        if uploaded and uploaded.name != st.session_state.fa_filename:
            with st.spinner("Loading…"):
                try:
                    df = load_uploaded_file(uploaded)
                    st.session_state.fa_df       = df
                    st.session_state.fa_filename = uploaded.name
                    st.session_state.fa_stats    = compute_stats(df)
                    st.session_state.fa_context  = build_dataset_context(uploaded.name, df, st.session_state.fa_stats)
                    st.session_state.fa_history  = []
                    st.session_state.fa_summary  = None
                    st.session_state.fa_summary_requested = False
                    st.session_state.fa_choice_made       = False
                except Exception as e:
                    st.error(f"Failed to load: {e}")

        if st.session_state.fa_df is not None:
            stats = st.session_state.fa_stats
            st.divider()
            st.caption(f"**{st.session_state.fa_filename}**")
            st.caption(f"{stats['total_rows']:,} rows · {stats['total_columns']} columns")
            with st.expander("Columns"):
                for col in stats["columns"]:
                    st.caption(f"`{col}` — {stats['dtypes'][col]}")
            st.divider()
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.fa_history  = []
                st.session_state.fa_summary  = None
                st.session_state.fa_summary_requested = False
                st.session_state.fa_choice_made       = False
                st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────

# ── Landing screen ──
if st.session_state.mode is None:
    st.title("📊 DataBot")
    st.markdown("#### What would you like to do?")
    st.divider()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### 🔍 Query Builder")
        st.markdown(
            "I have access to a database but can't write SQL.  \n"
            "**Generate a ready-to-run SQL query** I can paste into my DB tool."
        )
        if st.button("Build a Query →", use_container_width=True, type="primary"):
            st.session_state.mode = "query_builder"
            st.rerun()
    with c2:
        st.markdown("### 📊 File Analyst")
        st.markdown(
            "I have a data file (CSV, Excel, etc.) and want answers now.  \n"
            "**Upload the file — get real computed insights instantly.**"
        )
        if st.button("Analyse My File →", use_container_width=True, type="primary"):
            st.session_state.mode = "file_analyst"
            st.rerun()
    st.stop()

# ════════════════════════
# MODE 1 — QUERY BUILDER
# ════════════════════════
if st.session_state.mode == "query_builder":
    st.title("🔍 Query Builder")

    if not st.session_state.qb_schema:
        st.markdown("Set your table schema in the sidebar to get started.")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "**Upload a sample CSV**  \n"
                "Upload a few rows of your table — we auto-detect column names and types."
            )
        with c2:
            st.markdown(
                "**Enter manually**  \n"
                "Type your table name and column names — no file needed."
            )
    else:
        st.caption(f"Table: **{st.session_state.qb_table_name}** — ask what you want to find out and get SQL to run on your DB.")

        # Chat history
        for msg in st.session_state.qb_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("sql"):
                    st.code(msg["sql"], language="sql")
                    if msg.get("approach"):
                        st.caption(f"Approach: {msg['approach']}")
                else:
                    st.markdown(msg["content"])

        if user_input := st.chat_input("What do you want to find out? e.g. 'Top 5 departments by headcount'"):
            is_safe, reason = scan_input(user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            st.session_state.qb_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                if not is_safe:
                    msg = {
                        "injection": "My instructions are permanent and cannot be modified.",
                        "off_topic": "I only generate SQL queries. Please describe what you want to analyse.",
                        "too_long":  f"Input too long. Keep it under {MAX_INPUT_LENGTH} characters.",
                    }.get(reason, "Cannot process that input.")
                    st.warning(msg)
                    st.session_state.qb_history.append({"role": "assistant", "content": msg, "sql": None})
                else:
                    with st.spinner("Generating SQL…"):
                        output = generate_sql(st.session_state.qb_history, st.session_state.qb_schema)

                    if output["sql"]:
                        st.code(output["sql"], language="sql")
                        if output["approach"]:
                            st.caption(f"Approach: {output['approach']}")
                        st.session_state.qb_history.append({
                            "role":     "assistant",
                            "content":  output["raw"],
                            "sql":      output["sql"],
                            "approach": output["approach"],
                        })
                    else:
                        st.markdown(output["raw"])
                        st.session_state.qb_history.append({
                            "role": "assistant", "content": output["raw"], "sql": None,
                        })


# ══════════════════════
# MODE 2 — FILE ANALYST
# ══════════════════════
else:
    st.title("📊 File Analyst")

    if st.session_state.fa_df is None:
        st.markdown("Upload a CSV, Excel, TSV, JSON, or Parquet file from the sidebar.")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real computed answers**  \nActually runs code on your data — shows results, not instructions.")
        with c2:
            st.markdown("**Any format**  \nCSV · Excel · TSV · JSON · Parquet")

    else:
        stats = st.session_state.fa_stats
        st.caption(f"Dataset: **{st.session_state.fa_filename}** · {stats['total_rows']:,} rows · {stats['total_columns']} columns")

        # File acknowledgement
        if not st.session_state.fa_choice_made:
            st.success(f"**{st.session_state.fa_filename}** loaded successfully.")
            num_cols = list(stats["numeric"].keys())
            cat_cols = list(stats["categorical"].keys())
            st.markdown(
                f"- **{stats['total_rows']:,}** rows · **{stats['total_columns']}** columns\n"
                f"- Numeric: {', '.join(f'`{c}`' for c in num_cols) or 'none'}\n"
                f"- Categorical: {', '.join(f'`{c}`' for c in cat_cols) or 'none'}"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📋 Generate Executive Summary", use_container_width=True):
                    st.session_state.fa_summary_requested = True
                    st.session_state.fa_choice_made       = True
                    st.rerun()
            with c2:
                if st.button("💬 Start Asking Questions", use_container_width=True):
                    st.session_state.fa_choice_made = True
                    st.rerun()
            st.stop()

        # Executive summary
        if st.session_state.fa_summary_requested and st.session_state.fa_summary is None:
            st.subheader("Executive Summary")
            summary = st.write_stream(stream_summary(st.session_state.fa_filename, stats))
            st.session_state.fa_summary = summary
            st.divider()
        elif st.session_state.fa_summary:
            with st.expander("Executive Summary", expanded=False):
                st.markdown(st.session_state.fa_summary)
            st.divider()

        # Chat history
        for msg in st.session_state.fa_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("result_type") == "dataframe" and msg.get("result_data"):
                    st.dataframe(pd.read_json(io.StringIO(msg["result_data"])), use_container_width=True)
                elif msg.get("result_type") == "series" and msg.get("result_data"):
                    st.dataframe(pd.read_json(io.StringIO(msg["result_data"])), use_container_width=True)
                elif msg.get("result_type") == "scalar" and msg.get("result_data"):
                    st.metric(label="Result", value=msg["result_data"])

        # Chat input
        if user_input := st.chat_input("Ask anything about your data…"):
            is_safe, reason = scan_input(user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            st.session_state.fa_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                if not is_safe:
                    msg = {
                        "injection": "My instructions are permanent and cannot be modified.",
                        "off_topic": "I only answer questions about your dataset.",
                        "too_long":  f"Input too long. Keep it under {MAX_INPUT_LENGTH} characters.",
                    }.get(reason, "Cannot process that input.")
                    st.warning(msg)
                    st.session_state.fa_history.append({"role": "assistant", "content": msg})
                else:
                    with st.spinner("Analysing…"):
                        display = run_analysis(
                            st.session_state.fa_history,
                            st.session_state.fa_df,
                            st.session_state.fa_context,
                        )

                    history_entry = {"role": "assistant", "content": display["text"]}
                    st.markdown(display["text"])

                    if display["type"] == "result" and display.get("result") is not None:
                        result = display["result"]
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                            history_entry["result_type"] = "dataframe"
                            history_entry["result_data"] = result.to_json()
                        elif isinstance(result, pd.Series):
                            df_result = result.reset_index()
                            st.dataframe(df_result, use_container_width=True)
                            history_entry["result_type"] = "series"
                            history_entry["result_data"] = df_result.to_json()
                        elif isinstance(result, (int, float)):
                            val = f"{result:,}" if isinstance(result, int) else f"{result:,.4f}"
                            st.metric(label="Result", value=val)
                            history_entry["result_type"] = "scalar"
                            history_entry["result_data"] = val
                        else:
                            st.code(str(result))
                    elif display["type"] == "error":
                        st.error(f"Computation error: {display['error']}")

                    st.session_state.fa_history.append(history_entry)
