# Data-Software-Pt.2
"""
data_quality_report.py
Part 2 of the Data Quality & Analytics Platform: the Data Quality Report.
This module is READ-ONLY. It never deletes rows, deletes columns, corrects
values, or overwrites the DataFrame produced by Part 1 (Data Upload). Every
check works on a private, cleaned-up view of the data that is created inside
prepare_dataframe() and thrown away after the report is built.
The module has two layers:
1. ANALYSIS LAYER (pure Python + pandas + numpy, no Streamlit calls).
Functions such as calculate_missing_values(), detect_outliers() and
calculate_quality_score(). Because they do not touch Streamlit they are
easy to test.
2. DISPLAY LAYER (Streamlit).
Functions named display_*() that turn analysis results into headings,
metrics, tables and messages. display_quality_report() is the single
entry point that app.py calls.
Expected workflow:
Data Upload (Part 1) -> uploaded DataFrame -> run_quality_analysis()
-> display_quality_report()
"""
import datetime
import logging
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
# A module-level logger. Technical details of unexpected errors are written
# here (to the terminal running Streamlit) instead of being shown to the user.
logger = logging.getLogger(__name__)
# =============================================================================
# CONFIGURATION
# Every threshold used by the report lives here so it is easy to find, read,
# and change. These are data-quality INDICATORS chosen by this application;
# they are not universal rules.
# =============================================================================
# --- Part 1 integration: where the uploaded DataFrame is expected --------------
# Part 1 (Data Upload) is expected to store the DataFrame in st.session_state.
# get_uploaded_dataframe() checks these keys in order and uses the first one
# that holds a DataFrame. If your Part 1 code uses a different key, add it here
# (or, better, pass the DataFrame directly to display_quality_report()).
DATAFRAME_SESSION_KEYS = ("df", "dataframe", "uploaded_df", "uploaded_dataframe", "data")
DATASET_NAME_SESSION_KEYS = ("dataset_name", "file_name", "filename", "uploaded_file_name")
# --- Missing-value severity thresholds (percent of a column that is missing) --
MISSING_LOW_MAX = 5.0 # 0% < missing < 5% -> "Low concern"
MISSING_MODERATE_MAX = 20.0 # 5% <= missing < 20% -> "Moderate concern"; 20%+ -> "High conc
# --- Generic severity thresholds for "how many values are affected" issues ----
SEVERITY_HIGH_PCT = 10.0 # 10% or more of the values affected -> High
SEVERITY_MEDIUM_PCT = 1.0 # 1% or more of the values affected -> Medium (below that -> Lo
# --- Outlier detection (IQR method) -------------------------------------------
IQR_MULTIPLIER = 1.5 # Standard "Tukey fence" multiplier
MIN_VALUES_FOR_OUTLIERS = 4 # Fewer numeric values than this: quartiles are not meaningful
OUTLIER_MEDIUM_PCT = 5.0 # 5% or more potential outliers -> Medium issue (below -> Low)
# --- Type detection ------------------------------------------------------------
NUMERIC_TEXT_MIN_SHARE = 0.90 # A text column is "numeric-looking" if 90%+ of its values par
DATE_TEXT_MIN_SHARE = 0.50 # A text column is "date-looking" if 50%+ of its values DATE_NAME_HINT_MIN_SHARE = 0.10 # ...or 10%+ if the column NAME also suggests dates (e.g. "o
DATE_NAME_HINTS = {"date", "dob", "datetime", "timestamp", "time"}
have a
# --- Categorical consistency ----------------------------------------------------
MAX_UNIQUE_FOR_CONSISTENCY = 5000 # Skip columns with more distinct values than this (keeps
# --- Validity rules that depend on column NAMES (heuristics, clearly labelled) --
# If a column name contains one of these words, negative values are reported as
# "Invalid Values" because such quantities are normally not negative.
NON_NEGATIVE_NAME_TOKENS = {
"age", "price", "income", "salary", "quantity", "qty", "amount", "cost",
"revenue", "sales", "count", "population", "weight", "height", "distance", "duration",
}
MAX_REASONABLE_AGE = 120 # Values above this in a column named like "age" are reported as in
# Words in a column name that suggest the column is meant to hold unique identifiers.
ID_NAME_TOKENS = {"id", "uid", "uuid", "guid", "key", "pk"}
# Text values that often stand in for "no value" (compared case-insensitively).
PLACEHOLDER_TOKENS = {"n/a", "na", "#n/a", "null", "nan", "nil", "-", "--", "?", "missing", "
# --- Issue names (used in the issue table and to drive the recommendations) ----
ISSUE_MISSING = "Missing Values"
ISSUE_EMPTY = "Empty Column"
ISSUE_DUP_ROWS = "Duplicate Rows"
ISSUE_DUP_ID = "Duplicate Values in ID-like Column"
ISSUE_CONSTANT = "Constant Column"
ISSUE_OUTLIERS = "Potential Outliers"
ISSUE_INCONSISTENT = "Inconsistent Categorical Values"
ISSUE_NUMERIC_TEXT = "Numeric Values Stored as Text"
ISSUE_DATE_TEXT = "Dates Stored as Text"
ISSUE_MIXED_TYPES = "Mixed Data Types"
ISSUE_INVALID = "Invalid Values"
ISSUE_PLACEHOLDER = "Placeholder Text Values"
ISSUE_INVALID_DATES = "Invalid Dates"
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
ISSUE_COLUMNS = ["Severity", "Column", "Issue", "Details"]
# --- Date-shape patterns (used only to DECIDE whether text looks like a date) --
# Matches shapes such as 2023-05-01, 2023/5/1, 05/01/2023, 5.1.23, and optional times.
_NUMERIC_DATE_SHAPE = (
r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s?[AaPp][Mm]|Z|[+-]\d{2}:?\d{2})?)?"
)
# Matches shapes such as "March 5, 2020", "5 Mar 2020", "Mar 5 2020".
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_NAMED_DATE_SHAPE = (
rf"(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}"
rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{2,4}})"
)
DATE_SHAPE_PATTERN = rf"^(?:{_NUMERIC_DATE_SHAPE}|{_NAMED_DATE_SHAPE})$"
# =============================================================================
# SECTION 1: SMALL HELPER FUNCTIONS
# =============================================================================
def describe_dtype(dtype):
"""Return a friendlier text label for a pandas dtype (used in the Data Types table)."""
name = str(dtype)
if name == "object":
return "object (text or mixed)"
if name in ("str", "string"):
return "text (string)"
return name
def make_unique_labels(columns):
"""
Turn column labels into unique strings.
Real-world files sometimes contain duplicate or non-text column names. The
analysis needs every column to be addressable by a unique string, so the
analysis copy uses labels such as "Name", "Name.1". The user's original
DataFrame is not renamed.
"""
labels, seen = [], {}
for column in columns:
base = str(column)
if base in seen:
seen[base] += 1
labels.append(f"{base}.{seen[base]}")
else:
seen[base] = 0
labels.append(base)
return labels
def is_text_column(series):
"""True for object/string columns (the columns that can contain free text)."""
if isinstance(series.dtype, pd.CategoricalDtype):
return False
return pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.
def clean_blank_strings(series):
"""
Return a copy of a column in which blank/whitespace-only text becomes a
real missing value (NaN). Non-text columns are returned unchanged.
Why: a cell containing " " is not meaningful data. Treating blanks as
missing makes the completeness numbers reflect what a person would call
"empty". This only affects the private analysis copy.
"""
if isinstance(series.dtype, pd.CategoricalDtype):
series = series.astype(object)
if not is_text_column(series):
return series
try:
stripped = series.astype("string").str.strip()
is_blank = stripped.eq("").fillna(False).astype(bool)
return series.mask(is_blank)
except Exception: # unusual object contents: leave the column as it is
return series
def prepare_dataframe(df):
"""
Build the private analysis copy of the uploaded DataFrame.
Returns (analysis_df, original_dtypes):
* analysis_df: same data, unique string column labels, a clean 0..n-1
index, and blank strings turned into NaN. The uploaded DataFrame itself
is never modified.
* original_dtypes: {column label: friendly dtype label} taken from the
ORIGINAL data so the Data Types table reports what the user really has.
"""
# reset_index(drop=True) returns a NEW DataFrame, so renaming its columns
# below cannot affect the user's DataFrame.
work = df.reset_index(drop=True)
labels = make_unique_labels(df.columns)
work.columns = labels
original_dtypes = {label: describe_dtype(dtype) for label, dtype in zip(labels, df.dtypes
# Build the analysis DataFrame column by column (blank text -> NaN).
cleaned_columns = {label: clean_blank_strings(work[label]) for label in labels}
analysis_df = pd.DataFrame(cleaned_columns, index=work.index)
return analysis_df, original_dtypes
def check_dataframe(df):
"""
Validate the input before analysis.
Returns (level, message). level is None when the DataFrame is usable,
otherwise "error" or "warning", and message is the user-friendly text
that display_quality_report() shows.
"""
if df is None:
return "warning", ("No dataset is available yet. Please upload a dataset in the "
"Data Upload section first, then return to this page.")
if not isinstance(df, pd.DataFrame):
return "error", ("The uploaded data is not in a supported table format, so a Data "
"Quality Report cannot be created.")
if df.shape[1] == 0:
return "error", ("The dataset has no columns, so a Data Quality Report cannot be crea
if df.shape[0] == 0:
return "warning", ("The dataset has columns but no rows, so there is nothing to analy
"Please upload a file that contains data.")
return None, None
def column_name_tokens(name):
"""
Split a column name into lowercase words. Handles snake_case, spaces and
camelCase: "CustomerID" -> {"customer", "id"}, "annual_income" -> {"annual", "income"}.
"""
spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
return {token for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if token}
def is_id_like_column(name):
"""True if the column NAME suggests an identifier (e.g. "id", "customer_id", "OrderID")."
return bool(column_name_tokens(name) & ID_NAME_TOKENS)
def safe_nunique(series):
"""Count distinct non-null values; falls back to text comparison for unhashable values (l
try:
return int(series.nunique(dropna=True))
except TypeError:
return int(series.dropna().astype(str).nunique())
def first_examples(values, limit=3, width=24):
"""Return up to `limit` example values as a short quoted string, for issue details."""
examples = []
for value in pd.Series(values).dropna().astype(str).unique()[:limit]:
text = value if len(value) <= width else value[: width - 1] + "…"
examples.append(f"'{text}'")
return ", ".join(examples)
def placeholder_mask(series):
"""Boolean Series: True where a text value looks like a placeholder such as 'N/A' or 'nul
lowered = series.astype("string").str.strip().str.lower()
return lowered.isin(PLACEHOLDER_TOKENS).fillna(False).astype(bool)
def numeric_text_profile(series):
"""
Measure how numeric a TEXT column is.
Returns a dict with:
considered - non-missing, non-placeholder values examined
numeric share - how many of those can be read as numbers
- numeric / considered (0..1)
non_numeric_mask - Boolean Series (full length) marking the entries that are not number
"""
present = series.notna() & ~placeholder_mask(series)
texts = series[present].astype("string").str.strip().str.replace(",", "", regex=False)
converted = pd.to_numeric(texts, errors="coerce")
is_number = converted.notna()
non_numeric_mask = pd.Series(False, index=series.index)
non_numeric_mask.loc[texts.index[~is_number.to_numpy()]] = True
considered = int(len(texts))
numeric = int(is_number.sum())
return {
"considered": considered,
"numeric": numeric,
"share": (numeric / considered) if considered else 0.0,
"non_numeric_mask": non_numeric_mask,
}
def python_type_label(value):
"""Group Python value types into broad labels used to detect mixed data types."""
if isinstance(value, (bool, np.bool_)):
return "boolean"
if isinstance(value, (int, float, np.integer, np.floating)):
return "number"
if isinstance(value, str):
return "text"
if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
return "date/time"
return type(value).__name__
def get_numeric_values(series):
"""Return the finite numeric values of a column as floats (missing and infinite values re
try:
values = pd.to_numeric(series, errors="coerce").astype("float64")
except (TypeError, ValueError):
return pd.Series(dtype="float64")
return values.replace([np.inf, -np.inf], np.nan).dropna()
def parse_dates(texts):
"""
Parse text into datetimes WITHOUT raising errors: anything that cannot be
read as a real date becomes NaT (missing). Several strategies are tried so
that mixed formats and time zones are handled.
"""
with warnings.catch_warnings():
warnings.simplefilter("ignore")
for kwargs in ({"format": "mixed"}, {"format": "mixed", "utc": True}, {}):
try:
return pd.to_datetime(texts, errors="coerce", **kwargs)
except (TypeError, ValueError):
continue
return pd.Series(pd.NaT, index=texts.index)
def looks_like_date_column(series, column_name):
"""
Decide whether a TEXT column appears to contain dates.
It counts the values whose SHAPE looks like a date (e.g. 2023-05-01 or
"March 5, 2020"). This does not check that the dates are valid; invalid
ones are reported later by analyze_datetime_quality().
"""
present = series.notna() & ~placeholder_mask(series)
texts = series[present].astype(str).str.strip()
if texts.empty:
return False
share = texts.str.match(DATE_SHAPE_PATTERN, flags=re.IGNORECASE).mean()
if share >= DATE_TEXT_MIN_SHARE:
return True
return bool(share >= DATE_NAME_HINT_MIN_SHARE and (column_name_tokens(column_name) def severity_from_percentage(percentage):
"""Map 'percent of values affected' to High / Medium / Low."""
if percentage >= SEVERITY_HIGH_PCT:
return "High"
if percentage >= SEVERITY_MEDIUM_PCT:
return "Medium"
return "Low"
def format_timestamp(timestamp):
"""Format a Timestamp as a date, adding the time only when it is not midnight."""
if pd.isna(timestamp):
return "n/a"
try:
if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
& DATE
return timestamp.strftime("%Y-%m-%d")
return timestamp.strftime("%Y-%m-%d %H:%M:%S")
except Exception:
return str(timestamp)
# =============================================================================
# SECTION 2: COLUMN CLASSIFICATION AND DATASET OVERVIEW
# =============================================================================
def classify_columns(df):
"""
Sort every column into one group:
numeric - integer / float columns (booleans are NOT counted as numeric)
datetime - columns that pandas already stores as dates
date_text - text columns whose values look like dates
categorical - everything else (text, categories, booleans, ...)
"""
groups = {"numeric": [], "datetime": [], "date_text": [], "categorical": []}
for column in df.columns:
series = df[column]
if pd.api.types.is_bool_dtype(series.dtype):
groups["categorical"].append(column)
elif pd.api.types.is_numeric_dtype(series.dtype):
groups["numeric"].append(column)
elif pd.api.types.is_datetime64_any_dtype(series.dtype):
groups["datetime"].append(column)
elif is_text_column(series) and looks_like_date_column(series, column):
groups["date_text"].append(column)
else:
groups["categorical"].append(column)
return groups
def build_overview(df, groups, dataset_name):
"""Collect the high-level facts shown in the Dataset Overview section."""
rows, columns = df.shape
return {
"dataset_name": dataset_name,
"rows": int(rows),
"columns": int(columns),
"total_cells": int(rows * columns),
"numeric_columns": len(groups["numeric"]),
"categorical_columns": len(groups["categorical"]),
# Date/time columns include real datetime columns AND text columns that look like dat
"datetime_columns": len(groups["datetime"]) + len(groups["date_text"]),
}
# =============================================================================
# SECTION 3: MISSING VALUES, DUPLICATES, EMPTY AND CONSTANT COLUMNS
# =============================================================================
def classify_missing_severity(percentage):
"""Translate a missing-value percentage into this application's label (an indicator, not
if percentage == 0:
return "Complete"
if percentage < MISSING_LOW_MAX:
return "Low concern"
if percentage < MISSING_MODERATE_MAX:
return "Moderate concern"
return "High concern"
def calculate_missing_values(df):
"""Return one row per column: missing count, missing percentage and severity label."""
# Calculate the total number of missing values in each column.
# This information is used by the Data Quality Report to measure
# dataset completeness.
missing_counts = df.isnull().sum()
# Calculate the percentage of missing values in each column.
# This percentage allows the report to identify columns that
# may require additional data-quality attention.
total_rows = len(df)
missing_percentages = (missing_counts / total_rows * 100) if total_rows else missing_coun
table = pd.DataFrame({
"Column": missing_counts.index,
"Missing Count": missing_counts.to_numpy(dtype=int),
"Missing %": missing_percentages.to_numpy(dtype=float).round(2),
})
# Attach the severity label (Complete / Low / Moderate / High concern).
table["Severity"] = table["Missing %"].apply(classify_missing_severity)
return table.sort_values("Missing %", ascending=False, kind="stable").reset_index(drop=Tr
def calculate_duplicates(df):
"""Count rows that are exact repeats of an earlier row (the first occurrence is not count
try:
duplicate_flags = df.duplicated()
except TypeError:
# Unhashable cell contents (lists, dicts): compare the text form instead.
duplicate_flags = df.astype(str).duplicated()
# Number of duplicate rows, and that number as a percentage of all rows.
duplicate_rows = int(duplicate_flags.sum())
percentage = (duplicate_rows / len(df) * 100) if len(df) else 0.0
return {"duplicate_rows": duplicate_rows, "percentage": round(percentage, 2), "total_rows
def calculate_duplicate_values(df):
"""
Per column: how many values repeat an earlier value.
Duplicate Values = non-null values - distinct values, i.e. the number of
entries that repeat something already seen. A repeat is NOT automatically a
problem: repeated values are normal in columns like Country or Gender and
are only unexpected in identifier columns.
"""
records = []
for column in df.columns:
non_null = int(df[column].notna().sum())
unique = safe_nunique(df[column])
duplicates = non_null - unique
records.append({
"Column": column,
"Non-Null Values": non_null,
"Unique Values": unique,
"Duplicate Values": duplicates,
"Duplicate %": round(duplicates / non_null * 100, 2) if non_null else 0.0,
"Looks Like an ID Column": is_id_like_column(column),
})
return pd.DataFrame(records)
def detect_empty_columns(df):
"""Columns where every value is missing or blank."""
records = []
total_rows = len(df)
for column in df.columns:
missing = int(df[column].isna().sum())
if total_rows and missing == total_rows:
records.append({
"Column": column,
"Number of Values": total_rows,
"Missing Values": missing,
"Missing %": round(missing / total_rows * 100, 2),
})
return pd.DataFrame(records, columns=["Column", "Number of Values", "Missing Values", "Mi
def detect_constant_columns(df):
"""Columns whose non-null values are all identical (needs at least 2 rows to be meaningfu
columns = ["Column", "Unique Values", "Value", "Non-Null Values"]
if len(df) < 2:
return pd.DataFrame(columns=columns)
records = []
for column in df.columns:
non_null = df[column].dropna()
if len(non_null) > 0 and safe_nunique(non_null) == 1:
value = str(non_null.iloc[0])
records.append({
"Column": column,
"Unique Values": 1,
"Value": value if len(value) <= 60 else value[:59] + "…",
"Non-Null Values": int(len(non_null)),
})
return pd.DataFrame(records, columns=columns)
# =============================================================================
# SECTION 4: DATA TYPES, INVALID VALUES, DATES
# =============================================================================
def analyze_data_types(df, original_dtypes, groups):
"""
Build the Data Types table and detect type-related concerns.
Returns (table, findings, mixed_type_cells):
table - Column, Detected Data Type, Non-Null Values, Unique Values, Potential Concern
findings - issue dicts (Severity, Column, Issue, Details)
mixed_type_cells - number of cells in the minority type of mixed columns (feeds the Con
"""
records, findings, mixed_type_cells = [], [], 0
for column in df.columns:
series = df[column]
non_null = int(series.notna().sum())
concerns = []
if is_text_column(series) and non_null > 0:
# (a) Numeric-looking values stored as text.
profile = numeric_text_profile(series)
if profile["considered"] > 0 and profile["share"] >= NUMERIC_TEXT_MIN_SHARE and p
concerns.append("Numeric-looking values stored as text")
findings.append({
"Severity": "Low", "Column": column, "Issue": ISSUE_NUMERIC_TEXT,
"Details": (f"{profile['numeric']:,} of {profile['considered']:,} values
f"but are stored as text (may be intentional, e.g. ZIP codes
})
# (b) Date-looking values stored as strings.
if column in groups["date_text"]:
concerns.append("Date-looking values stored as text")
findings.append({
"Severity": "Low", "Column": column, "Issue": ISSUE_DATE_TEXT,
"Details": "Values look like dates but the column is stored as text, not
})
# (c) Mixed Python types inside one object column (e.g. numbers and text together
if pd.api.types.is_object_dtype(series.dtype):
type_counts = series.dropna().map(python_type_label).value_counts()
if len(type_counts) > 1:
minority = int(type_counts.sum() - type_counts.iloc[0])
mixed_type_cells += minority
summary = ", ".join(f"{label}: {count:,}" for label, count in type_counts
concerns.append("Mixed data types")
findings.append({
"Severity": "Medium", "Column": column, "Issue": ISSUE_MIXED_TYPES,
"Details": f"Column holds more than one kind of value ({summary})",
})
records.append({
"Column": column,
"Detected Data Type": original_dtypes.get(column, str(series.dtype)),
"Non-Null Values": non_null,
"Unique Values": safe_nunique(series),
"Potential Concern": "; ".join(concerns) if concerns else "None detected",
})
return pd.DataFrame(records), findings, mixed_type_cells
def detect_invalid_values(df, groups):
"""
Detect values that are unlikely to be valid. Rules used:
* Numeric columns: infinite values; negative values in columns whose NAME
suggests they should not be negative (age, price, income, ...); ages above 120.
* Text columns: placeholder text such as 'N/A' or 'null'; non-numeric
entries inside an otherwise numeric-looking column.
(Invalid dates are handled in analyze_datetime_quality().)
Returns (findings, invalid_counts) where invalid_counts maps
column -> number of distinct invalid cells (each cell counted once).
"""
findings, invalid_counts = [], {}
# ---- numeric columns ----------------------------------------------------
for column in groups["numeric"]:
try:
raw = pd.to_numeric(df[column], errors="coerce").astype("float64")
except (TypeError, ValueError):
continue
infinite = np.isinf(raw)
finite = raw.where(~infinite)
tokens = column_name_tokens(column)
invalid_mask = infinite.copy()
if infinite.sum() > 0:
findings.append({
"Severity": "High", "Column": column, "Issue": ISSUE_INVALID,
"Details": f"{int(infinite.sum()):,} infinite values",
})
if tokens & NON_NEGATIVE_NAME_TOKENS:
negative = finite < 0
if negative.sum() > 0:
invalid_mask |= negative
findings.append({
"Severity": "High", "Column": column, "Issue": ISSUE_INVALID,
"Details": (f"{int(negative.sum()):,} negative values (the column name su
f"values should not be negative; please verify)"),
})
if "age" in tokens:
too_old = finite > MAX_REASONABLE_AGE
if too_old.sum() > 0:
invalid_mask |= too_old
findings.append({
"Severity": "Medium", "Column": column, "Issue": ISSUE_INVALID,
"Details": f"{int(too_old.sum()):,} values above {MAX_REASONABLE_AGE} in
})
if invalid_mask.sum() > 0:
invalid_counts[column] = invalid_counts.get(column, 0) + int(invalid_mask.sum())
# ---- text columns (date-looking columns are checked separately) -----------
for column in groups["categorical"]:
series = df[column]
if not is_text_column(series) and not pd.api.types.is_object_dtype(series.dtype):
continue
non_null = int(series.notna().sum())
if non_null == 0:
continue
placeholders = placeholder_mask(series)
invalid_mask = placeholders.copy()
if placeholders.sum() > 0:
pct = placeholders.sum() / non_null * 100
findings.append({
"Severity": severity_from_percentage(pct), "Column": column, "Issue": ISSUE_P
"Details": (f"{int(placeholders.sum()):,} values such as "
f"{first_examples(series[placeholders])} may stand in for missing
})
profile = numeric_text_profile(series)
# Only treat leftovers as invalid if the column is overwhelmingly numeric.
if profile["considered"] > 0 and NUMERIC_TEXT_MIN_SHARE <= profile["share"] < 1.0:
bad = profile["non_numeric_mask"]
invalid_mask |= bad
pct = bad.sum() / non_null * 100
findings.append({
"Severity": severity_from_percentage(pct), "Column": column, "Issue": ISSUE_I
"Details": (f"{int(bad.sum()):,} entries are not numbers in a mostly numeric
f"(e.g. {first_examples(series[bad])})"),
})
if invalid_mask.sum() > 0:
invalid_counts[column] = invalid_counts.get(column, 0) + int(invalid_mask.sum())
return findings, invalid_counts
def analyze_datetime_quality(df, groups):
"""
Date/time quality for real datetime columns and for text columns that look like dates.
Returns (table, findings, invalid_counts). Nothing is converted in the
user's data; text dates are parsed only inside this function to measure quality.
"""
columns = ["Column", "Column Type", "Non-Null Values", "Missing Dates", "Invalid Dates",
"Minimum Date", "Maximum Date", "Unique Dates"]
records, findings, invalid_counts = [], [], {}
for column in groups["datetime"] + groups["date_text"]:
series = df[column]
missing = int(series.isna().sum())
non_null = int(series.notna().sum())
is_text = column in groups["date_text"]
invalid = 0
if is_text:
# Parse text values (excluding placeholders such as 'N/A', which are reported els
present = series.notna() & ~placeholder_mask(series)
texts = series[present].astype(str).str.strip()
parsed = parse_dates(texts)
bad = parsed.isna()
invalid = int(bad.sum())
valid = parsed.dropna()
if invalid > 0:
pct = invalid / max(len(texts), 1) * 100
invalid_counts[column] = invalid
findings.append({
"Severity": severity_from_percentage(pct), "Column": column, "Issue": ISS
"Details": (f"{invalid:,} of {len(texts):,} values could not be read as v
f"(e.g. {first_examples(texts[bad])})"),
})
else:
valid = series.dropna()
try:
minimum, maximum = format_timestamp(valid.min()), format_timestamp(valid.max())
except Exception:
minimum = maximum = "n/a"
try:
unique_dates = int(valid.dt.date.nunique())
except Exception:
unique_dates = int(len({v.date() for v in valid if hasattr(v, "date")}))
records.append({
"Column": column,
"Column Type": "Text that looks like dates" if is_text else "Date/time",
"Non-Null Values": non_null,
"Missing Dates": missing,
"Invalid Dates": invalid,
"Minimum Date": minimum if len(valid) else "n/a",
"Maximum Date": maximum if len(valid) else "n/a",
"Unique Dates": unique_dates if len(valid) else 0,
})
return pd.DataFrame(records, columns=columns), findings, invalid_counts
# =============================================================================
# SECTION 5: NUMERIC QUALITY AND OUTLIERS
# =============================================================================
def detect_outliers(df, numeric_columns):
"""
Potential outliers with the IQR method.
Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
Lower bound = Q1 - 1.5 * IQR, Upper bound = Q3 + 1.5 * IQR
A value outside [lower, upper] is a POTENTIAL outlier.
A potential outlier is not necessarily an error; it may be a legitimate observation.
"""
columns = ["Column", "Q1", "Q3", "IQR", "Lower Bound", "Upper Bound",
"Potential Outliers", "Outlier %", "Note"]
records = []
for column in numeric_columns:
values = get_numeric_values(df[column])
record = {"Column": column, "Q1": np.nan, "Q3": np.nan, "IQR": np.nan, "Lower Bound":
"Upper Bound": np.nan, "Potential Outliers": 0, "Outlier %": 0.0, "Note": "
if len(values) < MIN_VALUES_FOR_OUTLIERS:
record["Note"] = "Too few numeric values for the IQR method"
records.append(record)
continue
# Q1, Q3 and the interquartile range (IQR).
q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
iqr = q3 - q1
record.update({"Q1": q1, "Q3": q3, "IQR": iqr})
if iqr == 0:
# If the middle 50% of values are identical, every different value would be
# flagged, which is not informative. Report this instead of guessing.
record["Lower Bound"], record["Upper Bound"] = q1, q3
record["Note"] = "IQR is 0, so the IQR method is not informative for this column"
else:
lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
outliers = int(((values < lower) | (values > upper)).sum())
record.update({
"Lower Bound": lower, "Upper Bound": upper, "Potential Outliers": outliers,
"Outlier %": outliers / len(values) * 100,
})
records.append(record)
table = pd.DataFrame(records, columns=columns)
float_columns = ["Q1", "Q3", "IQR", "Lower Bound", "Upper Bound", "Outlier %"]
table[float_columns] = table[float_columns].astype(float).round(4)
return table
def analyze_numeric_quality(df, numeric_columns, outliers_table):
"""Summary statistics per numerical column (min, max, mean, median, std, missing, unique,
columns = ["Column", "Minimum", "Maximum", "Mean", "Median", "Std Deviation",
"Missing Values", "Unique Values", "Potential Outliers"]
outlier_lookup = {}
if not outliers_table.empty:
outlier_lookup = dict(zip(outliers_table["Column"], outliers_table["Potential Outlier
records = []
for column in numeric_columns:
values = get_numeric_values(df[column])
has_values = len(values) > 0
records.append({
"Column": column,
"Minimum": float(values.min()) if has_values else np.nan,
"Maximum": float(values.max()) if has_values else np.nan,
"Mean": float(values.mean()) if has_values else np.nan,
"Median": float(values.median()) if has_values else np.nan,
"Std Deviation": float(values.std()) if len(values) > 1 else np.nan,
"Missing Values": int(df[column].isna().sum()),
"Unique Values": safe_nunique(df[column]),
"Potential Outliers": int(outlier_lookup.get(column, 0)),
})
table = pd.DataFrame(records, columns=columns)
stat_columns = ["Minimum", "Maximum", "Mean", "Median", "Std Deviation"]
table[stat_columns] = table[stat_columns].astype(float).round(4)
return table
# =============================================================================
# SECTION 6: CATEGORICAL CONSISTENCY
# =============================================================================
def normalize_category(value):
"""
Reduce a category to a comparison key: lowercase, with spaces, punctuation
and underscores removed. "New York", "new york", "NEW YORK" and "NewYork"
all become "newyork".
"""
return re.sub(r"[\W_]+", "", str(value).lower())
def analyze_categorical_consistency(df, categorical_columns):
"""
Look for categories that probably mean the same thing but are written differently.
Returns (table, findings, inconsistent_cells):
inconsistent_cells counts rows written differently from the most common
spelling in their group. This feeds the Consistency score.
"""
columns = ["Column", "Comparison Key", "Variants Found", "Difference Type", "Rows Not Mat
records, findings, inconsistent_cells = [], [], 0
for column in categorical_columns:
series = df[column]
non_null = int(series.notna().sum())
if non_null == 0:
continue
try:
counts = series.dropna().astype(str).value_counts()
except Exception:
continue
if len(counts) > MAX_UNIQUE_FOR_CONSISTENCY:
continue # too many distinct values: likely free text or identifiers
# Group spellings that share the same comparison key.
grouped = {}
for variant, count in counts.items():
key = normalize_category(variant)
if not any(character.isalpha() for character in key):
continue # purely numeric keys are ignored (e.g. "1.0" vs "10" are different
grouped.setdefault(key, []).append((variant, int(count)))
column_groups, column_rows = 0, 0
example_group = ""
for key, variants in grouped.items():
if len(variants) < 2:
continue
variants.sort(key=lambda item: -item[1]) # most common spelling first
rows_differing = sum(count for _, count in variants[1:])
collapsed = {re.sub(r"\s+", " ", v.strip()).lower() for v, _ in variants}
difference = "Capitalization or extra whitespace" if len(collapsed) == 1 else "Sp
records.append({
"Column": column,
"Comparison Key": key,
"Variants Found": ", ".join(f"'{v}' ({c:,})" for v, c in variants),
"Difference Type": difference,
"Rows Not Matching Most Common Spelling": rows_differing,
})
column_groups += 1
column_rows += rows_differing
if not example_group:
example_group = " / ".join(f"'{v}'" for v, _ in variants[:3])
if column_groups:
inconsistent_cells += column_rows
pct = column_rows / non_null * 100
severity = "Medium" if (column_groups >= 3 or pct >= 5) else "Low"
findings.append({
"Severity": severity, "Column": column, "Issue": ISSUE_INCONSISTENT,
"Details": (f"{column_groups} group(s) of values may be the same category wri
f"(e.g. {example_group}); {column_rows:,} rows differ from the mo
})
return pd.DataFrame(records, columns=columns), findings, inconsistent_cells
# =============================================================================
# SECTION 7: OVERALL SCORE, ISSUE LIST, RECOMMENDATIONS
# =============================================================================
def calculate_quality_score(total_rows, total_columns, missing_cells, duplicate_rows,
id_uniqueness_percentages, invalid_cells, inconsistent_cells):
"""
Overall Data Quality Score (0-100). Four dimensions, equally weighted.
Every input is a plain count taken from the dataset, so the score is
reproducible: the same dataset always gives the same score.
"""
# Total number of cells, and how many of them contain something (are not missing).
total_cells = total_rows * total_columns
non_missing_cells = max(total_cells - missing_cells, 0)
# COMPLETENESS = share of cells that are NOT missing.
# completeness = (1 - missing cells / total cells) * 100
completeness = (1 - missing_cells / total_cells) * 100 if total_cells else 0.0
# UNIQUENESS = how free the dataset is of unexpected repeats. Two parts:
# (a) row uniqueness = (1 - duplicate rows / total rows) * 100
# (b) for columns NAMED like identifiers (id, uuid, key ...) the average
# share of values that are unique.
# If there are ID-like columns, uniqueness is the average of (a) and (b);
# otherwise it is just (a). Other columns are NOT expected to be unique,
# so repeated values in them do not reduce this score.
row_uniqueness = (1 - duplicate_rows / total_rows) * 100 if total_rows else 0.0
if id_uniqueness_percentages:
id_uniqueness = float(np.mean(id_uniqueness_percentages))
uniqueness = (row_uniqueness + id_uniqueness) / 2
else:
uniqueness = row_uniqueness
# VALIDITY = share of non-missing cells that are not flagged invalid
# (infinite/negative-where-unexpected/impossible ages, placeholder text,
# non-numeric entries in numeric columns, unparseable dates).
# validity = (1 - invalid cells / non-missing cells) * 100
validity = (1 - min(invalid_cells, non_missing_cells) / non_missing_cells) * 100 if non_m
# CONSISTENCY = share of non-missing cells that are not written inconsistently
# (category spellings that differ from the most common spelling of the
# same category, and minority types inside mixed-type columns).
# consistency = (1 - inconsistent cells / non-missing cells) * 100
consistency = (1 - min(inconsistent_cells, non_missing_cells) / non_missing_cells) * 100
# Keep every dimension between 0 and 100.
dimensions = {
"completeness": float(np.clip(completeness, 0, 100)),
"uniqueness": float(np.clip(uniqueness, 0, 100)),
"validity": float(np.clip(validity, 0, 100)),
"consistency": float(np.clip(consistency, 0, 100)),
}
# OVERALL SCORE = simple average of the four dimensions (equal weights).
overall = float(np.mean(list(dimensions.values())))
# Score bands: reading aid used by this application only.
if overall >= 90:
rating = "Excellent"
elif overall >= 75:
rating = "Good"
elif overall >= 50:
rating = "Fair"
else:
rating = "Needs attention"
return {**dimensions, "overall": overall, "rating": rating,
"missing_cells": int(missing_cells), "duplicate_rows": int(duplicate_rows),
"invalid_cells": int(invalid_cells), "inconsistent_cells": int(inconsistent_cells
def build_issue_list(missing, duplicates, duplicate_values, empty_columns, constant_columns,
outliers, extra_findings):
"""
Combine every detected problem into one table: Severity, Column, Issue, Details.
`extra_findings` holds findings already produced by the type, validity,
date and consistency checks. Wording is deliberately cautious ("potential",
"may") because the report detects patterns; it does not know the user's intent.
"""
issues = list(extra_findings)
# Empty columns (High).
for _, row in empty_columns.iterrows():
issues.append({"Severity": "High", "Column": row["Column"], "Issue": ISSUE_EMPTY,
"Details": f"All {int(row['Number of Values']):,} values are missing o
empty_names = set(empty_columns["Column"]) if not empty_columns.empty else set()
# Missing values (severity from the missing-value thresholds; fully empty columns already
severity_map = {"High concern": "High", "Moderate concern": "Medium", "Low concern": "Low
for _, row in missing.iterrows():
if row["Missing Count"] > 0 and row["Column"] not in empty_names:
issues.append({"Severity": severity_map[row["Severity"]], "Column": row["Column"]
"Details": f"{int(row['Missing Count']):,} missing values ({row['M
# Duplicate rows.
if duplicates["duplicate_rows"] > 0:
issues.append({"Severity": severity_from_percentage(duplicates["percentage"]), "Issue": ISSUE_DUP_ROWS,
"Details": f"{duplicates['duplicate_rows']:,} rows ({duplicates['perce
"Colum
# Duplicate values in ID-like columns.
if not duplicate_values.empty:
for _, row in duplicate_values[duplicate_values["Looks Like an ID Column"]].iterrows(
if row["Duplicate Values"] > 0:
issues.append({"Severity": "High", "Column": row["Column"], "Issue": ISSUE_DU
"Details": (f"{int(row['Duplicate Values']):,} repeated values
f"columns named like identifiers are usually expec
# Constant columns.
for _, row in constant_columns.iterrows():
issues.append({"Severity": "Low", "Column": row["Column"], "Issue": ISSUE_CONSTANT,
"Details": f"Every non-null value is '{row['Value']}'; may add limited
# Potential outliers.
if not outliers.empty:
for _, row in outliers[outliers["Potential Outliers"] > 0].iterrows():
severity = "Medium" if row["Outlier %"] >= OUTLIER_MEDIUM_PCT else "Low"
issues.append({"Severity": severity, "Column": row["Column"], "Issue": ISSUE_OUTL
"Details": (f"{int(row['Potential Outliers']):,} potential outlier
f"outside {row['Lower Bound']:.6g} to {row['Upper Boun
table = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
if table.empty:
return table
table["_order"] = table["Severity"].map(SEVERITY_ORDER)
table = table.sort_values(["_order", "Column"], kind="stable").drop(columns="_order")
return table.reset_index(drop=True)
def generate_recommendations(issues):
"""Turn the detected issues into practical, numbered actions. Only issues that exist prod
if issues.empty:
return ["No issues were detected by the checks in this report. Continue to review the
"sections above and confirm the data matches what you expect."]
def columns_for(*issue_names, severity=None):
subset = issues[issues["Issue"].isin(issue_names)]
if severity is not None:
subset = subset[subset["Severity"].isin(severity)]
names = list(dict.fromkeys(subset["Column"]))
shown = ", ".join(f"`{name}`" for name in names[:5])
return names, shown + (f" and {len(names) - 5} more" if len(names) > 5 else "")
recommendations = []
names, text = columns_for(ISSUE_EMPTY)
if names:
recommendations.append(f"Review empty columns ({text}). Confirm whether they were exp
names, text = columns_for(ISSUE_MISSING, severity=("High", "Medium"))
if names:
recommendations.append(f"Review columns with high or moderate missing-value percentag
f"(for example collect the data, fill it in, or exclude the co
if (issues["Issue"] == ISSUE_DUP_ROWS).any():
recommendations.append("Investigate potential duplicate records. Confirm whether repe
"or accidental copies before removing anything.")
names, text = columns_for(ISSUE_DUP_ID)
if names:
recommendations.append(f"Check ID-like columns ({text}) for repeated values and confi
names, text = columns_for(ISSUE_INVALID)
if names:
recommendations.append(f"Review invalid values in {text} (for example negative, infin
names, text = columns_for(ISSUE_INVALID_DATES)
if names:
recommendations.append(f"Check the date values in {text} that could not be read as va
names, text = columns_for(ISSUE_PLACEHOLDER)
if names:
recommendations.append(f"Decide how placeholder text such as N/A or null in {text} sh
names, text = columns_for(ISSUE_OUTLIERS)
if names:
recommendations.append(f"Review potential outliers in {text} before removing them; th
names, text = columns_for(ISSUE_INCONSISTENT)
if names:
recommendations.append(f"Standardize inconsistent categorical values in {text} so tha
names, text = columns_for(ISSUE_MIXED_TYPES)
if names:
recommendations.append(f"Review columns containing mixed data types ({text}) and deci
names, text = columns_for(ISSUE_NUMERIC_TEXT)
if names:
recommendations.append(f"Confirm whether numeric-looking text columns ({text}) should
f"Identifiers such as ZIP codes are often better kept as text.
names, text = columns_for(ISSUE_DATE_TEXT)
if names:
recommendations.append(f"Consider storing date-looking text columns ({text}) as a dat
names, text = columns_for(ISSUE_CONSTANT)
if names:
recommendations.append(f"Review constant columns ({text}). They add little analytical
return recommendations
# =============================================================================
# SECTION 8: RUN THE FULL ANALYSIS
# =============================================================================
def _safe_run(results, label, function, default, *args):
"""
Run one check. If it fails for any unexpected reason, record a friendly
warning, log the technical details privately, and continue with `default`
so one broken check cannot stop the whole report.
"""
try:
return function(*args)
except Exception:
logger.exception("Data quality check failed: %s", label)
results["warnings"].append(f"The '{label}' check could not be completed and was skipp
return default
def run_quality_analysis(df, dataset_name="Uploaded dataset"):
"""
Run every data-quality check and return one results dictionary.
The uploaded DataFrame is only READ. Raises ValueError if the DataFrame is
unusable (display_quality_report() checks this first and shows a message).
"""
level, message = check_dataframe(df)
if level is not None:
raise ValueError(message)
results = {"warnings": []}
empty = pd.DataFrame()
analysis_df, original_dtypes = prepare_dataframe(df)
groups = classify_columns(analysis_df)
results["groups"] = groups
results["overview"] = build_overview(analysis_df, groups, dataset_name)
# Individual checks (each protected by _safe_run).
missing = _safe_run(results, "Missing Values", calculate_missing_values, empty, analysis_
duplicates = _safe_run(results, "Duplicate Rows", calculate_duplicates,
{"duplicate_rows": 0, "percentage": 0.0, "total_rows": len(analysi
duplicate_values = _safe_run(results, "Duplicate Values", calculate_duplicate_values, emp
data_types, type_findings, mixed_cells = _safe_run(
results, "Data Types", analyze_data_types, (empty, [], 0), analysis_df, original_dtyp
empty_columns = _safe_run(results, "Empty Columns", detect_empty_columns, empty, analysis
constant_columns = _safe_run(results, "Constant Columns", detect_constant_columns, empty,
outliers = _safe_run(results, "Potential Outliers", detect_outliers, empty, analysis_df,
numeric_quality = _safe_run(results, "Numeric Data Quality", analyze_numeric_quality, emp
analysis_df, groups["numeric"], outliers)
categorical, category_findings, inconsistent_cells = _safe_run(
results, "Categorical Consistency", analyze_categorical_consistency, (empty, [], 0),
analysis_df, groups["categorical"])
invalid_findings, invalid_counts = _safe_run(
results, "Invalid Values", detect_invalid_values, ([], {}), analysis_df, groups)
datetime_quality, date_findings, invalid_date_counts = _safe_run(
results, "Date/Time Quality", analyze_datetime_quality, (empty, [], {}), analysis_df,
# Gather counts for the score. Invalid cells from the value checks and the date check are
# (placeholders are excluded from the date check), so they can be added.
missing_cells = int(missing["Missing Count"].sum()) if not missing.empty else 0
invalid_cells = sum(invalid_counts.values()) + sum(invalid_date_counts.values())
consistency_cells = inconsistent_cells + mixed_cells
id_uniqueness = []
if not duplicate_values.empty:
for _, row in duplicate_values[duplicate_values["Looks Like an ID Column"]].iterrows(
if row["Non-Null Values"] > 0:
id_uniqueness.append(100 - row["Duplicate Values"] / row["Non-Null Values"] *
score = calculate_quality_score(
total_rows=len(analysis_df), total_columns=analysis_df.shape[1], missing_cells=missin
duplicate_rows=duplicates["duplicate_rows"], id_uniqueness_percentages=id_uniqueness,
invalid_cells=invalid_cells, inconsistent_cells=consistency_cells)
extra = type_findings + invalid_findings + date_findings + category_findings
issues = _safe_run(results, "Issue List", build_issue_list, pd.DataFrame(columns=ISSUE_CO
missing, duplicates, duplicate_values, empty_columns, constant_columns
recommendations = _safe_run(results, "Recommendations", generate_recommendations, [], iss
results.update({
"missing": missing, "duplicates": duplicates, "duplicate_values": duplicate_values,
"data_types": data_types, "empty_columns": empty_columns, "constant_columns": constan
"outliers": outliers, "numeric_quality": numeric_quality, "categorical": categorical,
"datetime_quality": datetime_quality, "score": score, "issues": issues,
"recommendations": recommendations,
})
return results
# =============================================================================
# SECTION 9: CONNECTING TO PART 1 (DATA UPLOAD)
# =============================================================================
def get_uploaded_dataframe():
"""
Fetch the DataFrame created by Part 1 (Data Upload) from st.session_state.
Returns (DataFrame or None, dataset_name). None means Part 1 has not stored
a dataset yet. The DataFrame is returned as-is: this module never modifies it.
"""
dataframe = None
for key in DATAFRAME_SESSION_KEYS:
candidate = st.session_state.get(key)
if isinstance(candidate, pd.DataFrame):
dataframe = candidate
break
dataset_name = "Uploaded dataset"
for key in DATASET_NAME_SESSION_KEYS:
candidate = st.session_state.get(key)
if candidate:
dataset_name = str(candidate)
break
return dataframe, dataset_name
def dataframe_fingerprint(df):
"""
"""
try:
A cheap identity for a DataFrame, used to avoid recomputing the report when
Streamlit reruns the script without the data having changed.
content_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
except Exception: # unhashable cell contents: fall back to object identity
content_hash = id(df)
return (df.shape, tuple(map(str, df.columns)), tuple(map(str, df.dtypes)), content_hash)
def get_or_compute_results(df, dataset_name):
"""Return cached results for this exact dataset, or run the analysis and cache it in the
fingerprint = (dataset_name, dataframe_fingerprint(df))
cached = st.session_state.get("dq_report_cache")
if cached and cached["fingerprint"] == fingerprint:
return cached["results"]
results = run_quality_analysis(df, dataset_name)
st.session_state["dq_report_cache"] = {"fingerprint": fingerprint, "results": results}
return results
# =============================================================================
# SECTION 10: STREAMLIT DISPLAY FUNCTIONS
# Each function below draws ONE section of the report. The comments explain the
# link between the Python code, the user-facing text, and what appears on screen.
# =============================================================================
def display_overview(results):
"""Dataset Overview: name, size, and column-type counts."""
overview = results["overview"]
# The heading "Dataset Overview" appears as a section title at the top of the
# report and tells the user that the metrics below are high-level facts
# (size and column types) about the uploaded dataset.
st.subheader("Dataset Overview")
# The line "Dataset name: <file name>" appears under the heading and shows
# which uploaded dataset this report describes (the name Part 1 stored).
st.markdown(f"**Dataset name:** `{overview['dataset_name']}`")
# st.columns(4) makes four side-by-side slots. Each st.metric below shows a
# label (for example "Rows") with its number, computed from the DataFrame shape.
first_row = st.columns(4)
first_row[0].metric("Rows", f"{overview['rows']:,}") first_row[1].metric("Columns", f"{overview['columns']:,}") first_row[2].metric("Total Cells", f"{overview['total_cells']:,}") # visible label "Rows
# visible label "Col
# rows x columns
first_row[3].metric("Numerical Columns", f"{overview['numeric_columns']:,}")
second_row = st.columns(4)
second_row[0].metric("Categorical Columns", f"{overview['categorical_columns']:,}") # te
second_row[1].metric("Date/Time Columns", f"{overview['datetime_columns']:,}") # re
def display_score(results):
"""Overall Data Quality Score and its four dimensions."""
score = results["score"]
# The heading "Overall Data Quality Score" introduces the single 0-100 number
# that summarises the dataset, plus the four dimensions it is built from.
st.subheader("Overall Data Quality Score")
# The big metric shows the score as, for example, "87/100". It is the average
# of Completeness, Uniqueness, Validity and Consistency computed by
# calculate_quality_score().
st.metric("Overall Data Quality Score", f"{score['overall']:.0f}/100")
# A colored message shows the score band. Green ("success") for 90+, blue
# ("info") for 75-89, and yellow ("warning") below 75. The text tells the user
# the band name and reminds them these bands are this application's own scale.
message = f"Rating: {score['rating']} (score bands used by this application: 90+ Excellen
if score["overall"] >= 90:
st.success(message)
elif score["overall"] >= 75:
st.info(message)
else:
st.warning(message)
# Four metrics show each dimension as a percentage under the overall score.
columns = st.columns(4)
columns[0].metric("Completeness", f"{score['completeness']:.1f}%") # share of cells that
columns[1].metric("Uniqueness", f"{score['uniqueness']:.1f}%") # freedom from duplic
columns[2].metric("Validity", f"{score['validity']:.1f}%") # share of values not
columns[3].metric("Consistency", f"{score['consistency']:.1f}%") # share of values wri
# The expander titled "How is this score calculated?" is collapsed by default;
# clicking it reveals the plain-language formulas so the score is transparent.
with st.expander("How is this score calculated?"):
# This block of text appears inside the expander and explains each formula
# and shows the actual counts used for THIS dataset.
st.markdown(
"The score is the simple average of four dimensions (each 0-100%).\n\n"
f"- **Completeness** = (1 - missing cells / total cells) x 100. Missing cells her
f"- **Uniqueness** = (1 - duplicate rows / total rows) x 100. If some columns are
f"(id, key, uuid...), it is the average of that value and the share of unique val
f"Duplicate rows here: {score['duplicate_rows']:,}.\n"
f"- **Validity** = (1 - invalid cells / non-missing cells) x 100. Invalid cells h
f"(infinite values, negatives or impossible ages in columns whose names suggest t
f"placeholder text like N/A, non-numeric entries in numeric columns, unparseable
f"- **Consistency** = (1 - inconsistent cells / non-missing cells) x 100. Inconsi
f"{score['inconsistent_cells']:,} (category spellings that differ from the most c
f"minority types in mixed-type columns).\n\n"
"The same dataset always produces the same score. The rules and thresholds "application, not universal standards."
are in
)
def display_missing_values(results):
"""Completeness section: missing values per column."""
missing = results["missing"]
# The heading "Missing Values" (Completeness) marks the section that shows how
# much data is absent in each column.
st.subheader("Missing Values")
if missing.empty:
# This warning appears if the missing-value check failed, so the user knows the secti
st.warning("Missing-value information is not available for this dataset.")
return
with_missing = missing[missing["Missing Count"] > 0]
# A caption under the heading explains that the labels are this application's own indicat
st.caption("Severity labels are data-quality indicators used by this application "
f"(Complete = 0%, Low concern < {MISSING_LOW_MAX:g}%, Moderate concern < {MISS
"High concern otherwise). They are not universal rules.")
if with_missing.empty:
# This green message appears when every cell has a value.
st.success("No missing values were detected. Every column is complete.")
else:
# A summary sentence tells the user how many columns contain missing data.
st.markdown(f"**{len(with_missing)} of {len(missing)} columns contain missing values.
f"Columns with the highest percentage of missing data:")
# This table lists (up to) the five columns with the highest missing percentage.
st.dataframe(with_missing.head(5), hide_index=True)
# The expander "All columns: missing-value details" holds the full table (every column,
# including complete ones) so nothing is hidden from the user.
with st.expander("All columns: missing-value details"):
st.dataframe(missing, hide_index=True)
def display_duplicate_rows(results):
"""Duplicate rows count and percentage."""
duplicates = results["duplicates"]
# The heading "Duplicate Rows" starts the section about rows that repeat exactly.
st.subheader("Duplicate Rows")
# The metric shows the label "Duplicate Rows" with the count (for example 342)
# and a small note such as "2.4% of dataset" that gives the percentage.
st.metric("Duplicate Rows", f"{duplicates['duplicate_rows']:,}",
delta=f"{duplicates['percentage']:.1f}% of dataset", delta_color="off")
# A caption explains how duplicates are counted.
st.caption("A row is counted as a duplicate when every column matches an earlier row. The
def display_duplicate_values(results):
"""Duplicate values per column."""
table = results["duplicate_values"]
# The heading "Duplicate Values" starts the section about repeated values within each col
st.subheader("Duplicate Values")
# This caption explains that repeated values are not automatically a problem.
st.caption("Uniqueness expectations depend on the purpose of a column. Repeated values ar
"Country or Gender, but usually unexpected in an identifier column. Columns wh
"identifiers are marked in the last column.")
if table.empty:
st.warning("Duplicate-value information is not available for this dataset.")
return
# The table shows, for each column, the unique values, duplicate values and duplicate per
st.dataframe(table, hide_index=True)
def display_data_types(results):
"""Data types and potential type concerns."""
table = results["data_types"]
# The heading "Data Types" starts the section describing how each column is stored.
st.subheader("Data Types")
# The caption reminds the user that no data has been converted.
st.caption("The report only detects potential type problems. Your original data has not b
if table.empty:
st.warning("Data-type information is not available for this dataset.")
return
# The table lists each column's detected type, non-null count, unique count and any poten
st.dataframe(table, hide_index=True)
def display_empty_columns(results):
"""Empty columns."""
table = results["empty_columns"]
# The heading "Empty Columns" starts the section about columns with no data.
st.subheader("Empty Columns")
if table.empty:
# This green message appears when every column contains at least one value.
st.success("No empty columns were detected.")
else:
# This warning appears when at least one column has no data; the table below names th
st.warning(f"{len(table)} empty column(s) detected: every value is missing or blank."
st.dataframe(table, hide_index=True)
def display_constant_columns(results):
"""Constant columns."""
table = results["constant_columns"]
# The heading "Constant Columns" starts the section about columns with only one distinct
st.subheader("Constant Columns")
if table.empty:
# This green message appears when no column has just one repeated value.
st.success("No constant columns were detected.")
else:
# This information message explains what a constant column means and that it is not a
st.info(f"{len(table)} constant column(s) detected. A column where every value is the
"limited analytical value, but it should not be deleted automatically: it may
st.dataframe(table, hide_index=True)
def display_numeric_quality(results):
"""Numeric data quality summary."""
table = results["numeric_quality"]
# The heading "Numeric Data Quality" starts the section with summary statistics for numer
st.subheader("Numeric Data Quality")
if table.empty:
# This message appears when the dataset has no numerical columns.
st.info("No numerical columns were found in this dataset.")
return
# The table shows min, max, mean, median, standard deviation, missing, unique and potenti
st.dataframe(table, hide_index=True)
def display_outliers(results):
"""Potential outliers by IQR."""
table = results["outliers"]
# The heading "Potential Outliers" starts the section that lists unusually high/low numer
st.subheader("Potential Outliers")
if table.empty:
# This message appears when there are no numerical columns to check.
st.info("No numerical columns were found, so outlier detection was skipped.")
return
# The caption explains the method and that outliers are not automatically errors.
st.caption("Method: IQR. Values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR are flagged
"A potential outlier is not necessarily an error; it may be a legitimate obser
# The table shows Q1, Q3, IQR, bounds, the number and percentage of potential outliers fo
st.dataframe(table, hide_index=True)
def display_categorical_consistency(results):
"""Categorical consistency."""
table = results["categorical"]
# The heading "Categorical Consistency" starts the section that looks for the same st.subheader("Categorical Consistency")
if table.empty:
# This green message appears when no suspicious spelling variations were found.
st.success("No potentially inconsistent categorical values were detected.")
return
catego
# This warning explains what the table shows and that nothing was changed.
st.warning("Some values may represent the same category written differently (for example
"'new york', 'NEW YORK'). These are possibilities, not certainties. Nothing ha
st.dataframe(table, hide_index=True)
def display_datetime_quality(results):
"""Date/time quality."""
table = results["datetime_quality"]
# The heading "Date/Time Quality" starts the section about date columns.
st.subheader("Date/Time Quality")
if table.empty:
# This message appears when no date or date-looking columns were detected.
st.info("No date/time columns were detected in this dataset.")
return
# The caption clarifies that text dates are only parsed temporarily for measurement.
st.caption("Text columns that look like dates are parsed only to measure quality. The ori
st.dataframe(table, hide_index=True)
def display_issues(results):
"""Centralized data-quality issue table with a severity filter."""
issues = results["issues"]
# The heading "Data Quality Issues" starts the central list of everything the checks dete
st.subheader("Data Quality Issues")
if issues.empty:
# This green message appears when the checks found nothing to report.
st.success("No data quality issues were detected by the checks in this report.")
return
# Three metrics show how many High / Medium / Low severity issues were found.
counts = issues["Severity"].value_counts()
columns = st.columns(3)
columns[0].metric("High Severity", int(counts.get("High", 0)))
columns[1].metric("Medium Severity", int(counts.get("Medium", 0)))
columns[2].metric("Low Severity", int(counts.get("Low", 0)))
# The caption reminds the user that severity is an indicator.
st.caption("Severity is an indicator based on thresholds used by this application. "in the data; please review them in the context of your dataset.")
Findin
# The multiselect labelled "Show severities" lets the user choose which severities appear
selected = st.multiselect("Show severities", ["High", "Medium", "Low"], default=["High",
key="dq_severity_filter")
filtered = issues[issues["Severity"].isin(selected)]
# The table lists Severity, Column, Issue and Details for every issue that matches st.dataframe(filtered, hide_index=True)
the fi
def display_recommendations(results):
"""Recommended actions."""
# The heading "Recommendations" starts the list of suggested next steps.
st.subheader("Recommendations")
# The heading "Recommended Actions" (bold text) introduces the numbered list below.
st.markdown("**Recommended Actions**")
# Each recommendation appears as a numbered line. They are generated from the issues actu
# detected in this dataset (see generate_recommendations()).
st.markdown("\n".join(f"{number}. {text}" for number, text in enumerate(results["recommen
# This caption confirms the report is read-only: it never edits the user's data.
st.caption("This report is read-only. Your uploaded dataset has not been modified. Apply
def display_quality_report(df, dataset_name="Uploaded dataset"):
"""
Entry point for Part 2. app.py calls this with the DataFrame from Part 1.
Order of the page: overview -> score -> completeness -> duplicates -> types ->
empty/constant -> numeric/outliers -> categories -> dates -> issues -> recommendations.
"""
# The heading "Data Quality Report" is the main title of this page, so the
# user immediately knows which module they are viewing.
st.header("Data Quality Report")
# Validate the input first. Depending on the problem, the message below is shown as an
# error (red) or warning (yellow) box, and the report stops there.
level, message = check_dataframe(df)
if level == "error":
st.error(message) # red box: the data cannot be used at all (e.g. no columns
return
if level == "warning":
st.warning(message) # yellow box: e.g. no dataset uploaded yet, or no rows
if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
# A short line lists the columns so the user can see what was uploaded.
st.caption("Columns found: " + ", ".join(map(str, df.columns)))
return
# A spinner with the text "Analyzing dataset quality..." is visible while the checks run.
try:
with st.spinner("Analyzing dataset quality..."):
results = get_or_compute_results(df, dataset_name)
except Exception:
# Full technical details go to the log, not to the user.
logger.exception("Data quality analysis failed")
# This red box appears if something unexpected stops the entire analysis.
st.error("The data quality analysis could not be completed for this dataset. "
"Please check that the file was read correctly and try again.")
return
# This green message confirms that the analysis finished and the report below is ready.
st.success("Data quality analysis completed.")
# Any individual check that failed is listed here as a warning, so the user knows a secti
for warning_text in results["warnings"]:
st.warning(warning_text)
display_overview(results)
st.divider()
display_score(results)
st.divider()
display_missing_values(results)
st.divider()
display_duplicate_rows(results)
display_duplicate_values(results)
st.divider()
display_data_types(results)
st.divider()
display_empty_columns(results)
display_constant_columns(results)
st.divider()
display_numeric_quality(results)
display_outliers(results)
st.divider()
display_categorical_consistency(results)
st.divider()
display_datetime_quality(results)
st.divider()
display_issues(results)
st.divider()
display_recommendations(results)
def main():
"""Optional standalone mode: `streamlit run data_quality_report.py` (reads the dataset Pa
st.set_page_config(page_title="Data Quality Report", layout="wide")
# The title "Data Quality & Analytics Platform" appears at the top of the page in standal
st.title("Data Quality & Analytics Platform")
dataframe, dataset_name = get_uploaded_dataframe()
display_quality_report(dataframe, dataset_name)
if __name__ == "__main__":
main()