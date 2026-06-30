"""Reusable analysis for the UK road accident analytics project.

The functions here are designed to support the Jupyter notebook. 
They assume that the project database is available locally.
"""

from __future__ import annotations

# Standard libraries for database access, file paths, data containers and optional type hints
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Libraries for mapping, data manipulation, association-rule, mining, clustering, classification, preprocessing and time-series forecasting.
import folium
import numpy as np
import pandas as pd
from folium.plugins import Fullscreen, HeatMap, MarkerCluster, MiniMap
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, 
                             confusion_matrix, f1_score, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fix week day order to keep visualisations in a natural Monday to Sunday sequence instead of alphabetical order
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# The default database path
DEFAULT_DB_PATH = Path("data/raw/accident_data_v1.0.0_2023.db")

# Lookup dictionaries translate STATS19 (police reporting) numeric category codes into readable labels. 
# This makes the notebook outputs and visualisations easier to interpret.
LOOKUPS = {
    "accident_severity": {1: "Fatal", 2: "Serious", 3: "Slight"},
    "road_type": {
        1: "Roundabout",
        2: "One way street",
        3: "Dual carriageway",
        6: "Single carriageway",
        7: "Slip road",
        9: "Unknown"
    },
    "weather_conditions": {
        1: "Fine, no high winds",
        2: "Raining, no high winds",
        3: "Snowing, no high winds",
        4: "Fine with high winds",
        5: "Raining with high winds",
        6: "Snowing with high winds",
        7: "Fog or mist",
        8: "Other",
        9: "Unknown"
    },
    "road_surface_conditions": {
        1: "Dry",
        2: "Wet or damp",
        3: "Snow",
        4: "Frost or ice",
        5: "Flood over 3cm deep"
    },
    "light_conditions": {
        1: "Daylight",
        4: "Darkness, lights lit",
        5: "Darkness, lights unlit",
        6: "Darkness, no lighting",
        7: "Darkness, lighting unknown",
    },
    "urban_or_rural_area": {
        1: "Urban",
        2: "Rural",
        3: "Unallocated"
    },
    "sex_of_casualty": {1: "Male", 2: "Female", 9: "Unknown", -1: "Unknown"},
    "vehicle_type": {
        1: "Pedal cycle",
        2: "Motorcycle 50cc and under",
        3: "Motorcycle 125cc and under",
        4: "Motorcycle over 125cc and up to 500cc",
        5: "Motorcycle over 500cc",
        8: "Taxi / private hire car",
        9: "Car",
        10: "Minibus (8 to 16 passenger seats)",
        11: "Bus or coach (17 or more passenger seats)",
        16: "Ridden horse",
        17: "Agricultural vehicle",
        18: "Tram/Light rail",
        19: "Van / goods 3.5 tonnes mgw or under",
        20: "Goods over 3.5t and under 7.5t",
        21: "Goods 7.5 tonnes mgw and over",
        22: "Mobility scooter",
        23: "Electric motorcycle",
        90: "Other vehicle",
        97: "Motorcycle cc unknown",
        98: "Goods, unknown weight"
    },
    "police_force": {13: "West Yorkshire", 14: "South Yorkshire", 16: "Humberside"},
}


@dataclass
class ForecastResult:
    """Container for forecast outputs and metrics."""

    # Observed values for the test period.
    actual: pd.Series
    # Forecast from the SARIMAX model.
    sarimax_forecast: pd.Series
    # Seasonal-naive baseline used for comparison.
    naive_forecast: pd.Series
    # Error metrics comparing model forecasts against the observed values.
    metrics: pd.DataFrame


def connect_database(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection and raise a clear error if the DB is missing."""
    # Convert string paths to Path objects so validation works consistently
    path = Path(db_path)

    # Fail early with a clear message when the raw database file is missing.
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Place database in 'data/raw/' or pass a valid path.")
    return sqlite3.connect(path)


def _read_sql(db_path: str | Path, query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Read a SQL query from the accident database."""
    # The context manager closes the SQLite connection automatically after the query has been executed.
    with connect_database(db_path) as conn:
        # Parameters are passed separately to keep SQL queries reusable and safer than string interpolation.
        return pd.read_sql_query(query, conn, params=params or {})


def add_time_features(df: pd.DataFrame, date_col: str = "date", time_col: str = "time") -> pd.DataFrame:
    """Add parsed date, hour, weekday name, month and week columns."""
    # Duplicate the dataframe so the original remains unchanged.
    out = df.copy()

    if date_col in out.columns:
        # STATS19 dates are commonly stored in UK format, 'dayfirst=True' avoids silently misreading dates such as 05/04/2019
        out[date_col] = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce")
        out["day_of_week_name"] = out[date_col].dt.day_name()
        out["month"] = out[date_col].dt.month
        out["week"] = out[date_col].dt.isocalendar().week.astype("Int64")
    if time_col in out.columns:
        # Extracting hour for peak-time analysis and day-hour heatmaps.
        out["hour"] = pd.to_datetime(out[time_col], format="%H:%M", errors="coerce").dt.hour
    return out


def decode_columns(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Append columns using STATS19 style lookup dictionaries."""
    out = df.copy()
    # Default to all known lookup columns, but allow for a subset to be decoded when working with narrower dataframes.
    cols = columns or LOOKUPS.keys()
    for col in cols:
        # Only decode columns that exist in the current dataframe.
        if col in out.columns and col in LOOKUPS:
            out[f"{col}_label"] = out[col].map(LOOKUPS[col]).fillna(out[col].astype(str))
    return out


def load_accidents(db_path: str | Path = DEFAULT_DB_PATH, year: int = 2019) -> pd.DataFrame:
    """Load accidents for a given year with parsed time and decoded fields."""
    # SQL parameters keep the year configurable while avoiding manual string formatting inside the query.
    query = """
        SELECT *
        FROM accident
        WHERE accident_year = :year
    """
    df = _read_sql(db_path, query, {"year": year})
    df = add_time_features(df)
    df = decode_columns(df)
    return df


def load_all_accidents(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load all accident years available in the SQLite database."""
    # Useful for multi-year forecasting or cross-year exploratory analysis.
    df = _read_sql(db_path, "SELECT * FROM accident")
    df = add_time_features(df)
    df = decode_columns(df)
    return df


def load_motorcycle_accidents(db_path: str | Path = DEFAULT_DB_PATH, year: int = 2019) -> pd.DataFrame:
    """Load motorcycle-related accident records and add engine-size categories."""
    # Join accidents to vehicles so the analysis keeps the accident context while filtering specifically for motorcycle vehicle types.
    query = """
        SELECT
            a.accident_index,
            a.accident_year,
            a.date,
            a.time,
            a.accident_severity,
            a.latitude,
            a.longitude,
            a.road_type,
            a.speed_limit,
            a.weather_conditions,
            a.light_conditions,
            v.vehicle_reference,
            v.vehicle_type
        FROM accident a
        JOIN vehicle v ON a.accident_index = v.accident_index
        WHERE a.accident_year = :year
          AND v.vehicle_type IN (2, 3, 4, 5)
    """
    df = _read_sql(db_path, query, {"year": year})
    # Consolidate motorcycle vehicle codes into broader engine-size bands for cleaner charts and comparisons
    cc_map = {2: "≤125cc", 3: "≤125cc", 4: "125–500cc", 5: ">500cc"}
    df["cc_rating"] = df["vehicle_type"].map(cc_map)
    df = add_time_features(df)
    df = decode_columns(df)
    return df


def load_pedestrian_casualties(db_path: str | Path = DEFAULT_DB_PATH, year: int = 2019) -> pd.DataFrame:
    """Load pedestrian casualties for a given accident year."""
    # Join casualties back to accidents to retain location, road, weather and lighting info for pedestrian-specific analysis.
    query = """
        SELECT
            c.*,
            a.date,
            a.time,
            a.accident_severity,
            a.latitude,
            a.longitude,
            a.road_type,
            a.speed_limit,
            a.weather_conditions,
            a.light_conditions
        FROM casualty c
        JOIN accident a ON c.accident_index = a.accident_index
        WHERE c.accident_year = :year
          AND c.casualty_class = 3
    """
    df = _read_sql(db_path, query, {"year": year})
    df = add_time_features(df)
    df = decode_columns(df)
    if "sex_of_casualty_label" in df.columns:
        # Create a clearer alias for dashboard labels while preserving the original decoded STATS19 column.
        df["casualty_gender"] = df["sex_of_casualty_label"]
    return df


def load_regional_accidents(db_path: str | Path = DEFAULT_DB_PATH, year: int = 2019) -> pd.DataFrame:
    """Load accidents in Kingston upon Hull and East Riding/Humberside LSOAs."""
    # Joining with the LSOA lookup allows the regional filter to use place names rather than only coded geography identifiers.
    query = """
        SELECT
            a.*,
            l.lsoa01cd,
            l.lsoa01nm
        FROM accident a
        JOIN lsoa l ON a.lsoa_of_accident_location = l.lsoa01cd
        WHERE a.accident_year = :year
          AND (
              l.lsoa01nm LIKE '%Kingston upon Hull%'
              OR l.lsoa01nm LIKE '%East Riding of Yorkshire%'
              OR l.lsoa01nm LIKE '%Humberside%'
          )
          AND a.latitude IS NOT NULL
          AND a.longitude IS NOT NULL
    """
    df = _read_sql(db_path, query, {"year": year})
    df = add_time_features(df)
    df = decode_columns(df)
    return df


def day_hour_matrix(df: pd.DataFrame, id_col: str = "accident_index") -> pd.DataFrame:
    """Return a day-of-week by hour matrix for heatmap visualisation."""
    # Count accidents for each weekday-hour combination.
    pivot = df.pivot_table(
        index="day_of_week_name",
        columns="hour",
        values=id_col,
        aggfunc="count",
        fill_value=0,
    )
    # Reindex to DAY_ORDER so missing weekdays appear as zero rows and the output is always ordered consistently for visualisation.
    return pivot.reindex(DAY_ORDER).fillna(0).astype(int)


def accident_summary(df: pd.DataFrame) -> dict:
    """Return headline metrics for dashboard cards"""
    if df.empty:
        return {}
    # Use decoded severity labels where available
    severity_counts = df.get("accident_severity_label", pd.Series(dtype=object)).value_counts() # df.get keeps the function safe if an undecoded dataframe is passed
    serious_fatal = int(severity_counts.get("Serious", 0) + severity_counts.get("Fatal", 0))
    total = len(df)
    # Check the mode calculations so empty or missing time columns do not break dashboard metric rendering.
    peak_hour = int(df["hour"].mode().iloc[0]) if "hour" in df and not df["hour"].dropna().empty else None
    peak_day = df["day_of_week_name"].mode().iloc[0] if "day_of_week_name" in df and not df["day_of_week_name"].dropna().empty else None
    return {
        "total_accidents": total,
        "serious_fatal_accidents": serious_fatal,
        "serious_fatal_share": serious_fatal / total if total else np.nan,
        "peak_hour": peak_hour,
        "peak_day": peak_day,
        "date_min": df["date"].min() if "date" in df else None,
        "date_max": df["date"].max() if "date" in df else None,
    }


def run_apriori_severity(
    accident_df: pd.DataFrame,
    min_support: float = 0.03,
    min_lift: float = 1.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """Find association rules where accident severity is the consequent."""
    # These attributes are used as antecedents for rules that may explain patterns associated with accident severity.
    required = [
        "accident_severity_label",
        "light_conditions_label",
        "weather_conditions_label",
        "road_surface_conditions_label",
        "road_type_label",
        "speed_limit",
        "urban_or_rural_area_label",
    ]
    # Keep only columns present in the supplied dataframe so the function remains reusable across filtered datasets.
    available = [c for c in required if c in accident_df.columns]
    if "accident_severity_label" not in available:
        return pd.DataFrame()

    df = accident_df[available].dropna().copy()
    if df.empty:
        return pd.DataFrame()

    if "speed_limit" in df.columns:
        # Convert continuous speed limits into categorical bands, which are more suitable for one-hot transaction-style association mining.
        df["speed_limit_band"] = pd.cut(
            df["speed_limit"],
            bins=[0, 20, 30, 40, 50, 60, 70, 200],
            labels=["≤20", "21–30", "31–40", "41–50", "51–60", "61–70", ">70"],
            include_lowest=True,
        )
        df = df.drop(columns=["speed_limit"])

    # Apriori expects a transaction-style boolean matrix, each one-hot column is treated as an item that can appear in a rule.
    basket = pd.get_dummies(df.astype("category"), dtype=bool)
    if basket.empty:
        return pd.DataFrame()

    # First identify sufficiently common itemsets before deriving association rules from them.
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    if rules.empty:
        return pd.DataFrame()

    # Keep only rules where the consequent is a single severity class, this makes the output focused on interpretable severity drivers.
    severity_tokens = ("accident_severity_label_Fatal", "accident_severity_label_Serious", "accident_severity_label_Slight")
    rules = rules[rules["consequents"].apply(lambda items: len(items) == 1 and next(iter(items)) in severity_tokens)]
    if rules.empty:
        return pd.DataFrame()

    # Rank stronger, more reliable rules first and convert frozensets into readable text
    out = rules.sort_values(["lift", "confidence", "support"], ascending=False).head(top_n).copy()
    out["antecedents_text"] = out["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    out["consequent_text"] = out["consequents"].apply(lambda x: next(iter(x)).replace("accident_severity_label_", ""))
    return out[["antecedents_text", "consequent_text", "support", "confidence", "lift"]]


def cluster_locations_kmeans(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Cluster regional accident points using KMeans on latitude and longitude."""
    # Remove records without coordinates, spatial clustering requires valid latitude and longitude values.
    out = df.dropna(subset=["latitude", "longitude"]).copy()
    if out.empty:
        out["kmeans_cluster"] = pd.Series(dtype=int)
        return out
    # Clamp k so KMeans is always valid, even when the filtered dataset is small.
    k = min(max(1, int(k)), len(out))
    coords = out[["latitude", "longitude"]]
    # Standardise coordinates so latitude and longitude contribute on comparable scales
    scaled = StandardScaler().fit_transform(coords)
    out["kmeans_cluster"] = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(scaled)
    return out


def cluster_locations_dbscan(df: pd.DataFrame, eps_meters: float = 350, min_samples: int = 8) -> pd.DataFrame:
    """Cluster accident points using DBSCAN with haversine distance.
    """
    # Remove records without coordinates because spatial clustering requires
    # valid latitude and longitude values.
    out = df.dropna(subset=["latitude", "longitude"]).copy()
    if out.empty:
        out["dbscan_cluster"] = pd.Series(dtype=int)
        return out
    # Haversine distance expects coordinates in radians. Convert the user-facing epsilon from metres into radians using Earth's approximate radius.
    coords_rad = np.radians(out[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_meters / 6_371_000
    labels = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine").fit_predict(coords_rad)
    out["dbscan_cluster"] = labels
    return out


def describe_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Summarise cluster size, severity, timing, road and weather profile."""
    rows = []
    if df.empty or cluster_col not in df.columns:
        return pd.DataFrame()

    for cluster_id, group in df.groupby(cluster_col):
        # DBSCAN labels noise points as -1; give that group a name so tables do not imply it is a normal hotspot cluster.
        if cluster_id == -1:
            label = "Noise / dispersed points"
        else:
            label = f"Cluster {cluster_id}"
        rows.append(
            {
                "cluster": cluster_id,
                "cluster_label": label,
                "accidents": len(group),
                "avg_severity_code": round(group["accident_severity"].mean(), 2) if "accident_severity" in group else np.nan,
                "most_common_severity": _safe_mode(group, "accident_severity_label"),
                "peak_day": _safe_mode(group, "day_of_week_name"),
                "peak_hour": _safe_mode(group, "hour"),
                "common_road_type": _safe_mode(group, "road_type_label"),
                "common_weather": _safe_mode(group, "weather_conditions_label"),
                "mean_latitude": group["latitude"].mean() if "latitude" in group else np.nan,
                "mean_longitude": group["longitude"].mean() if "longitude" in group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("accidents", ascending=False)


def _safe_mode(df: pd.DataFrame, col: str):
    # Return None instead of raising an error when a summary column is missing or contains only null values.
    if col not in df.columns or df[col].dropna().empty:
        return None
    return df[col].mode(dropna=True).iloc[0]


def create_folium_accident_map(
    df: pd.DataFrame,
    cluster_col: Optional[str] = None,
    max_markers: int = 3000,
    include_heatmap: bool = True,
) -> folium.Map:
    """Create an interactive Folium map with marker clustering and optional heatmap."""
    # Folium can only plot records with valid coordinates.
    plot_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if plot_df.empty:
        return folium.Map(location=[53.744, -0.332], zoom_start=10)

    # Use the average coordinate as the map centre so the map adapts to the filtered region being displayed.
    center = [plot_df["latitude"].mean(), plot_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    if include_heatmap:
        # Heatmaps show density patterns while clustered markers preserve access to individual accident details.
        heat_data = plot_df[["latitude", "longitude"]].dropna().values.tolist()
        HeatMap(heat_data, name="Accident density heatmap", radius=13, blur=18, min_opacity=0.25).add_to(m)

    marker_layer = MarkerCluster(name="Clustered accident markers")
    # Limit marker volume for browser performance on large datasets 
    marker_sample = plot_df.sample(n=min(max_markers, len(plot_df)), random_state=42) if len(plot_df) > max_markers else plot_df
    severity_colors = {"Fatal": "red", "Serious": "orange", "Slight": "blue"}

    for _, row in marker_sample.iterrows():
        # Colour markers by severity to make the most harmful incidents stand out visually on the interactive map.
        severity = row.get("accident_severity_label", "Unknown")
        color = severity_colors.get(severity, "gray")
        cluster_value = row.get(cluster_col) if cluster_col else None
        # Popup content gives the key accident attributes without inspecting the raw table.
        popup = folium.Popup(
            f"""
            <b>Accident ID:</b> {row.get('accident_index', 'N/A')}<br>
            <b>Date:</b> {row.get('date', 'N/A')}<br>
            <b>Time:</b> {row.get('time', 'N/A')}<br>
            <b>Severity:</b> {severity}<br>
            <b>Road type:</b> {row.get('road_type_label', 'N/A')}<br>
            <b>Weather:</b> {row.get('weather_conditions_label', 'N/A')}<br>
            <b>Speed limit:</b> {row.get('speed_limit', 'N/A')}<br>
            <b>Cluster:</b> {cluster_value if cluster_value is not None else 'N/A'}
            """,
            max_width=330,
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=popup,
        ).add_to(marker_layer)

    marker_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def train_severity_model(accident_df: pd.DataFrame, random_state: int = 42) -> dict:
    """Train a baseline severity classifier with balanced evaluation metrics."""
    # Candidate predictors combine temporal, environmental, road and spatial attributes that are associated with accident severity.
    candidate_features = [
        "hour",
        "day_of_week_name",
        "road_type_label",
        "weather_conditions_label",
        "road_surface_conditions_label",
        "light_conditions_label",
        "urban_or_rural_area_label",
        "speed_limit",
        "latitude",
        "longitude",
    ]
    # Use only features present in the input. This supports modelling on partial or filtered datasets without rewriting the pipeline.
    features = [c for c in candidate_features if c in accident_df.columns]
    target = "accident_severity_label"
    df = accident_df[features + [target]].dropna(subset=[target]).copy()
    # Avoid training a model when the sample is too small or contains only one severity class.
    if len(df) < 100 or df[target].nunique() < 2:
        return {"model": None, "metrics": pd.DataFrame(), "report": "Insufficient data for model training."}

    X = df[features]
    y = df[target]
    # Split features by type so each group receives suitable preprocessing.
    categorical = [c for c in features if X[c].dtype == "object" or c.endswith("_label") or c == "day_of_week_name"]
    numeric = [c for c in features if c not in categorical]

    # Numeric values are median-imputed and scaled; categorical values are imputed then one-hot encoded to support unseen categories at inference time.
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)])
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    min_samples_leaf=3,
                ),
            ),
        ]
    )

    # Stratification preserves class balance in the test set when every class has enough examples to split safely.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=stratify)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Report both ordinary accuracy and class-sensitive metrics because severity classes are usually imbalanced.
    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_test, preds),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                "macro_f1": f1_score(y_test, preds, average="macro"),
                "weighted_f1": f1_score(y_test, preds, average="weighted"),
            }
        ]
    )
    return {
        "model": model,
        "metrics": metrics,
        "classification_report": classification_report(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=sorted(y.unique())),
        "labels": sorted(y.unique()),
    }


def weekly_series_for_force(accidents: pd.DataFrame, force_code: int) -> pd.Series:
    """Build a complete weekly accident count series for a police force."""
    # Filter to one police force and remove undated records before resampling.
    df = accidents[(accidents["police_force"] == force_code) & accidents["date"].notna()].copy()
    if df.empty:
        return pd.Series(dtype=float)

    # Resample to weekly counts ending on Monday, then fill any missing weeks with zero so the time series has a regular frequency.
    weekly = df.set_index("date").resample("W-MON")["accident_index"].count().astype(float)
    full_idx = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
    return weekly.reindex(full_idx, fill_value=0)


def compare_weekly_forecasts(
    accidents: pd.DataFrame,
    force_code: int,
    train_years: Iterable[int] = (2017, 2018),
    test_year: int = 2019,
) -> ForecastResult | None:
    """Fit SARIMAX and seasonal-naive weekly forecasts for a police force."""
    series = weekly_series_for_force(accidents, force_code)
    if series.empty:
        return None

    # Split the series by calendar year to mimic a realistic forecasting setup: train on past years, evaluate on a future year.
    train = series[series.index.year.isin(list(train_years))]
    actual = series[series.index.year == test_year]
    if len(train) < 52 or actual.empty:
        return None

    # Seasonal-naive baseline uses the closest corresponding week from the previous year. This gives the SARIMAX model a meaningful benchmark.
    naive_values = []
    for date in actual.index:
        previous_year_date = date - pd.DateOffset(years=1)
        nearest = train.index[np.argmin(np.abs((train.index - previous_year_date).days))]
        naive_values.append(train.loc[nearest])
    naive = pd.Series(naive_values, index=actual.index, name="seasonal_naive")

    try:
        # SARIMAX captures short-term autocorrelation and yearly seasonality in weekly accident counts.
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        sarimax = fit.get_forecast(steps=len(actual)).predicted_mean
        sarimax.index = actual.index
        sarimax = sarimax.clip(lower=0)
    except Exception:
        # Fall back to the seasonal-naive forecast if SARIMAX cannot converge for a small or noisy subset.
        sarimax = naive.copy()
        sarimax.name = "sarimax_fallback"

    # Compare SARIMAX against the seasonal-naive baseline using forecast error metrics that are easy to interpret
    metrics = pd.DataFrame(
        [
            _forecast_metrics("SARIMAX", actual, sarimax),
            _forecast_metrics("Seasonal naive", actual, naive),
        ]
    )
    return ForecastResult(actual=actual, sarimax_forecast=sarimax, naive_forecast=naive, metrics=metrics)


def _forecast_metrics(model_name: str, actual: pd.Series, predicted: pd.Series) -> dict:
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    # Use max(actual, 1) in the denominator to avoid division-by-zero when a week has no recorded accidents.
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def hull_top_lsoa_daily_forecast(
    db_path: str | Path = DEFAULT_DB_PATH,
    year: int = 2019,
    top_n: int = 30,
    forecast_days: int = 31,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """Forecast daily accidents for Hull's top Q1 LSOAs using Jan-Jun data."""
    # Pull Hull accident counts at LSOA level so the forecast focuses on local areas with the highest early-year accident activity.
    query = """
        SELECT
            a.accident_index,
            a.date,
            a.lsoa_of_accident_location,
            l.lsoa01nm
        FROM accident a
        JOIN lsoa l ON a.lsoa_of_accident_location = l.lsoa01cd
        WHERE a.accident_year = :year
          AND l.lsoa01nm LIKE '%Kingston upon Hull%'
    """
    df = _read_sql(db_path, query, {"year": year})
    df = add_time_features(df)
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    # Identify the highest-volume LSOAs in Q1, then train on the first six months for those same areas.
    q1 = df[df["month"].isin([1, 2, 3])]
    top_lsoas = q1["lsoa01nm"].value_counts().head(top_n).index.tolist()
    train_df = df[df["lsoa01nm"].isin(top_lsoas) & df["month"].isin([1, 2, 3, 4, 5, 6])]
    # Convert filtered accidents into a complete daily count series.
    daily = train_df.set_index("date").resample("D")["accident_index"].count().astype(float)
    daily = daily.asfreq("D", fill_value=0)
    if len(daily) < 30:
        return daily, pd.Series(dtype=float), top_lsoas

    try:
        # SARIMAX captures short-term autocorrelation and weekly seasonality in daily accident counts.
        model = SARIMAX(daily, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=forecast_days).predicted_mean.clip(lower=0)
    except Exception:
        # If the time-series model fails, use the recent two-week daily average as a transparent fallback forecast.
        forecast = pd.Series([daily.tail(14).mean()] * forecast_days)

    # Label forecast values with the actual future dates after the training period to simplify plotting.
    forecast.index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    return daily, forecast, top_lsoas


def save_map(map_object: folium.Map, output_path: str | Path) -> Path:
    """Save a Folium map to HTML and return the path."""
    path = Path(output_path)
    # Create the output directory if it does not exist, allowing for nested directories to be created automatically.
    path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(str(path))
    return path
