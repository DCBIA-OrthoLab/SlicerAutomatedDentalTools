#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import glob
import re
import warnings
from pathlib import Path
import pandas as pd
import joblib

# The deployed sklearn version commonly differs from the one used to train the models.
# This is expected (models are tested for compatibility before being shipped) and not
# an actual error, so it shouldn't be surfaced as a CLI failure.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("SurgMovPred_CLI")

logger.info("SurgMovPred_CLI.py initialization")

def clean_name(name: str) -> str:
    """Cleans a column name so it exactly matches the training-time format."""
    try:
        name = str(name).strip()
        name = re.sub(r'[\r\n\t]', ' ', name)

        # 1. Strip problematic quotes and brackets
        name = name.replace('"', '').replace('\\', '').replace('[', '').replace(']', '')

        # 2. Replace spaces and dashes with underscores
        # (keeping the apostrophe (') for Jarabak's and SM_A'_CP!)
        name = re.sub(r'[^0-9a-zA-Z_\']+', '_', name)

        # 3. Collapse multiple underscores
        name = re.sub(r'_+', '_', name).strip('_')

        # 4. Model-specific adjustment
        name = name.replace("total", "Total")
        
        if re.match(r'^\d', name):
            name = f'f_{name}'
            
        return name or 'f_unnamed'
    except Exception:
        logger.exception(f"Error cleaning column name '{name}'")
        return 'f_unnamed'


# Common naming conventions used by different users for the patient identifier column.
# Matched case-insensitively, with optional separator (space/underscore/dash) between words.
ID_COLUMN_PATTERNS = [
    r'^#$',
    r'^id$',
    r'^id[\s_-]?patient$',
    r'^patient[\s_-]?id$',
    r'^patient[\s_-]?(num|number|no)$',
    r'^subject[\s_-]?id$',
    r'^patient$',
    r'^subject$',
]


def find_id_column(columns) -> str:
    """
    Tries to identify which input column holds the patient identifier.
    Naming conventions vary a lot between users (e.g. '#', 'ID', 'PatientID', 'Patient Number'...),
    so this matches a broad set of common patterns instead of a single fixed name.

    Returns the matching column name, or None if nothing matched.
    """
    normalized = [(col, str(col).strip().lower()) for col in columns]

    for pattern in ID_COLUMN_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for col, norm in normalized:
            if regex.fullmatch(norm):
                return col

    # Fallback: any column whose name contains both "patient" and "id"
    for col, norm in normalized:
        if 'patient' in norm and 'id' in norm:
            return col

    return None


def load_all_model_packages(base_model_folder: str) -> dict:
    """
    Recursively walks the models folder to load every 'stacking_package.pkl' file.

    Returns:
        dict: A dictionary { target_name: package_dict }
    """
    packages = {}
    base_path = Path(base_model_folder)

    if not base_path.exists():
        raise FileNotFoundError(f"Models folder does not exist: {base_model_folder}")

    # Find every stacking_package.pkl file in the folder and its subfolders
    package_files = list(base_path.glob("**/stacking_package.pkl"))

    if not package_files:
        raise FileNotFoundError(f"No 'stacking_package.pkl' model package found in {base_model_folder}")

    logger.info(f"Loading {len(package_files)} model(s)...")

    for pkl_path in package_files:
        try:
            package = joblib.load(pkl_path)
            target_name = package['target_name']
            packages[target_name] = package
        except Exception:
            logger.exception(f"Unable to load package {pkl_path}")

    if not packages:
        raise RuntimeError(f"None of the {len(package_files)} model package(s) found in {base_model_folder} could be loaded.")

    logger.info(f"Successfully loaded {len(packages)} model package(s).")
    return packages


def load_data_from_directory(directory_path: str) -> pd.DataFrame:
    """Loads and combines the CSV, XLSX, or ODS files of a folder."""
    try:
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        logger.info(f"Scanning directory for data files: {directory_path}")

        extensions = ['*.csv', '*.xlsx', '*.ods']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(directory_path, ext)))
            all_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        all_files = list(set(all_files))
        if not all_files:
            raise FileNotFoundError(f"No valid CSV, XLSX, or ODS files found in: {directory_path}")
            
        logger.info(f"Found {len(all_files)} file(s) to process.")
        
        df_list = []
        for file_path in all_files:
            try:
                logger.info(f"Loading file: {file_path}")
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.csv':
                    df = pd.read_csv(file_path)
                elif ext == '.xlsx':
                    df = pd.read_excel(file_path)
                elif ext == '.ods':
                    df = pd.read_excel(file_path, engine='odf')
                else:
                    continue
                
                logger.info(f"Successfully loaded {file_path} ({len(df)} rows)")
                df_list.append(df)

            except Exception:
                logger.exception(f"Failed to load file {file_path}")

        if not df_list:
            raise RuntimeError(f"No data could be successfully loaded from any of the {len(all_files)} file(s) found in {directory_path}.")

        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"All files combined successfully. Total: {len(combined_df)} rows")
        return combined_df

    except Exception:
        logger.exception(f"Error loading data from directory: {directory_path}")
        raise


def predict_all_targets(df: pd.DataFrame, packages: dict) -> pd.DataFrame:
    """
    Predicts the values for each loaded model, dynamically adapting to the input data.
    """
    try:
        # 1. Clean the input file's column names so they match the training-time format
        df_cleaned = df.copy()
        df_cleaned.columns = [clean_name(col) for col in df_cleaned.columns]

        # Collect predictions in a plain dict and build the DataFrame once at the end,
        # instead of inserting one column at a time (which fragments the DataFrame
        # and triggers pandas' PerformanceWarning).
        predictions_by_target = {}

        logger.info("Starting predictions for all target variables...")

        for target_name, pack in packages.items():
            try:
                expected_features = pack['features_names']
                scaler = pack['scaler']
                model = pack['model']

                # Resolve each expected feature to an actual column in the input file.
                # Some files provide T0 measurements without the "_T0" suffix used at training time.
                feature_source = {}
                missing_features = []
                for f in expected_features:
                    if f in df_cleaned.columns:
                        feature_source[f] = f
                    elif f.endswith('_T0') and f[:-3] in df_cleaned.columns:
                        feature_source[f] = f[:-3]
                    else:
                        missing_features.append(f)

                if missing_features:
                    logger.warning(f"⚠️ Model '{target_name}' skipped: missing {len(missing_features)} input feature(s) (e.g. {missing_features[:3]})")
                    continue

                # Extract and order the data according to this model's specific needs
                X_target = df_cleaned[[feature_source[f] for f in expected_features]]
                X_target.columns = expected_features

                # Standardize using the model's own scaler
                X_scaled = scaler.transform(X_target)
                X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features, index=df_cleaned.index)

                # Predict
                predictions_by_target[target_name] = model.predict(X_scaled_df)

            except Exception:
                logger.exception(f"Error predicting target '{target_name}'")

        results_df = pd.DataFrame(predictions_by_target, index=df_cleaned.index)

        logger.info(f"Predictions complete. {len(results_df.columns)} target(s) predicted out of {len(packages)} available.")
        return results_df

    except Exception:
        logger.exception("General error during prediction")
        raise


def save_results(df: pd.DataFrame, output_folder: str) -> str:
    """Saves the predictions table as Excel and CSV."""
    try:
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        excel_output = out_path / "predictions_outputs.xlsx"
        csv_output = out_path / "predictions_outputs.csv"

        logger.info(f"Saving results to: {out_path}")
        df.to_excel(excel_output, index=True)
        df.to_csv(csv_output, index=True)

        logger.info("Results saved successfully!")
        return str(excel_output)

    except Exception:
        logger.exception(f"Error saving results to: {output_folder}")
        raise


def main(args):
    try:
        logger.info("=== Surgical Movements Prediction Engine (Stacking Deploy) ===")

        # 1. Load every model present in the specified folder
        # (Can point to "saved_models/", or directly to "saved_models/class_1")
        packages = load_all_model_packages(args.modelPath)

        # 2. Load the input data (folder of CSV / Excel / ODS files)
        df_input = load_data_from_directory(args.inputFolder)

        # 2b. Recover the patient identifier from the input so it can be carried over to the output
        id_column = find_id_column(df_input.columns)
        if id_column is not None:
            logger.info(f"Detected patient ID column: '{id_column}'")
            id_values = df_input[id_column].reset_index(drop=True)
        else:
            logger.warning("Could not detect a patient ID column in the input data; 'IDPatient' will be left empty in the output.")
            id_values = pd.Series([pd.NA] * len(df_input))

        # 3. Predict every target
        df_results = predict_all_targets(df_input, packages)
        df_results.insert(0, 'IDPatient', id_values.values)

        # 4. Save
        save_results(df_results, args.outputFolder)

        logger.info("=== Process completed successfully ===")
        return 0

    except Exception:
        # Each step above already logs its own exception with a full traceback;
        # this is the final safety net so the process always exits with a clear status.
        logger.exception("Process failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool to make predictions using trained Stacking packages.")

    parser.add_argument('inputFolder', type=str, help="Path to the folder containing the files (csv, xlsx, ods) to predict")
    parser.add_argument('modelPath', type=str, help="Path to the root folder of the saved models (e.g. saved_models/class_None)")
    parser.add_argument("outputFolder", type=str, help="Folder where the generated result files will be exported")

    args = parser.parse_args()
    sys.exit(main(args))