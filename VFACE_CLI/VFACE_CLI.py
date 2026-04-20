#!/usr/bin/env python-real
"""
VFACE CLI (Vertical Facial Asymmetry Classification Engine - Command Line Interface)

This script performs facial asymmetry classification using machine learning models.
It loads predictions from trained models and classifies asymmetric cases.

Author: Alexandre Buisson (University of North Carolina at Chapel Hill)
"""

import argparse
import joblib
import os
import re
import pandas as pd
import lightgbm as lgb
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_name(name: str) -> str:
    """
    Normalize column names by removing special characters and standardizing format.
    
    Args:
        name: The column name to clean
        
    Returns:
        str: Cleaned column name
    """
    try:
        name = str(name).strip()
        # Replace whitespace characters with underscore
        name = re.sub(r'[\r\n\t]', ' ', name)
        # Remove quotes and backslashes
        name = re.sub(r'["\'\\]', '', name)
        # Replace special characters with underscore
        name = re.sub(r'[^0-9a-zA-Z_]+', '_', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name).strip('_')
        # Prefix names starting with digits
        if re.match(r'^\d', name):
            name = f'f_{name}'
        return name or 'f_unnamed'
    except Exception as e:
        logger.error(f"Error cleaning name '{name}': {e}")
        return 'f_unnamed'


def load_models(model_path: str) -> tuple:
    """
    Load pre-trained machine learning models for asymmetry classification.
    
    Args:
        model_path: Path to the directory containing model files
        
    Returns:
        tuple: Tuple of (model_sym_asym, model_mand_asym, model_max_asym)
        
    Raises:
        FileNotFoundError: If any required model file is missing
        Exception: If model loading fails
    """
    try:
        logger.info(f"Loading models from: {model_path}")
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Define model file paths
        sym_asym_path = os.path.join(model_path, "sym_asymm.txt")
        mand_asym_path = os.path.join(model_path, "mand_asym.txt")
        max_asym_path = os.path.join(model_path, "max_asym.txt")
        
        # Verify all model files exist
        for model_file in [sym_asym_path, mand_asym_path, max_asym_path]:
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Required model file not found: {model_file}")
        
        # Load models
        model_sym_asym = joblib.load(sym_asym_path)
        model_mand_asym = joblib.load(mand_asym_path)
        model_max_asym = joblib.load(max_asym_path)
        
        logger.info("All models loaded successfully")
        return model_sym_asym, model_mand_asym, model_max_asym
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def load_data(excel_path: str) -> pd.DataFrame:
    """
    Load measurement data from Excel file.
    
    Args:
        excel_path: Path to the Excel file containing measurements
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If Excel file is not found
        Exception: If loading fails
    """
    try:
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        logger.info(f"Loading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        logger.info(f"Excel file loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise


def classify_symmetry(df: pd.DataFrame, model_sym_asym, model_mand_asym, model_max_asym) -> pd.DataFrame:
    """
    Classify facial asymmetry using loaded models.
    
    Args:
        df: Input dataframe with measurement features
        model_sym_asym: Symmetry/Asymmetry classifier model
        model_mand_asym: Mandible asymmetry classifier model
        model_max_asym: Maxilla asymmetry classifier model
        
    Returns:
        pd.DataFrame: DataFrame with classification results
        
    Raises:
        Exception: If classification fails
    """
    try:
        # Define translation dictionaries
        symmetry_translation = {0: "Asymmetric", 1: "Symmetric"}
        binary_translation = {0: "False", 1: "True"}
        
        # Clean column names
        for col in df.columns:
            if "/" in col:
                df = df.rename(columns={col: clean_name(col)})
        
        # Predict symmetry/asymmetry
        logger.info("Making symmetry/asymmetry predictions...")
        y_sym = model_sym_asym.predict(df[model_sym_asym.feature_name_])
        df["Asymmetry"] = pd.Series(y_sym, index=df.index).map(symmetry_translation)
        logger.info("Symmetry predictions completed")
        
        # Initialize sub-classification columns
        df["Mand"] = "False"
        df["Max"] = "False"
        
        # Identify asymmetric cases
        mask_asym = (y_sym == 0)
        if mask_asym.any():
            num_asymmetric = mask_asym.sum()
            logger.info(f"Processing {num_asymmetric} asymmetric cases for sub-classification...")
            
            df_asym = df.loc[mask_asym]
            
            # Classify mandible asymmetry
            try:
                y_mand_sub = model_mand_asym.predict(df_asym[model_mand_asym.feature_name_])
                df.loc[mask_asym, "Mand"] = pd.Series(y_mand_sub, index=df_asym.index).map(binary_translation)
            except Exception as e:
                logger.warning(f"Error in mandible classification: {e}")
            
            # Classify maxilla asymmetry
            try:
                y_max_sub = model_max_asym.predict(df_asym[model_max_asym.feature_name_])
                df.loc[mask_asym, "Max"] = pd.Series(y_max_sub, index=df_asym.index).map(binary_translation)
            except Exception as e:
                logger.warning(f"Error in maxilla classification: {e}")
                
        else:
            logger.info("No asymmetric cases detected. Skipping sub-model classification.")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise


def save_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save classification results to Excel file.
    
    Args:
        df: DataFrame containing classification results
        output_path: Path where results should be saved
        
    Raises:
        Exception: If saving fails
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        logger.info(f"Saving results to: {output_path}")
        df.to_excel(output_path, index=False)
        logger.info("Results saved successfully!")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def main(args):
    """
    Main function for facial asymmetry classification.
    
    Args:
        args: Command-line arguments containing model_path, excel_path, and output_path
        
    Returns:
        int: 0 if successful, 1 if error occurred
    """
    try:
        logger.info("=== Facial Asymmetry Classification Engine (VFACE) ===")
        
        # Load models
        model_sym_asym, model_mand_asym, model_max_asym = load_models(args.model_path)
        
        # Load data
        df = load_data(args.excel_path)
        
        # Classify symmetry
        df_results = classify_symmetry(df, model_sym_asym, model_mand_asym, model_max_asym)
        
        # Save results
        save_results(df_results, args.output_path)
        
        logger.info("=== Classification process completed successfully ===")
        return 0
        
    except Exception as e:
        logger.error(f"Classification process failed: {e}")
        return 1


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Facial Asymmetry Classification Engine - Classifies facial asymmetry based on trained ML models"
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the directory containing trained model files'
    )
    parser.add_argument(
        'excel_path',
        type=str,
        help='Path to the Excel file containing measurement features'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where classification results should be saved'
    )
    
    args = parser.parse_args()
    
    logger.info("Input arguments:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Excel path: {args.excel_path}")
    logger.info(f"  Output path: {args.output_path}")
    
    exit_code = main(args)
    exit(exit_code)
