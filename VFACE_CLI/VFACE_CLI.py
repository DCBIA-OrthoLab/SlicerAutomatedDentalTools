#!/usr/bin/env python-real

import argparse
import joblib
import os
import re
import pandas as pd
import lightgbm as lgb

def clean_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[\r\n\t]', ' ', name)
    name = re.sub(r'["\'\\]', '', name)
    name = re.sub(r'[^0-9a-zA-Z_]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if re.match(r'^\d', name):
        name = f'f_{name}'
    return name or 'f_unnamed'

def main(args):
    dic_translation = {0: "Asym", 1: "Sym"}
    dic_translation_bin = {0: "False", 1: "True"}
    
    print(f"Loading Excel file: {args.excel_path}")
    df = pd.read_excel(args.excel_path)
    print("Excel file loaded successfully")
    
    print(f"Loading models from: {args.model_path}")
    model_sym_asym = joblib.load(os.path.join(args.model_path, "sym_asymm.txt"))
    model_mand_asym = joblib.load(os.path.join(args.model_path, "mand_asym.txt"))
    model_max_asym = joblib.load(os.path.join(args.model_path, "max_asym.txt"))
    print("Models loaded successfully")

    mand_features = model_mand_asym.feature_name_
    max_features = model_max_asym.feature_name_

    print("Making predictions...")

    for col in df.columns:
        if "/" in col:
            df = df.rename(columns={col: clean_name(col)})

    y_sym = model_sym_asym.predict(df[model_sym_asym.feature_name_])
    df["Asymmetry"] = pd.Series(y_sym, index=df.index).map(dic_translation)
    print("Predictions done")
    df["Mand"] = "False"
    df["Max"] = "False"

    mask_asym = (y_sym == 0)
    if mask_asym.any():
        print(f"Processing {mask_asym.sum()} asymmetric cases...")

        df_asym = df.loc[mask_asym]
        
        y_mand_sub = model_mand_asym.predict(df_asym[mand_features])
        df.loc[mask_asym, "Mand"] = pd.Series(y_mand_sub, index=df_asym.index).map(dic_translation_bin)
        
        y_max_sub = model_max_asym.predict(df_asym[max_features])
        df.loc[mask_asym, "Max"] = pd.Series(y_max_sub, index=df_asym.index).map(dic_translation_bin)
    else:
        print("No asymmetric cases detected. Skipping sub-models.")

    print(f"Saving results to: {args.output_path}")
    df.to_excel(args.output_path, index=False)
    print("Results saved successfully!")

if __name__ == "__main__":

    print("=== Asymmetry Classification CLI ===")
    parse = argparse.ArgumentParser()
    parse.add_argument('model_path',type = str)
    parse.add_argument('excel_path',type = str)
    parse.add_argument('output_path', type = str)

    args = parse.parse_args()
    
    print(f"Input arguments:")
    print(f"Model path: {args.model_path}")
    print(f"Excel path: {args.excel_path}")
    print(f"Output path: {args.output_path}")

    main(args)
