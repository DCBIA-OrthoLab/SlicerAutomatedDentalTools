# Utilities for generating landmark JSON outputs and label translation
import json
import os

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger
import re

logger = get_logger("ALI_IOS_IO")

def GenControlPoint(group_data, selected_lm):
    """Generate control points for landmarks with error handling."""
    try:
        if not group_data:
            logger.warning("group_data is empty")
            return []
        
        if not selected_lm:
            logger.warning("selected_lm is empty")
            return []
        
        lm_lst = []
        for i, (label, data) in enumerate(group_data.items(), 1):
            try:
                if label in selected_lm:
                    if not all(k in data for k in ["x", "y", "z"]):
                        logger.warning(f"Missing coordinates for landmark {label}")
                        continue
                    
                    lm_lst.append({
                        "id": str(i),
                        "label": label,
                        "description": data.get("desc", ""),
                        "associatedNodeID": "",
                        "position": [data["x"], data["y"], data["z"]],
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": True,
                        "visibility": True,
                        "positionStatus": "defined"
                    })
            except Exception as e:
                logger.error(f"Error processing landmark {label}: {e}")
                continue
        
        logger.debug(f"Generated {len(lm_lst)} control points")
        return lm_lst
    except Exception as e:
        logger.error(f"Error in GenControlPoint: {e}")
        raise

def WriteJson(lm_lst, out_path):
    """Write landmarks to JSON file with error handling."""
    try:
        if not out_path:
            logger.error("Output path cannot be empty")
            raise ValueError("Output path is empty")
        
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
                logger.debug(f"Created output directory: {out_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory {out_dir}: {e}")
                raise
        
        content = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
            "markups": [{
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": lm_lst,
                "measurements": [],
                "display": {
                    "visibility": False,
                    "opacity": 1.0,
                    "color": [0.5, 0.5, 0.5],
                    "selectedColor": [0.27, 0.67, 0.39],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 2.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "snapMode": "toVisibleSurface"
                }
            }]
        }
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4)
        
        logger.info(f"Successfully wrote JSON to {out_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file {out_path}: {e}")
        raise

def TradLabelMG(teeth_list):
    """Translate mucogingival label names to lower universal ids.

    MG labels are centered on the midline: L0MG is the midline label
    (universal id 25) and LL/LR numbering runs distally from it, so the
    shared FDI mapping of TradLabel does not apply. Beware the collision
    with training names: LR1MG here is tooth 26, while the training name
    LR1MG designates tooth 25 — this table only speaks the output naming,
    which is the only one the GUI and the CLI exchange.
    """
    mapping = {
        'LL6MG': 19, 'LL5MG': 20, 'LL4MG': 21, 'LL3MG': 22, 'LL2MG': 23, 'LL1MG': 24,
        'L0MG': 25,
        'LR1MG': 26, 'LR2MG': 27, 'LR3MG': 28, 'LR4MG': 29, 'LR5MG': 30, 'LR6MG': 31
    }

    if not teeth_list:
        logger.warning("MG teeth_list is empty")
        return {'Lower': [], 'Upper': []}

    result = {'Lower': [], 'Upper': []}
    unknown = []
    for tooth in teeth_list:
        if tooth in mapping:
            result['Lower'].append(mapping[tooth])
        else:
            logger.warning(f"Unknown MG label: {tooth}")
            unknown.append(tooth)

    if unknown:
        logger.warning(f"Failed to map MG labels: {unknown}")

    logger.info(f"Translated MG labels - Lower: {result['Lower']}")
    return result

def TradLabel(teeth_list):
    """Translate tooth labels to FDI numbering system with error handling."""
    try:
        mapping = {
            'LL7': 18, 'LL6': 19, 'LL5': 20, 'LL4': 21, 'LL3': 22, 'LL2': 23, 'LL1': 24,
            'LR1': 25, 'LR2': 26, 'LR3': 27, 'LR4': 28, 'LR5': 29, 'LR6': 30, 'LR7': 31,
            'UL7': 15, 'UL6': 14, 'UL5': 13, 'UL4': 12, 'UL3': 11, 'UL2': 10, 'UL1': 9,
            'UR1': 8, 'UR2': 7, 'UR3': 6, 'UR4': 5, 'UR5': 4, 'UR6': 3, 'UR7': 2
        }
        
        if not teeth_list:
            logger.warning("teeth_list is empty")
            return {'Lower': [], 'Upper': []}
        
        result = {'Lower': [], 'Upper': []}
        unknown_teeth = []
        
        for tooth in teeth_list:
            try:
                logger.debug(f"Processing tooth: {tooth}")
                if tooth in mapping:
                    fdi_number = mapping[tooth]
                    if tooth.startswith('L'):
                        result['Lower'].append(fdi_number)
                    else:
                        result['Upper'].append(fdi_number)
                else:
                    logger.warning(f"Unknown tooth notation: {tooth}")
                    unknown_teeth.append(tooth)
            except Exception as e:
                logger.error(f"Error processing tooth {tooth}: {e}")
                unknown_teeth.append(tooth)
        
        if unknown_teeth:
            logger.warning(f"Failed to map teeth: {unknown_teeth}")
        
        logger.info(f"Translated teeth - Upper: {result['Upper']}, Lower: {result['Lower']}")
        return result
    except Exception as e:
        logger.error(f"Error in TradLabel: {e}")
        raise


def ScanJawFromName(path):
    """'Upper', 'Lower' or None, read from the tokens of an input scan's name.

    The letter has to be a token of its own so that "Dupont_003_T1_L.vtk" is not
    read as upper on the u of Dupont, and a name claiming both jaws is refused
    rather than resolved to one of them.

    Deliberately not ASO_IOS_utils.JawFromFileName, which answers a different
    question on the same input. That one reads names ALI_IOS has already
    written, where the jaw of the model is appended after the jaw of the scan,
    so a name holding both markers means "this model, on that scan" and its
    last marker is the answer. This one reads the scan before anything is
    appended: a second marker there is a name nobody can interpret, and
    renumbering an arch on a guess is worse than leaving it alone. Same refusal
    as AREG_IOSCBCT.extract_jaw, which reads the same files further down.
    """

    name = os.path.basename(path)
    upper = re.search(r'(?:^|_)(?:u|upper|max|mx)(?=_|\.|$)', name, re.IGNORECASE)
    lower = re.search(r'(?:^|_)(?:l|lower|mand|md)(?=_|\.|$)', name, re.IGNORECASE)
    if upper and lower:
        logger.warning(f"{name} names both jaws, so it cannot say which one it is")
        return None
    if upper:
        return "Upper"
    if lower:
        return "Lower"
    return None
