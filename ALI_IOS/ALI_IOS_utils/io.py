# Utilities for generating landmark JSON outputs and label translation
import json

def GenControlPoint(group_data, selected_lm):
    lm_lst = []
    for i, (label, data) in enumerate(group_data.items(), 1):
        if label in selected_lm:
            lm_lst.append({
                "id": str(i),
                "label": label,
                "description": "",
                "associatedNodeID": "",
                "position": [data["x"], data["y"], data["z"]],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": True,
                "visibility": True,
                "positionStatus": "defined"
            })
    return lm_lst

def WriteJson(lm_lst, out_path):
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

def TradLabel(teeth_list):
    mapping = {
        'LL7':18,'LL6':19,'LL5':20,'LL4':21,'LL3':22,'LL2':23,'LL1':24,
        'LR1':25,'LR2':26,'LR3':27,'LR4':28,'LR5':29,'LR6':30,'LR7':31,
        'UL7':15,'UL6':14,'UL5':13,'UL4':12,'UL3':11,'UL2':10,'UL1':9,
        'UR1':8,'UR2':7,'UR3':6,'UR4':5,'UR5':4,'UR6':3,'UR7':2
    }
    result = {'Lower': [], 'Upper': []}
    
    for tooth in teeth_list:
        print(f"Processing tooth: {tooth}")
        if tooth in mapping.keys():
            print(f"Tooth {tooth} found in mapping.")
            if tooth.startswith('L'):
                result['Lower'].append(mapping[tooth])
            else:
                result['Upper'].append(mapping[tooth])
    return result
