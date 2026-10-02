"""The landmark files, read and written in a single place.

`LoadJsonLandmarks` existed in three strictly identical copies
(`AREG_IOS_utils`, `ASO_IOS_utils`, `FlexReg_Method`) and `WriteJson` in two
(`AREG_CBCT_utils`, `ASO_CBCT_utils`). The variants that diverge -- the
ASO_CBCT one, the ALI_CBCT one, the ALI_IOS one -- stay where they are.

The mutable default `list_landmark=[]` that the three copies carried becomes
`None`: the list was never written to, only iterated over, so the change is
invisible, but the trap is gone.

Imported from the Conda environment: neither Slicer nor Qt here.
"""
import json

import numpy as np


def LoadJsonLandmarks(ldmk_path, full_landmark=True, list_landmark=None):
    """
    Load landmarks from json file

    Parameters
    ----------
    img : sitk.Image
        Image to which the landmarks belong

    Returns
    -------
    dict
        Dictionary of landmarks

    Raises
    ------
    ValueError
        If the json file is not valid
    """

    with open(ldmk_path) as f:
        data = json.load(f)

    markups = data["markups"][0]["controlPoints"]

    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array(
            [markup["position"][0], markup["position"][1], markup["position"][2]]
        )
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord

    if not full_landmark:
        out = {}
        for lm in list_landmark or ():
            out[lm] = landmarks[lm]
        landmarks = out
    return landmarks


def GenControlePoint(landmarks):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark, data in landmarks.items():
        id += 1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data[0], data[1], data[2]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "defined",
        }
        lm_lst.append(controle_point)

    return lm_lst


def WriteJson(landmarks, out_path):
    false = False
    true = True
    file = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": false,
                "labelFormat": "%N-%d",
                "controlPoints": GenControlePoint(landmarks),
                "measurements": [],
                "display": {
                    "visibility": false,
                    "opacity": 1.0,
                    "color": [0.5, 0.5, 0.5],
                    "selectedColor": [
                        0.26666666666666669,
                        0.6745098039215687,
                        0.39215686274509806,
                    ],
                    "propertiesLabelVisibility": false,
                    "pointLabelsVisibility": true,
                    "textScale": 2.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": true,
                    "sliceProjection": false,
                    "sliceProjectionUseFiducialColor": true,
                    "sliceProjectionOutlinedBehindSlicePlane": false,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": false,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(file, f, ensure_ascii=False, indent=4)


def ListLandmarksJson(json_file):
    """The landmark labels of a markups file, in order."""
    with open(json_file) as f:
        data = json.load(f)

    return [
        data["markups"][0]["controlPoints"][i]["label"]
        for i in range(len(data["markups"][0]["controlPoints"]))
    ]
