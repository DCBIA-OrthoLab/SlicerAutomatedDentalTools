import os
import json
import glob

# The same body as the fourteen other copies, down to the spacing.
from ADTLib.io.fs import search  # noqa: F401  (re-exporte)

def GenControlPoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data["x"], data["y"], data["z"]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "defined"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
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
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.5, 0.5, 0.5],
                "selectedColor": [0.26666666666666669, 0.6745098039215687, 0.39215686274509806],
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
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)


def GetBrain(dir_path):
    brain_dic = {}
    normpath = os.path.normpath("/".join([dir_path, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and ".pth" in img_fn:
            lab = os.path.basename(os.path.dirname(os.path.dirname(img_fn)))
            num = os.path.basename(os.path.dirname(img_fn))
            if lab in brain_dic.keys():
                brain_dic[lab][num] = img_fn
            else:
                network = {num : img_fn}
                brain_dic[lab] = network

    return brain_dic
