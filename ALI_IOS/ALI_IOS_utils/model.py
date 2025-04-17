# Constants and lookup dictionaries for landmark mappings and model configurations
import numpy as np
from scipy import linalg

LOWER_DENTAL = ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7']
UPPER_DENTAL = ['UL7','UL6','UL5','UL4','UL3','UL2','UL1','UR1','UR2','UR3','UR4','UR5','UR6','UR7']
TYPE_LM = ['O','MB','DB','CL','CB']

LANDMARKS = {
    "L": [tooth + lm for tooth in LOWER_DENTAL for lm in TYPE_LM],
    "U": [tooth + lm for tooth in UPPER_DENTAL for lm in TYPE_LM]
}

LABEL_L = [str(x) for x in range(18, 32)]  # "18" to "31"
LABEL_U = [str(x) for x in range(2, 16)]   # "2" to "15"

dic_label = {
    'O': {
        **{str(15 - i): LANDMARKS["U"][i*5:i*5+3] for i in range(14)},  # teeth 15 to 2
        **{str(18 + i): LANDMARKS["L"][i*5:i*5+3] for i in range(14)}   # teeth 18 to 31
    },
    'C': {
        **{str(15 - i): LANDMARKS["U"][i*5+3:i*5+5] for i in range(14)},
        **{str(18 + i): LANDMARKS["L"][i*5+3:i*5+5] for i in range(14)}
    }
}

dic_cam = {
    'O': {
        'L': ([0,0,1],
              np.array([0.5,0.,1.0])/linalg.norm([0.5,0.5,1.0]),
              np.array([-0.5,0.,1.0])/linalg.norm([-0.5,-0.5,1.0]),
              np.array([0,0.5,1])/linalg.norm([1,0,1]),
              np.array([0,-0.5,1])/linalg.norm([0,1,1])),
        'U': ([0,0,-1],
              np.array([0.5,0.,-1])/linalg.norm([0.5,0.5,-1]),
              np.array([-0.5,0.,-1])/linalg.norm([-0.5,-0.5,-1]),
              np.array([0,0.5,-1])/linalg.norm([1,0,-1]),
              np.array([0,-0.5,-1])/linalg.norm([0,1,-1]))
    },
    'C': {
        'L': tuple(np.array(vec)/linalg.norm(vec) for vec in [
            [1,0,0], [-1,0,0], [1,-1,0], [-1,-1,0], [1,1,0], [-1,1,0],
            [1,0,0.5], [-1,0,0.5], [1,-1,0.5], [-1,-1,0.5], [1,1,0.5], [-1,1,0.5]
        ]),
        'U': tuple(np.array(vec)/linalg.norm(vec) for vec in [
            [1,0,0], [-1,0,0], [1,-1,0], [-1,-1,0], [1,1,0], [-1,1,0],
            [1,0,-0.5], [-1,0,-0.5], [1,-1,-0.5], [-1,-1,-0.5], [1,1,-0.5], [-1,1,-0.5]
        ])
    }
}

MODELS_DICT = {
    'O': {'O': 0, 'MB': 1, 'DB': 2},
    'C': {'CL': 0, 'CB': 1}
}