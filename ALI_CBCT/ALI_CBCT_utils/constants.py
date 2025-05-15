import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUP_LABELS = {
    'CB': ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],
    'U': ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],
    'L': ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],
    'CI': ['UR3OIP', 'UL3OIP', 'UR3RIP', 'UL3RIP']
}

LABEL_GROUPS = {}
LABELS = []
for group, labels in GROUP_LABELS.items():
    for label in labels:
        LABEL_GROUPS[label] = group
        LABELS.append(label)

SCALE_KEYS = ['1', '0-3']

MOVEMENT_MATRIX_6 = np.array([
    [1, 0, 0],   # MoveUp
    [-1, 0, 0],  # MoveDown
    [0, 1, 0],   # MoveBack
    [0, -1, 0],  # MoveFront
    [0, 0, 1],   # MoveLeft
    [0, 0, -1],  # MoveRight
])
MOVEMENT_ID_6 = ["Up", "Down", "Back", "Front", "Left", "Right"]

MOVEMENTS = {
    "id": MOVEMENT_ID_6,
    "mat": MOVEMENT_MATRIX_6
}

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def GetTargetOutputFromAction(mov_mat, action):
    target = np.zeros((1, len(mov_mat)))[0]
    target[action] = 1
    return target
