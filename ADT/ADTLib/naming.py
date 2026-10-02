r"""Reading a patient identifier off a file name.

Six sites built the same identifier by chaining fourteen `.split(...)[0]` calls
in the same order, and the order is load-bearing: a longer marker has to be cut
before a shorter one it contains, or `_Scanreg` loses its tail to `_Scan` and
two files stop pairing. Written once, that ordering is a property of this
module instead of something each copy has to remember.

Standard library only: the CLIs call this from the Conda environment.

TIMEPOINT-SUFFIX -- still not implemented, but now a single line away from
here. The identifier is what is left once the markers are cut off, and the only
timepoints in the marker sets below are _T1 and _T2: an input named
P001_T3.nii.gz keeps the identifier "P001_T3" and never meets the
P001_T4.nii.gz of the T2 folder. The pair is dropped, not reported.

Accepting it takes one line in `patient_id`, after the loop:

    name = re.sub(r"_[Tt]\d+$", "", name)

and it then holds for **every** marker set, which was not the case when ten
sites each carried their own chain. Added after the loop, it is strictly
additive: a name the loop already shortened is untouched. Mind _T10 all the
same, which "_T1" cuts today down to "P001" -- that defect exists before the
change and does not go away with it.

Left as it is on purpose: accepting _T3 changes which scans get paired, which
is a decision and not a refactor. The supported answer is still to name the
inputs _T1/_T2.
"""

# Cut longest-first where one marker contains another: _Scanreg before _Scan,
# _MAND before _MD, _MAX before _MX. This is the order the six chains used.
PATIENT_ID_MARKERS = (
    "_Scan", "_scan", "_Or", "_OR", "_MAND", "_MD", "_MAX", "_MX",
    "_CB", "_lm", "_T2", "_T1", "_Cl", ".",
)


def patient_id(name, markers=PATIENT_ID_MARKERS):
    """The identifier that pairs a scan with its follow-up.

    `name` is a base name, not a path: the callers pass os.path.basename.
    """
    for marker in markers:
        name = name.split(marker)[0]
    return name

# The repository's other marker sets. They are not merged into the one above
# because they do not describe the same files: the same name does not give the
# same identifier in them, and changing that would decide which scans get
# paired. Naming them here at least makes them comparable side by side, and
# makes the algorithm -- the order of the cuts, the fragile part -- exist only
# once.

#: ASO, CBCT orientation. Cuts `_Scanreg` before `_Scan`, and ignores the
#: anatomy markers (`_MAND`, `_MAX`, `_CB`) the default set strips.
ASO_CBCT_MARKERS = (
    "_scan", "_Scanreg", "_Scan", "_Or", "_OR", "_lm", "_T1", "_T2", ".",
)

#: ASO_CBCT, the CLI. Same vocabulary as above but in a different order --
#: `_Or` first -- and without the timepoints, which therefore stay in the id.
ASO_CBCT_CLI_MARKERS = (
    "_Or", "_OR", "_scan", "_Scanreg", "_Scan", "_lm", ".",
)

#: AREG, IOS/CBCT pairing. Deliberately short: it pairs surfaces whose name
#: carries neither anatomy nor timepoint.
AREG_IOSCBCT_MARKERS = ("_scan", "_Scanreg", "_lm")

#: MRI2CBCT, TMJ crop. The default set plus seventeen markers specific to this
#: pipeline: segmentation, mask, prediction, crop, side, modality.
TMJ_CROP_MARKERS = PATIENT_ID_MARKERS[:-1] + (
    "_seg", "_Seg", "_mask", "_Mask", "_pred", "_Pred", "_crop", "_Crop",
    "_Left", "_left", "_Right", "_right", "_approximate", "_Approximate",
    "_CBCT", "_MRI", "_MR", ".",
)

#: Strip the suffix of a landmark file, without touching the rest of the name.
#: Used where the full identifier is not what is being looked for.
LANDMARK_SUFFIX_MARKERS = ("_lm", "_Or", ".")
