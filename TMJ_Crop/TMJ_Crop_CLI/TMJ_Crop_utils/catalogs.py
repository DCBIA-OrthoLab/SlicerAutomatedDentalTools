# The names of the Extra outputs, as SADT-VISOR's TMJ_Crop spells them
# (tools/TMJ_Crop/src/sadt_tmj_crop/catalogs.py, commit 11f3cb8). The UI offers
# exactly these, and the CLI refuses any other.

OUTPUT_ORIENTED_SCAN = "Oriented scan"
OUTPUT_MANDIBLE = "Mandible segmentation"

OUTPUT_LABEL = "Extra outputs"
OUTPUT_NOTE = ("Saved beside the crops, in the oriented frame. The crops, the "
               "report and the quick view are always written.")
OUTPUT_HELP = {
    OUTPUT_ORIENTED_SCAN: "The whole scan in the Frankfurt horizontal frame the "
                          "crops were cut from, as <scan>_oriented.nii.gz.",
    OUTPUT_MANDIBLE: "The mandible mask (AMASSS) in that same frame, as "
                     "<scan>_mandible.nii.gz.",
}

# Which joints to crop. The first is the default, as in the VISOR tool.
SIDES = {"Left and Right": ("R", "L"), "Left": ("L",), "Right": ("R",)}

# What `scale_by` may be. Neither it nor `scale` is offered by the panel: a
# clinician should not change the size of the box. They stay on the CLI for
# research use, exactly as the VISOR tool keeps them in run().
SCALE_BY = ("none", "mandible-volume")

# The seven landmarks ASO orients on by default, the ones the Frankfurt
# horizontal frame needs (ASO's own default for a CBCT).
ORIENTATION_LANDMARKS = ("Ba", "S", "N", "RPo", "LPo", "ROr", "LOr")

# What the chain's steps are told to name their files.
ORIENTED_SUFFIX = "Or"
PREDICTION_ID = "Pred"
