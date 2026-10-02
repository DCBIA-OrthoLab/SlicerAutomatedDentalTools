"""What each tool family accepts in its `Process`, declared once.

The `Process(**kwargs)` contract is the same word in all six `Method.py`, but
nobody knows what each one expects: the keys only show up deep inside the
bodies, in the form `kwargs["..."]`. Four hundred and seventy-three accesses,
no list anywhere, no checking, and a typo that only shows at run time, on the
code path where it is read.

Here each family declares its fields. Three things follow: the list exists and
can be read at a glance, an unknown key is rejected **at construction** rather
than ignored, and the editor can complete.

Why an absence marker rather than default values
-------------------------------------------------
Callers deliberately pass different subsets: a `TestProcess` does not need what
`Process` needs, and AREG's internal helpers make do with two or three keys.
Giving everything a default would therefore turn today's failure -- `KeyError`
on the absent key -- into an empty value that travels through the run without
saying a word.

A field that is not supplied is therefore `MISSING`, and reading it raises
`KeyError` with the field name: **exactly the error from before, at the same
moment, with the same message**. Moving to the dataclass adds no tolerance; it
only adds the declaration and the rejection of unknown keys.

The only fields with a real default value are those the code already read with
`kwargs.get("...", default)`: the default declared here is the one written
there, and `check_request_swap.py` rejects the conversion if they differ.

Standard library only.
"""
from dataclasses import dataclass, fields
from typing import Any


class _Missing:
    """Value of a field the caller did not supply."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<not provided>"

    def __bool__(self):
        raise KeyError(
            "a field that was not supplied is being tested as a boolean: "
            "pass the key, or declare a default on the field")


MISSING = _Missing()


@dataclass
class ProcessRequest:
    """What every tool receives. The families add their own fields."""

    input_folder: str = MISSING
    output_folder: str = MISSING
    log_path: str = MISSING

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if value is MISSING:
            # The same KeyError `kwargs["name"]` raised before, so that nothing
            # changes for the caller that caught it -- or did not catch it.
            raise KeyError(name)
        return value

    def given(self, name):
        """Was the field supplied? Without raising, unlike attribute access."""
        return object.__getattribute__(self, name) is not MISSING

    def with_(self, **changes):
        """A copy of the request, with a few fields replaced.

        `dataclasses.replace` does not fit here: it reads every field back to
        rebuild the object, including those that were not supplied -- and
        reading them raises, which is precisely the point. This copy reads the
        raw values, absence marker included.
        """
        raw = {f.name: object.__getattribute__(self, f.name) for f in fields(self)}
        raw.update(changes)
        return type(self)(**raw)

    def provided(self):
        """The names of the fields that were actually supplied."""
        return sorted(f.name for f in fields(self) if self.given(f.name))


@dataclass
class ASORequest(ProcessRequest):
    """The keys the four ASO methods read."""

    gold_folder: str = MISSING
    add_in_namefile: str = MISSING
    dic_checkbox: Any = MISSING
    is_dicom_input: str = MISSING
    model_folder_ali: str = MISSING
    model_folder_segor: str = MISSING
    smallFOV: str = MISSING


@dataclass
class ALIRequest(ProcessRequest):
    """The keys the two ALI methods read."""

    model_folder: str = MISSING
    lm_type: Any = MISSING
    teeth: Any = MISSING
    teeth_mg: Any = "None"          # read by `kwargs.get`, default kept as it was
    is_dicom_input: str = MISSING


@dataclass
class AREGRequest(ProcessRequest):
    """The keys the three AREG families read, and their helpers.

    `input_folder` is unused there: AREG reasons over two time points, T1 and T2.
    """

    input_t1_folder: str = ""
    input_t2_folder: str = ""
    input_t1_mask: str = MISSING
    input_t2_landmarks: str = MISSING
    model_folder_1: str = MISSING
    model_folder_2: str = MISSING
    model_folder_3: str = MISSING
    add_in_namefile: str = MISSING
    is_dicom_input: str = MISSING
    dic_checkbox: Any = MISSING
    merge_seg: Any = MISSING
    OrientReference: str = MISSING
    ApproxStep: Any = MISSING
    slicerDownload: Any = MISSING
    LabelSeg: Any = MISSING
    mgl_landmarks: str = ""
    patch_radius: str = "5.0"
    reg_type: Any = None            # these three were read by `kwargs.get`: the
                                    # default declared here is the one used there


@dataclass
class VFACERequest(ProcessRequest):
    """The keys `CreateListProcess`, VFACE's single entry point, reads."""

    gold_folder: str = MISSING
    t2_folder: str = MISSING
    measurements_folder: str = MISSING
    model_folder: str = MISSING
    model_folder_ali: str = MISSING
    model_vface: str = MISSING
    mirror_matrix: Any = MISSING
    mode: str = MISSING
    mode2: str = MISSING
    reg_type: str = MISSING
    bool_quantification: Any = MISSING
    bool_visualization: Any = MISSING
