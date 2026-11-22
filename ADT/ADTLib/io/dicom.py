"""Convert one patient's DICOM folder without seeing anybody else's output.

`dicom2nifti.convert_directory` does not say what it wrote: it names each file
after the series, and logs a failed series instead of raising. Looking for the
result in the folder the patients share found another patient's volume.

Standard library only, and pydicom once there is a DICOM file to read: it comes
with dicom2nifti, whose converter the caller passes in.
"""
import os
import shutil
import tempfile

#: dicom2nifti converts no series this short: it takes it for a localizer.
LOCALIZER_MAX_SLICES = 3


def _count_volume_series(folder):
    """How many series below `folder` dicom2nifti would take for a volume.

    The files it writes cannot tell: two series with the same number and
    description get the same name, and the second replaces the first.
    """
    slices = {}
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as handle:
                    if handle.read(132)[128:] != b"DICM":
                        continue
            except OSError:
                continue  # unreadable: the converter passes it by as well

            import pydicom
            from pydicom.errors import InvalidDicomError

            try:
                header = pydicom.dcmread(path, stop_before_pixels=True)
            except (InvalidDicomError, OSError, EOFError):
                continue
            series = header.get("SeriesInstanceUID")
            frames = int(header.get("NumberOfFrames") or 1)
            # A report or a dose sheet has a series too, but no position.
            if series and (frames > 1 or "ImagePositionPatient" in header):
                slices[series] = slices.get(series, 0) + frames
    return sum(1 for count in slices.values() if count > LOCALIZER_MAX_SLICES)


def convert_single_volume(dicom_folder, output_file, convert):
    """Save the one volume of `dicom_folder` as `output_file`, or raise.

    `convert(dicom_folder, folder)` is `dicom2nifti.convert_directory`. It
    writes into a folder of its own, so what is found there can only be this
    patient's. No volume, or more than one, is a RuntimeError and saves nothing.
    """
    count = _count_volume_series(dicom_folder)
    if count > 1:
        raise RuntimeError(
            f"{count} volume series in {dicom_folder}: keep one series per patient folder"
        )

    with tempfile.TemporaryDirectory(prefix="adt_dicom_") as private:
        convert(dicom_folder, private)
        written = [name for name in os.listdir(private) if name.endswith(".nii.gz")]
        if len(written) != 1:
            raise RuntimeError(
                f"expected one volume from {dicom_folder}, the conversion gave {len(written)}"
            )

        # An existing output is skipped on the next run, so it only gets its
        # name once whole. The suffix also keeps it out of the searches for scans.
        partial = output_file + ".part"
        try:
            shutil.copyfile(os.path.join(private, written[0]), partial)
            os.replace(partial, output_file)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
