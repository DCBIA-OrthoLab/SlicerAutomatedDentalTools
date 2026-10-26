"""MRI preparation: optional resampling to 0.3 mm, then left/right separation.

Everything here works in world (LPS, SimpleITK) coordinates read from the
NATIVE header. The old MRI2CBCT "Orient" step overwrote the direction matrix
with a fixed one; that throws away the only record of which way the slices
run, so it is not used. Measured on 355 single-joint slabs with known side
(97 patients): "LPS slab centre x > 0 => LEFT" was right 355/355.
"""
import numpy as np
import SimpleITK as sitk

#: Dataset516 was trained on 0.3 mm isotropic volumes.
TARGET_SPACING = 0.3
#: Wider than this along the patient's left-right axis, the volume covers both
#: joints (merged bilateral slab or whole head). Single-joint slabs measured at
#: most 98 mm (median 55); the merged slabs at 120-130 mm.
BILATERAL_MM = 110.0


def resample_iso(image, spacing=TARGET_SPACING):
    """Resample to isotropic spacing, keeping the PHYSICAL extent.

    The old resampler kept the voxel count when changing the spacing, which
    shrank a 100x100x30 mm slab to 76.8x76.8x3 mm. Here size = extent / spacing.
    """
    old_sp = np.array(image.GetSpacing())
    old_sz = np.array(image.GetSize())
    D = np.array(image.GetDirection()).reshape(3, 3)
    extent = old_sp * old_sz
    new_sp = np.full(3, float(spacing))
    new_sz = np.maximum(1, np.round(extent / new_sp)).astype(int)
    # keep the outer corner of the field of view where it was
    corner = np.array(image.GetOrigin()) - D @ (old_sp / 2)
    new_origin = corner + D @ (new_sp / 2)
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(new_sp.tolist())
    f.SetSize([int(s) for s in new_sz])
    f.SetOutputDirection(image.GetDirection())
    f.SetOutputOrigin(new_origin.tolist())
    f.SetInterpolator(sitk.sitkLinear)
    f.SetDefaultPixelValue(0)
    return f.Execute(image)


def lr_axis(image):
    """Index axis that runs closest to the patient's left-right (LPS x)."""
    D = np.array(image.GetDirection()).reshape(3, 3)
    return int(np.argmax(np.abs(D[0, :])))


def world_centre(image):
    D = np.array(image.GetDirection()).reshape(3, 3)
    sz = np.array(image.GetSize())
    sp = np.array(image.GetSpacing())
    return np.array(image.GetOrigin()) + D @ ((sz - 1) / 2.0 * sp)


def lr_extent_mm(image):
    D = np.array(image.GetDirection()).reshape(3, 3)
    ext = np.abs(D) @ (np.array(image.GetSize()) * np.array(image.GetSpacing()))
    return float(ext[0])


def side_from_header(image):
    """"L" or "R" from where the volume sits; LPS x > 0 is the patient's left."""
    return "L" if world_centre(image)[0] > 0 else "R"


def split_lr(image):
    """Cut at the middle of the left-right index axis.

    Each half is named by where it sits relative to the other, in world space,
    so the answer does not depend on which way the slice index runs.
    """
    axis = lr_axis(image)
    size = list(image.GetSize())
    mid = size[axis] // 2
    lo_size, hi_size = list(size), list(size)
    lo_size[axis] = mid
    hi_size[axis] = size[axis] - mid
    hi_index = [0, 0, 0]
    hi_index[axis] = mid
    lo = sitk.RegionOfInterest(image, lo_size, [0, 0, 0])
    hi = sitk.RegionOfInterest(image, hi_size, hi_index)
    if world_centre(lo)[0] > world_centre(hi)[0]:
        return {"L": lo, "R": hi}
    return {"L": hi, "R": lo}


def split_gapped(image, acq_spacing):
    """Split a merged bilateral slab acquired as two stacks with a gap.

    NIfTI cannot store the gap, so the file spreads all slices evenly (B038:
    18 slices at 6.876 mm, really 9 + 9 at 3.0 mm with 69 mm between them).
    Only the two END slices are in the right place. Each half is rebuilt at the
    acquisition spacing, anchored on its own end slice: the first on slice 0,
    the second on the last slice. Same assumption as the old tool: the two
    stacks have the same number of slices.
    """
    axis = lr_axis(image)
    size = list(image.GetSize())
    n = size[axis]
    h = n // 2
    D = np.array(image.GetDirection()).reshape(3, 3)
    step = D[:, axis]
    file_sp = image.GetSpacing()[axis]
    origin = np.array(image.GetOrigin())
    last = origin + step * (n - 1) * file_sp

    halves = []
    for start, count in ((0, h), (h, n - h)):
        roi_size = list(size)
        roi_size[axis] = count
        index = [0, 0, 0]
        index[axis] = start
        part = sitk.RegionOfInterest(image, roi_size, index)
        sp = list(part.GetSpacing())
        sp[axis] = float(acq_spacing)
        part.SetSpacing(sp)
        if start == 0:
            part.SetOrigin(origin.tolist())
        else:
            part.SetOrigin((last - step * (count - 1) * float(acq_spacing)).tolist())
        halves.append(part)
    a, b = halves
    if world_centre(a)[0] > world_centre(b)[0]:
        return {"L": a, "R": b}
    return {"L": b, "R": a}


def has_gap(image, acq_spacing):
    """The slices are spread wider than the acquisition spacing allows."""
    return image.GetSpacing()[lr_axis(image)] > 1.2 * float(acq_spacing)


def plan_joints(image, separate, acq_spacing=3.0):
    """Decide how one MRI volume becomes joints.

    Returns (joints, note) with joints = {side: image}; side is "L", "R", or
    "B" (both joints in one volume, registered against the whole CBCT).
    """
    width = lr_extent_mm(image)
    wide = width > BILATERAL_MM
    file_sp = image.GetSpacing()[lr_axis(image)]
    if separate and wide and has_gap(image, acq_spacing):
        n = image.GetSize()[lr_axis(image)]
        note = (f"{width:.0f} mm wide, both joints, slices {file_sp:.2f} mm apart in the file: "
                f"two stacks with a gap; split into {n // 2} + {n - n // 2} slices "
                f"rebuilt at {float(acq_spacing):g} mm")
        if n % 2:
            note += " (odd slice count: check the split)"
        return split_gapped(image, acq_spacing), note
    if separate and wide:
        return split_lr(image), f"{width:.0f} mm wide, both joints, contiguous: split into left and right"
    if separate:
        side = side_from_header(image)
        return {side: image}, (f"{width:.0f} mm wide, a single joint: not split; "
                               f"side {side} from its position in the header")
    if wide:
        note = f"{width:.0f} mm wide, both joints: kept whole, registered to the whole CBCT"
        if has_gap(image, acq_spacing):
            note += (f". WARNING: slices are {file_sp:.2f} mm apart, more than the "
                     f"{float(acq_spacing):g} mm acquisition spacing - if this is a merged "
                     "bilateral slab, run it with 'Separate left and right'")
        return {"B": image}, note
    side = side_from_header(image)
    return {side: image}, f"{width:.0f} mm wide, single joint; side {side} from its position in the header"
