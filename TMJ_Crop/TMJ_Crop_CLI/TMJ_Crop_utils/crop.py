# Copied without change from the SADT-VISOR tool TMJ_Crop (tools/TMJ_Crop/src/sadt_tmj_crop/crop.py,
# last changed in commit 11f3cb8, read at upstream/main 9e86702). A change to the algorithm
# belongs in that repository first, so the two tools keep cutting the same arrays.
"""The crop itself: geometry and resampling, in the frame ASO oriented the scan to.

Everything here works in RAS millimetres (x right, y anterior, z superior), the
frame of a NIfTI affine. Slicer markups files are LPS, so landmarks are flipped
on the way in (`lps_to_ras`) and nowhere else.

ASO rotates the scan onto the reference but does not translate it, so the
oriented frame differs from the reference's by a translation only. That
translation is the mean difference between the reference landmarks and the same
landmarks ASO kept for this scan -- the "centres matched" comparison ASO's own
report uses. Landmarks ASO dropped as outliers do not take part.

Heavy imports (nibabel, scipy, matplotlib) live inside the functions that need
them, so importing this package for its schema costs none of them.
"""

from dataclasses import dataclass, field

# What `scale_by` may be; `run()` publishes the same two as its choices.
SCALE_BY = ("none", "mandible-volume")

# The two porions, which anchor each side's search window.
PORION = {"R": "RPo", "L": "LPo"}


@dataclass(frozen=True)
class CropSettings:
    """What the crop does, in millimetres unless it says voxels.

    The defaults are the ones validated on a clinical cohort of 198 CBCTs
    (396 of 396 sides found). `shape` is in voxels, each a multiple of 16 so a
    network can halve it four times.
    """

    shape: tuple = (96, 112, 80)
    spacing: float = 0.3
    # Top voxel centre above the condyle top (fossa roof), before scaling.
    above_top_mm: float = 10.8
    # Front voxel centre ahead of the head centre, before scaling; the rest of
    # the box length goes posterior.
    anterior_mm: float = 14.25
    # The search window around the porion. It stops short of the coronoid
    # process, which lies further anterior.
    lateral_mm: float = 30.0
    behind_mm: float = 10.0
    ahead_mm: float = 25.0
    # The head is the mask within `head_mm` below the top and `head_radius_mm`
    # of it in the axial plane.
    head_mm: float = 8.0
    head_radius_mm: float = 12.0
    # A top further than this from the porion is reported, not hidden.
    max_porion_mm: float = 25.0
    # Mandible volume that gets scale 1 (median of six public test scans).
    reference_volume_ml: float = 54.6
    scale_by: str = "none"
    # A fixed factor overrides `scale_by`; 0 means "not given".
    scale: float = 0.0

    def __post_init__(self):
        if self.scale_by not in SCALE_BY:
            raise ValueError(f"scale_by must be one of {', '.join(SCALE_BY)}")
        try:
            scale = float(self.scale)
        except (TypeError, ValueError):
            raise ValueError("scale must be a number") from None
        if scale != scale or scale in (float("inf"), float("-inf")):
            raise ValueError("scale must be a finite number")
        # 0 = not given; anything else is a factor on a 0.3 mm voxel, so a tiny or a
        # negative one would write a degenerate affine rather than a crop.
        if scale != 0.0 and not 0.1 <= scale <= 10.0:
            raise ValueError("scale must be 0 (not given) or between 0.1 and 10")
        object.__setattr__(self, "scale", scale)


def lps_to_ras(point):
    """A markups position (LPS) in RAS."""
    import numpy as np

    return np.asarray(point, dtype=float) * np.array([-1.0, -1.0, 1.0])


def read_landmarks(path):
    """{label: RAS position} from a Slicer markups file."""
    import json

    import numpy as np

    with open(path) as handle:
        markup = json.load(handle)["markups"][0]
    lps = markup.get("coordinateSystem", "LPS") == "LPS"
    points = {}
    for control_point in markup["controlPoints"]:
        label = control_point.get("label")
        position = control_point.get("position")
        if not label or position is None or len(position) < 3:
            continue
        point = np.asarray(position[:3], dtype=float)
        points[label] = lps_to_ras(point) if lps else point
    return points


def largest_component(mask):
    """The biggest connected piece of a boolean mask."""
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (np.argmax(sizes) + 1)


def translation_to_reference(scan_landmarks, reference, used):
    """The translation that carries the kept landmarks onto the reference.

    Returns (translation, residuals, names): the mean difference, the distance
    left on each landmark after it, and which landmarks took part. Raises
    ValueError when no landmark is shared, because a crop placed on a guess
    would be a wrong crop that looks right.
    """
    import numpy as np

    names = [name for name in used if name in scan_landmarks and name in reference]
    if not names:
        raise ValueError("no landmark is shared by the scan and the reference")
    source = np.array([scan_landmarks[name] for name in names])
    target = np.array([reference[name] for name in names])
    translation = (target - source).mean(axis=0)
    residual = np.linalg.norm(source + translation - target, axis=1)
    return translation, residual, names


def find_condyle(points, porion, side_sign, settings):
    """Condyle top and head centre among the oriented mandible points of one side.

    The window keeps the region on that side of the midline, near the porion in
    x, and from `behind_mm` posterior to `ahead_mm` anterior of it. Returns
    None when the mandible has nothing there.
    """
    import numpy as np

    dx = points[:, 0] - porion[0]
    dy = points[:, 1] - porion[1]
    selected = (
        (np.sign(points[:, 0]) == side_sign)
        & (np.abs(dx) < settings.lateral_mm)
        & (dy > -settings.behind_mm)
        & (dy < settings.ahead_mm)
    )
    region = points[selected]
    if len(region) == 0:
        return None
    top = region[np.argmax(region[:, 2])]
    head = region[region[:, 2] > top[2] - settings.head_mm]
    head = head[np.linalg.norm(head[:, :2] - top[:2], axis=1) < settings.head_radius_mm]
    return top, head.mean(0), len(region)


def sample_box(volume, inverse_affine, translation, centre, shape, spacing, mirror, fill):
    """Resample a box of `shape` voxels centred on `centre`, in the oriented frame."""
    import numpy as np
    from scipy import ndimage

    axes = [(np.arange(n) - (n - 1) / 2.0) * spacing for n in shape]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    if mirror:
        gx = -gx
    oriented = np.stack([gx, gy, gz], -1).reshape(-1, 3) + centre
    world = oriented - translation  # the rotation is the identity: ASO already applied it
    ijk = world @ inverse_affine[:3, :3].T + inverse_affine[:3, 3]
    out = ndimage.map_coordinates(volume, ijk.T, order=1, mode="constant", cval=fill)
    return out.reshape(shape)


def oriented_slice(volume, inverse_affine, translation, x, ylim, zlim, spacing, fill):
    """A sagittal slice of the oriented scan at x, for the QC context view."""
    import numpy as np
    from scipy import ndimage

    ys = np.arange(ylim[0], ylim[1], spacing)
    zs = np.arange(zlim[0], zlim[1], spacing)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    oriented = np.stack([np.full_like(gy, x), gy, gz], -1).reshape(-1, 3)
    world = oriented - translation
    ijk = world @ inverse_affine[:3, :3].T + inverse_affine[:3, 3]
    out = ndimage.map_coordinates(volume, ijk.T, order=1, mode="constant", cval=fill)
    return out.reshape(gy.shape)


@dataclass
class CropResult:
    """What one scan produced: the JSON report and the crops for the QC image."""

    report: dict
    crops: dict = field(default_factory=dict)
    context: object = None


def crop_scan(scan_path, landmarks_path, mask_path, reference, used, dropped,
              settings, name, output_dir, sides=("R", "L")):
    """Crop the requested joints of one oriented scan and write `<name>_R/_L.nii.gz`.

    `sides` is the joints to cut, any of "R" and "L"; the other is neither
    searched for, written, nor reported.

    `reference` is {label: RAS position} of the reference landmarks; `used` and
    `dropped` come from ASO's report for this scan. Writes the two crops and
    the JSON report, and returns the QC inputs. Does not draw the QC image.
    """
    import json
    import os

    import nibabel as nib
    import numpy as np

    image = nib.load(scan_path)
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    inverse_affine = np.linalg.inv(image.affine)
    fill = float(np.percentile(volume, 0.5))

    scan_landmarks = read_landmarks(landmarks_path)
    translation, residual, common = translation_to_reference(scan_landmarks, reference, used)

    mask_image = nib.load(mask_path)
    mask = largest_component(np.asanyarray(mask_image.dataobj) > 0)
    if not mask.any():
        raise ValueError("the mandible mask is empty")
    mandible_ml = float(mask.sum()) * abs(np.linalg.det(mask_image.affine[:3, :3])) / 1000.0
    if settings.scale > 0:
        scale = settings.scale
    elif settings.scale_by == "mandible-volume":
        scale = (mandible_ml / settings.reference_volume_ml) ** (1.0 / 3.0)
    else:
        scale = 1.0
    spacing = settings.spacing * scale

    ijk = np.argwhere(mask)
    world = ijk @ mask_image.affine[:3, :3].T + mask_image.affine[:3, 3]
    oriented = world + translation

    shape = tuple(settings.shape)
    half = (np.array(shape) - 1) / 2.0 * spacing  # centre to the outer voxel centres
    report = {
        "scan": name,
        "landmarks_used": common,
        "landmarks_dropped": dropped,
        "fit_residual_mm": {key: round(float(r), 2) for key, r in zip(common, residual)},
        "box_mm": [round(n * spacing, 1) for n in shape],
        "spacing_mm": round(spacing, 4),
        "scale": round(scale, 4),
        "mandible_volume_ml": round(mandible_ml, 1),
        "translation": translation.round(4).tolist(),
        "sides_requested": list(sides),
        "sides": {},
    }
    crops = {}
    for side, sign in (("R", 1), ("L", -1)):
        if side not in sides:
            continue
        porion = reference[PORION[side]]
        found = find_condyle(oriented, porion, sign, settings)
        if found is None:
            report["sides"][side] = {"status": "no mandible near porion"}
            continue
        top, head, _ = found
        centre = head.copy()
        centre[1] = head[1] + settings.anterior_mm * scale - half[1]
        centre[2] = top[2] + settings.above_top_mm * scale - half[2]
        crop = sample_box(volume, inverse_affine, translation, centre, shape, spacing,
                          mirror=(side == "L"), fill=fill)
        distance = float(np.linalg.norm(top - porion))
        status = "ok" if distance <= settings.max_porion_mm else "check: condyle far from porion"
        # Box corners back in the scan's own RAS space, for a viewer; corner k
        # has bit 0/1/2 set when it lies on the high x/y/z face of the box.
        extent = np.array(shape) / 2.0 * spacing
        signs = np.array([[(k >> a) & 1 for a in range(3)] for k in range(8)]) * 2 - 1
        corners = centre + signs * extent - translation
        report["sides"][side] = {
            "status": status,
            "condyle_top": [round(float(v), 1) for v in top],
            "box_centre": [round(float(v), 1) for v in centre],
            "top_to_porion_mm": round(distance, 1),
            "top_minus_porion_mm": [round(float(v), 1) for v in top - porion],
            "box_corners_scan_ras": [[round(float(v), 2) for v in c] for c in corners],
            "condyle_top_scan_ras": [round(float(v), 2) for v in top - translation],
            "box_centre_scan_ras": [round(float(v), 2) for v in centre - translation],
        }
        # `centre` is in the oriented frame, which is the scan's own frame shifted
        # by `translation`. The crop is placed in the SCAN's frame, so it opens
        # on top of the oriented scan this tool can hand back; the array itself
        # is the same either way.
        affine = np.diag([spacing] * 3 + [1.0])
        affine[:3, 3] = centre - half - translation
        if side == "L":
            affine[0, 0] = -spacing
            affine[0, 3] = centre[0] + half[0] - translation[0]
        nib.save(nib.Nifti1Image(np.clip(np.rint(crop), -32768, 32767).astype(np.int16), affine),
                 os.path.join(output_dir, f"{name}_{side}.nii.gz"))
        crops[side] = (crop, centre, top)

    with open(os.path.join(output_dir, f"{name}.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    return CropResult(report=report, crops=crops,
                      context=(volume, inverse_affine, translation, fill, shape, spacing))


def _window(array, reference):
    """Scale to [0, 1] with the 1st-99.5th percentiles of `reference` (the whole crop)."""
    import numpy as np

    low, high = np.percentile(reference, [1, 99.5])
    return np.clip((array - low) / max(high - low, 1e-6), 0, 1)


def write_quality_image(result, reference, name, output_dir):
    """One PNG: three orthogonal slices per crop, and the box on a sagittal slice."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    volume, inverse_affine, translation, fill, shape, spacing = result.context
    wanted = [side for side in ("R", "L") if side in result.report["sides_requested"]]
    figure, axes = plt.subplots(len(wanted), 4, figsize=(16, 4.2 * len(wanted)), squeeze=False)
    size = np.array(shape) * spacing
    for row, side in enumerate(wanted):
        axes_row = axes[row]
        for axis in axes_row:
            axis.set_axis_off()
        if side not in result.crops:
            axes_row[0].set_title(f"{side}: not found")
            continue
        crop, centre, top = result.crops[side]
        middle = [n // 2 for n in shape]
        views = [
            ("axial", crop[:, :, middle[2]].T, size[0], size[1]),
            ("coronal", crop[:, middle[1], :].T, size[0], size[2]),
            ("sagittal", crop[middle[0], :, :].T, size[1], size[2]),
        ]
        for axis, (label, picture, width, height) in zip(axes_row[:3], views):
            axis.imshow(_window(picture, crop), cmap="gray", origin="lower",
                        extent=[0, width, 0, height])
            axis.set_title(f"{side} {label}")
        ylim = (centre[1] - 35, centre[1] + 35)
        zlim = (centre[2] - 35, centre[2] + 35)
        context = oriented_slice(volume, inverse_affine, translation, top[0], ylim, zlim, 0.3, fill)
        axis = axes_row[3]
        axis.imshow(_window(context, crop).T, cmap="gray", origin="lower",
                    extent=[ylim[0], ylim[1], zlim[0], zlim[1]])
        axis.add_patch(patches.Rectangle(
            (centre[1] - size[1] / 2, centre[2] - size[2] / 2), size[1], size[2],
            fill=False, color="red", lw=1.5))
        porion = reference[PORION[side]]
        axis.plot([porion[1]], [porion[2]], "c+", ms=10)
        axis.plot([top[1]], [top[2]], "y.", ms=6)
        axis.set_title(f"{side} context (sagittal, anterior right)")
    figure.suptitle("crops in ASO orientation" + (", left side mirrored" if "L" in wanted else ""))
    figure.tight_layout()
    path = os.path.join(output_dir, f"{name}_qc.png")
    figure.savefig(path, dpi=90)
    plt.close(figure)
    return path
