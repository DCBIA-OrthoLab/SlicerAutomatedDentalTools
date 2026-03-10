#!/usr/bin/env python
"""
Worker script for batch_process.
Each invocation processes a SINGLE pair of VTK files in its own process.
When this process exits, ALL memory (including VTK C++ allocations) is
reclaimed by the operating system — no leaks possible.

Usage:
    python _batch_worker.py --file1 path/to/t1.vtk --file2 path/to/t2.vtk --output path/to/out.vtk [--signed|--unsigned]
"""
import sys
import argparse
import traceback


def read_vtk(file_path):
    """Read a VTK/VTP file and return a vtkPolyData."""
    import vtk
    file_path = str(file_path)
    if file_path.lower().endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(reader.GetOutput())
    return output


def write_vtk(polydata, file_path):
    """Write a vtkPolyData to a VTK/VTP file."""
    import vtk
    file_path = str(file_path)
    if file_path.lower().endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()


def clean_and_triangulate(polydata):
    """Clean and triangulate a vtkPolyData."""
    import vtk
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.SetAbsoluteTolerance(1e-6)
    cleaner.Update()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(cleaner.GetOutput())
    triangle_filter.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(triangle_filter.GetOutput())
    return result


def subsample_polydata(polydata, target_reduction=0.5):
    """Reduce the number of points in a polydata using quadric decimation."""
    import vtk
    num_points = polydata.GetNumberOfPoints()
    if num_points < 1000:
        return polydata

    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputData(polydata)
    decimation.SetTargetReduction(target_reduction)
    decimation.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(decimation.GetOutput())
    return result


def interpolate_distance_to_original(original, subsampled):
    """
    Interpolate the distance array from a subsampled mesh back to the original mesh
    using vtkPointLocator for efficiency.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    import numpy as np

    sub_array = subsampled.GetPointData().GetArray("Distance")
    if sub_array is None:
        print("Warning: No 'Distance' array found on subsampled mesh.")
        return original

    sub_distances = vtk_to_numpy(sub_array)

    # Use vtkPointLocator for O(n log n) instead of O(n*m) brute force
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(subsampled)
    locator.BuildLocator()

    num_original_points = original.GetNumberOfPoints()
    orig_distances = np.zeros(num_original_points)

    for i in range(num_original_points):
        pt = original.GetPoint(i)
        closest_id = locator.FindClosestPoint(pt)
        orig_distances[i] = sub_distances[closest_id]

    distance_vtk = numpy_to_vtk(orig_distances, deep=True)
    distance_vtk.SetName("Distance")
    original.GetPointData().AddArray(distance_vtk)
    original.GetPointData().SetActiveScalars("Distance")

    return original


def compute_distance(pd1, pd2, signed=True):
    """
    Compute distance between two polydata meshes.
    Uses subsampling for large meshes to reduce memory usage.
    """
    import vtk

    n1 = pd1.GetNumberOfPoints()
    n2 = pd2.GetNumberOfPoints()
    total_points = n1 + n2
    complexity = n1 * n2  # Rough indicator of vtkDistancePolyDataFilter memory usage

    print(f"  Points: pd1={n1}, pd2={n2}, total={total_points}, complexity={complexity:.2e}")

    # Thresholds for subsampling
    SUBSAMPLE_THRESHOLD = 1_000_000       # 1M points combined
    HEAVY_SUBSAMPLE_THRESHOLD = 2_000_000 # 2M points combined

    if total_points > HEAVY_SUBSAMPLE_THRESHOLD:
        print(f"  -> HEAVY subsampling (target_reduction=0.7)")
        pd1_sub = subsample_polydata(pd1, target_reduction=0.7)
        pd2_sub = subsample_polydata(pd2, target_reduction=0.7)
        result_sub = compute_distance_direct(pd1_sub, pd2_sub, signed)
        result = interpolate_distance_to_original(pd1, result_sub)
        return result

    elif total_points > SUBSAMPLE_THRESHOLD:
        print(f"  -> MODERATE subsampling (target_reduction=0.5)")
        pd1_sub = subsample_polydata(pd1, target_reduction=0.5)
        pd2_sub = subsample_polydata(pd2, target_reduction=0.5)
        result_sub = compute_distance_direct(pd1_sub, pd2_sub, signed)
        result = interpolate_distance_to_original(pd1, result_sub)
        return result

    else:
        print(f"  -> Direct computation (no subsampling needed)")
        return compute_distance_direct(pd1, pd2, signed)


def compute_distance_direct(pd1, pd2, signed=True):
    """Compute distance using vtkDistancePolyDataFilter."""
    import vtk

    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, pd1)
    distance_filter.SetInputData(1, pd2)
    if signed:
        distance_filter.SignedDistanceOn()
    else:
        distance_filter.SignedDistanceOff()
    distance_filter.ComputeSecondDistanceOff()
    distance_filter.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(distance_filter.GetOutput())

    # Rename the distance array for consistency
    if result.GetPointData().GetArray("Distance"):
        pass  # Already named correctly
    else:
        for i in range(result.GetPointData().GetNumberOfArrays()):
            arr = result.GetPointData().GetArray(i)
            if arr and arr.GetName() and "istance" in arr.GetName():
                arr.SetName("Distance")
                break

    return result


def main():
    parser = argparse.ArgumentParser(description="Process a single VTK pair for distance computation")
    parser.add_argument("--file1", required=True, help="Path to T1 VTK file")
    parser.add_argument("--file2", required=True, help="Path to T2 VTK file")
    parser.add_argument("--output", required=True, help="Path to output VTK file")
    parser.add_argument("--signed", action="store_true", default=True, help="Compute signed distance")
    parser.add_argument("--unsigned", action="store_true", default=False, help="Compute unsigned distance")
    args = parser.parse_args()

    signed = not args.unsigned

    try:
        print(f"Reading {args.file1}...")
        pd1 = read_vtk(args.file1)
        print(f"Reading {args.file2}...")
        pd2 = read_vtk(args.file2)

        print(f"Cleaning and triangulating...")
        pd1 = clean_and_triangulate(pd1)
        pd2 = clean_and_triangulate(pd2)

        print(f"Computing distance (signed={signed})...")
        result = compute_distance(pd1, pd2, signed=signed)

        print(f"Writing {args.output}...")
        write_vtk(result, args.output)

        print("SUCCESS")
        sys.exit(0)

    except Exception as e:
        print(f"FAILURE: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
