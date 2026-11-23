"""Dividing a folder of inputs into batches, and packing one of them.

A cohort of 20 CBCTs is ~2 GB in one archive: the server's card waits for its
last byte, a connection dropped at 95% starts again from zero, and it is over
MAX_UPLOAD_MB anyway. The server publishes how much to send at once (see its
`GET /tools` `batch` field); everything here is the half that acts on it.

Two rules carry the whole thing, and both are tested by what they REFUSE to do:
a top-level entry is never opened, and nothing is ever lost between the batches.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import types
import zipfile

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: F401,E402 - installs the stubs

from ServerToolsCoreLib import design, slicer_io  # noqa: E402


MB = 1024 * 1024


class _Cohort:
    """A folder on disk, built entry by entry."""

    def __init__(self):
        self.path = tempfile.mkdtemp(prefix="cohort_")

    def close(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def scan(self, name: str, size: int = 32) -> str:
        """One file of `size` bytes. Sparse above a few kB, so a 50 MB scan
        costs an inode and nothing else -- these tests are about the arithmetic
        of sizing a batch, not about moving bytes."""
        path = os.path.join(self.path, name)
        with open(path, "wb") as handle:
            handle.truncate(size)
        return path

    def patient(self, name: str, slices: int = 3, size: int = 32) -> str:
        """A subfolder, the shape a DICOM series and a per-patient cohort take."""
        folder = os.path.join(self.path, name)
        os.makedirs(folder, exist_ok=True)
        for index in range(slices):
            with open(os.path.join(folder, f"{index:03d}.dcm"), "wb") as handle:
                handle.truncate(size)
        return folder


class SplitCohortTest(unittest.TestCase):
    def setUp(self):
        self.cohort = _Cohort()
        self.addCleanup(self.cohort.close)

    def split(self, max_mb=400, max_files=25):
        return slicer_io.split_cohort(self.cohort.path, max_mb, max_files)

    # --- the invariant that must never break --------------------------

    def test_a_folder_under_both_caps_is_one_batch(self):
        """Which is what makes this change invisible for every cohort small
        enough not to need it: one batch is one run, byte for byte today's."""
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.assertEqual(len(self.split()), 1)

    def test_every_entry_lands_in_exactly_one_batch(self):
        """Nothing lost, nothing sent twice. A patient silently dropped here is
        a patient missing from a report that says the run succeeded."""
        for index in range(23):
            self.cohort.scan(f"scan{index:02d}.nii.gz", size=3 * MB)
        self.cohort.patient("dicom_patient")

        batches = self.split(max_mb=10, max_files=4)
        seen = [name for batch in batches for name in batch]

        self.assertEqual(sorted(seen), sorted(os.listdir(self.cohort.path)))
        self.assertEqual(len(seen), len(set(seen)))

    # --- what decides where the cut falls ------------------------------

    def test_megabytes_are_what_normally_binds(self):
        """The reason the server sizes in bytes rather than in files: 8 CBCTs
        of 50 MB fill a 400 MB batch, and nobody had to write "CBCT" anywhere
        for that to happen."""
        for index in range(20):
            self.cohort.scan(f"cbct{index:02d}.nii.gz", size=50 * MB)

        batches = self.split(max_mb=400, max_files=25)

        self.assertEqual([len(batch) for batch in batches], [8, 8, 4])

    def test_small_files_batch_far_more_of_themselves(self):
        """Same two numbers, same code, intraoral surfaces instead: 5 MB each,
        so the FILE cap is what binds and a batch holds 25 of them."""
        for index in range(60):
            self.cohort.scan(f"arch{index:02d}.vtk", size=5 * MB)

        batches = self.split(max_mb=400, max_files=25)

        self.assertEqual([len(batch) for batch in batches], [25, 25, 10])

    def test_the_file_cap_alone_can_bind(self):
        """5 000 clinical notes are 100 MB -- one batch by bytes, and not one
        partial result until the last note is done."""
        for index in range(100):
            self.cohort.scan(f"note{index:03d}.txt")

        self.assertEqual([len(batch) for batch in self.split(max_files=25)], [25] * 4)

    def test_a_cap_of_zero_does_not_bind(self):
        for index in range(40):
            self.cohort.scan(f"scan{index:02d}.nii.gz", size=50 * MB)

        self.assertEqual([len(batch) for batch in self.split(max_mb=0, max_files=10)],
                         [10, 10, 10, 10])
        self.assertEqual(len(self.split(max_mb=0, max_files=0)), 1)

    def test_an_entry_larger_than_a_whole_batch_travels_alone(self):
        """Refused, dropped or split are all worse. It goes on its own and the
        server's upload limit is what has the last word on it."""
        self.cohort.scan("small_a.nii.gz", size=1 * MB)
        self.cohort.scan("enormous.nii.gz", size=900 * MB)
        self.cohort.scan("small_b.nii.gz", size=1 * MB)

        batches = self.split(max_mb=400)

        self.assertEqual(batches, [["enormous.nii.gz"], ["small_a.nii.gz", "small_b.nii.gz"]])

    # --- a patient is never opened --------------------------------------

    def test_a_subfolder_is_one_unit_however_many_files_it_holds(self):
        """A DICOM series is hundreds of slices that mean nothing apart, and a
        cohort filed per patient is the same shape. Splitting inside one hands
        the tool half a patient and calls it a batch."""
        for index in range(6):
            self.cohort.patient(f"patient{index}", slices=200)

        batches = self.split(max_files=2)

        self.assertEqual([len(batch) for batch in batches], [2, 2, 2])
        self.assertEqual(batches[0], ["patient0", "patient1"])

    def test_a_subfolder_is_sized_by_everything_inside_it(self):
        """Counted as one entry, weighed as all of it: a 300 MB series must not
        ride along as if it were a single small file."""
        self.cohort.patient("heavy", slices=3, size=150 * MB)
        self.cohort.scan("light.nii.gz", size=1 * MB)

        batches = self.split(max_mb=400)

        self.assertEqual(batches, [["heavy"], ["light.nii.gz"]])

    # --- the same cohort divides the same way every time ----------------

    def test_the_split_is_sorted_and_repeatable(self):
        """A rerun after a failure resends the same batches, and two timepoints
        of one patient stay adjacent rather than landing in different runs."""
        for name in ("b_T2", "a_T2", "b_T1", "a_T1"):
            self.cohort.scan(f"{name}.nii.gz")

        first = self.split(max_files=2)

        self.assertEqual(first, [["a_T1.nii.gz", "a_T2.nii.gz"],
                                 ["b_T1.nii.gz", "b_T2.nii.gz"]])
        self.assertEqual(self.split(max_files=2), first)

    def test_an_empty_or_unreadable_folder_divides_into_nothing(self):
        """The caller then sends what it has, and the server reports an empty
        cohort the way it always did."""
        self.assertEqual(self.split(), [])
        self.assertEqual(slicer_io.split_cohort("/no/such/folder", 400, 25), [])


class ZipSubsetTest(unittest.TestCase):
    def setUp(self):
        self.cohort = _Cohort()
        self.addCleanup(self.cohort.close)
        self.dest = tempfile.mkdtemp(prefix="batch_zip_")
        self.addCleanup(shutil.rmtree, self.dest, True)

    def _members(self, archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            return sorted(info.filename for info in archive.infolist() if not info.is_dir())

    def test_members_are_named_as_if_the_whole_folder_had_been_sent(self):
        """What lets a batch be an ordinary run: the server unpacks the same
        tree shape, so a tool mirroring its input tree keeps working per batch."""
        self.cohort.scan("a.nii.gz")
        self.cohort.patient("patient1", slices=2)
        self.cohort.scan("b.nii.gz")
        out = os.path.join(self.dest, "batch.zip")

        slicer_io.zip_subset(self.cohort.path, ["a.nii.gz", "patient1"], out, compress=False)

        self.assertEqual(
            self._members(out),
            # Forward slashes on every system: that is how a zip names members.
            ["a.nii.gz", "patient1/000.dcm", "patient1/001.dcm"],
        )

    def test_nothing_outside_the_batch_is_packed(self):
        self.cohort.scan("wanted.nii.gz")
        self.cohort.scan("not_this_time.nii.gz")
        out = os.path.join(self.dest, "batch.zip")

        slicer_io.zip_subset(self.cohort.path, ["wanted.nii.gz"], out, compress=False)

        self.assertEqual(self._members(out), ["wanted.nii.gz"])

    def test_packing_every_entry_gives_what_zipping_the_folder_gives(self):
        """The refactor that made the two share a writer must not have changed
        the whole-folder path, which every un-batched run still takes."""
        self.cohort.scan("a.nii.gz")
        self.cohort.patient("patient1", slices=2)
        whole = os.path.join(self.dest, "whole.zip")
        pieces = os.path.join(self.dest, "pieces.zip")

        slicer_io.zip_folder(self.cohort.path, whole, compress=False)
        slicer_io.zip_subset(self.cohort.path, ["a.nii.gz", "patient1"], pieces, compress=False)

        self.assertEqual(self._members(whole), self._members(pieces))

    def test_an_already_compressed_member_is_stored_rather_than_deflated(self):
        """Pinned because it survives a refactor only if someone checks: a
        .nii.gz re-deflated is CPU spent to save nothing, and it is what every
        member of a CBCT cohort is."""
        self.cohort.scan("scan.nii.gz", size=4096)
        self.cohort.scan("landmarks.json", size=4096)
        out = os.path.join(self.dest, "batch.zip")

        slicer_io.zip_subset(
            self.cohort.path, ["scan.nii.gz", "landmarks.json"], out, compress=True)

        with zipfile.ZipFile(out) as archive:
            kinds = {info.filename: info.compress_type for info in archive.infolist()}
        self.assertEqual(kinds["scan.nii.gz"], zipfile.ZIP_STORED)
        self.assertEqual(kinds["landmarks.json"], zipfile.ZIP_DEFLATED)

    def test_a_path_that_is_not_a_folder_is_refused(self):
        with self.assertRaises(IOError):
            slicer_io.zip_subset("/no/such/folder", ["a"], os.path.join(self.dest, "x.zip"))


if __name__ == "__main__":
    unittest.main()


class _Widget:
    """An input row as `_prepareOneInputFile` reads one."""

    def __init__(self, path):
        self.currentPath = path

    def is_folder(self):
        return os.path.isdir(self.currentPath)


class ApplyQueuesOneRunPerBatchTest(unittest.TestCase):
    """The whole feature, from the panel's side: one Apply, several runs.

    Nothing below stubs `prepareInputFiles` -- that is the code under test.
    `_pumpRuns` is stubbed, because whether a queued run STARTS is test_runs.py's
    subject and has nothing to learn from a cohort.
    """

    PLAN = {"axis": "scans", "max_mb": 0, "max_files": 2}

    def setUp(self):
        from ServerToolsCoreLib.base_widget import ServerToolWidgetBase

        self.cohort = _Cohort()
        self.addCleanup(self.cohort.close)
        self.output = tempfile.mkdtemp(prefix="out_")
        self.addCleanup(shutil.rmtree, self.output, True)

        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel.TOOL_NAME = "AMASSS"
        panel._runs = []
        panel._runsStarted = 0
        panel._schema = {"arguments": {"scans": {"type": "path"}}, "batch": dict(self.PLAN)}
        panel._inputModes = {"scans": "folder_zip"}
        panel._inputWidgets = {"scans": _Widget(self.cohort.path)}
        panel._hiddenArgs = set()
        panel._outputFolderWidget = _Widget(self.output)
        panel.collectArgs = lambda: {"suffix": "_seg"}
        panel._pumpRuns = lambda: None
        # A server that predates the batch policy endpoint: the panel keeps
        # its own concurrency.
        panel.client = types.SimpleNamespace(batch_policy=lambda: None)
        self.panel = panel
        # The workspaces outlive onApplyButton -- a run holds one until it ends.
        self.addCleanup(self._cleanWorkspaces)

    def _cleanWorkspaces(self):
        for run in self.panel._runs:
            run.workspace.__exit__(None, None, None)

    def _packed(self, run):
        with zipfile.ZipFile(run.files["scans"]) as archive:
            return sorted(info.filename for info in archive.infolist() if not info.is_dir())

    def test_one_apply_queues_one_run_per_batch(self):
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.panel.onApplyButton()

        self.assertEqual(len(self.panel._runs), 3)
        self.assertEqual([self._packed(run) for run in self.panel._runs],
                         [["patient0.nii.gz", "patient1.nii.gz"],
                          ["patient2.nii.gz", "patient3.nii.gz"],
                          ["patient4.nii.gz"]])

    def test_a_batch_writes_apart_while_the_others_are_still_running(self):
        """Into the folder the clinician picked, one level down.

        Sharing one folder meant a finished batch's results sitting among a
        running one's half-written files with nothing saying which was which,
        and two batches writing one name overwriting in silence. They are
        folded back into that folder when the last batch lands, so what the
        clinician ends up with is what they asked for.
        """
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.panel.onApplyButton()

        self.assertEqual([run.output_dir for run in self.panel._runs],
                         [os.path.join(self.output, "batch_%02d" % n)
                          for n in (1, 2, 3)])
        for run in self.panel._runs:
            self.assertTrue(os.path.isdir(run.output_dir),
                            "the batch folder was named but never created")
        self.assertEqual({run.cohort.root for run in self.panel._runs},
                         {self.output})

    def test_each_batch_says_which_one_it_is(self):
        for index in range(3):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.panel.onApplyButton()

        labels = [run.label for run in self.panel._runs]
        self.assertTrue(all("(%d/2)" % (index + 1) in labels[index] for index in range(2)), labels)
        self.assertEqual([run.cohort_index for run in self.panel._runs], [1, 2])
        self.assertEqual({run.cohort.total for run in self.panel._runs}, {2})
        # One cohort, not one per batch: it is what the batches have in common.
        self.assertEqual(len({id(run.cohort) for run in self.panel._runs}), 1)

    def test_each_batch_gets_its_own_copy_of_the_arguments(self):
        """One dict shared by three runs is one dict any of them could still be
        reading when another is written to."""
        for index in range(3):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.panel.onApplyButton()
        self.panel._runs[0].args["suffix"] = "_touched"

        self.assertEqual(self.panel._runs[1].args["suffix"], "_seg")

    def test_the_arguments_are_read_once_for_the_whole_cohort(self):
        """Read at Apply, like every other input: five batches must not pick up
        a value the user changed while the first was uploading."""
        reads = []
        self.panel.collectArgs = lambda: reads.append(1) or {"suffix": "_seg"}
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.panel.onApplyButton()

        self.assertEqual(len(reads), 1)

    # --- everything that must stay exactly one run ---------------------

    def _assertOneWholeRun(self):
        self.panel.onApplyButton()
        self.assertEqual(len(self.panel._runs), 1)
        run = self.panel._runs[0]
        self.assertIsNone(run.cohort_index)
        self.assertNotIn("/", run.label)
        return run

    def test_a_cohort_that_fits_in_one_batch_is_one_run(self):
        """The invariant: below the caps, this feature is not observable."""
        self.cohort.scan("only.nii.gz")

        self.assertEqual(self._packed(self._assertOneWholeRun()), ["only.nii.gz"])

    def test_a_single_file_is_sent_as_the_file_it_is(self):
        """Not zipped and not divided: `file_or_folder` reads which one it was
        given off the path, and a cohort of one is not a cohort."""
        scan = self.cohort.scan("single.nii.gz")
        self.panel._inputModes = {"scans": "file_or_folder"}
        self.panel._inputWidgets = {"scans": _Widget(scan)}

        self.assertEqual(self._assertOneWholeRun().files["scans"], scan)

    def test_a_server_that_publishes_no_plan_sends_the_cohort_whole(self):
        """An older server, or a tool pairing two folders per patient."""
        self.panel._schema.pop("batch")
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.assertEqual(len(self._packed(self._assertOneWholeRun())), 5)

    def test_a_panel_with_nowhere_to_write_sends_the_cohort_whole(self):
        """Without one output folder the batches' results land in per-run
        temporary directories that are removed as each run ends."""
        self.panel._outputFolderWidget = None
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self._assertOneWholeRun()

    def test_a_module_that_builds_its_own_inputs_sends_the_cohort_whole(self):
        """Dividing what an override produces is a guess about somebody else's
        work. Note it is still called the way it always was -- with one
        argument -- which is what stops this from breaking such a module."""
        seen = []
        self.panel.prepareInputFiles = lambda workspace: seen.append(workspace) or {"scans": "x"}
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")

        self.panel.onApplyButton()

        self.assertEqual(len(self.panel._runs), 1)
        self.assertEqual(len(seen), 1)

    def test_an_axis_that_is_not_a_folder_on_this_machine_sends_one_run(self):
        """A single scan, a volume picked out of the scene, a name the server
        hosts: there is no cohort here to divide."""
        self.panel._inputModes = {"scans": "file_or_folder"}
        self.panel._inputWidgets = {"scans": _Widget(self.cohort.scan("single.nii.gz"))}

        self._assertOneWholeRun()

    def test_nothing_is_queued_when_one_batch_cannot_be_packed(self):
        """All or nothing: half a cohort queued and half of it reported as an
        error is the one outcome nobody can act on."""
        for index in range(5):
            self.cohort.scan(f"patient{index}.nii.gz")
        packed = []
        real = self.panel._zipFolder

        def failing(workspace, arg_name, folder, entries=None):
            if len(packed) == 2:
                raise IOError("the disk filled up")
            packed.append(entries)
            return real(workspace, arg_name, folder, entries)

        self.panel._zipFolder = failing

        self.panel.onApplyButton()

        self.assertEqual(self.panel._runs, [])


class MergedReportTest(unittest.TestCase):
    """Folding several batches' run reports back into one.

    The failure this prevents is the quietest one the feature could produce: a
    cohort of forty patients, every result on disk, and a summary describing the
    four the last batch happened to hold.
    """

    def merge(self, first, second):
        from ServerToolsCoreLib.base_widget import _merged_report
        return _merged_report(first, second)

    def test_counters_add_up(self):
        merged = self.merge({"summary": {"cropped": 4, "scans_found": 5}},
                            {"summary": {"cropped": 3, "scans_found": 3}})

        self.assertEqual(merged["summary"], {"cropped": 7, "scans_found": 8})

    def test_per_scan_lists_are_concatenated(self):
        merged = self.merge({"without_a_roi": [{"patient": "a"}]},
                            {"without_a_roi": [{"patient": "b"}]})

        self.assertEqual(merged["without_a_roi"], [{"patient": "a"}, {"patient": "b"}])

    def test_a_flag_is_a_property_of_the_run_not_a_counter(self):
        """A bool IS an int in Python, so two `true`s add to 2 unless this is
        checked first -- and `"gpu_resampling": 2` is not a thing."""
        merged = self.merge({"gpu_resampling": True}, {"gpu_resampling": True})

        self.assertIs(merged["gpu_resampling"], True)

    def test_what_names_the_run_keeps_the_first_batch_s_word(self):
        merged = self.merge({"tool": "AMASSS", "model_bundle": "AMASSS_Models"},
                            {"tool": "AMASSS", "model_bundle": "AMASSS_Models"})

        self.assertEqual(merged["model_bundle"], "AMASSS_Models")

    def test_a_key_only_one_batch_produced_survives(self):
        """A batch where nothing failed writes no `failures` key at all; the
        one that did must not lose it to the one that did not."""
        self.assertEqual(self.merge({"summary": {}}, {"failures": ["x"]})["failures"], ["x"])
        self.assertEqual(self.merge({"failures": ["x"]}, {"summary": {}})["failures"], ["x"])

    def test_it_does_not_care_what_a_tool_puts_in_its_report(self):
        """The rule is on the JSON types, never on a field name: every tool
        writes its own shape and this is one function for all of them."""
        merged = self.merge(
            {"deep": {"nested": {"count": 1, "names": ["a"]}}},
            {"deep": {"nested": {"count": 2, "names": ["b"]}}},
        )

        self.assertEqual(merged["deep"]["nested"], {"count": 3, "names": ["a", "b"]})


class CohortReportOnDiskTest(unittest.TestCase):
    """The merge where it matters: the file every module reads afterwards."""

    def setUp(self):
        from ServerToolsCoreLib.base_widget import ServerToolWidgetBase, _Cohort, _Run

        self.output = tempfile.mkdtemp(prefix="out_")
        self.addCleanup(shutil.rmtree, self.output, True)

        class _Panel(ServerToolWidgetBase):
            RUN_REPORT = "AMASSS_report.json"

        self.panel = _Panel.__new__(_Panel)
        self.cohort = _Cohort(2)
        self.run = _Run(1, "cohort (1/2)", {}, {}, self.output, None,
                        cohort=self.cohort, cohort_index=1)
        self.lone = _Run(2, "one.nii.gz", {}, {}, self.output, None)

    def _write(self, payload):
        path = os.path.join(self.output, "AMASSS_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_the_report_on_disk_ends_up_describing_the_whole_cohort(self):
        self.panel._runInHand = self.run
        self._write({"summary": {"segmented": 4}, "scans": ["a", "b"]})
        self.panel._mergeRunReport(self.output)
        # The second batch overwrites the file, exactly as the server's archive
        # does -- which is the whole reason this exists.
        self._write({"summary": {"segmented": 3}, "scans": ["c"]})
        self.panel._mergeRunReport(self.output)

        report = self.panel._readRunReport(self.output)

        self.assertEqual(report["summary"]["segmented"], 7)
        self.assertEqual(report["scans"], ["a", "b", "c"])

    def test_it_is_complete_at_every_step_not_only_at_the_end(self):
        """A cohort abandoned halfway leaves a report of exactly what ran."""
        self.panel._runInHand = self.run
        self._write({"summary": {"segmented": 4}})
        self.panel._mergeRunReport(self.output)

        self.assertEqual(self.panel._readRunReport(self.output)["summary"]["segmented"], 4)

    def test_an_ordinary_run_s_report_is_left_exactly_as_the_tool_wrote_it(self):
        self.panel._runInHand = self.lone
        self._write({"summary": {"segmented": 4}})
        self.panel._mergeRunReport(self.output)

        self.assertIsNone(self.cohort.report)
        self.assertEqual(self.panel._readRunReport(self.output), {"summary": {"segmented": 4}})

    def test_a_batch_that_produced_no_report_costs_the_summary_and_not_the_run(self):
        self.panel._runInHand = self.run

        self.panel._mergeRunReport(self.output)  # nothing on disk at all

        self.assertIsNone(self.cohort.report)

    def test_a_report_that_cannot_be_parsed_is_not_fatal(self):
        self.panel._runInHand = self.run
        with open(os.path.join(self.output, "AMASSS_report.json"), "w") as handle:
            handle.write("{ this is not json")

        self.panel._mergeRunReport(self.output)

        self.assertIsNone(self.cohort.report)


class CohortFoldedBackTest(unittest.TestCase):
    """The batches write apart, and are put back together when the last lands.

    Apart while they run, because a finished batch's results among a running
    one's half-written files say nothing about which is which, and two
    batches writing one name overwrite in silence. Together at the end,
    because a clinician looking for a patient should not have to know which
    batch the transfer happened to put them in.
    """

    def setUp(self):
        from ServerToolsCoreLib.base_widget import ServerToolWidgetBase, _Cohort, _Run

        self.output = tempfile.mkdtemp(prefix="out_")
        self.addCleanup(shutil.rmtree, self.output, True)

        class _Panel(ServerToolWidgetBase):
            RUN_REPORT = "AMASSS_report.json"
            TOOL_NAME = "AMASSS"

        self.panel = _Panel.__new__(_Panel)
        self.cohort = _Cohort(2)
        self.cohort.root = self.output
        self.runs = [_Run(n, "cohort (%d/2)" % n, {}, {},
                          os.path.join(self.output, "batch_%02d" % n), None,
                          cohort=self.cohort, cohort_index=n) for n in (1, 2)]

    def _produced(self, batch, relative, data="result"):
        path = os.path.join(self.output, "batch_%02d" % batch, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(data)
        return path

    def _here(self):
        """Everything under the output folder, relative and sorted."""
        found = []
        for root, _dirs, names in os.walk(self.output):
            for name in names:
                found.append(os.path.relpath(os.path.join(root, name), self.output))
        return sorted(found)

    def _endCohort(self):
        self.cohort.finished = self.cohort.total
        self.panel._finishCohort(self.runs[-1])

    def test_the_batches_are_folded_back_into_the_folder_that_was_picked(self):
        self._produced(1, "p1_Or.nii.gz")
        self._produced(1, "p2_Or.nii.gz")
        self._produced(2, "p3_Or.nii.gz")

        self._endCohort()

        self.assertEqual(self._here(),
                         ["p1_Or.nii.gz", "p2_Or.nii.gz", "p3_Or.nii.gz"])
        self.assertEqual([name for name in os.listdir(self.output)
                          if name.startswith("batch_")], [],
                         "an empty batch folder was left behind")

    def test_a_tree_is_merged_rather_than_replaced(self):
        """A tool mirrors its input tree, so two batches legitimately write
        into subfolders of the same name -- and moving one over the other
        would take the whole of the first with it."""
        self._produced(1, os.path.join("Segmentations", "p1.nii.gz"))
        self._produced(2, os.path.join("Segmentations", "p3.nii.gz"))

        self._endCohort()

        self.assertEqual(self._here(), [os.path.join("Segmentations", "p1.nii.gz"),
                                        os.path.join("Segmentations", "p3.nii.gz")])

    def test_what_survives_is_the_cohort_report_not_whichever_batch_moved_first(self):
        """Every batch writes the same report name. Folding them up on file
        order would leave a report that describes one batch and is named as
        if it described the run."""
        self._produced(1, "AMASSS_report.json", json.dumps({"scans": ["a"]}))
        self._produced(2, "AMASSS_report.json", json.dumps({"scans": ["c"]}))
        self.cohort.report = {"scans": ["a", "b", "c"]}

        self._endCohort()

        with open(os.path.join(self.output, "AMASSS_report.json")) as handle:
            self.assertEqual(json.load(handle), {"scans": ["a", "b", "c"]})

    def test_one_name_produced_by_two_batches_keeps_the_first(self):
        """Two batches hold different patients, so this means two runs wrote
        one name -- and the one already there is what the merged report and
        any loaded node already refer to."""
        self._produced(1, "shared.nii.gz", "from the first")
        self._produced(2, "shared.nii.gz", "from the second")

        self._endCohort()

        with open(os.path.join(self.output, "shared.nii.gz")) as handle:
            self.assertEqual(handle.read(), "from the first")

    def test_the_reviews_come_up_with_the_results_they_belong_to(self):
        """A correction a clinician made is not scratch: it is the record of
        what they changed, and it has to survive the merge like everything
        else."""
        self._produced(1, os.path.join("quality_control", "batch_01",
                                       "ALI_CBCT", "p1_lm_Pred.mrk.json"))
        self._produced(2, os.path.join("quality_control", "batch_02",
                                       "ALI_CBCT", "p3_lm_Pred.mrk.json"))

        self._endCohort()

        self.assertEqual(self._here(), [
            os.path.join("quality_control", "batch_01", "ALI_CBCT", "p1_lm_Pred.mrk.json"),
            os.path.join("quality_control", "batch_02", "ALI_CBCT", "p3_lm_Pred.mrk.json"),
        ])

    def test_nothing_is_folded_while_a_batch_is_still_running(self):
        """Half a cohort merged reads as a finished one that lost patients."""
        self._produced(1, "p1_Or.nii.gz")
        self.cohort.finished = 1

        self.panel._finishCohort(self.runs[0])

        self.assertEqual(self._here(), [os.path.join("batch_01", "p1_Or.nii.gz")])

    def test_a_folder_of_the_clinicians_own_is_not_moved_by_us(self):
        """Only what this panel named. Someone who keeps a `notes` folder
        beside their results keeps it exactly where it is."""
        os.makedirs(os.path.join(self.output, "notes"))
        with open(os.path.join(self.output, "notes", "mine.txt"), "w") as handle:
            handle.write("kept")
        self._produced(1, "p1_Or.nii.gz")

        self._endCohort()

        self.assertEqual(self._here(), [os.path.join("notes", "mine.txt"),
                                        "p1_Or.nii.gz"])

    def test_an_undivided_run_has_no_cohort_and_nothing_is_touched(self):
        from ServerToolsCoreLib.base_widget import _Run
        with open(os.path.join(self.output, "p1_Or.nii.gz"), "w") as handle:
            handle.write("result")

        self.panel._finishCohort(_Run(1, "one", {}, {}, self.output, None))

        self.assertEqual(self._here(), ["p1_Or.nii.gz"])


class OneDialogPerCohortTest(unittest.TestCase):
    """Five modal dialogs for one Apply are four clicks nobody asked for, each
    one interrupting the upload of the next batch."""

    def setUp(self):
        from ServerToolsCoreLib.base_widget import ServerToolWidgetBase, _Cohort, _Run
        import slicer

        self.panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        self.cohort = _Cohort(3)
        self.runs = [_Run(index, "c", {}, {}, "/out", None,
                          cohort=self.cohort, cohort_index=index)
                     for index in (1, 2, 3)]

        self.dialogs, self.status = [], []
        self.addCleanup(setattr, slicer.util, "infoDisplay", slicer.util.infoDisplay)
        self.addCleanup(setattr, slicer.util, "showStatusMessage", slicer.util.showStatusMessage)
        slicer.util.infoDisplay = lambda message, *a, **k: self.dialogs.append(message)
        slicer.util.showStatusMessage = lambda message, *a, **k: self.status.append(message)

    def _announce(self, run):
        self.panel._runInHand = run
        self.panel._countBatch(run)
        self.panel._announce("done")

    def test_only_the_last_batch_opens_a_dialog(self):
        for run in self.runs:
            self._announce(run)

        self.assertEqual(self.dialogs, ["done"])
        self.assertEqual(self.status, ["done", "done"])

    def test_a_failed_batch_still_counts_so_the_last_one_still_speaks(self):
        """`complete` asks whether anything is still coming, not whether
        everything worked."""
        self.panel._countBatch(self.runs[0])  # failed: counted, never announced
        self._announce(self.runs[1])
        self._announce(self.runs[2])

        self.assertEqual(self.dialogs, ["done"])

    def test_an_ordinary_run_always_opens_its_dialog(self):
        from ServerToolsCoreLib.base_widget import _Run

        self._announce(_Run(9, "one.nii.gz", {}, {}, "/out", None))

        self.assertEqual(self.dialogs, ["done"])


class CohortPanelTest(unittest.TestCase):
    """What a cohort in flight looks like.

    Five lines of the same sentence and five Cancel buttons read as five
    unrelated jobs someone started by accident. A cohort is ONE piece of work
    made of parts, and the panel has to say so -- in scans, which is the unit
    the work is actually in.
    """

    def setUp(self):
        import qt_stubs
        qt_stubs.install()
        from ServerToolsCoreLib.base_widget import ServerToolWidgetBase, _Cohort, _Run
        import qt

        self.qt = qt
        self.Run, self.Cohort = _Run, _Cohort
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel.TOOL_NAME = "ALI"
        panel._runs = []
        panel._cohortView = None
        panel._runControlsLayout = qt.QVBoxLayout()
        panel._runControlsWidget = None
        panel._progressBar = None
        panel._progressLabel = None
        panel.applyButton = qt.QPushButton("Apply")
        panel.cancelButton = qt.QPushButton("Cancel")
        self.phases = []
        panel._showPhase = self.phases.append
        self.panel = panel

    def _cohort(self, batches=5, scans_per_batch=4, started=1):
        """A cohort of `batches`, the first `started` of them running."""
        cohort = self.Cohort(batches, batches * scans_per_batch)
        for index in range(1, batches + 1):
            run = self.Run(index, f"cohort ({index}/{batches})", {}, {}, "/out", None,
                           cohort=cohort, cohort_index=index, scan_count=scans_per_batch)
            if index <= started:
                run.started_at = 0.0
            self.panel._runs.append(run)
        return cohort

    def _view(self):
        self.panel._syncRunControls()
        return self.panel._cohortView

    # --- no cancelling one batch of five -------------------------------

    def test_a_cohort_offers_no_per_batch_cancel_button(self):
        """Abandoning batch 3 of 5 leaves results covering an arbitrary part of
        the cohort. Nobody wants that outcome, so the panel does not offer it."""
        self._cohort()

        self._view()

        buttons = [w for w in self.panel._runControlsWidget.layout.widgets
                   if isinstance(w, self.qt.QPushButton)]
        self.assertEqual(buttons, [])

    def test_unrelated_runs_keep_their_own_cancel_buttons(self):
        """The regression that matters: a queue of independent runs is not a
        cohort, and cancelling the one that is stuck is the whole point there."""
        for number in (1, 2):
            self.panel._runs.append(
                self.Run(number, f"patient_{number}.nii.gz", {}, {}, "/out", None))

        self._view()

        buttons = [w for w in self.panel._runControlsWidget.layout.widgets
                   if isinstance(w, self.qt.QPushButton)]
        self.assertEqual(len(buttons), 2)

    def test_the_panel_s_cancel_speaks_for_the_cohort(self):
        self._cohort()
        self._view()

        self.assertIn("cohort", self.panel.cancelButton.text.lower())

    # --- the number the user asked for ---------------------------------

    def test_the_headline_counts_scans_and_not_batches(self):
        """"2 of 5" would be true and useless: nobody has five batches of work
        to do, they have twenty scans."""
        cohort = self._cohort(batches=5, scans_per_batch=4)
        cohort.scans_done = 8
        view = self._view()

        self.panel._renderProgress()

        self.assertIn("8", view.total.text)
        self.assertIn("20", view.total.text)
        self.assertIn("scans", view.total.text)

    def test_scans_lost_to_a_failed_batch_are_named_not_folded_in(self):
        """"16 of 20" with four lost in silence is the report this whole
        feature exists not to produce."""
        cohort = self._cohort()
        cohort.scans_done, cohort.scans_failed = 12, 4
        view = self._view()

        self.panel._renderProgress()

        self.assertIn("12", view.total.text)
        self.assertIn("4", view.total.text)
        self.assertIn("failed", view.total.text.lower())

    def test_nothing_failed_says_nothing_about_failures(self):
        cohort = self._cohort()
        cohort.scans_done = 8
        view = self._view()

        self.panel._renderProgress()

        self.assertNotIn("failed", view.total.text.lower())

    # --- the bars ------------------------------------------------------

    def test_the_cohort_bar_counts_the_batch_in_flight(self):
        """Otherwise it steps five times over an hour and looks frozen between
        them, which is the complaint progress reporting exists to answer."""
        cohort = self._cohort(batches=5, scans_per_batch=4)
        cohort.scans_done = 8            # two batches finished: 8/20 = 40%
        self.panel._runs[0].fraction = 0.5   # half of a third batch: +2 scans
        view = self._view()

        self.panel._renderProgress()

        self.assertEqual(view.bar.value, 50)

    def test_a_queued_batch_shows_no_bar_of_its_own(self):
        """Three empty bars are three things that look stuck."""
        self._cohort(batches=3, started=1)
        self.panel._runs[0].fraction = 0.4
        view = self._view()

        self.panel._renderProgress()

        self.assertTrue(view.rows[1][1].isVisible())
        self.assertFalse(view.rows[2][1].isVisible())
        self.assertFalse(view.rows[3][1].isVisible())

    def test_a_running_batch_with_nothing_to_report_shows_no_bar_either(self):
        """Most tools report no fraction at all. A bar inventing motion to look
        busy is worse than the elapsed time beside it."""
        self._cohort(batches=3, started=1)
        view = self._view()

        self.panel._renderProgress()

        self.assertFalse(view.rows[1][1].isVisible())

    # --- the lines -----------------------------------------------------

    def test_a_batch_is_numbered_by_its_place_in_the_cohort(self):
        """"Batch 2 of 5" is where the user is; "Run 7" is bookkeeping."""
        self._cohort(batches=5, started=1)
        view = self._view()

        self.panel._renderProgress()

        self.assertIn("2", view.rows[2][0].text)
        self.assertIn("5", view.rows[2][0].text)
        self.assertIn("queued", view.rows[2][0].text.lower())

    def test_the_box_says_it_all_so_the_label_above_stays_empty(self):
        """The same sentence twice reads as two different runs."""
        self._cohort()
        self._view()

        self.panel._renderProgress()

        self.assertEqual(self.phases[-1], "")

    # --- what is not a cohort ------------------------------------------

    def test_a_cohort_with_a_stranger_queued_behind_it_is_not_drawn_as_one(self):
        """It is a queue that happens to contain a cohort. Drawing it as one
        would put another run's progress inside the box and its scans outside
        the count."""
        self._cohort(batches=2)
        self.panel._runs.append(
            self.Run(99, "other.nii.gz", {}, {}, "/out", None))

        self.assertIsNone(self._view())

    # --- a cohort bigger than the box ----------------------------------

    def test_a_long_queue_lists_the_next_few_and_counts_the_rest(self):
        """A hundred scans is twenty-five batches. Listed in full they would be
        the tallest thing on the panel and would say nothing the headline
        count does not."""
        self._cohort(batches=12, started=1)
        view = self._view()

        self.panel._renderProgress()

        self.assertEqual(len(view.rows), 4)
        self.assertIn("8", view.remainder.text)
        self.assertIn("more", view.remainder.text.lower())

    def test_the_batches_listed_are_the_ones_in_flight_and_next(self):
        """`_runs` is in queue order and a finished run leaves it, so the fold
        always falls after what is happening, never before it."""
        self._cohort(batches=12, started=2)
        view = self._view()

        self.assertEqual(sorted(view.rows), [1, 2, 3, 4])

    def test_a_short_cohort_has_no_remainder_line_at_all(self):
        """Three batches, three lines, and nothing saying "+0 more"."""
        self._cohort(batches=3)
        view = self._view()

        self.panel._renderProgress()

        self.assertIsNone(view.remainder)

    def test_the_count_shrinks_as_batches_finish(self):
        """Rebuilt whenever the run set changes, so the fold moves with it."""
        self._cohort(batches=12, started=1)
        view = self._view()
        self.panel._renderProgress()
        self.assertIn("8", view.remainder.text)

        del self.panel._runs[:5]
        view = self._view()
        self.panel._renderProgress()

        self.assertIn("3", view.remainder.text)

    # --- it has to look like it belongs in Slicer ----------------------

    def test_the_box_is_the_same_outlined_surface_as_every_other(self):
        """It has been a hairline with no fill, then a fill with no hairline,
        and is now what the whole panel is: a surface with a line round it.
        The thing it must never be is a shape the panel uses nowhere else --
        this box appears mid-run, and a reader should not have to work out
        what kind of object just arrived."""
        self._cohort()
        view = self._view()

        self.assertIn(design.tokens()["SURFACE"], view.frame.styleSheet)
        self.assertIn("1px solid {}".format(design.tokens()["BORDER"]),
                      view.frame.styleSheet)
