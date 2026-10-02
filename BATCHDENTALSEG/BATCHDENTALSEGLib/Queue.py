"""
Persistent processing queue for BatchDentalSegmentator.

The queue is flat at the *scan* level (not the folder level): every entry carries
its own model / device / output folder, so several folders with different models
can be stacked in a single session. The state is written to disk after every
scan, which makes an interrupted run resumable.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

logger = logging.getLogger("BatchDentalSeg_Queue")

VOLUME_PATTERNS = ("*.nii", "*.nii.gz", "*.gipl", "*.gipl.gz")
VOLUME_SUFFIXES = (".nii.gz", ".gipl.gz", ".nii", ".gipl")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

STATE_FILE_NAME = ".batchdentalseg_queue.json"


def listVolumes(folder):
    """Sorted, de-duplicated list of the volume files contained in a folder."""
    folder = Path(folder)
    files = set()
    for pattern in VOLUME_PATTERNS:
        files.update(folder.glob(pattern))
    return sorted(files)


def volumeStem(path):
    """``case01.nii.gz`` -> ``case01`` (``Path.stem`` would leave ``case01.nii``)."""
    name = Path(path).name
    for suffix in VOLUME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def expectedOutputPath(input_path, output_dir):
    """Path of the NIfTI written by the widget for a given input scan."""
    return Path(output_dir).joinpath(f"{volumeStem(input_path)}_Segmentation.nii.gz")


@dataclass
class QueueItem:
    inputPath: str
    outputDir: str
    model: str
    device: str
    status: str = STATUS_PENDING
    error: str = ""
    durationSec: float = 0.0
    autoCrop: bool = False      # crop to the patient before inference (RAM retry)

    @property
    def name(self):
        return Path(self.inputPath).name


class SegmentationQueue:
    """Ordered list of scans to segment, with on-disk state and chunk boundaries."""

    def __init__(self, statePath=None, chunkSize=5):
        self.items = []
        self.index = 0
        self.chunkSize = chunkSize
        self.statePath = Path(statePath) if statePath else None

    # ─── Building ──────────────────────────────────────────────────────────────

    def addFolder(self, input_folder, output_dir, model, device, skipExisting=True):
        """Append every volume of a folder. Returns (added, skipped)."""
        added = skipped = 0
        for file_path in listVolumes(input_folder):
            if skipExisting and expectedOutputPath(file_path, output_dir).exists():
                skipped += 1
                continue
            if any(item.inputPath == str(file_path) for item in self.items):
                skipped += 1
                continue
            self.items.append(QueueItem(str(file_path), str(output_dir), model, device))
            added += 1
        self.save()
        return added, skipped

    def clear(self):
        self.items = []
        self.index = 0
        self.save()

    def removeAt(self, indices):
        """Remove pending entries by row index; entries already consumed are kept."""
        removable = sorted((i for i in indices if i >= self.index), reverse=True)
        for i in removable:
            del self.items[i]
        self.save()
        return len(removable)

    def retryFailed(self, shouldAutoCrop=None):
        """
        Re-queue every failed entry at the end of the list.

        ``shouldAutoCrop`` is an optional predicate taking the failed entry and
        returning True when its retry must run on a cropped scan.
        """
        failed = [item for item in self.items if item.status == STATUS_FAILED]
        for item in failed:
            self.items.append(QueueItem(
                item.inputPath, item.outputDir, item.model, item.device,
                autoCrop=bool(shouldAutoCrop(item)) if shouldAutoCrop else False))
        self.save()
        return len(failed)

    # ─── Consuming ─────────────────────────────────────────────────────────────

    def current(self):
        return self.items[self.index] if self.index < len(self.items) else None

    def advance(self, status, error="", duration_sec=0.0):
        item = self.current()
        if item is not None:
            item.status = status
            item.error = error
            item.durationSec = duration_sec
        self.index += 1
        self.save()
        return item

    def isChunkBoundary(self):
        """True when the next scan starts a new chunk (deep cleanup point)."""
        return self.index > 0 and self.chunkSize > 0 and self.index % self.chunkSize == 0

    # ─── Reporting ─────────────────────────────────────────────────────────────

    def counts(self):
        result = {STATUS_PENDING: 0, STATUS_RUNNING: 0, STATUS_DONE: 0, STATUS_FAILED: 0}
        for item in self.items:
            result[item.status] = result.get(item.status, 0) + 1
        return result

    def summary(self):
        counts = self.counts()
        return (
            f"{counts[STATUS_DONE]} done, {counts[STATUS_FAILED]} failed, "
            f"{counts[STATUS_PENDING]} pending (total {len(self.items)})"
        )

    def isEmpty(self):
        return not self.items

    def isFinished(self):
        return self.index >= len(self.items)

    # ─── Persistence ───────────────────────────────────────────────────────────

    def setStatePath(self, folder):
        self.statePath = Path(folder).joinpath(STATE_FILE_NAME) if folder else None

    def save(self):
        if not self.statePath:
            return
        try:
            self.statePath.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "index": self.index,
                "chunkSize": self.chunkSize,
                "items": [asdict(item) for item in self.items],
            }
            self.statePath.write_text(json.dumps(payload, indent=2))
        except Exception as e:  # persistence must never break a running batch
            logger.warning(f"Could not save queue state to {self.statePath}: {e}")

    def load(self):
        """Restore a previous run. Returns True when a state file was read."""
        if not self.statePath or not self.statePath.exists():
            return False
        try:
            payload = json.loads(self.statePath.read_text())
        except Exception as e:
            logger.warning(f"Could not read queue state from {self.statePath}: {e}")
            return False

        known = set(QueueItem.__dataclass_fields__.keys())
        self.items = [
            QueueItem(**{k: v for k, v in raw.items() if k in known})
            for raw in payload.get("items", [])
        ]
        self.index = min(int(payload.get("index", 0)), len(self.items))
        self.chunkSize = int(payload.get("chunkSize", self.chunkSize))

        # A scan interrupted mid-inference is left as "running": rewind to it.
        for i, item in enumerate(self.items):
            if item.status == STATUS_RUNNING:
                item.status = STATUS_PENDING
                self.index = min(self.index, i)
        return True
