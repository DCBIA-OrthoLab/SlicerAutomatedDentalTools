from dataclasses import dataclass, field, asdict
from typing import Union, List
import os
from itertools import chain

from ASO_IOS_utils.utils import JawFromFileName, StripJawFromFileName
from ADTLib.io.fs import search


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("ASO_IOS_datafile")


@dataclass(init=True)
class Upper:
    name1: str = field(repr=False, default="Upper")
    name2: str = field(repr=False, default="_U_")

    def __str__(self) -> str:
        return "Upper"

    def __eq__(self, __o: object) -> bool:
        out = False
        if isinstance(__o, Upper):
            out = True

        elif isinstance(__o, str):
            if __o == "Upper":
                out = True
        return out

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


@dataclass(init=True)
class Lower:
    name1: str = field(repr=False, default="Lower")
    name2: str = field(repr=False, default="_L_")

    def __str__(self) -> str:
        return "Lower"

    def __eq__(self, __o: object) -> bool:
        out = False
        if isinstance(__o, Lower):
            out = True

        elif isinstance(__o, str):
            if __o == "Lower":
                out = True
        return out

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


@dataclass(init=True, repr=True)
class Jaw:
    upper: Upper = field(init=False, repr=False, default_factory=Upper)
    lower: Lower = field(init=False, repr=False, default_factory=Lower)
    actual: Union[Upper, Lower]

    def __init__(self, actual) -> None:
        assert isinstance(actual, (Upper, Lower, str))
        self.upper = Upper()
        self.lower = Lower()
        if isinstance(actual, str):
            actual = Files.TypeOfJaw(actual)
        self.actual = actual

    def inv(self):
        out = self.upper
        if isinstance(self.actual, Upper):
            out = self.lower

        return str(out)

    def __str__(self) -> str:
        return str(self.actual)

    def __eq__(self, other):
        out = False
        if isinstance(other.actual, self.actual):
            out = True
        return out

    def __call__(self):

        return str(self.actual)


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Jaw_File:

    vtk: str
    jaw: Jaw
    name: str
    json: Union[str, None] = field(default=None)


@dataclass(init=True, repr=True, eq=True)
class Mouth_File:
    Upper: Union[Jaw_File, str]
    Lower: Union[Jaw_File, str]
    name: str


class Files:
    def __init__(self, folder: str) -> None:

        self.list_file: List[Union[Mouth_File, Jaw_File]] = []
        self.folder: str = folder
        self.extension = [".vtk", ".vtp", ".stl", ".off", ".obj"]

    def __name_file__(self, name_file: str):
        name_file = os.path.basename(name_file)
        name_file, _ = os.path.splitext(name_file)
        jaw = self.TypeOfJaw(name_file)
        name_file = self.__remove_jaw__(name_file, jaw)
        if "_out" in name_file:
            name_file = name_file.replace("_out", "").replace("Or", "")

        return jaw, name_file

    def __remove_jaw__(self, name_file: str, jaw: Union[Upper, Lower] = None):
        """The pairing key: the name with every jaw marker taken out.

        Both arches have to reduce to the same string, so the marker of either
        jaw is removed and not just the one this file carries. The recursive
        call this used to make threw its own result away, so a name holding the
        marker twice kept one of them.
        """
        return StripJawFromFileName(name_file)

    @staticmethod
    def TypeOfJaw(name_file: str):
        jaw = JawFromFileName(name_file)

        if jaw is None:
            raise ValueError(
                f"dont found the jaw's type to {name_file}. Name the arch in the "
                "file name, as U / L or Upper / Lower set off by an underscore "
                "(P1_T1_U.vtk, P1_T1_Lower_Seg.vtk)")
        return Upper() if jaw == "Upper" else Lower()

    def __len__(self):
        return len(self.list_file)

    def __iter__(self):
        self.iter = -1
        return self

    def __next__(self):
        self.iter += 1
        if self.iter >= len(self.list_file):
            raise StopIteration

        return asdict(self.list_file[self.iter])

    def search(self, path, *args):
        """The files under `path` by extension, directories excluded.

        The `files_only` filter is what set this variant apart from the
        fourteen others: see `ADTLib.io.fs`.
        """
        return search(path, *args, files_only=True)


class Files_vtk_link(Files):
    """
    From folder path,find lower upper jaw belong to the same patient

    So, list_file get Mouth file. In Mouth file there are upper lower jaw and name of the pattient

    Args:
        Files (_type_): _description_
    """

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.organise(folder)

    def organise(self, folder):
        list_vtk = list(
            chain.from_iterable(self.search(folder, self.extension).values())
        )

        # A mouth is one upper and one lower, and it is the jaw read from each
        # name that says which is which. Counting the files instead only checked
        # that a patient had two of them, then took the first as the upper: a
        # patient with two uppers and no lower (a leftover scan, the same arch
        # in two formats) was paired as a mouth whose lower arch is a maxilla,
        # and nothing said so.
        dic = {}
        for vtk in list_vtk:
            jaw, name = self.__name_file__(vtk)
            dic.setdefault(name, {"Upper": [], "Lower": []})[str(jaw)].append(vtk)

        def names(paths):
            return ", ".join(sorted(os.path.basename(p) for p in paths)) or "none"

        for name, by_jaw in dic.items():
            upper, lower = by_jaw["Upper"], by_jaw["Lower"]

            if len(upper) == 1 and len(lower) == 1:
                self.list_file.append(Mouth_File(upper[0], lower[0], name))
                continue

            # Say what is wrong with the patient rather than dropping it in
            # silence: the run that follows will simply not mention it.
            if not upper or not lower:
                missing = "upper" if not upper else "lower"
                logger.warning(
                    f"{name}: no {missing} arch, the pair is skipped. "
                    f"Upper: {names(upper)} | Lower: {names(lower)}")
            else:
                logger.warning(
                    f"{name}: {len(upper)} upper and {len(lower)} lower files, "
                    "cannot tell which two are the mouth, the pair is skipped. "
                    f"Upper: {names(upper)} | Lower: {names(lower)}")

        return self.list_file

class Files_vtk_json(Files):
    """
    From path folder, find landmark(json) and jaw(vtk) matche together.
    So, in list_files there are Jaw_file with landmark(json), jaw(vtk), lower/upper and name of patient
    There is only one landmark by Jaw_file
    Args:
        Files (_type_): _description_
    """

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.list_file = self.organise(folder)

    def organise(self, folder):
        list_file = []
        dic = self.search(folder, self.extension, ".json")

        list_json = dic[".json"]
        list_json.append("Upper_nioegfjhdfjkdffdhjmndfhnmdfhj")
        list_vtk = list(chain.from_iterable(map(dic.get, self.extension)))
        json_remove = None
        for vtk in list_vtk:
            vtk_jaw, vtk_name = self.__name_file__(vtk)
            for json in list_json:
                json_jaw, json_name = self.__name_file__(json)
                if vtk_name in json_name and vtk_jaw == json_jaw:
                    fil = Jaw_File(json=json, vtk=vtk, jaw=json_jaw, name=vtk_name)
                    list_file.append(fil)
                    json_remove = json
                    break

            if json_remove is not None:
                list_json.remove(json_remove)

            json_remove = None

        return list_file


class Files_vtk_json_link(Files_vtk_json):
    """
    From  folder, match files belong to the same patient: upper jaw(vtk) , upper landmark(json),  lower jaw(vtk) and lower landmark(json).
    So, in list_files there are Mouth_file with 2 Jaw_file and name of patient. Each Jaw_file contain landmark file, jaw file and upper or lower.
    Only one json file is taken by jaw


    Args:
        Files_vtk_json (_type_): _description_
    """

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.organise(folder)

    def organise(self, folder):
        list_file = super().organise(folder)
        list_upper = []
        list_lower = []
        for fil in list_file:
            if isinstance(fil.jaw, Upper):
                list_upper.append(fil)
            else:
                list_lower.append(fil)

        lower_remove = None
        for upper in list_upper:
            for lower in list_lower:
                if upper.name == lower.name:
                    fil = Mouth_File(upper, lower, upper.name)
                    self.list_file.append(fil)
                    lower_remove = lower
                    break
            if lower_remove is not None:
                list_lower.remove(lower_remove)

            lower_remove = None

        return self.list_file


class Files_vtk_json_semilink(Files):
    """
    From  folder, match files belong to the same patient: upper jaw(vtk) , upper landmark(json),  lower jaw(vtk) and lower landmark(json). the json files are not required to use this class unlike to Files_vtk_json_link
    So, in list_files there are Mouth_file with 2 Jaw_file and name of patient. Each Jaw_file contain landmark file, jaw file and upper or lower.
    Only one json file is taken by jaw


    Args:
        Files (_type_): _description_
    """

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.list_file = self.organise(folder)

    def organise(self, folder):
        list_file = []
        dic = self.search(folder, self.extension, ".json")
        list_vtk = list(chain.from_iterable(map(dic.get, self.extension)))
        list_json = dic[".json"]

        fil = {"Upper": [], "Lower": []}
        json_remove = None
        for vtk in list_vtk:
            vtk_jaw, vtk_name = self.__name_file__(vtk)

            for json in list_json:
                json_jaw, json_name = self.__name_file__(json)

                if vtk_name in json_name and vtk_jaw == json_jaw:
                    fil[str(vtk_jaw)].append(
                        Jaw_File(json=json, vtk=vtk, name=vtk_name, jaw=json_jaw)
                    )
                    json_remove = json

                    break

            if json_remove is not None:
                list_json.remove(json_remove)

            else:
                fil[str(vtk_jaw)].append(Jaw_File(vtk=vtk, name=vtk_name, jaw=vtk_jaw))

            json_remove = None

        for upper in fil["Upper"]:
            for lower in fil["Lower"]:
                if upper.name == lower.name:
                    list_file.append(
                        Mouth_File(Upper=upper, Lower=lower, name=upper.name)
                    )
                    break

        return list_file
