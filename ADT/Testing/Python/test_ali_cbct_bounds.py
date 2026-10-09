# Ou l'agent d'ALI CBCT a le droit de se poser.
#
# Une seule ligne decidait :
#
#     new_pos.all() > 0 and (new_pos < GetSize(scale)).all()
#
# et elle se trompait des deux cotes.
#
# En bas, `new_pos.all()` reduit le tableau a UN booleen avant la comparaison.
# `True > 0` vaut True, `False > 0` vaut False : la ligne demandait donc
# « aucune coordonnee n'est exactement nulle », jamais « toutes sont
# positives ». Deux fautes opposees d'un coup. Les coordonnees loin dans le
# negatif passaient -- et ce sont les cheres : SpatialCrop ramene a zero un
# debut de decoupe negatif, donc elles lisent toutes la MEME zone, le reseau
# repond toujours le meme deplacement, et la recherche epuise son budget.
# Pendant ce temps un pas sur une coordonnee exactement NULLE -- un vrai
# voxel, que GetZone lit tres bien -- etait refuse et coutait un des trois
# essais. Mesure sur MG_test_scan : la recherche de `Me` s'est vu refuser
# trois fois le pas du voxel 1 vers le voxel 0, puis a rendu -1.
#
# En haut, `GetSize` est la taille AVANT rembourrage, alors que GetZone
# decoupe le tenseur rembourre : l'agent s'interdisait une zone qu'il sait
# lire, et chaque pas dedans lui coutait un essai aussi.
#
# Les deux moities vont ensemble : reparer la borne basse sans elargir la
# haute ferait abandonner l'agent PLUS TOT pres d'un bord.
import os
import sys
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ALI_CBCT")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# `ALI_CBCT_utils.preprocess` importe dicom2nifti, que le Python de Slicer ne
# sait pas charger hors de l'application (son pydicom tire gdcm, qui casse).
# Il ne sert qu'a convertir du DICOM et rien ici n'en convertit.
sys.modules.setdefault("dicom2nifti", types.ModuleType("dicom2nifti"))

import numpy as np  # noqa: E402

from ALI_CBCT_utils.agent import InsideSamplableZone  # noqa: E402
from ALI_CBCT_utils.environment import Environment  # noqa: E402

# Ce qu'ALI passe vraiment : agent_FOV [64,64,64], donc padding FOV/2 + 1.
FOV = np.array([64, 64, 64])
PADDING = (FOV / 2 + 1).astype(np.int16)
SIZE = np.array([120, 130, 140])


def old_predicate(position, size):
    """L'expression d'avant, recopiee telle quelle."""
    return bool(position.all() > 0 and (position < size).all())


def samplable_bounds():
    """Les deux bornes, calculees par le vrai code de l'Environment.

    La methode ne lit que `self.padding` et `self.GetSize`, donc un objet nu
    qui porte ces deux-la suffit : pas d'image, pas de transformation monai,
    pas de scan a charger.
    """
    stub = types.SimpleNamespace(padding=PADDING, GetSize=lambda scale: SIZE)
    return Environment.GetSamplableBounds(stub, "1", FOV)


# (position, accepte par l'ancienne, accepte par la nouvelle, pourquoi)
CASES = [
    ([60, 60, 60], True, True, "au centre : les deux acceptent"),
    ([1, 60, 60], True, True, "premier voxel strictement positif"),
    ([119, 60, 60], True, True, "dernier voxel du scan"),

    ([0, 60, 60], False, True, "voxel 0, lisible, et l'ancienne le refusait"),
    ([0, 0, 0], False, True, "le coin du scan, refuse lui aussi"),
    # -1 : la decoupe y est encore juste, les deux l'acceptent -- mais
    # l'ancienne pour la mauvaise raison, « aucune coordonnee nulle ».
    ([-1, 60, 60], True, True, "derniere position basse a zone entiere"),

    ([-2, 60, 60], True, False, "dehors, et l'ancienne l'acceptait"),
    ([-5, 60, 60], True, False, "plus loin dehors, meme zone lue qu'a -1"),
    ([-50, 60, 60], True, False, "tres loin dehors, l'ancienne dit oui"),
    ([60, -3, 70], True, False, "negative sur le deuxieme axe"),
    ([-2, -2, -2], True, False, "les trois negatives, l'ancienne dit oui"),

    ([120, 60, 60], False, True, "dans la marge rembourree, lisible"),
    ([121, 60, 60], False, True, "derniere position a zone entiere"),
    ([60, 131, 60], False, True, "meme marge sur un axe plus long"),

    ([122, 60, 60], False, False, "au-dela : la zone serait tronquee"),
    ([60, 60, 300], False, False, "tres au-dela"),
]


class SamplableZoneTest(unittest.TestCase):

    def test_the_bounds_are_the_padded_extent_not_GetSize(self):
        """Un voxel de part et d'autre des faces, pas 0 .. size-1."""
        low, high = samplable_bounds()
        np.testing.assert_array_equal(low, np.array([-1, -1, -1]))
        np.testing.assert_array_equal(high, SIZE + 1)

    def test_the_case_table(self):
        low, high = samplable_bounds()
        for coords, old_ok, new_ok, why in CASES:
            position = np.array(coords, dtype=np.int32)
            with self.subTest(position=coords, why=why):
                self.assertEqual(old_predicate(position, SIZE), old_ok,
                                 f"ancienne expression, {why}")
                self.assertEqual(InsideSamplableZone(position, low, high),
                                 new_ok, f"nouvelle expression, {why}")

    def test_the_old_expression_let_runaway_negatives_through(self):
        """Le defaut cher : aucune coordonnee nulle suffisait a passer."""
        low, high = samplable_bounds()
        for coords in ([-2, 60, 60], [-5, 60, 60], [-50, 60, 60],
                       [-2, -2, -2], [60, -3, 70]):
            position = np.array(coords)
            self.assertTrue(old_predicate(position, SIZE), coords)
            self.assertFalse(InsideSamplableZone(position, low, high), coords)

    def test_the_voxels_the_old_expression_refused_are_now_reachable(self):
        """Le voxel 0 en bas -- celui qui tenait `Me` hors d'atteinte -- et
        la marge rembourree en haut, que la seule reparation du bas aurait
        laissee fermee."""
        low, high = samplable_bounds()
        for reachable in (np.array([0, 60, 60]), np.zeros(3, dtype=int),
                          SIZE, SIZE + 1):
            self.assertFalse(old_predicate(reachable, SIZE), reachable)
            self.assertTrue(InsideSamplableZone(reachable, low, high),
                            reachable)

    def test_a_float_position_is_handled(self):
        """SetPosAtCenter pose des flottants : le predicat doit les prendre."""
        low, high = samplable_bounds()
        self.assertTrue(InsideSamplableZone(SIZE / 2, low, high))
        self.assertFalse(
            InsideSamplableZone(np.array([-1.5, 60.0, 60.0]), low, high))


if __name__ == "__main__":
    unittest.main()
