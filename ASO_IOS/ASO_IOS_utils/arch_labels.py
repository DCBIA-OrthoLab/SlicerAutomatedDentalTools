"""Repair of a scan the segmentation numbered in both jaws at once.

The crown segmentation names every point on its own, and nothing in a local
neighbourhood says which jaw the scan is: a maxilla and a mirrored mandible have
the same shape, only the palate tells them apart. On an arch it cannot place, it
splits each tooth between its own number and the same rank in the other arch.

That costs this module its orientation. PRE_ASO_IOS fits an arch on three or
four named teeth, and a tooth whose points all went to the other jaw's number is
simply gone: the ICP raises ToothNoExist and the whole arch is dropped, before
ALI_IOS -- which carries the same repair for the scans it is handed directly --
ever sees it.

Kept in step with ALI_IOS_utils.surface.UnifyArchLabels on purpose. The two CLIs
are separate Slicer modules with no shared package, so the logic is duplicated
rather than imported; change one and change the other.
"""
import logging

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)

# Universal numbering runs 1-16 over the upper arch from the patient's right,
# then 17-32 over the lower from the left. Tooth t and tooth t+16 hold the same
# rank in their own arch, which puts them on opposite sides of the mouth.
ARCH_OFFSET = 16
# Two labels whose points share a centre this closely are one tooth, not two.
# Measured on the scan this was written for, the two families sit 2.8 to 8.0 mm
# apart while every other pairing of the same labels starts at 18.9 mm, so the
# threshold sits in the middle of that gap. A scan genuinely holding both arches
# puts t and t+16 on opposite sides AND opposite jaws, far past this.
SAME_TOOTH_MM = 12.0
# One coincidence is a stray patch. A numbering that has split shows on the arch.
MIN_SPLIT_TEETH = 3
# Second motif, voisin mais distinct : l arcade entiere porte les numeros de
# l autre machoire. Rien n est coupe en deux, donc ArchLabelSplit ne trouve
# rien et le seuil ci-dessus n est jamais atteint -- pourtant chaque dent
# nommee est introuvable et l arcade est abandonnee. Mesure du 2026-09-24,
# patient pt_040 : l arcade BASSE porte 42 773 points en numeros hauts contre
# 241 en numeros bas, LL6 (19) n existe pas et son homologue haut (3) porte
# 7198 points.
#
# Le seuil est loin de tout ce qui a ete mesure : quatre arcades correctes
# -- la reference du corpus, la meme sur prod, et l arcade haute de ce meme
# patient -- portent 100,00 % de leur propre famille ; celle-ci en porte
# 0,56 %. Aucun scan connu ne se situe entre les deux.
MOSTLY_OTHER_ARCH = 0.05
# En dessous, il n y a pas assez de dents segmentees pour conclure quoi que
# ce soit : une arcade partielle n est pas une arcade mal numerotee.
MIN_TOOTH_POINTS = 2000


def ArchLabelSplit(labels, points, max_distance=SAME_TOOTH_MM):
    """Teeth carrying both an upper and a lower number, as [(upper, mm), ...]."""
    labels = np.asarray(labels).ravel()
    points = np.asarray(points)
    split = []
    for upper in range(1, ARCH_OFFSET + 1):
        lower = upper + ARCH_OFFSET
        here, there = labels == upper, labels == lower
        if not here.any() or not there.any():
            continue
        gap = float(np.linalg.norm(points[here].mean(axis=0) - points[there].mean(axis=0)))
        if gap <= max_distance:
            split.append((upper, gap))
    return split


def ArchNumberingFamilies(labels, jaw):
    """(points portant la numerotation de `jaw`, points portant l autre)."""
    labels = np.asarray(labels).ravel()
    dents = labels[(labels >= 1) & (labels <= 2 * ARCH_OFFSET)]
    bas = (dents > ARCH_OFFSET)
    propre = bas if jaw == "Lower" else ~bas
    return int(propre.sum()), int((~propre).sum())


def ArchIsOtherJawNumbering(labels, jaw):
    """L arcade est-elle numerotee, presque entierement, dans l autre famille ?"""
    propre, autre = ArchNumberingFamilies(labels, jaw)
    total = propre + autre
    if total < MIN_TOOTH_POINTS:
        return False
    return propre <= MOSTLY_OTHER_ARCH * total


def UnifyArchLabels(surf, jaw, required=(), property_name="Universal_ID"):
    """Renumber a doubly-numbered arch into `jaw`'s numbering, in place.

    Returns the number of points moved, 0 when there was nothing to repair.
    `jaw` is the arbiter and has to come from outside the geometry: an isolated
    arch does not determine its own left and right until the jaw is known.

    Every point of the losing family is moved, not only those on the teeth
    caught carrying both numbers. Where a tooth was named in the wrong
    numbering alone it is missing from the split entirely, and those are exactly
    the ones whose absence stops this module.
    """
    if jaw not in ("Upper", "Lower"):
        return 0
    array = surf.GetPointData().GetScalars(property_name)
    if array is None:
        array = surf.GetPointData().GetArray(property_name)
    if array is None:
        return 0

    labels = vtk_to_numpy(array)
    points = vtk_to_numpy(surf.GetPoints().GetData())
    split = ArchLabelSplit(labels, points)
    basculee = ArchIsOtherJawNumbering(labels, jaw)
    if len(split) < MIN_SPLIT_TEETH and not basculee:
        return 0

    if jaw == "Lower":
        wrong = (labels >= 1) & (labels <= ARCH_OFFSET)
        shift = ARCH_OFFSET
    else:
        wrong = (labels > ARCH_OFFSET) & (labels <= 2 * ARCH_OFFSET)
        shift = -ARCH_OFFSET

    moved = int(wrong.sum())
    if not moved:
        return 0

    # Ne pas deplacer un probleme faute de savoir le resoudre : on ne
    # renumerote que si le decalage fait APPARAITRE les dents que l appelant
    # reclame. Une reparation qui ne peut pas montrer qu elle repare ne
    # s applique pas -- l echec reste alors celui d avant, lisible.
    # `required` vide : l appelant n a pas de liste (ALI_IOS predit des
    # reperes, il n exige aucune dent nommee), la preuve se limite aux
    # etiquettes.
    if required:
        apres = np.where(wrong, labels + shift, labels)
        manquantes = [t for t in required if not (apres == t).any()]
        if manquantes:
            logger.warning(
                "Not renumbering this %s arch: even shifted, %s would still be "
                "missing, so its numbering is not simply the other jaw's."
                % (jaw.lower(), ", ".join(str(t) for t in manquantes)))
            return 0

    # Lus AVANT le decalage : apres, les deux familles ont fusionne et le
    # message annoncerait « 0 contre 43 014 », ce qui ne decrit plus rien.
    propre_avant, autre_avant = ArchNumberingFamilies(labels, jaw)

    labels[wrong] += shift
    array.Modified()
    # Dire QUEL motif a declenche la reparation. Les deux se soignent pareil
    # mais ne se diagnostiquent pas pareil, et annoncer « une dent numerotee
    # deux fois » sur une arcade entierement basculee envoie le lecteur
    # chercher un decoupage qui n existe pas.
    if len(split) >= MIN_SPLIT_TEETH:
        logger.warning(
            "%d teeth are numbered twice, once in each arch (%s), so the segmentation "
            "could not tell which jaw this scan is. Reading it as %s from its name and "
            "moving %d point(s) onto that numbering."
            % (len(split), ", ".join("%d/%d at %.1f mm" % (t, t + ARCH_OFFSET, d)
                                     for t, d in split[:4]), jaw.lower(), moved))
    else:
        logger.warning(
            "This arch carries almost only the other jaw's numbers (%d point(s) "
            "against %d), so the segmentation read it as the other arch entirely "
            "rather than splitting it. Reading it as %s from its name and moving "
            "%d point(s) onto that numbering."
            % (autre_avant, propre_avant, jaw.lower(), moved))
    return moved
