# Which torch wheel channel a GPU gets, and which ones it must never get.
#
# Every pin in the extension used to name a channel outright -- cu118 in four
# modules, PyPI's default cu121 in AREG -- and none of them could be right for
# the whole fleet, because no published channel covers every card:
#
#     cu118 (2.2.0)           sm_37 50 60 70 75 80 86 90
#     cu121 (2.2.0)           sm_50 60 70 75 80 86 90
#     cu128 (2.7.1, 2.11.0)   sm_75 80 86 90 100 120 + compute_120
#
# cu128 is the only one with Blackwell, the only one carrying any PTX, and the
# only one without Maxwell, Pascal and Volta. The cases below pin both halves of that: a 5070 Ti has to
# move to cu128, and a GTX 1080 has to stay away from it.
import os
import sys
import unittest

_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from ADTLib.env.cuda import (  # noqa: E402
    CHANNELS, parse_architecture, parse_ptx, select_channel, split_arch_list)


class ParseArchitectureTest(unittest.TestCase):

    def test_the_last_digit_is_the_minor(self):
        self.assertEqual(parse_architecture("sm_86"), (8, 6))
        self.assertEqual(parse_architecture("sm_90"), (9, 0))

    def test_a_three_digit_architecture_is_not_a_minor_of_twenty(self):
        """sm_120 is Blackwell 12.0, not 1.20.

        Read the other way round it lands on capability (1, 20), which no card
        reports, and every Blackwell GPU is then excluded from the one channel
        that serves it.
        """
        self.assertEqual(parse_architecture("sm_100"), (10, 0))
        self.assertEqual(parse_architecture("sm_120"), (12, 0))

    def test_a_ptx_entry_is_not_a_cubin(self):
        """They are read apart: one is compiled code, the other is source."""
        self.assertIsNone(parse_architecture("compute_120"))
        self.assertEqual(parse_ptx("compute_120"), (12, 0))
        self.assertIsNone(parse_ptx("sm_120"))

    def test_anything_else_is_not_an_architecture(self):
        self.assertIsNone(parse_architecture(""))
        self.assertIsNone(parse_ptx(""))


class ArchListTest(unittest.TestCase):

    def test_the_real_answer_of_the_wheel_this_installs(self):
        """What torch 2.7.1+cu128 reports on the RTX 5070 Ti of the report."""
        cubins, ptx = split_arch_list(
            ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"])
        self.assertEqual(cubins[-1], (12, 0))
        self.assertEqual(ptx, (12, 0))

    def test_a_cubin_only_wheel_has_no_ptx(self):
        """cu121, which is why a card it does not list fails outright."""
        cubins, ptx = split_arch_list(
            ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])
        self.assertEqual(len(cubins), 7)
        self.assertIsNone(ptx)


class ChannelSelectionTest(unittest.TestCase):

    def channel_for(self, capability):
        chosen = select_channel(capability)
        return chosen.name if chosen else None

    def test_blackwell_gets_the_only_channel_that_carries_it(self):
        """The RTX 5070 Ti of the report: sm_120, on a cu121 torch."""
        self.assertEqual(self.channel_for((12, 0)), "cu128")
        self.assertEqual(self.channel_for((10, 0)), "cu128")

    def test_pascal_and_maxwell_are_kept_off_cu128(self):
        """cu128 dropped them. A blanket bump would have bricked these."""
        for capability in ((5, 0), (5, 2), (6, 0), (6, 1)):
            self.assertEqual(self.channel_for(capability), "cu121", capability)

    def test_volta_is_kept_off_cu128_too(self):
        """cu128 starts at sm_75, so a V100 is not covered by it either."""
        self.assertEqual(self.channel_for((7, 0)), "cu121")

    def test_kepler_falls_back_to_the_only_channel_left(self):
        self.assertEqual(self.channel_for((3, 7)), "cu118")

    def test_every_card_that_works_today_is_left_where_it_is(self):
        """Turing through Hopper: this change must move none of them.

        The channels are ordered most-conservative first for exactly this
        reason. Ordered newest-first instead, every one of these would be
        dragged onto cu128 and a four-minor-version torch upgrade, which is a
        fleet migration and not a bug fix.
        """
        for capability in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0)):
            self.assertEqual(self.channel_for(capability), "cu121", capability)

    def test_a_card_no_channel_covers_gets_none(self):
        """None means the CPU build, not a channel picked at random."""
        self.assertIsNone(self.channel_for((2, 0)))

    def test_ada_runs_on_kernels_built_for_ampere(self):
        """The compatibility rule this all rests on.

        CUDA guarantees binary compatibility within a major version, so the
        sm_86 cubin in cu121 runs on a capability 8.9 card -- which is why
        every RTX 6000 Ada in the lab works on a wheel whose list stops at
        sm_86, and why sm_90 does nothing at all for a 12.0 card.
        """
        cu121 = next(c for c in CHANNELS if c.name == "cu121")
        self.assertTrue(cu121.runs_on((8, 9)))    # sm_86 covers it
        self.assertTrue(cu121.runs_on((8, 0)))    # sm_80 exactly
        self.assertFalse(cu121.runs_on((12, 0)))  # another major entirely
        self.assertFalse(cu121.runs_on((3, 5)))   # no sm_3x in this channel

        # Within one major it only reaches upwards: cu128's lowest 7.x is
        # sm_75, which does nothing for a 7.0 card.
        cu128 = next(c for c in CHANNELS if c.name == "cu128")
        self.assertFalse(cu128.runs_on((7, 0)))
        self.assertTrue(cu128.runs_on((7, 5)))

    def test_ptx_serves_a_card_newer_than_anything_compiled(self):
        """cu128 carries compute_120, so the generation after Blackwell is
        JIT-compiled rather than dropped to the CPU. Slow first launch, but a
        run, and no new pin needed the day that card appears."""
        cu128 = next(c for c in CHANNELS if c.name == "cu128")
        self.assertTrue(cu128.runs_on((13, 0)))
        self.assertEqual(self.channel_for((13, 0)), "cu128")

    def test_ptx_does_not_reach_downwards(self):
        """compute_120 says nothing for a card below 12.0."""
        cu128 = next(c for c in CHANNELS if c.name == "cu128")
        self.assertFalse(cu128.runs_on((6, 1)))

    def test_nothing_is_selected_without_a_capability(self):
        self.assertFalse(any(channel.runs_on(None) for channel in CHANNELS))


class ChannelRequirementsTest(unittest.TestCase):

    def test_the_three_wheels_are_pinned_to_one_build(self):
        """Resolved separately, pip is free to satisfy one by replacing torch."""
        cu128 = next(c for c in CHANNELS if c.name == "cu128")
        self.assertEqual(cu128.requirements(), [
            "torch==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "torchaudio==2.7.1+cu128",
        ])

    def test_pypi_stays_reachable_but_not_for_torch(self):
        """--index-url, or pip answers torch==2.7.1 with the default variant."""
        arguments = next(c for c in CHANNELS if c.name == "cu128").pip_arguments()
        self.assertIn("--index-url https://download.pytorch.org/whl/cu128", arguments)
        self.assertIn("--extra-index-url https://pypi.org/simple", arguments)


if __name__ == "__main__":
    unittest.main()
