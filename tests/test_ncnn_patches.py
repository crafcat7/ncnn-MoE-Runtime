"""Regression tests for preparing patched ncnn sources without building ncnn."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "cmake" / "ncnn_moe_prepare_ncnn.cmake"
CMAKE = shutil.which("cmake")
GIT = shutil.which("git")


def cmake_argument(value):
    value = str(value).replace("\\", "/")
    delimiter = "="
    while f"]{delimiter}]" in value:
        delimiter += "="
    return f"[{delimiter}[{value}]{delimiter}]"


def patch_text(before="before", after="after"):
    return ("diff --git a/value.txt b/value.txt\n"
            "--- a/value.txt\n"
            "+++ b/value.txt\n"
            "@@ -1 +1 @@\n"
            f"-{before}\n"
            f"+{after}\n")


@unittest.skipUnless(CMAKE and GIT, "cmake and git are required")
class NcnnPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ncnn patch tests ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "upstream source"
        self.build = self.root / "build output"
        self.patches = self.root / "patch series"
        self.prepared = self.build / "_deps" / "ncnn-src"
        self.source.mkdir()
        self.patches.mkdir()
        self.write(self.source / "CMakeLists.txt",
                   "cmake_minimum_required(VERSION 3.21)\nproject(upstream_fixture NONE)\n")
        self.write(self.source / "value.txt", "before\n")
        self.patch = self.patches / "0001-change-value.patch"
        self.write(self.patch, patch_text())

    @staticmethod
    def write(path, contents):
        path.write_text(contents, encoding="utf-8", newline="\n")

    def run_cmake(self, *arguments, succeeds=True):
        result = subprocess.run(
            [CMAKE, *map(str, arguments)], capture_output=True,
            encoding="utf-8", errors="replace", timeout=60, check=False,
        )
        output = result.stdout + result.stderr
        if succeeds:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return result

    def prepare(self, succeeds=True):
        script = self.root / "prepare sources.cmake"
        self.write(script,
                   f"cmake_minimum_required(VERSION 3.21)\n"
                   f"include({cmake_argument(HELPER)})\n"
                   f"ncnn_moe_prepare_ncnn({cmake_argument(self.source)} "
                   f"{cmake_argument(self.build)} {cmake_argument(self.patches)} prepared)\n"
                   f"if(NOT prepared STREQUAL {cmake_argument(self.prepared)})\n"
                   f"  message(FATAL_ERROR \"Unexpected prepared source path: ${{prepared}}\")\n"
                   f"endif()\n")
        return self.run_cmake("-P", script, succeeds=succeeds)

    def test_first_apply_keeps_source_and_excludes_git_metadata(self):
        self.write(self.source / ".git", "gitdir: deliberately-unavailable\n")
        nested = self.source / "nested dependency"
        nested.mkdir()
        (nested / ".git").mkdir()
        self.write(nested / ".git" / "config", "not source\n")
        self.write(nested / "source.txt", "nested source\n")

        self.prepare()

        self.assertEqual((self.prepared / "value.txt").read_text(), "after\n")
        self.assertEqual((self.source / "value.txt").read_text(), "before\n")
        self.assertFalse((self.prepared / ".git").exists())
        self.assertFalse((self.prepared / "nested dependency" / ".git").exists())
        self.assertEqual((self.prepared / "nested dependency" / "source.txt").read_text(),
                         "nested source\n")
        self.assertEqual((nested / ".git" / "config").read_text(), "not source\n")

    def test_unchanged_inputs_leave_prepared_files_untouched(self):
        self.prepare()
        target = self.prepared / "value.txt"
        marker = self.prepared / "test marker.txt"
        self.write(marker, "preserve unchanged prepared tree\n")
        before = (target.stat().st_mtime_ns, marker.stat().st_mtime_ns)

        self.prepare()

        self.assertTrue(marker.exists(), "An unchanged prepare replaced the source tree")
        self.assertEqual((target.stat().st_mtime_ns, marker.stat().st_mtime_ns), before)
        self.assertEqual(target.read_text(), "after\n")

    def test_changed_patch_rebuilds_from_clean_source(self):
        self.prepare()
        self.write(self.patch, patch_text(after="updated patch"))

        self.prepare()

        self.assertEqual((self.prepared / "value.txt").read_text(), "updated patch\n")
        self.assertEqual((self.source / "value.txt").read_text(), "before\n")

    def test_source_updates_and_removals_refresh_output(self):
        self.write(self.source / "additional.txt", "old source\n")
        self.write(self.source / "removed.txt", "obsolete source\n")
        self.prepare()
        self.write(self.source / "additional.txt", "new source\n")
        (self.source / "removed.txt").unlink()
        self.write(self.source / "added.txt", "added source\n")

        self.prepare()

        self.assertEqual((self.prepared / "additional.txt").read_text(), "new source\n")
        self.assertEqual((self.prepared / "added.txt").read_text(), "added source\n")
        self.assertFalse((self.prepared / "removed.txt").exists())
        self.assertEqual((self.prepared / "value.txt").read_text(), "after\n")

    def test_patch_conflict_preserves_previous_tree_and_can_recover(self):
        self.prepare()
        self.write(self.patch, patch_text(before="upstream changed", after="cannot apply"))

        self.prepare(succeeds=False)

        self.assertEqual((self.prepared / "value.txt").read_text(), "after\n")
        self.assertEqual((self.source / "value.txt").read_text(), "before\n")
        self.write(self.patch, patch_text(after="recovered"))
        self.prepare()
        self.assertEqual((self.prepared / "value.txt").read_text(), "recovered\n")

    def test_missing_patches_preserve_previous_tree(self):
        self.prepare()
        self.patch.unlink()

        self.prepare(succeeds=False)

        self.assertEqual((self.prepared / "value.txt").read_text(), "after\n")
        self.assertEqual((self.source / "value.txt").read_text(), "before\n")

    def test_patch_series_uses_filename_order(self):
        self.write(self.patches / "0002-follow-up.patch", patch_text(before="after", after="final"))

        self.prepare()

        self.assertEqual((self.prepared / "value.txt").read_text(), "final\n")
        self.assertEqual((self.source / "value.txt").read_text(), "before\n")

    def test_patch_change_triggers_reconfigure_on_build(self):
        project = self.root / "configure project"
        project.mkdir()
        self.write(project / "CMakeLists.txt",
                   "cmake_minimum_required(VERSION 3.21)\n"
                   "project(patch_reconfigure NONE)\n"
                   f"include({cmake_argument(HELPER)})\n"
                   f"ncnn_moe_prepare_ncnn({cmake_argument(self.source)} "
                   f'"${{CMAKE_BINARY_DIR}}" {cmake_argument(self.patches)} prepared)\n'
                   'file(READ "${prepared}/value.txt" contents)\n'
                   'file(WRITE "${CMAKE_BINARY_DIR}/observed.txt" "${contents}")\n'
                   "add_custom_target(check_patched ALL\n"
                   '  COMMAND "${CMAKE_COMMAND}" -E compare_files\n'
                   '    "${prepared}/value.txt" "${CMAKE_BINARY_DIR}/observed.txt")\n')
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []
        self.run_cmake("-S", project, "-B", self.build, *generator)
        self.assertEqual((self.build / "observed.txt").read_text(), "after\n")
        self.write(self.patch, patch_text(after="rebuilt automatically"))

        self.run_cmake("--build", self.build, "--config", "Release")

        self.assertEqual((self.build / "observed.txt").read_text(), "rebuilt automatically\n")
        self.assertEqual((self.prepared / "value.txt").read_text(), "rebuilt automatically\n")
        self.assertEqual((self.source / "value.txt").read_text(), "before\n")


if __name__ == "__main__":
    unittest.main()
