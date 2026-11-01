"""Regression tests for preparing patched ncnn sources without building ncnn."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "cmake" / "ncnn_moe_prepare_ncnn.cmake"
CMAKE = shutil.which("cmake")
GIT = shutil.which("git")
CXX = next((shutil.which(name) for name in ("c++", "g++", "clang++", "cl")
            if shutil.which(name)), None)
REPOSITORY = HELPER.parents[1]


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


@unittest.skipUnless(GIT and CXX, "git and a C++ compiler are required")
class NcnnCooperativeLocalSizeTests(unittest.TestCase):
    def test_rewriter_defines_local_size_before_workgroup_matrix_types(self):
        """Run the patched production rewriter; no Vulkan device or SDK is needed."""
        with tempfile.TemporaryDirectory(prefix="ncnn local size tests ") as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            upstream = REPOSITORY / "third_party" / "ncnn" / "src" / "gpu.cpp"
            original = upstream.read_bytes()
            source = root / "src" / "gpu.cpp"
            source.write_bytes(original)
            patch = REPOSITORY / "third_party" / "patches" / "ncnn" / "0003-vulkan-workgroup-cooperative-local-size.patch"
            environment = os.environ.copy()
            for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
                environment.pop(name, None)
            environment["GIT_CEILING_DIRECTORIES"] = str(root.parent)
            applied = subprocess.run([GIT, "apply", "--", str(patch)], cwd=root, env=environment,
                                     capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
            self.assertEqual(upstream.read_bytes(), original, "Applying the regression patch changed upstream")
            patched = source.read_text(encoding="utf-8")
            begin = patched.index("static void inject_local_size_xyz(")
            end = patched.index("\nVkShaderModule VulkanDevice::compile_shader_module(", begin)
            implementation = patched[begin:end]
            unpatched = original.decode("utf-8")
            original_begin = unpatched.index("static void inject_local_size_xyz(")
            original_end = unpatched.index("\nVkShaderModule VulkanDevice::compile_shader_module(", original_begin)
            original_implementation = unpatched[original_begin:original_end].replace(
                "static void inject_local_size_xyz(", "static void inject_original_local_size_xyz(", 1)
            harness = root / "local_size_test.cpp"
            harness.write_text(
                "#include <cstdint>\n#include <cstring>\n#include <vector>\n"
                "#include <utility>\n#include <iostream>\n#include <initializer_list>\n"
                + implementation + original_implementation + r'''

static void emit(std::vector<uint32_t>& code, uint32_t opcode,
                 std::initializer_list<uint32_t> operands)
{
    code.push_back((uint32_t(operands.size() + 1) << 16) | opcode);
    code.insert(code.end(), operands.begin(), operands.end());
}

static bool exercise(bool matrix, bool unsigned_type, bool legacy, bool late_unsigned_type,
                     uint32_t scope, bool specialization_scope)
{
    // An instruction fixture for the local-size transformation. Matrix scope
    // may use a signed integer type, so the missing-uint case is intentional.
    std::vector<uint32_t> source = {0x07230203, legacy ? 0x00010500u : 0x00010600u, 0, 40, 0};
    emit(source, 16, {1, 17, 8, 8, 1}); // OpExecutionMode LocalSize
    for (uint32_t i = 0; i < 3; ++i)
        emit(source, 71, {11 + i, 1, 233 + i}); // reserved SpecIds
    emit(source, 21, {2, 32, unsigned_type ? 0u : 1u});
    for (uint32_t i = 0; i < 3; ++i)
        emit(source, 50, {2, 11 + i, 8});
    if (matrix)
    {
        emit(source, specialization_scope ? 50 : 43, {2, 4, scope});
        emit(source, 4456, {20, 3, 4, 5, 5, 6});
        emit(source, 4456, {21, 3, 4, 5, 5, 6});
    }
    if (late_unsigned_type)
        emit(source, 21, {7, 32, 0});
    emit(source, 54, {8, 1, 0, 9});
    const auto before = source;
    std::vector<uint32_t> destination(source.size() + 28);
    size_t bytes = 0;
    inject_local_size_xyz(source.data(), source.size() * sizeof(uint32_t), 256, 1, 1,
                          destination.data(), &bytes);
    destination.resize(bytes / sizeof(uint32_t));
    if (source != before || bytes > (source.size() + 28) * sizeof(uint32_t)) return false;
    if (!matrix || scope != 2 || legacy)
    {
        std::vector<uint32_t> original_destination(source.size() + 28);
        size_t original_bytes = 0;
        inject_original_local_size_xyz(source.data(), source.size() * sizeof(uint32_t), 256, 1, 1,
                                       original_destination.data(), &original_bytes);
        original_destination.resize(original_bytes / sizeof(uint32_t));
        if (destination != original_destination) return false;
    }
    const uint32_t expected_bound = legacy ? 40 : (unsigned_type || late_unsigned_type) ? 43 : 44;
    if (destination[3] != expected_bound) return false;
    uint32_t constants = 0, unsigned_types = 0, reserved_decorations = 0, replaced = 0;
    bool encountered_type_or_function = false, execution_mode = false;
    for (size_t offset = 5; offset < destination.size();)
    {
        const uint32_t* p = destination.data() + offset;
        const uint32_t count = p[0] >> 16, op = p[0] & 65535;
        if (count == 0 || offset + count > destination.size()) return false;
        if (op == 21 && p[2] == 32 && p[3] == 0) ++unsigned_types;
        if ((op == 4456 && scope == 2) || op == 54)
        {
            if (!legacy && constants != 3) return false;
            encountered_type_or_function = true;
        }
        if (op == 50 && p[2] >= 40 && p[2] <= 42)
        {
            if (legacy || encountered_type_or_function || unsigned_types != 1
                || p[1] != (unsigned_type ? 2u : late_unsigned_type ? 7u : 43u)
                || p[3] != (p[2] == 40 ? 256u : 1u)) return false;
            ++constants;
        }
        if (op == 71 && p[2] == 1 && p[3] >= 233 && p[3] <= 235)
        {
            if (p[1] != 40 + p[3] - 233) return false;
            ++reserved_decorations;
        }
        if ((op == 50 || op == 43) && p[2] >= 11 && p[2] <= 13)
        {
            if (legacy || op != 43 || p[3] != (p[2] == 11 ? 256u : 1u)) return false;
            ++replaced;
        }
        if ((op == 16 && p[2] == 17) || (op == 331 && p[2] == 38))
        {
            if (legacy ? (op != 16 || p[3] != 256 || p[4] != 1 || p[5] != 1)
                       : (op != 331 || p[3] != 40 || p[4] != 41 || p[5] != 42)) return false;
            execution_mode = true;
        }
        offset += count;
    }
    return execution_mode && encountered_type_or_function
           && constants == (legacy ? 0u : 3u)
           && reserved_decorations == constants && replaced == constants
           && unsigned_types == (unsigned_type || late_unsigned_type || !legacy ? 1u : 0u);
}

int main()
{
    struct Case
    {
        bool matrix, unsigned_type, legacy, late_unsigned_type;
        uint32_t scope;
        bool specialization_scope;
    };
    // Workgroup-matrix regressions plus unchanged scalar, subgroup, device
    // and legacy paths; unrelated dimensions need not form a Cartesian product.
    const Case cases[] = {
        {true, true, false, false, 2, false},
        {true, true, false, false, 2, true},
        {true, false, false, false, 2, false},
        {true, false, false, true, 2, true},
        {false, true, false, false, 2, false},
        {false, false, false, true, 2, false},
        {true, true, false, false, 3, true},
        {true, true, false, false, 1, false},
        {true, true, true, false, 2, false},
        {false, false, true, false, 2, false},
    };
    for (const auto& test : cases)
        if (!exercise(test.matrix, test.unsigned_type, test.legacy, test.late_unsigned_type,
                       test.scope, test.specialization_scope))
        {
            std::cerr << "LocalSizeId ordering/rewrite failed: matrix=" << test.matrix
                      << " uint=" << test.unsigned_type << " legacy=" << test.legacy
                      << " late_uint=" << test.late_unsigned_type << " scope=" << test.scope
                      << " spec=" << test.specialization_scope << '\n';
            return 1;
        }
    return 0;
}
''', encoding="utf-8", newline="\n")
            executable = root / ("local_size_test.exe" if os.name == "nt" else "local_size_test")
            if Path(CXX).name.lower() in ("cl", "cl.exe", "clang-cl", "clang-cl.exe"):
                command = [CXX, "/nologo", "/EHsc", "/std:c++14", str(harness),
                           f"/Fe:{executable}", f"/Fo:{root / 'local_size_test.obj'}"]
            else:
                command = [CXX, "-std=c++11", str(harness), "-o", str(executable)]
            compiled = subprocess.run(command, cwd=root, capture_output=True,
                                      encoding="utf-8", errors="replace", timeout=60, check=False)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(executable)], cwd=root, capture_output=True,
                                      encoding="utf-8", errors="replace", timeout=30, check=False)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
