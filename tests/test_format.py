"""Regression tests for the source-format check; no runtime build is needed."""

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "cmake" / "check_format.py"
SPEC = importlib.util.spec_from_file_location("check_format", MODULE_PATH)
check_format = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_format)


class FirstArgumentTests(unittest.TestCase):
    def test_aligned_calls_and_declarations(self):
        self.assertEqual(check_format.first_argument_breaks(
            "void f(int a,\n       int b);\n"
            "promise->set_value(run(model,\n                      input));\n"), [])

    def test_detached_first_argument(self):
        for text, position in [
            ("void f(\n    int a,\n    int b);", (1, 7)),
            ("promise->set_value(run(\n    model,\n    input));", (1, 23)),
            ('log(\n    "message");', (1, 4)),
            ("f(\r\n    a);", (1, 2)),
            ("f(\n    /* explanation */ a,\n    g(\n        b));", (3, 6)),
        ]:
            with self.subTest(text=text):
                self.assertEqual(check_format.first_argument_breaks(text), [position])

    def test_literals_and_comments_are_not_code(self):
        text = ('const char* a = R"tag(run(\n    "quoted"))tag";\n'
                'const char* b = u8R"(call(\n    value))";\n'
                'const char* c = "run(\\n value)";\n'
                "const char d = '(';\n"
                "const auto count = 1'024;\n"
                "// run(\n// value)\n"
                "/* run(\n    value) */\n")
        self.assertEqual(check_format.first_argument_breaks(text), [])

    def test_leading_comment_or_directive(self):
        self.assertEqual(check_format.first_argument_breaks(
            "f(\n    // why this argument\n    value);\n"
            "f(\n#if FEATURE\n    a,\n#else\n    b,\n#endif\n    c);\n"), [])

    def test_empty_parentheses(self):
        self.assertEqual(check_format.first_argument_breaks("f(\n);"), [])

    def test_macro_body(self):
        self.assertEqual(check_format.first_argument_breaks(
            "#define F(x) f(\\\n    #x)\n"), [(1, 15)])
        self.assertEqual(check_format.first_argument_breaks(
            "#define F(x) f(\\\n    x)\n"), [(1, 15)])
        self.assertEqual(check_format.first_argument_breaks(
            "#define F(x) f(x, \\\n                  2)\n"), [])

    @unittest.skipUnless(shutil.which("clang-format"), "clang-format is required")
    def test_cli_rejects_detached_or_misaligned_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(MODULE_PATH.parents[1] / ".clang-format", root / ".clang-format")
            source = root / "src" / "sample.cpp"
            source.parent.mkdir()
            for call, expected in [
                ("    run(a,\n        b);", 0),
                ("    run(\n        a,\n        b);", 1),
                ("    run(a,\n       b);", 1),
            ]:
                with self.subTest(call=call):
                    contents = "void f()\n{\n" + call + "\n}\n"
                    source.write_text(contents, encoding="utf-8", newline="\n")
                    result = subprocess.run(
                        [sys.executable, str(MODULE_PATH), "--source-dir", str(root)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertEqual(source.read_bytes(), contents.encode("utf-8"))
                    if "run(\n" in call:
                        self.assertIn("sample.cpp:3:8: keep the first argument", result.stderr)

    def test_cli_does_not_silently_skip_a_missing_formatter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "sample.cpp"
            source.parent.mkdir()
            source.write_text("void f();\n", encoding="utf-8", newline="\n")
            result = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--source-dir", str(root),
                 "--clang-format", str(root / "missing-clang-format")],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("was not found", result.stderr)

    def test_all_project_layers_and_shaders_are_scanned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = ["include/api.h", "src/engine/a.cpp", "src/kernels/vulkan/a.comp",
                        "examples/a.cc", "tests/a.cxx"]
            for name in expected + ["third_party/ncnn/a.cpp", "build/a.cpp", "src/README.md"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(check_format.source_files(root), sorted(root / name for name in expected))


if __name__ == "__main__":
    unittest.main()
