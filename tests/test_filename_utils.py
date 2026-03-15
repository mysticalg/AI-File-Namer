import tempfile
import unittest
from pathlib import Path

from src.ai_file_namer import (
    collect_media_files,
    collect_subfolders,
    compute_target_stem_length,
    format_date,
    sanitize_filename_stem,
)


class FilenameUtilsTests(unittest.TestCase):
    def test_sanitize_filename_stem_defaults_to_underscores(self):
        raw = "A Cute Cat!!!\n(indoors)"
        self.assertEqual(sanitize_filename_stem(raw), "a_cute_cat_indoors")

    def test_sanitize_filename_stem_supports_spaces_title_case_and_length(self):
        raw = "A very descriptive filename for a coastal sunset with waves and orange sky"
        value = sanitize_filename_stem(raw, separator=" ", capitalization="title", max_length=40)
        self.assertEqual(value, "A Very Descriptive Filename For A Coasta")
        self.assertLessEqual(len(value), 40)

    def test_format_date(self):
        result = format_date("%Y-%m-%d")
        self.assertEqual(len(result), 10)

    def test_collect_media_files_recursive_option(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "photo.jpg").write_bytes(b"x")
            (root / "notes.txt").write_text("ignore")
            sub = root / "sub"
            sub.mkdir()
            (sub / "clip.mp4").write_bytes(b"y")

            non_recursive = collect_media_files(root, recursive=False)
            recursive = collect_media_files(root, recursive=True)

            self.assertEqual([path.name for path in non_recursive], ["photo.jpg"])
            self.assertEqual({path.name for path in recursive}, {"photo.jpg", "clip.mp4"})

    def test_collect_subfolders_recursive_depth_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            level1 = root / "music"
            level2 = level1 / "midi"
            level3 = level2 / "classical"
            level3.mkdir(parents=True)

            folders = collect_subfolders(root, recursive=True)
            self.assertGreater(len(folders), 2)
            self.assertEqual(folders[0].name, "classical")

    def test_compute_target_stem_length_includes_date_and_extension(self):
        path = Path("example.jpeg")
        stem_length = compute_target_stem_length(
            path=path,
            include_date=True,
            date_text="2026-03-15",
            max_filename_length=40,
            date_separator=" ",
        )
        # 40 total - len("2026-03-15 ") 11 - len(".jpeg") 5 = 24
        self.assertEqual(stem_length, 24)


if __name__ == "__main__":
    unittest.main()
