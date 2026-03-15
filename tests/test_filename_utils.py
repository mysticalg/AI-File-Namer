import tempfile
import unittest
from pathlib import Path

from src.ai_file_namer import (
    collect_media_files,
    collect_subfolders,
    format_date,
    sanitize_filename_stem,
)


class FilenameUtilsTests(unittest.TestCase):
    def test_sanitize_filename_stem(self):
        raw = "A Cute Cat!!!\n(indoors)"
        self.assertEqual(sanitize_filename_stem(raw), "a_cute_cat_indoors")

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


if __name__ == "__main__":
    unittest.main()
