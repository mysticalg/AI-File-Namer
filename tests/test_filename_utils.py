import tempfile
import unittest
from pathlib import Path

from src.ai_file_namer import (
    append_hashtags,
    collect_media_files,
    collect_subfolders,
    compute_target_stem_length,
    format_date,
    group_duplicate_files,
    group_duplicate_folders,
    remove_empty_folders,
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

    def test_append_hashtags_respects_max_length(self):
        result = append_hashtags("My Song Name", separator=" ", hashtag_count=3, max_length=22)
        self.assertLessEqual(len(result), 22)
        self.assertIn("#", result)

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
        self.assertEqual(stem_length, 24)

    def test_group_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            a = root / "a.jpg"
            b = root / "b.jpg"
            c = root / "c.jpg"
            a.write_bytes(b"same")
            b.write_bytes(b"same")
            c.write_bytes(b"different")

            groups = group_duplicate_files([a, b, c])
            self.assertEqual(len(groups), 1)
            self.assertEqual({item.name for item in groups[0]}, {"a.jpg", "b.jpg"})


    def test_remove_empty_folders_recursive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            empty_parent = root / "empty_parent"
            empty_child = empty_parent / "empty_child"
            non_empty = root / "non_empty"
            empty_child.mkdir(parents=True)
            non_empty.mkdir()
            (non_empty / "keep.txt").write_text("x")

            removed = remove_empty_folders(root, recursive=True)

            self.assertEqual(removed, 2)
            self.assertFalse(empty_parent.exists())
            self.assertTrue(non_empty.exists())

    def test_remove_empty_folders_non_recursive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            top_empty = root / "top_empty"
            nested = root / "outer" / "inner"
            top_empty.mkdir()
            nested.mkdir(parents=True)

            removed = remove_empty_folders(root, recursive=False)

            self.assertEqual(removed, 1)
            self.assertFalse(top_empty.exists())
            self.assertTrue((root / "outer").exists())

    def test_group_duplicate_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            f1 = root / "one"
            f2 = root / "two"
            f1.mkdir()
            f2.mkdir()
            (f1 / "x.txt").write_text("same")
            (f2 / "x.txt").write_text("same")

            groups = group_duplicate_folders([f1, f2])
            self.assertEqual(len(groups), 1)
            self.assertEqual({item.name for item in groups[0]}, {"one", "two"})


if __name__ == "__main__":
    unittest.main()
