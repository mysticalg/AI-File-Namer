import unittest

from src.ai_file_namer import format_date, sanitize_filename_stem


class FilenameUtilsTests(unittest.TestCase):
    def test_sanitize_filename_stem(self):
        raw = "A Cute Cat!!!\n(indoors)"
        self.assertEqual(sanitize_filename_stem(raw), "a_cute_cat_indoors")

    def test_format_date(self):
        result = format_date("%Y-%m-%d")
        self.assertEqual(len(result), 10)


if __name__ == "__main__":
    unittest.main()
