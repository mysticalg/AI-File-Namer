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
    FilenamePreferences,
    extract_json_object,
    extract_restructure_plan,
    extract_partial_restructure_operations,
    format_restructure_preview_paths,
    summarize_debug_headers,
    summarize_debug_payload,
    remove_empty_folders,
    sanitize_category_path,
    sanitize_restructure_operations,
    normalize_ai_source_relative_path,
    normalize_ai_destination_relative_path,
    build_folder_inventory,
    find_missing_restructure_sources,
    sanitize_filename_stem,
    build_ollama_tags_endpoint,
    parse_ollama_model_names,
    load_app_settings,
    save_app_settings,
    clamp_ai_timeout_seconds,
)


class FilenameUtilsTests(unittest.TestCase):

    def test_clamp_ai_timeout_seconds_respects_bounds(self):
        self.assertEqual(clamp_ai_timeout_seconds(120), 120)
        self.assertEqual(clamp_ai_timeout_seconds(5), 30)
        self.assertEqual(clamp_ai_timeout_seconds("9999"), 3600)
        self.assertEqual(clamp_ai_timeout_seconds("bad"), 120)

    def test_load_app_settings_returns_empty_for_missing_or_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            self.assertEqual(load_app_settings(settings_path), {})

            settings_path.write_text("not-json", encoding="utf-8")
            self.assertEqual(load_app_settings(settings_path), {})

    def test_save_and_load_app_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            payload = {
                "provider_mode": "Remote (OpenAI-compatible /v1/chat/completions)",
                "endpoint": "https://example.test/v1/chat/completions",
                "recursive_scan": False,
                "max_filename_length": 88,
                "ai_timeout_seconds": 240,
            }

            save_app_settings(payload, settings_path)
            loaded = load_app_settings(settings_path)

            self.assertEqual(loaded, payload)

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

    def test_sanitize_category_path_limits_depth_and_cleans_segments(self):
        path = sanitize_category_path(
            raw="Music > Classical / Bach / Organ / Extra",
            separator="_",
            capitalization="lower",
            max_segment_length=12,
            max_depth=3,
        )
        self.assertEqual(path, "music/classical/bach")

    def test_sanitize_category_path_supports_title_case_with_spaces(self):
        path = sanitize_category_path(
            raw="personal photos/travel shots",
            separator=" ",
            capitalization="title",
            max_segment_length=20,
            max_depth=3,
        )
        self.assertEqual(path, "Personal Photos/Travel Shots")

    def test_extract_json_object_reads_wrapped_json(self):
        payload = "```json\n{\"operations\": [], \"dedupe_files\": true}\n```"
        result = extract_json_object(payload)
        self.assertIn("operations", result)
        self.assertTrue(result.get("dedupe_files"))

    def test_sanitize_restructure_operations_filters_invalid_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "messy folder"
            folder.mkdir()
            file_path = folder / "track 01.mp3"
            file_path.write_text("x")

            preferences = FilenamePreferences(
                separator=" ",
                capitalization="title",
                max_filename_length=96,
                max_folder_name_length=32,
                include_hashtags=False,
                hashtag_count=3,
            )
            operations = [
                {"type": "folder", "source": "messy folder", "destination": "Music/Archive"},
                {"type": "file", "source": "messy folder/track 01.mp3", "destination": "Music/Singles"},
                {"type": "folder", "source": "../escape", "destination": "Bad"},
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].item_type, "folder")
            self.assertIn("Music", suggestions[0].target_relative)

    def test_normalize_ai_source_relative_path_strips_root_prefix(self):
        root = Path("/tmp/Bach")
        self.assertEqual(normalize_ai_source_relative_path("/Bach/Works", root), "Works")
        self.assertEqual(normalize_ai_source_relative_path("Bach/Works", root), "Works")

    def test_normalize_ai_destination_relative_path_supports_root_destination(self):
        root = Path("/tmp/Bach")
        preferences = FilenamePreferences(
            separator=" ",
            capitalization="title",
            max_filename_length=96,
            max_folder_name_length=32,
            include_hashtags=False,
            hashtag_count=3,
        )
        self.assertEqual(normalize_ai_destination_relative_path("/Bach", root, preferences), "")
        self.assertEqual(normalize_ai_destination_relative_path("", root, preferences), "")

    def test_sanitize_restructure_operations_handles_absolute_and_root_destination(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "Bach"
            works = root / "Works for Solo Lute"
            works.mkdir(parents=True)

            preferences = FilenamePreferences(
                separator=" ",
                capitalization="title",
                max_filename_length=96,
                max_folder_name_length=64,
                include_hashtags=False,
                hashtag_count=3,
            )
            operations = [
                {"type": "folder", "source": "/Bach/Works for Solo Lute", "destination": "/Bach"},
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            # Root destination is accepted and converted into a valid target path.
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].original_relative, "Works for Solo Lute")
            self.assertEqual(suggestions[0].target_relative, "Works For Solo Lute")

    def test_sanitize_restructure_operations_handles_nested_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "Bach"
            child = root / "Works for Solo Lute" / "Partita in C minor - BWV 997"
            child.mkdir(parents=True)

            preferences = FilenamePreferences(
                separator=" ",
                capitalization="title",
                max_filename_length=96,
                max_folder_name_length=64,
                include_hashtags=False,
                hashtag_count=3,
            )
            operations = [
                {
                    "type": "folder",
                    "source": "/Bach/Works for Solo Lute/Partita in C minor - BWV 997",
                    "destination": "/Bach/Partitas",
                },
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].original_relative, "Works for Solo Lute/Partita in C minor - BWV 997")
            self.assertEqual(suggestions[0].target_relative, "Partitas/Partita In C Minor Bwv 997")

    def test_sanitize_restructure_operations_ignores_file_operations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "media"
            folder.mkdir()
            (folder / "clip.mp4").write_text("x")

            preferences = FilenamePreferences(
                separator="_",
                capitalization="lower",
                max_filename_length=96,
                max_folder_name_length=32,
                include_hashtags=False,
                hashtag_count=3,
            )
            operations = [
                {"type": "file", "source": "media/clip.mp4", "destination": "video/clips"},
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(suggestions, [])

    def test_build_folder_inventory_returns_folder_only_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "photos" / "travel").mkdir(parents=True)
            (root / "photos" / "family").mkdir(parents=True)
            (root / "photos" / "image1.jpg").write_text("x")

            inventory = build_folder_inventory(root, recursive=True)

            self.assertEqual(inventory["root"], root.name)
            self.assertIn("root_full_path", inventory)
            self.assertIn("root_parent_name", inventory)
            self.assertNotIn("files", inventory)
            self.assertNotIn("file_count", inventory)
            self.assertIn("all_folder_paths", inventory)
            self.assertGreaterEqual(len(inventory["all_folder_paths"]), 3)
            first_row = inventory["folders"][0]
            self.assertIn("direct_subfolder_count", first_row)
            self.assertIn("sample_subfolders", first_row)
            self.assertNotIn("sample_files", first_row)

    def test_format_restructure_preview_paths_outputs_old_new_and_transition(self):
        old_path, new_path, transition = format_restructure_preview_paths(
            "Old Folder/Sub",
            "Music/Archive/Old Folder",
        )
        self.assertEqual(old_path, "Old Folder/Sub")
        self.assertEqual(new_path, "Music/Archive/Old Folder")
        self.assertIn("→", transition)

    def test_format_restructure_preview_paths_normalizes_slashes(self):
        old_path, new_path, transition = format_restructure_preview_paths(
            r"old\nested",
            r"new\bucket\old",
        )
        self.assertEqual(old_path, "old/nested")
        self.assertEqual(new_path, "new/bucket/old")
        self.assertEqual(transition, "old/nested → new/bucket/old")

    def test_summarize_debug_headers_redacts_authorization(self):
        headers = {"Authorization": "Bearer secret-token", "Content-Type": "application/json"}
        result = summarize_debug_headers(headers)
        self.assertEqual(result["Authorization"], "Bearer ***redacted***")
        self.assertEqual(result["Content-Type"], "application/json")

    def test_summarize_debug_payload_truncates_large_content(self):
        payload = {"x": "a" * 6000}
        result = summarize_debug_payload(payload, max_chars=120)
        self.assertIn("truncated", result)


    def test_find_missing_restructure_sources_identifies_uncovered_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "photos").mkdir()
            (root / "videos").mkdir()

            preferences = FilenamePreferences(
                separator="_",
                capitalization="lower",
                max_filename_length=96,
                max_folder_name_length=32,
                include_hashtags=False,
                hashtag_count=3,
            )
            operations = [{"type": "folder", "source": "photos", "destination": "media"}]
            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)

            missing = find_missing_restructure_sources(
                suggestions=suggestions,
                root=root,
                candidate_folders=collect_subfolders(root, recursive=False),
            )
            self.assertEqual(missing, ["videos"])


    def test_parse_ollama_model_names_sorts_and_deduplicates(self):
        payload = {
            "models": [
                {"name": "llava"},
                {"name": "mistral"},
                {"name": "LLaVA"},
                {"name": ""},
                {},
            ]
        }
        names = parse_ollama_model_names(payload)
        self.assertEqual(names, ["llava", "mistral"])

    def test_build_ollama_tags_endpoint_maps_generate_to_tags(self):
        self.assertEqual(
            build_ollama_tags_endpoint("http://localhost:11434/api/generate"),
            "http://localhost:11434/api/tags",
        )
        self.assertEqual(
            build_ollama_tags_endpoint("http://host:11434/custom"),
            "http://host:11434/custom/api/tags",
        )


    def test_extract_restructure_plan_reads_nested_response_envelope(self):
        payload = {
            "model": "mistral",
            "response": "{\"operations\":[{\"type\":\"folder\",\"source\":\"a\",\"destination\":\"music\"}],\"dedupe_files\":true}",
        }
        result = extract_restructure_plan(payload)
        self.assertIn("operations", result)
        self.assertEqual(len(result["operations"]), 1)

    def test_extract_restructure_plan_returns_empty_for_non_plan_payload(self):
        payload = {"model": "mistral", "response": "done"}
        result = extract_restructure_plan(payload)
        self.assertEqual(result, {})


    def test_extract_partial_restructure_operations_recovers_complete_items(self):
        raw = """```json
{
  "operations": [
    {"type": "folder", "source": "Bach", "destination": "Classical"},
    {"type": "folder", "source": "Mozart", "destination": "Classical"},
    {"type": "folder", "source": "Incomplete", "destination"
```"""
        ops = extract_partial_restructure_operations(raw)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0]["source"], "Bach")

    def test_extract_restructure_plan_recovers_from_truncated_nested_response(self):
        payload = {
            "response": """{"model":"mistral","response":"```json\n{\n  \"operations\": [\n    {\"type\": \"folder\", \"source\": \"Bach\", \"destination\": \"Classical\"},\n    {\"type\": \"folder\", \"source\": \"Mozart\", \"destination\": \"Classical\"},\n    {\"type\": \"folder\", \"source\": \"Cut\", \"destination\"\n```"}"""
        }
        plan = extract_restructure_plan(payload)
        self.assertIn("operations", plan)
        self.assertEqual(len(plan["operations"]), 2)


if __name__ == "__main__":
    unittest.main()
