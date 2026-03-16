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
    build_ollama_missing_guidance,
    extract_openai_text_content,
    build_openai_models_endpoint,
    parse_openai_model_names,
    parse_retry_after_seconds,
    build_openai_rate_limit_guidance,
)


class FilenameUtilsTests(unittest.TestCase):



    def test_build_openai_models_endpoint_from_chat_completions(self):
        endpoint = build_openai_models_endpoint("https://api.openai.com/v1/chat/completions")
        self.assertEqual(endpoint, "https://api.openai.com/v1/models")

    def test_build_openai_models_endpoint_from_nested_path(self):
        endpoint = build_openai_models_endpoint("https://proxy.example.com/openai/v1/chat/completions")
        self.assertEqual(endpoint, "https://proxy.example.com/openai/v1/models")

    def test_build_openai_models_endpoint_rejects_ollama_generate_path(self):
        endpoint = build_openai_models_endpoint("http://localhost:11434/api/generate")
        self.assertEqual(endpoint, "")

    def test_parse_retry_after_seconds_handles_numeric_and_invalid_values(self):
        self.assertEqual(parse_retry_after_seconds("3"), 3.0)
        self.assertEqual(parse_retry_after_seconds("1.5"), 1.5)
        self.assertIsNone(parse_retry_after_seconds(""))
        self.assertIsNone(parse_retry_after_seconds("soon"))

    def test_build_openai_rate_limit_guidance_explains_single_request_limit(self):
        message = build_openai_rate_limit_guidance(2)
        self.assertIn("single request", message)
        self.assertIn("Retry after about 2 second(s)", message)

    def test_parse_openai_model_names_filters_non_chat_ids(self):
        payload = {
            "data": [
                {"id": "gpt-4o-mini"},
                {"id": "whisper-1"},
                {"id": "text-embedding-3-small"},
                {"id": "gpt-4.1"},
                {"id": "gpt-4o-mini"},
            ]
        }
        self.assertEqual(parse_openai_model_names(payload), ["gpt-4.1", "gpt-4o-mini"])

    def test_extract_openai_text_content_supports_chat_completions_string(self):
        payload = {
            "choices": [
                {"message": {"content": "final_name"}},
            ]
        }
        self.assertEqual(extract_openai_text_content(payload), "final_name")

    def test_extract_openai_text_content_supports_list_content_blocks(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "output_text", "text": "first"},
                            {"type": "output_text", "text": "second"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(extract_openai_text_content(payload), "first\nsecond")

    def test_extract_openai_text_content_supports_responses_api_output(self):
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "structured_result"}
                    ],
                }
            ]
        }
        self.assertEqual(extract_openai_text_content(payload), "structured_result")


    def test_clamp_ai_timeout_seconds_respects_bounds(self):
        self.assertEqual(clamp_ai_timeout_seconds(120), 120)
        self.assertEqual(clamp_ai_timeout_seconds(5), 30)
        self.assertEqual(clamp_ai_timeout_seconds("9999"), 3600)
        self.assertEqual(clamp_ai_timeout_seconds("bad"), 120)


    def test_build_ollama_missing_guidance_when_ollama_unreachable(self):
        message = build_ollama_missing_guidance("HTTPConnectionPool(host=localhost): Max retries exceeded")
        self.assertIn("https://ollama.com/download", message)

    def test_build_ollama_missing_guidance_for_other_errors(self):
        message = build_ollama_missing_guidance("unexpected payload")
        self.assertEqual(message, "Could not fetch Ollama models. You can still type a model manually.")

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

    def test_sanitize_filename_stem_supports_natural_language_case(self):
        raw = "A visual representation of United States against United Kingdom's empire"
        value = sanitize_filename_stem(raw, separator=" ", capitalization="natural", max_length=120)
        self.assertEqual(value, "A visual representation of United States against United Kingdom's empire")

    def test_sanitize_filename_stem_supports_natural_language_with_underscores(self):
        raw = "United Kingdom's map overview"
        value = sanitize_filename_stem(raw, separator="_", capitalization="natural", max_length=120)
        self.assertEqual(value, "United_Kingdom's_map_overview")

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
            # Invalid path is filtered, while valid folder+file operations are preserved.
            self.assertEqual(len(suggestions), 2)
            self.assertEqual(sorted(item.item_type for item in suggestions), ["file", "folder"])
            self.assertTrue(all("Music" in item.target_relative for item in suggestions))

    def test_build_folder_inventory_can_exclude_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "album").mkdir()
            (root / "album" / "track01.mp3").write_text("x")

            inventory = build_folder_inventory(root, recursive=True, include_files=False)

            self.assertEqual(inventory.get("total_file_count"), 0)
            self.assertEqual(inventory.get("file_paths"), [])

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

    def test_normalize_ai_destination_relative_path_strips_windows_absolute_prefix(self):
        root = Path("/tmp/Bach")
        preferences = FilenamePreferences(
            separator=" ",
            capitalization="title",
            max_filename_length=96,
            max_folder_name_length=64,
            include_hashtags=False,
            hashtag_count=3,
        )
        value = normalize_ai_destination_relative_path(
            "D:/OneDrive/Music/midi/classical/Bach/Works/Invent",
            root,
            preferences,
        )
        self.assertEqual(value, "Works/Invent")

    def test_normalize_ai_destination_relative_path_drops_unmatched_absolute_path(self):
        root = Path("/tmp/Bach")
        preferences = FilenamePreferences(
            separator=" ",
            capitalization="title",
            max_filename_length=96,
            max_folder_name_length=64,
            include_hashtags=False,
            hashtag_count=3,
        )
        value = normalize_ai_destination_relative_path(
            "D:/Completely/Other/Tree/NotRelated",
            root,
            preferences,
        )
        self.assertEqual(value, "")


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

    def test_sanitize_restructure_operations_handles_windows_absolute_destination(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "Bach"
            source = root / "Works" / "Invent"
            source.mkdir(parents=True)

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
                    "source": "Works/Invent",
                    "destination": "D:/OneDrive/Music/midi/classical/Bach",
                },
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(len(suggestions), 1)
            # Ensure absolute drive prefixes are stripped instead of creating `D/...` folders.
            self.assertEqual(suggestions[0].target_relative, "Invent")

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

    def test_sanitize_restructure_operations_avoids_double_folder_suffix_when_ai_returns_full_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache = root / "iTunes" / "Album Artwork" / "Cache"
            cache.mkdir(parents=True)

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
                    "source": "iTunes/Album Artwork/Cache",
                    # Model incorrectly returns final path instead of parent path.
                    "destination": "Media/Artwork/Cache",
                },
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].target_relative, "Media/Artwork/Cache")

    def test_sanitize_restructure_operations_collapses_duplicate_trailing_segment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "Accapella" / "Artists (Alphabetical)" / "0"
            source.mkdir(parents=True)

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
                    "source": "Accapella/Artists (Alphabetical)/0",
                    "destination": "Music/Vocal/Acapella/Artists Alphabetical/0",
                },
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].target_relative, "Music/Vocal/Acapella/Artists Alphabetical/0")

    def test_sanitize_restructure_operations_supports_file_operations(self):
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
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].item_type, "file")
            self.assertEqual(suggestions[0].target_relative, "video/clips/clip.mp4")

    def test_sanitize_restructure_operations_supports_file_rename_destination(self):
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
                {"type": "file", "source": "media/clip.mp4", "destination": "video/highlights/my best clip.mp4"},
            ]

            suggestions = sanitize_restructure_operations(operations, root=root, preferences=preferences)
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0].target_relative, "video/highlights/my_best_clip.mp4")

    def test_build_folder_inventory_returns_compact_tree_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "photos" / "travel").mkdir(parents=True)
            (root / "photos" / "family").mkdir(parents=True)
            (root / "photos" / "image1.jpg").write_text("x")

            inventory = build_folder_inventory(root, recursive=True)

            self.assertEqual(inventory["root"], root.name)
            self.assertIn("root_full_path", inventory)
            self.assertIn("root_parent_name", inventory)
            self.assertIn("file_paths", inventory)
            self.assertIn("total_file_count", inventory)
            self.assertIn("photos/image1.jpg", inventory["file_paths"])
            self.assertIn("folder_paths", inventory)
            self.assertGreaterEqual(len(inventory["folder_paths"]), 3)
            self.assertEqual(inventory["total_folder_count"], len(inventory["folder_paths"]))
            self.assertIn("direct_children", inventory)
            self.assertIn("photos", inventory["direct_children"])
            self.assertEqual(sorted(inventory["direct_children"]["photos"]), ["family", "travel"])

    def test_build_folder_inventory_honors_sampling_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for idx in range(6):
                (root / f"folder_{idx}").mkdir(parents=True)
                (root / f"folder_{idx}" / f"file_{idx}.jpg").write_text("x")

            inventory = build_folder_inventory(root, recursive=True, max_nodes=3, max_files=2)
            self.assertLessEqual(len(inventory["folder_paths"]), 3)
            self.assertLessEqual(len(inventory["file_paths"]), 2)

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

    def test_extract_restructure_plan_supports_single_operation_shape(self):
        payload = {
            "response": """```json
{
  "operation": "folder",
  "source": "Bach",
  "destination": "Classical/Bach"
}
```"""
        }
        plan = extract_restructure_plan(payload)
        self.assertIn("operations", plan)
        self.assertEqual(len(plan["operations"]), 1)
        self.assertEqual(plan["operations"][0]["type"], "folder")

    def test_extract_partial_restructure_operations_supports_operation_key(self):
        raw = """{
  "operation": "file",
  "source": "old/song.mid",
  "destination": "classical/Bach/song.mid"
}"""
        ops = extract_partial_restructure_operations(raw)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["type"], "file")


if __name__ == "__main__":
    unittest.main()
