"""cx_Freeze setup script for building a Windows MSI installer."""

from cx_Freeze import Executable, setup

build_exe_options = {
    "packages": ["tkinter", "PIL", "cv2", "requests"],
    "include_msvcr": True,
    # Keep build lightweight and deterministic.
    "excludes": ["unittest", "pydoc"],
}

executables = [
    Executable(
        script="src/ai_file_namer.py",
        base="Win32GUI",
        target_name="AIFileNamer.exe",
    )
]

setup(
    name="AI File Namer",
    version="1.0.0",
    description="AI-powered filename suggestions for image and video files.",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": {
            "summary_data": {
                "author": "AI File Namer",
                "comments": "Desktop AI-powered bulk file renaming tool",
            },
            "upgrade_code": "{F042B994-C0E8-4F03-8CA8-25D2361C071E}",
            "install_icon": None,
        },
    },
    executables=executables,
)
