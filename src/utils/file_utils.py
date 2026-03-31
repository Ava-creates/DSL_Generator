#!/usr/bin/env python3
"""
File utility functions for versioning and file operations.
"""

import os
import re


def version_file(file_path: str, keep_original: bool = False) -> None:
    """Version a file by renaming existing versions and creating a new numbered version.
    
    The current file is always the original name (e.g., cfg_output.json).
    Older versions are numbered: cfg_output_0.json, cfg_output_1.json, etc.
    
    Example flow:
    - First CFG: cfg_output.json (no versioning needed)
    - Second CFG: cfg_output.json -> cfg_output_0.json, new -> cfg_output.json
    - Third CFG: cfg_output.json -> cfg_output_1.json, new -> cfg_output.json
    
    Args:
        file_path: Path to the file to version
        # keep_original: (unused)
    """
    if not os.path.exists(file_path):
        return
    
    # Extract directory, base name, and extension
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Split filename into base and extension
    if '.' in filename:
        base_name, ext = os.path.splitext(filename)
        ext_with_dot = ext
    else:
        base_name = filename
        ext_with_dot = ""
    
    # Find the highest existing version number
    max_version = -1
    # Pattern to match: base_name_N.ext (e.g., cfg_output_0.json, cfg_output_1.json)
    pattern = f"^{re.escape(base_name)}_(\\d+){re.escape(ext_with_dot)}$"
    
    if directory:
        files = os.listdir(directory)
    else:
        files = os.listdir('.')
    
    for f in files:
        match = re.match(pattern, f)
        if match:
            version_num = int(match.group(1))
            max_version = max(max_version, version_num)
    
    # Calculate new version number
    new_version = max_version + 1
    
    # Create versioned filename
    versioned_filename = f"{base_name}_{new_version}{ext_with_dot}"
    if directory:
        versioned_path = os.path.join(directory, versioned_filename)
    else:
        versioned_path = versioned_filename
    
    # Rename the existing file to the versioned name
    try:
        os.rename(file_path, versioned_path)
        print(f"   Versioned {filename} -> {versioned_filename}")
    except OSError as e:
        print(f"   Error versioning file {filename}: {e}")
        raise

