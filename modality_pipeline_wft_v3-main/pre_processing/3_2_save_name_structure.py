#!/usr/bin/env python3
import os
import argparse

def write_yaml(root_dir, output_file, use_tabs=True):
    """
    Walks a directory and writes its structure into a YAML-like file.
    Directories become keys, and files are listed as sequences under them.
    """
    indent_unit = "\t" if use_tabs else "  "  # tab vs 2 spaces

    def walk(dir_path, depth, f):
        entries = sorted(os.listdir(dir_path))
        for entry in entries:
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                f.write(f"{indent_unit * depth}{entry}:\n")
                walk(full_path, depth + 1, f)
            else:
                f.write(f"{indent_unit * (depth+1)}- {entry}\n")

    root_name = os.path.basename(os.path.normpath(root_dir))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{root_name}:\n")
        walk(root_dir, 1, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML-like file tree")
    parser.add_argument("input_dir", help="Input directory to scan")
    parser.add_argument("output_file", help="Output YAML file path")
    parser.add_argument("--spaces", action="store_true",
                        help="Use spaces instead of tabs (valid YAML)")

    args = parser.parse_args()
    write_yaml(args.input_dir, args.output_file, use_tabs=not args.spaces)
