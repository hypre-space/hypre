# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import argparse
import os

def extract_files_from_section(input_text, section_start):
    files = []
    collect_files = False
    for line in input_text.splitlines():
        if line.strip().startswith(section_start):
            collect_files = True
            continue

        if collect_files:
            if line.strip() == "":
                break  # Assuming an empty line marks the end of the section

            # Extract the file name, assuming no spaces in file names
            file_name = line.strip().split("\\")[0]

            # Filter only source files
            if file_name.endswith('.c') or file_name.endswith('.cpp'):
                files.append(file_name)

    return files

def are_files_in_list(file_list_a, file_list_b):
    return [fn for fn in file_list_a if fn not in file_list_b]

def add_missing_files(file_content, section_start, missing_files):
    lines = file_content.split('\n')
    updated_lines = []
    in_section_start = -1
    in_section_end = -1

    # Identify the start and end of the SRCS block
    for i, line in enumerate(lines):
        trimmed_line = line.strip()
        if trimmed_line.startswith(section_start):
            in_section_start = i

        if in_section_start != -1 and trimmed_line.endswith(")") and in_section_end == -1:
            in_section_end = i
            break

    if in_section_start == -1 or in_section_end == -1:
        print("Error: Could not find a complete SRCS block in file B.")
        return file_content

    # Add all lines up to the end of the section, excluding the closing parenthesis
    updated_lines.extend(lines[:in_section_end])

    # Add missing files
    for missing_file in missing_files:
        updated_lines.append(f"  {missing_file}")

    # Add closing parenthesis
    updated_lines.append(lines[in_section_end])

    # Add remaining lines
    updated_lines.extend(lines[in_section_end + 1:])

    updated_content = '\n'.join(updated_lines)
    return updated_content

def process(args, section_A, section_B):
    # Read file contents
    file_A_path = os.path.join(args.folder, "Makefile")
    file_B_path = os.path.join(args.folder, "CMakeLists.txt")
    with open(file_A_path, 'r') as file_A, open(file_B_path, 'r') as file_B:
        file_A_content = file_A.read()
        file_B_content = file_B.read()

    # Extract files
    files_A = extract_files_from_section(file_A_content, section_A)
    files_B = extract_files_from_section(file_B_content, section_B)
    if args.verbose:
        print(f"{files_A = }\n")
        print(f"{files_B = }")

    # Check if all files in A are in B
    missing = are_files_in_list(files_A, files_B)

    if missing:
        print("\nAdded to FILE B:", missing)
        new_file_B_content = add_missing_files(file_B_content, section_B, missing)
        with open(file_B_path, 'w') as file_B:
            file_B.write(new_file_B_content)

def main():
    parser = argparse.ArgumentParser(description="Check and update CMakeLists based on the contents of Makefile")
    parser.add_argument("-f", "--folder", required=True, help="Folder path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Turn on verbose mode")
    args = parser.parse_args()

    # Validate folder path
    args.folder = os.path.normpath(args.folder)
    if not os.path.isdir(args.folder):
        print("The specified folder does not exist or is not a directory.")
        return

    # Process source files
    process(args, "FILES =", "set(SRCS")

    # Process GPU source files
    process(args, "CUFILES =", "set(GPU_SRCS")

    # Done!
    print(f"Done with {args.folder = }...")

if __name__ == "__main__":
    main()
