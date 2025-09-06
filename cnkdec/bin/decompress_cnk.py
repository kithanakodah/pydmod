#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path


def decompress_cnk_files(source_dir, dest_dir, cnkdec_path="cnkdec.exe"):
    """
    Decompress all CNK0 files using cnkdec.exe

    Args:
        source_dir: Directory containing compressed CNK0 files
        dest_dir: Directory to output decompressed files
        cnkdec_path: Path to cnkdec.exe executable
    """

    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_path}")
        return False

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)

    # Find all CNK0 files
    cnk_files = list(source_path.glob("*.cnk0"))

    if not cnk_files:
        print(f"No .cnk0 files found in {source_path}")
        return False

    print(f"Found {len(cnk_files)} CNK0 files to decompress")
    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    print(f"Using cnkdec: {cnkdec_path}")
    print()

    success_count = 0
    failed_files = []

    for cnk_file in cnk_files:
        output_file = dest_path / cnk_file.name

        print(f"Processing: {cnk_file.name}")

        try:
            # Run cnkdec.exe d input output
            result = subprocess.run([
                cnkdec_path, 'd', str(cnk_file), str(output_file)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print(f"  SUCCESS: {cnk_file.name} decompressed")
                success_count += 1

                # Verify the output file exists and has reasonable size
                if output_file.exists() and output_file.stat().st_size > 8:
                    print(f"    Output size: {output_file.stat().st_size:,} bytes")
                else:
                    print(f"    WARNING: Output file seems too small")
            else:
                print(f"  ERROR: Failed to decompress {cnk_file.name}")
                print(f"    Return code: {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                failed_files.append(cnk_file.name)

        except subprocess.TimeoutExpired:
            print(f"  ERROR: Timeout decompressing {cnk_file.name}")
            failed_files.append(cnk_file.name)
        except FileNotFoundError:
            print(f"ERROR: cnkdec.exe not found at: {cnkdec_path}")
            print("Please download cnkdec.exe from: https://github.com/psemu/cnkdec")
            return False
        except Exception as e:
            print(f"  ERROR: Unexpected error with {cnk_file.name}: {e}")
            failed_files.append(cnk_file.name)

    print()
    print(f"Decompression complete!")
    print(f"  Successful: {success_count}/{len(cnk_files)}")
    print(f"  Failed: {len(failed_files)}")

    if failed_files:
        print("Failed files:")
        for failed in failed_files:
            print(f"  - {failed}")

    return len(failed_files) == 0


def main():
    if len(sys.argv) < 3:
        print("Usage: python decompress_cnk.py <source_dir> <dest_dir> [cnkdec_path]")
        print()
        print("Example:")
        print("  python decompress_cnk.py M:/H1Z1_assets/compressed M:/H1Z1_assets/decompressed")
        print("  python decompress_cnk.py ./compressed ./decompressed ./cnkdec.exe")
        sys.exit(1)

    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    cnkdec_path = sys.argv[3] if len(sys.argv) > 3 else "cnkdec.exe"

    success = decompress_cnk_files(source_dir, dest_dir, cnkdec_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()