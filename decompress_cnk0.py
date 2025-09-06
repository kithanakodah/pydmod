import os
import subprocess
import glob
import sys
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def decompress_cnk0_files():
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cnk0_decompress_log_{timestamp}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger

    print(f"CNK0 Decompression Log - {datetime.now()}")
    print("=" * 50)

    try:
        # Define paths
        source_dir = r"M:\H1Z1_assets"
        output_dir = r"M:\H1Z1_assets\decompressed"
        cnkdec_exe = r"cnkdec.exe"  # Adjust path if needed

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all .cnk0 files in the source directory
        search_pattern = os.path.join(source_dir, "*.cnk0")
        cnk0_files = glob.glob(search_pattern)

        if not cnk0_files:
            print("No .cnk0 files found in the source directory.")
            return

        print(f"Found {len(cnk0_files)} .cnk0 files to decompress")

        # Check if cnkdec.exe exists
        if not os.path.exists(cnkdec_exe):
            print(
                f"Error: {cnkdec_exe} not found. Please ensure it's in the current directory or provide the full path.")
            return

        successful = 0
        failed = 0

        for i, cnk_file in enumerate(cnk0_files, 1):
            # Get just the filename without path
            filename = os.path.basename(cnk_file)

            # Create output path with same filename (keeping .cnk0 extension)
            output_file = os.path.join(output_dir, filename)

            print(f"[{i}/{len(cnk0_files)}] Processing {filename}...")

            try:
                # Run cnkdec.exe with 'd' command for decompress
                result = subprocess.run([cnkdec_exe, "d", cnk_file, output_file],
                                        capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    print(f"  ✓ Successfully decompressed to {filename}")
                    successful += 1
                else:
                    print(f"  ✗ Failed to decompress {filename}")
                    if result.stdout:
                        print(f"    Stdout: {result.stdout.strip()}")
                    if result.stderr:
                        print(f"    Stderr: {result.stderr.strip()}")
                    failed += 1

            except subprocess.TimeoutExpired:
                print(f"  ✗ Timeout while processing {filename}")
                failed += 1
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {str(e)}")
                failed += 1

        print(f"\nDecompression complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Decompressed files saved to: {output_dir}")
        print(f"Log saved to: {log_filename}")

    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nComplete log saved to: {log_filename}")


if __name__ == "__main__":
    decompress_cnk0_files()