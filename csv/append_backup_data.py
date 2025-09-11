#!/usr/bin/env python3
"""
Script to append data from backup_csv directory to the corresponding CSV files in csv directory.
This script will merge data while avoiding duplicates based on all columns.
"""

import os
import csv
from pathlib import Path


def read_csv_rows(file_path):
    """Read CSV file and return header and data rows as strings."""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return None, []
            return rows[0], rows[1:]  # header, data
    except FileNotFoundError:
        return None, []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, []


def write_csv_data(file_path, header, data_rows):
    """Write header and data rows to CSV file."""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


def append_csv_data(csv_file_path, backup_file_path, output_file_path):
    """
    Append data from backup CSV to main CSV, avoiding duplicates.
    Preserves exact string format of all data.
    
    Args:
        csv_file_path: Path to the current CSV file
        backup_file_path: Path to the backup CSV file
        output_file_path: Path where the merged CSV should be saved
    """
    # Read current CSV
    current_header, current_rows = read_csv_rows(csv_file_path)
    if current_header is not None:
        print(f"Current CSV has {len(current_rows)} rows")
    else:
        print("Current CSV not found or empty")
        current_rows = []
    
    # Read backup CSV
    backup_header, backup_rows = read_csv_rows(backup_file_path)
    if backup_header is None:
        print(f"Backup file not found or empty: {backup_file_path}")
        return False
    
    print(f"Backup CSV has {len(backup_rows)} rows")
    
    # Use backup header as the reference (they should be the same)
    header = backup_header
    if current_header and current_header != backup_header:
        print(f"Warning: Headers differ between files!")
        print(f"Current: {current_header}")
        print(f"Backup:  {backup_header}")
    
    # Combine rows and remove duplicates while preserving exact format
    all_rows = current_rows + backup_rows
    initial_count = len(all_rows)
    
    # Remove duplicates by converting rows to tuples and using a set
    unique_rows = []
    seen = set()
    for row in all_rows:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)
    
    final_count = len(unique_rows)
    print(f"Merged: {initial_count} total rows, {final_count} unique rows ({initial_count - final_count} duplicates removed)")
    
    # Sort rows for consistent output
    # Try to sort by the first few columns (assuming they are numeric/sortable)
    try:
        def sort_key(row):
            # Create a sort key from the first few columns
            key = []
            for i in range(min(3, len(row))):  # Use first 3 columns for sorting
                try:
                    # Try to convert to number for proper sorting, but keep original string
                    if row[i].replace('.', '').replace('-', '').isdigit():
                        key.append(float(row[i]))
                    else:
                        key.append(row[i])
                except (ValueError, AttributeError):
                    key.append(row[i])
            return key
        
        unique_rows.sort(key=sort_key)
    except Exception as e:
        print(f"Warning: Could not sort rows: {e}")
    
    # Write merged data
    write_csv_data(output_file_path, header, unique_rows)
    print(f"Merged data written to {output_file_path} ({len(unique_rows)} rows)")
    return True


def main():
    # Define directories
    csv_dir = Path("/users/xilin/TinyPtr/latex/csv")
    backup_dir = Path("/users/xilin/TinyPtr/latex/backup_csv")
    
    print(f"CSV directory: {csv_dir}")
    print(f"Backup directory: {backup_dir}")
    print()
    
    if not csv_dir.exists():
        print(f"Error: CSV directory does not exist: {csv_dir}")
        return
    
    if not backup_dir.exists():
        print(f"Error: Backup directory does not exist: {backup_dir}")
        return
    
    # Get list of CSV files in backup directory
    backup_files = list(backup_dir.glob("*.csv"))
    
    if not backup_files:
        print("No CSV files found in backup directory")
        return
    
    print(f"Found {len(backup_files)} CSV files in backup directory")
    print()
    
    # Process each CSV file
    success_count = 0
    for backup_file in backup_files:
        filename = backup_file.name
        csv_file = csv_dir / filename
        
        print(f"Processing: {filename}")
        print("-" * 50)
        
        success = append_csv_data(csv_file, backup_file, csv_file)
        if success:
            success_count += 1
        
        print()
    
    print(f"Summary: Successfully processed {success_count}/{len(backup_files)} files")


if __name__ == "__main__":
    main()
