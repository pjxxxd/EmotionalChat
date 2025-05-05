#!/usr/bin/env python3
import os
import json

def remove_annotations(input_dir='./data', output_dir='./results'):
    """
    Reads all .json files in the input directory, removes the 'annotation' field
    from each entry, and writes the cleaned data to the output directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, 'r', encoding='utf-8') as infile:
                try:
                    data = json.load(infile)
                except json.JSONDecodeError as e:
                    print(f"Skipping {filename}: invalid JSON ({e})")
                    continue

            # Remove 'annotation' field from each entry
            cleaned_data = [
                {
                    'role': entry.get('role'),
                    'content': entry.get('content')
                }
                for entry in data if isinstance(entry, dict)
            ]

            # Write the cleaned data to the output directory
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(cleaned_data, outfile, ensure_ascii=False, indent=2)

    print(f"Processed JSON files from '{input_dir}' and saved results to '{output_dir}'.")

if __name__ == "__main__":
    remove_annotations()
