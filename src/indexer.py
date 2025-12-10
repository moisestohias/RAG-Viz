""" 
This Python script scans the current directory and its subdirectories for Markdown (.md) files. It creates a JSON file where each key is the relative file path and the value is the first 200 words of the file's content, excluding lines containing base64 image data.
""" 

import os,re,json
# import yaml

def get_description_yaml_metadata(f):
    if f.readline().strip() != '---':
        return False
    yaml_lines = []
    for line in f:
        if line.strip() == '---':
            break
        yaml_lines.append(line)
    else:
        return False  # no closing ---
    try:
        metadata = yaml.safe_load(''.join(yaml_lines))
        description = metadata.get('description') if isinstance(metadata, dict) else None
        return description
    except yaml.YAMLError:
        return False

# def has_yaml_frontmatter(content):
    # Regex to check for '---' at the start, followed by content, followed by '---'
    # Use re.DOTALL to match across newlines
    # yaml_regex = r'^\s*---\s*$.*?^\s*---\s*$'
    # return bool(re.match(yaml_regex, content, re.MULTILINE | re.DOTALL))


# Robust pattern to catch base64-encoded image data in Markdown
BASE64_PATTERN = re.compile(
    r"data:image[^;]*;base64,|]\(data:image/svg\+xml;base64"
)

def get_first_n_words(filepath: str, word_count: int = 100, min_len: int = 30) -> str:
    """Read file, skip lines with base64 images, return first N words."""
    filtered_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if description := get_description_yaml_metadata(f):
                return description
            else:
                f.seek(0)  # Reset to start of file
                for line in f:
                    if not BASE64_PATTERN.search(line):
                        filtered_lines.append(line)
                        words = " ".join(filtered_lines).split()
                        if len(words) >= word_count:
                            return " ".join(words[:word_count])
                content = " ".join(words[:word_count])
                return content if len(content) < min_len else None
    except Exception as e: 
        print(f"Skipping {filepath}: {e}")
        return None


def construct_md_json(root_dir: str = '.', file_path: str = 'indexed.json') -> None:
    md_map: dict[str, dict[str:str]] = {}

    # Load existing mapping if file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: md_map = json.load(f)
            print(f"Loaded existing mapping from: {file_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing mapping: {e}")
            md_map = {}
    
    abs_root = os.path.abspath(root_dir)

    print(f"Scanning for .md files starting at: {abs_root}")

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.md'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir).replace(os.sep, '/')
                if rel_path in md_map.keys(): continue

                snippet = get_first_n_words(full_path, 200)
                if snippet:
                    md_map[rel_path] = {"content": snippet}
                    # print(f"Indexed: {rel_path}")

    with open(file_path, 'w', encoding='utf-8') as out:
        json.dump(md_map, out, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Indexed {len(md_map)} files into '{file_path}'.")

if __name__ == "__main__":
    construct_md_json()