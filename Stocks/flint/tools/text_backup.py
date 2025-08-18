import os
from pathlib import Path
from typing import Union, List, Tuple

def convert_files_to_text(
    root_dir: str,
    output_dir: str,
    include_subdirs: Union[bool, List[str]] = True,
    exclude_dirs: List[str] = None,
    extensions: Tuple[str, ...] = ('.py', '.yml', '.toml'),
    exclude_files: List[str] = None
):
    """
    Convert files to text with flexible directory processing options.
    
    Args:
        root_dir: Directory containing source files
        output_dir: Where to save text versions
        include_subdirs: True=all, False=none, List=specific subdirs + root
        exclude_dirs: Directories to always exclude
        extensions: File extensions to process
        exclude_files: Specific filenames to exclude (including extensions)
    """
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Default exclude directories and files
    if exclude_dirs is None:
        exclude_dirs = [".venv", ".vscode", ".git", "__pycache__"]
    if exclude_files is None:
        exclude_files = [".python-version", "*.ipynb", "*.md"]
    if extensions is None:
        extensions = '*'
    
    # Prepare files to process
    files_to_process = []
    
    # Always include root directory files
    files_to_process.extend(Path(root_dir).glob('*'))
    
    # Add subdirectories if specified
    if isinstance(include_subdirs, list):
        for subdir in include_subdirs:
            subdir_path = Path(root_dir) / subdir
            if subdir_path.exists():
                files_to_process.extend(subdir_path.rglob('*'))
    elif include_subdirs is True:
        # Include all subdirectories recursively
        files_to_process.extend(Path(root_dir).rglob('*'))
    
    # Track output filenames to prevent overwrites
    output_files = set()
    
    # NEW: Handle wildcard extension logic
    process_all_files = False
    if extensions == '*' or extensions == ('*',) or extensions is None:
        process_all_files = True
    
    for filepath in files_to_process:
        # Skip directories and excluded files
        if (not filepath.is_file() or 
            any(excluded in filepath.parts for excluded in exclude_dirs) or
            filepath.name in exclude_files):
            continue
            
        # MODIFIED: Extension check logic
        if process_all_files or filepath.suffix.lower() in extensions:
            # Get relative path to maintain structure
            rel_path = filepath.relative_to(root_dir)
            
            # Create unique output filename that preserves original extension
            output_filename = f"{rel_path.stem}_{rel_path.suffix[1:]}.txt"
            output_path = Path(output_dir) / rel_path.with_name(output_filename)
            
            # Ensure we don't overwrite files
            if output_path in output_files:
                print(f"Warning: Skipping potential overwrite of {output_path}")
                continue
                
            output_files.add(output_path)
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read and write the file
            try:
                with open(filepath, 'r', encoding='utf-8') as infile, \
                     open(output_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(infile.read())
                
                print(f"Converted: {filepath} → {output_path}")
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")


# Example usage
if __name__ == "__main__":
    convert_files_to_text(
        root_dir="/workspaces/smart_dev/projects/Notebooks/Stocks",
        output_dir="/workspaces/smart_dev/projects/Notebooks/Stocks/text_output",
        include_subdirs=['flint'],
        extensions=None,
        exclude_dirs=[".venv", ".vscode", ".git", "__pycache__", ".ruff_cache", "data", "results", "ml-unified"],
        exclude_files=[".python-version", "local_settings.py", "stock_algo.ipynb", "ruff_check.log", "README.md"]  # Additional excludes
    )