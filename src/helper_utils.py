import os

def expand_full_path(path:str):
  return os.path.expanduser(path) if path.startswith("~") else os.path.abspath(path)

def expand_full_path_and_ensure_file_exist(file_path:str):
  file_path = expand_full_path(file_path)
  if not os.path.exists(file_path): 
    raise FileNotFoundError(f"File: {file_path!r} doesn't exist")
  return file_path

