import requests, json, time
from functools import wraps

# This code embedes text data (from JSON) using a (local) API endpoint, saving the results to a JSON file. There are two decorators one for retry, and one for timeing the execution. We save after each successful embedding iteration to prevent data loss if something goes wrong. Finally we have a counter to stopp after a set batch size.

# The reason we are embedding one entry at time, and saving is for two main reasons
# 1. To avoid hanging up the system 
# 2. Accidental exit (lossing work)

# --- Helpers Decorator ---
def retry(exceptions: tuple[type[Exception], ...] | type[Exception], tries:int=3, delay:int=1, backoff:int=2):
    """A decorator to retry a function if it raises a specific exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    _tries -= 1
                    if _tries == 0:
                        raise
                    msg = f"Retrying in {_delay}s... ({_tries} tries left)"
                    print(f"Exception: {e}. {msg}")
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator

def timing(func):
    """A timing decorator compute execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Function {func.__name__!r} took {run_time:.4f} seconds")
        return result
    return wrapper
# ------------------------

EMBEDDING_MODELS = {
    "Qwen3-Embedding": {
        "full_model_name": "dengcao/Qwen3-Embedding-0.6B:Q8_0",
        "suffix": "<|endoftext|>",
        "task_prompts": {
            "clustering": "Identify the topic or theme of the given text",
            "retrieval_query": "Given a search query, retrieve relevant passages",
            "retrieval_doc": "",
        }
    },
}

def apply_embedding_template(content: str, model:str="Qwen3-Embedding", task: str = "clustering") -> str:
    config = EMBEDDING_MODELS[model]
    if not config: raise(f"{model} not found ")
    instruction = config["task_prompts"].get(task, "")
    suffix = config["suffix"]
    return f"Instruct: {instruction}\nQuery: {content}{suffix}" if instruction else  f"{content}{suffix}"


@retry(exceptions=requests.exceptions.Timeout)
def post(end_point_url:str, headers:dict, data:dict):
  # print(end_point_url, headers, data, sep="\n---\n")
  response = requests.post(end_point_url, headers=headers, json=data)
  # print("\n---\n", response.json())
  try:
    return response.json().get("embeddings", [None])[0]
  except json.JSONDecodeError as e:
    raise e(f"Error decoding JSON response for content starting with: {content[:20]}...")
  return response

def embed(content:str, model:str="Qwen3-Embedding", end_point_url:str = "http://localhost:11434/api/embed"):
  # Maybe update this to take a list, even if it's one, due to standard (avoid deviation)
  if not isinstance(content, str): 
    raise TypeError(f"content must be str got {type(content)!r}")
  if model not in EMBEDDING_MODELS.keys(): 
    raise KeyError(f"Specified model not in available models: {EMBEDDING_MODELS.keys()}")

  full_model_name = EMBEDDING_MODELS.get(model).get("full_model_name")
  content = apply_embedding_template(content)
  headers = {"Content-Type": "application/json"}
  data = {"model": full_model_name, "input": content }
  embeddings = post(end_point_url, headers=headers, data=data)
  return embeddings

