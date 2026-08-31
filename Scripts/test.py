import json

path = "data/processed/chunks/chunks_Cours_Oracle_.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)

        if "proc" in chunk["text"].lower():
            print("TROUVE:")
            print(chunk["text"])