import os, json, sys, hashlib, re

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def extract_user_assistant(messages):
    user = None
    assistant = None
    for m in messages:
        if isinstance(m, dict):
            if m.get("role") == "user" and user is None:
                user = m.get("content")
            if m.get("role") == "assistant" and assistant is None:
                assistant = m.get("content")
    return norm(user), norm(assistant)

def main(path):
    base = os.path.basename(path)
    out_dir = os.path.join("outputs_jsonl", base.replace(".jsonl",""))
    os.makedirs(out_dir, exist_ok=True)

    seen_ans = set()
    kept = 0
    total = 0
    dropped = 0

    out_path = os.path.join(out_dir, base.replace(".jsonl","") + "_dedupByAnswer.jsonl")

    with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            user, assistant = extract_user_assistant(obj.get("messages", []))
            h = sha1(assistant.lower())
            if h in seen_ans:
                dropped += 1
                continue
            seen_ans.add(h)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input: {path}")
    print(f"Output: {out_path}")
    print(f"Total: {total}, Kept: {kept}, Dropped: {dropped}, KeptRate: {kept/total:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/dedup_jsonl_by_answer.py <file.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
