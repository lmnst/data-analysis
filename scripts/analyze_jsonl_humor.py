import os, re, json, sys, hashlib
from collections import Counter, defaultdict

def norm_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_word(text: str, w: str) -> bool:
    # case-insensitive, allow punctuation boundaries
    if not text or not w:
        return False
    t = text.lower()
    w2 = w.lower().strip()
    if not w2:
        return False
    # word boundary-ish: allow letters around? use regex boundaries for safety
    pattern = r"(?<!\w)" + re.escape(w2) + r"(?!\w)"
    return re.search(pattern, t) is not None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def extract_user_assistant(messages):
    user = None
    assistant = None
    if not isinstance(messages, list):
        return None, None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "user" and user is None:
            user = content
        if role == "assistant" and assistant is None:
            assistant = content
    return user, assistant

def main(path):
    base = os.path.basename(path)
    out_dir = os.path.join("outputs_jsonl", base.replace(".jsonl",""))
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    n_parse_err = 0
    n_missing_fields = 0
    n_missing_ua = 0
    n_empty_answer = 0
    n_short_answer = 0
    n_word_ok = 0

    # duplicates
    prompt_hashes = Counter()
    answer_hashes = Counter()
    pair_hashes = Counter()

    # length stats
    ans_lens = []
    prompt_lens = []

    bad_samples = []
    rag_docs = []
    sft_pairs = []

    # simple “red flag” keywords (very lightweight)
    red_flags = ["hitler","nazi","kkk","genocide","rape","suicide","kill yourself","terrorist"]

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_parse_err += 1
                continue

            messages = obj.get("messages")
            w1 = obj.get("word1")
            w2 = obj.get("word2")

            if messages is None or w1 is None or w2 is None:
                n_missing_fields += 1
                bad_samples.append({"line_no": line_no, "reason":"missing_fields", "raw": obj})
                continue

            user, assistant = extract_user_assistant(messages)
            user = norm_text(user)
            assistant = norm_text(assistant)

            if not user or not assistant:
                n_missing_ua += 1
                bad_samples.append({"line_no": line_no, "reason":"missing_user_or_assistant", "raw": obj})
                continue

            if len(assistant) == 0:
                n_empty_answer += 1
                bad_samples.append({"line_no": line_no, "reason":"empty_answer", "raw": obj})
                continue

            if len(assistant.split()) < 4:
                n_short_answer += 1  # not necessarily invalid, but low quality indicator

            ok1 = contains_word(assistant, str(w1))
            ok2 = contains_word(assistant, str(w2))
            word_ok = ok1 and ok2
            if word_ok:
                n_word_ok += 1
            else:
                bad_samples.append({
                    "line_no": line_no,
                    "reason": "missing_word1_or_word2_in_answer",
                    "word1": w1, "word2": w2,
                    "answer": assistant[:300],
                })

            # red-flag scan
            lower = assistant.lower()
            if any(rf in lower for rf in red_flags):
                bad_samples.append({"line_no": line_no, "reason":"red_flag_keyword", "answer": assistant[:300]})

            # stats
            prompt_lens.append(len(user))
            ans_lens.append(len(assistant))

            ph = sha1(user)
            ah = sha1(assistant)
            pph = sha1(user + "\n" + assistant)

            prompt_hashes[ph] += 1
            answer_hashes[ah] += 1
            pair_hashes[pph] += 1

            # RAG doc: index assistant joke as document
            rag_docs.append({
                "id": f"{base}:{line_no}",
                "text": assistant,
                "meta": {"word1": w1, "word2": w2, "prompt": user}
            })

            # SFT pair
            sft_pairs.append({
                "prompt": user,
                "response": assistant,
                "word1": w1,
                "word2": w2
            })

    def top_dup_rate(counter: Counter):
        if not counter:
            return 0.0
        dup = sum(c-1 for c in counter.values() if c > 1)
        total = sum(counter.values())
        return dup / total if total else 0.0

    summary = {
        "file": base,
        "n_lines": n,
        "parse_error": n_parse_err,
        "missing_fields": n_missing_fields,
        "missing_user_or_assistant": n_missing_ua,
        "empty_answer": n_empty_answer,
        "short_answer_<4words": n_short_answer,
        "word_constraint_pass": n_word_ok,
        "word_constraint_pass_rate": (n_word_ok / n) if n else None,
        "prompt_dup_rate": top_dup_rate(prompt_hashes),
        "answer_dup_rate": top_dup_rate(answer_hashes),
        "pair_dup_rate": top_dup_rate(pair_hashes),
        "prompt_len_mean": float(sum(prompt_lens)/len(prompt_lens)) if prompt_lens else None,
        "answer_len_mean": float(sum(ans_lens)/len(ans_lens)) if ans_lens else None,
        "answer_len_p50": int(sorted(ans_lens)[len(ans_lens)//2]) if ans_lens else None,
        "answer_len_p90": int(sorted(ans_lens)[int(len(ans_lens)*0.9)]) if ans_lens else None,
        "notes": [
            "word_constraint_pass_rate should be close to 1.0 for SFT",
            "answer_dup_rate/pair_dup_rate should be low for RAG usefulness",
        ]
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # save bad samples (cap to keep file manageable)
    with open(os.path.join(out_dir, "bad_samples.jsonl"), "w", encoding="utf-8") as f:
        for x in bad_samples[:2000]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "rag_corpus.jsonl"), "w", encoding="utf-8") as f:
        for x in rag_docs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "sft_pairs.jsonl"), "w", encoding="utf-8") as f:
        for x in sft_pairs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("Wrote:", out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_jsonl_humor.py <file.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
