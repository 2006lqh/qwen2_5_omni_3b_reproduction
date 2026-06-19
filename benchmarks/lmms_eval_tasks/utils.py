import ast
import re
from collections import defaultdict


def _librispeech_utils():
    from lmms_eval.tasks.librispeech import utils as u

    return u


def _omni_bench_utils():
    from lmms_eval.tasks.omni_bench import utils as u

    return u


def librispeech_doc_to_audio(doc):
    return _librispeech_utils().librispeech_doc_to_audio(doc)


def librispeech_doc_to_text(doc, lmms_eval_specific_kwargs):
    return _librispeech_utils().librispeech_doc_to_text(doc, lmms_eval_specific_kwargs)


def librispeech_process_result(doc, result):
    return _librispeech_utils().librispeech_process_result(doc, result)


def librispeech_wer(results, args):
    return _librispeech_utils().librispeech_wer(results, args)


def omni_bench_original_doc_to_visual(doc):
    return _omni_bench_utils().omni_bench_original_doc_to_visual(doc)


def omni_bench_original_doc_to_text(doc, lmms_eval_specific_kwargs):
    return _omni_bench_utils().omni_bench_original_doc_to_text(doc, lmms_eval_specific_kwargs)


def omni_bench_doc_to_choice(doc):
    return _omni_bench_utils().omni_bench_doc_to_choice(doc)


def omni_bench_doc_to_target(doc):
    return _omni_bench_utils().omni_bench_doc_to_target(doc)


def omni_bench_process_results(doc, results):
    return _omni_bench_utils().omni_bench_process_results(doc, results)


def omni_bench_aggregate_results(results):
    return _omni_bench_utils().omni_bench_aggregate_results(results)


def _coerce_options(options):
    if isinstance(options, str):
        try:
            return ast.literal_eval(options)
        except (SyntaxError, ValueError):
            return [item.strip() for item in options.split("\n") if item.strip()]
    return list(options)


def parse_options(options):
    choices = _coerce_options(options)
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))


def mmmu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    open_ended_prompt = kwargs.get("open_ended_prompt", "")
    question = doc["question"]
    options = parse_options(doc.get("options", []))

    if doc.get("question_type") == "multiple-choice":
        return f"{pre_prompt}{question}\nOptions:\n{options}\n{post_prompt}"
    return f"{pre_prompt}{question}\nOptions:\n{options}\n{open_ended_prompt}"


def mmmu_doc_to_visual(doc):
    tokens = sorted(set(re.findall(r"<image \d+>", doc.get("question", ""))))
    visuals = []
    for token in tokens:
        key = token.strip("<>").replace(" ", "_")
        image = doc.get(key)
        if image is not None:
            visuals.append(image.convert("RGB") if hasattr(image, "convert") else image)
    return visuals


def mmmu_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    content = []
    for image in mmmu_doc_to_visual(doc):
        content.append({"type": "image", "url": image})
    content.append({"type": "text", "text": mmmu_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def _extract_subset_name(sample_id):
    sample_id = str(sample_id)
    match = re.search(r"^[^_]+_(.+)_\d+$", sample_id)
    return match.group(1) if match else "unknown"


def _parse_choice_response(response, choices, options):
    text = f" {str(response).strip()} "
    candidates = []
    for choice in choices:
        if re.search(rf"[\s\(\[]({choice})[\s\)\]\.:,]", text, flags=re.IGNORECASE):
            candidates.append(choice)
    if candidates:
        return candidates[-1].upper()

    lower_text = str(response).lower()
    for choice, option in zip(choices, options):
        if str(option).lower() in lower_text:
            return choice
    return str(response).strip()[:1].upper()


def _normalize_open(value):
    value = str(value).strip().lower()
    return re.sub(r"\s+", " ", value)


def mmmu_process_results(doc, results):
    response = results[0] if results else ""
    if doc.get("question_type") == "multiple-choice":
        options = _coerce_options(doc.get("options", []))
        choices = [chr(ord("A") + i) for i in range(len(options))]
        parsed_pred = _parse_choice_response(response, choices, options)
    else:
        parsed_pred = _normalize_open(response)

    item = {
        "id": doc.get("id", ""),
        "subdomain": _extract_subset_name(doc.get("id", "")),
        "question_type": doc.get("question_type", ""),
        "answer": doc.get("answer", ""),
        "parsed_pred": [parsed_pred],
    }
    return {"mmmu_acc": item, "submission": {doc.get("id", ""): parsed_pred}}


def _is_mmmu_correct(item):
    answer = item.get("answer", "")
    preds = item.get("parsed_pred", [])
    if item.get("question_type") == "multiple-choice":
        gold = answer if isinstance(answer, list) else [answer]
        return any(str(pred).strip().upper() in {str(a).strip().upper() for a in gold} for pred in preds)

    gold = answer if isinstance(answer, list) else [answer]
    gold_norm = [_normalize_open(x) for x in gold]
    return any(g in _normalize_open(pred) for pred in preds for g in gold_norm if g)


def mmmu_aggregate_results(results):
    if not results:
        return 0.0
    by_subset = defaultdict(list)
    for item in results:
        by_subset[item.get("subdomain", "unknown")].append(item)

    total = 0
    correct = 0
    for items in by_subset.values():
        total += len(items)
        correct += sum(1 for item in items if _is_mmmu_correct(item))
    return round(correct / total, 5) if total else 0.0


def _get_column_value(doc, candidates):
    for candidate in candidates:
        if candidate in doc and doc[candidate] is not None:
            return doc[candidate]
    return ""


def _tokenize_mixed_text(text):
    tokens = []
    i = 0
    text = str(text).lower()
    while i < len(text):
        char = text[i]
        if "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
            i += 1
        else:
            match = re.match(r"[a-zA-Z']+\w*", text[i:])
            if match:
                tokens.append(match.group(0))
                i += match.end()
            else:
                i += 1
    return tokens


def _edit_distance(ref_tokens, hyp_tokens):
    rows, cols = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        dp[i][0] = i
    for j in range(cols + 1):
        dp[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[rows][cols]


def _compute_mer(ref, hyp):
    ref_tokens = _tokenize_mixed_text(ref)
    hyp_tokens = _tokenize_mixed_text(hyp)
    denom = max(len(ref_tokens), len(hyp_tokens))
    return _edit_distance(ref_tokens, hyp_tokens) / denom if denom else 0.0


def wenet_speech_doc_to_audio(doc):
    audio = _get_column_value(doc, ["audio"])
    if not audio:
        return []
    if isinstance(audio, dict):
        return [audio]
    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        array = decoded.data.cpu().numpy()
        return [{"array": array, "sampling_rate": decoded.sample_rate}]
    return [audio]


def wenet_speech_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = "Please listen to the audio and transcribe what you hear. Please only provide the transcription without any additional commentary. Do not include any punctuation."
    return f"{pre_prompt}{prompt}{post_prompt}"


def wenet_speech_process_results(doc, results):
    reference = _get_column_value(doc, ["text"])
    hypothesis = results[0] if results else ""
    if not reference:
        return {"mer": 100.0}
    return {"mer": _compute_mer(reference, hypothesis) * 100}
