from typing import List, Dict

from app import MAX_PAUSE


def group_words(words: List[Dict]) -> List[Dict]:
    grouped, current = [], {"speaker": None, "start": None, "end": None, "text": []}

    for w in sorted(words, key=lambda x: x["start"]):
        spk, start, end, txt = w["speaker"], w["start"], w["end"], w["word"]
        if current["speaker"] != spk or (current["end"] and start - current["end"] > MAX_PAUSE):
            if current["text"]:
                grouped.append(current)
            current = {"speaker": spk, "start": start, "end": end, "text": [txt]}
        else:
            current["end"] = end
            current["text"].append(txt)

    if current["text"]:
        grouped.append(current)

    return grouped