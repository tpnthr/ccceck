from typing import List, Dict

from config import MAX_PAUSE


def     group_words(words: List[Dict]) -> List[Dict]:
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

def render_stereo_dialogue_lines(grouped_dialogue):
    # format: [timestamps (start-end)] (speaker) - phrase
    lines = []
    for entry in grouped_dialogue:
        start = f"{entry['start']:.2f}"
        end = f"{entry['end']:.2f}"
        speaker = entry['speaker']
        phrase = " ".join(entry['text'])
        lines.append(f"[{start}-{end}] ({speaker}) - {phrase}")
    return lines

def render_mono_dialogue_lines(words):
    # Group words by segment/time if needed. If not, just one full dialogue line.
    lines = []
    if all(k in words[0] for k in ("start", "end", "word")):
        phrases = []
        cur_start, cur_end = words[0]["start"], words[0]["end"]
        for w in words:
            phrases.append(w["word"])
            cur_end = w["end"]
        phrase_text = " ".join(phrases)
        lines.append(f"[{cur_start:.2f}-{cur_end:.2f}] - {phrase_text}")
    else:
        phrase_text = " ".join([w["word"] for w in words])
        lines.append(f"(speaker) - {phrase_text}")
    return lines
