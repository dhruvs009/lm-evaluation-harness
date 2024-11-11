import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>Answer the following question.<|eot_id|><|start_header_id|>user<|end_header_id|>{preprocess(doc["activity_label"] + ": " + ctx)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>''',
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
