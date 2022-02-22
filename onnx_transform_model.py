from transformers import AutoTokenizer
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

convert(framework="pt",
        model="HoC_models/HoC_bert_multi_task/",  # CHANGED: refer to custom model
        tokenizer=tokenizer,  # <-- CHANGED: add tokenizer
        output=Path("onnx/HoC_bert_multi_task.onnx"),
        opset=12)
