import json

from docarray import BaseDoc
from docarray.typing import NdArray


class ToyDoc(BaseDoc):
    text: str = ""
    embedding: NdArray[1536]  # NdArray[4096], 1536 for open ai
    reference: str = ""
    title: str = ""
    article: str = ""
    
    
class CustomDataset:
    def __init__(self, file_path):
        # Load the JSON data from the file
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        all_data_values = [(i["text"], i["reference"],i["article"],i["title"]) for i in data]

        self.documents_data = all_data_values