from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import SimpleDirectoryReader
import os
import torch
from typing import Any, List
from pydantic import PrivateAttr
from transformers import BertModel, AutoTokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding

class EmbedModel(BaseEmbedding):
    _device : Any = PrivateAttr()
    _token : Any = PrivateAttr()
    _model : Any = PrivateAttr()

    def __init__(self, model_path):
        super().__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = BertModel.from_pretrained(model_path).to(self._device)
        self._token = AutoTokenizer.from_pretrained(model_path)
        self._model.eval()

    async def _aget_query_embedding(self, query: str):
        pass

    def _get_query_embedding(self, query: str):
        pass

    def _get_text_embedding(self, text: str):
        pass

    @torch.no_grad()
    def get_text_embedding_batch(self, texts: List[str], batch_size=512, max_length=1024, **kwargs: Any) -> List[List[float]] :
        if isinstance(texts, str): texts = [texts]
        
        vector_buff = []
        for pos in range(0, len(texts), batch_size):
            inputs = self._token.batch_encode_plus(
                texts[pos: pos + batch_size], 
                padding="longest", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(self._device)

            model_output = self._model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            vectors = torch.nn.functional.normalize(vectors, 2.0, dim=1)
            vector_buff.append(vectors.to('cpu'))
        vectors = torch.vstack(vector_buff)
        return vectors.tolist()
 


os.environ["OPENAI_API_KEY"] = "sk-..."
embed_model = EmbedModel("/home/guodewen/research/IRTrain/models/stella-large-zh-v2")
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=40, embed_model=embed_model, validata_assignment=False
)

documents = SimpleDirectoryReader(input_files=["temp.txt"]).load_data()
# also baseline splitter
base_splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
for node in nodes:
    print(node.get_content(), end="\n--------------------------------\n")