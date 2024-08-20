from transformers import AutoTokenizer, AutoModel, BertModel
import torch
from typing import List

class BGEEmbed(object):
    ''' 使用模型将文本表征为向量
    '''
    def __init__(self, path, device='cuda', batch_size = 256) -> None:
        self.token = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = BertModel.from_pretrained(path, trust_remote_code=True, device_map=device)
        self.device = self.model.device
        self.batch_size = batch_size

    def emdbed(self, text: List[str], max_length: int = 1024) -> torch.Tensor :
        if isinstance(text, str): text = [text]
        
        vector_buff = []
        # 当batch size 过大的时候分批计算
        for pos in range(0, len(text), self.batch_size):
            prein = self.token.batch_encode_plus(
                text[pos: pos + self.batch_size], 
                padding="longest", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**prein)
                model_output = model_output[0][:, 0]
                vectors = torch.nn.functional.normalize(model_output, 2.0, dim=1)
            vector_buff.append(vectors.to('cpu'))
        vectors = torch.vstack(vector_buff)
        return vectors.to(self.device)


if __name__ == '__main__':
    # Sentences we want sentence embeddings for
    sentences = ["下面介绍合肥天气：合肥晴天，阴转多云", "合肥天气介绍"]

    encode = BGEEmbed('/home/guodewen/research/IRTrain/models/bge-large-en-v1.5', device='cpu')
    a = encode.emdbed(sentences)
    print(a[0,:] @ a[1,:])
    exit()

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('/home/guodewen/research/IRTrain/models/bge-large-en-v1.5')
    model = AutoModel.from_pretrained('/home/guodewen/research/IRTrain/models/bge-large-en-v1.5')
    model.eval()

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print("Sentence embeddings:", sentence_embeddings)
    print(sentence_embeddings[0, :] @ sentence_embeddings[1, :])
    print(sentence_embeddings.shape)
