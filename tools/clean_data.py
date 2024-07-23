import logging
import jsonlines
from random import randint, shuffle
import torch
from tqdm import tqdm
from index import ModelEmbed

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)  


def clean_data_intbatch_false_neg(src_train_file, dst_train_file, batch_size=256):
    ''' batch_size: 这里指的是多机多卡的batch_size， 假如说每张卡batch_size=128, 有两张卡，那么这里的batch_size=256
    '''
    with jsonlines.open(src_train_file, "r") as reader:
        src_dataset = list(reader)
        src_size = len(src_dataset)
    logging.info(f"train_data size: {src_size}")

    encoder = ModelEmbed("/home/guodewen/research/models/stella_v2_large/", device="cuda")

    dst_dataset = []
    src_dataset.sort(key=lambda x: x["task_id"])

    for pos in tqdm(range(0, src_size, batch_size)):
        batch = src_dataset[pos: pos + batch_size]

        # 先按照字面意思去重过滤，避免伪负例，如果遇到相同的就从后面随机抽一个替换，降低冲突概率
        uniq_text = set()
        for cur, item in enumerate(batch):
            # q, pd, nd = item["query"][-1], item["pos"][-1], set(item["neg"][1:])
            cand_text = set([item["query"][-1], item["pos"][-1]] + item["neg"][1:])
            # while text & uniq_text:
            if cand_text & uniq_text:
                idx = randint(pos + batch_size, min(pos + batch_size * 2, src_size) - 1)
                batch[cur], src_dataset[idx] = src_dataset[idx], batch[cur]
                # q, pd, nd = item["query"][-1], item["pos"], set(item["neg"])
                logging.info(f"swap idx {cur}-{idx} in phase 1")
            
            uniq_text |= cand_text
        
        # 再按照向量得分过滤，避免伪负例, 如果遇到就从后面随机抽一个替换，降低冲突概率
        q =  [item["query"][-1] for item in batch]
        pd = [item["pos"][-1] for item in batch]
        for item in batch:
            pd.extend(item["neg"][1:])
        q_emb = encoder.emdbed(q)
        pd_emb = encoder.emdbed(pd)

        score_matrix = q_emb @ pd_emb.T

        # 将每一列填充为第一列元素
        max_score = score_matrix.max(dim=-1)
        std_score = torch.diag(score_matrix)

        flags = torch.nonzero(max_score > std_score + 0.1).squeeze().tolist()
        for cur in flags:
            idx = randint(pos + batch_size, min(pos + batch_size * 2, src_size) - 1)
            batch[cur], src_dataset[idx] = src_dataset[idx], batch[cur]
            logging.info(f"swap idx {cur}-{idx} in phase 2")
        dst_dataset.append(batch)
        
    shuffle(dst_dataset)

    with jsonlines.open(dst_train_file, 'w') as writer:
        for item in dst_dataset:
            writer.write(item)


clean_data_intbatch_false_neg("/data/public/data_20240710/dataset/medi2-data-jsonl/cool_0621/train.jsonl", "/home/guodewen/research/dataset/train.jsonl")
