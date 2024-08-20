# coding=utf8

import os
import logging
import yaml
import torch
import shutil
from os.path import join
from copy import deepcopy
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, BertModel

os.environ["WANDB_DISABLED"] = "true"
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
import torch.nn.functional as F
from loguru import logger

from src import (
    InBatchDataSet,
    in_batch_collate_fn,
    VecDataSet,
    SaveModelCallBack,
)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def get_vecs_e5(ipt):
            attention_mask = ipt["attention_mask"]
            model_output = model(**ipt)
            last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            vectors = F.normalize(vectors, 2.0, dim=1)
            return vectors

        # Step1 计算inbatch loss
        q_num = inputs[-1]
        name = inputs[0]
        inputs = inputs[1:-1]

        in_batch_loss = torch.tensor(0.0)
        vectors = [get_vecs_e5(ipt) for ipt in inputs]
        vectors = torch.cat(vectors, dim=0)
        vecs1, vecs2 = vectors[:q_num, :], vectors[q_num:, :]
        logits = torch.mm(vecs1, vecs2.t())

        LABEL = torch.LongTensor(list(range(q_num))).to(vectors.device)
        in_batch_loss = F.cross_entropy(logits * in_batch_ratio, LABEL)
        logger.info(f"step-{self.state.global_step}, {name}-loss:{in_batch_loss.item()}")

        return (in_batch_loss, None) if return_outputs else in_batch_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 读取参数并赋值
    config_path = 'conf.yml'
    with open(config_path, "r", encoding="utf8") as fr:
        conf = yaml.safe_load(fr)

    # args of hf trainer
    hf_args = deepcopy(conf["train_args"])
    in_batch_bsz = conf["in_batch_bsz"]
    if hf_args.get("deepspeed") and conf["use_deepspeed"]:
        hf_args["deepspeed"]["gradient_accumulation_steps"] = hf_args["gradient_accumulation_steps"]
        hf_args["deepspeed"]["train_micro_batch_size_per_gpu"] = hf_args["per_device_train_batch_size"]
        hf_args["deepspeed"]["optimizer"]["params"]["lr"] = hf_args["learning_rate"]
    else:
        hf_args.pop("deepspeed", None)

    model_name = conf["model_name"]

    grad_checkpoint = hf_args["gradient_checkpointing"]
    in_batch_train_paths = conf["in_batch_train_paths"]
    max_length = conf["max_length"]
    model_dir = conf["model_dir"]
    in_batch_ratio = conf["in_batch_ratio"]
    hard_neg_ratio = conf["hard_neg_ratio"]

    output_dir = hf_args["output_dir"]

    # 拷贝 config
    if local_rank == 0:
        if not os.path.exists(hf_args["output_dir"]):
            os.makedirs(hf_args["output_dir"], exist_ok=True)
    
        shutil.copy('conf.yml', os.path.join(output_dir, "train_config.yml"))
        # 初始化log
        logger.add(join(output_dir, "train_log.txt"), level="INFO", compression="zip", rotation="500 MB",
                   format="{message}")
    # in-batch 数据集
    in_batch_data_loaders = []
    if in_batch_train_paths:
        for data_name, data_paths in in_batch_train_paths.items():
            logger.info(f"添加数据迭代器，data_name:{data_name}, data_paths:{data_paths}")
            in_batch_data_loaders.append(
                DataLoader(
                    dataset=InBatchDataSet(data_paths=data_paths, data_name=data_name, model_name=model_name),
                    shuffle=True,
                    collate_fn=lambda x: in_batch_collate_fn(x, tokenizer, max_length),
                    drop_last=True,
                    batch_size=in_batch_bsz,
                    num_workers=2
                )
            )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = BertModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.train()

    torch_compile = torch.__version__.startswith("2")
    args = TrainingArguments(**hf_args, torch_compile=torch_compile, prediction_loss_only=True)
    trainer = MyTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=lambda x: x[0],
        train_dataset=VecDataSet(in_batch_data_loaders),
        callbacks=[SaveModelCallBack(output_dir=output_dir, local_rank=local_rank)]
    )
    trainer.train()
