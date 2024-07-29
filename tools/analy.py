import jsonlines

qset = set()
dset = set()
task_id = set()
# with jsonlines.open('/home/guodewen/research/dataset/train_shuf.jsonl', 'r') as f:
with jsonlines.open('/data/public/data_20240710/dataset/medi2-data-jsonl/cool_0621/train.jsonl', 'r') as f:
    for obj in f:
        qset.add(obj['query'][-1])
        dset |= set([obj['pos'][-1]] + obj['neg'][1:])
        task_id.add(obj['task_id'])
print(f"query nums {len(qset)}")
print(f"docs nums {len(dset)}")
print(len(qset | dset))
print(f'task_id num: {len(task_id)}')