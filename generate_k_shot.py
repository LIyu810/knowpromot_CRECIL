import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import json
from collections import Counter, OrderedDict
import logging
logger = logging.getLogger(__name__)




def get_labels(path, name,  negative_label="no_relation"):
    """See base class."""

    count = Counter()
    with open(path + "/" + name, "r") as f:
        features = []
        # for line in f.readlines():
        #      line = line.rstrip()
        #      if len(line) > 1:
        #         # count[line['relation']] += 1
        #         features.append(eval(line))
        features = json.load(f)




    # logger.info("label distribution as list: %d labels" % len(count))
    # # Make sure the negative label is alwyas 0
    #     labels = []
    # for label, count in count.most_common():
    #     logger.info("%s: %d 个 %.2f%%" % (label, count,  count * 100.0 / len(dataset)))
    #     if label not in labels:
    #         labels.append(label)
    return features

# list 转成Json格式数据
def listToJson(data):
    list_json = data
    str_json = json.dumps(list_json, indent=4, ensure_ascii=False)              # json转为string
    return str_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    # parser.add_argument("--task", type=str, nargs="+",
    #     default=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'],
    #     help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[14,15,16],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="/workspace/knowprompt/", help="Path to original data")
    parser.add_argument("--dataset", type=str, default="/workspace/knowprompt/dataset/chinese_dialogue", help="Path to original data")
    parser.add_argument("--data_file", type=str, default='train.json', choices=['train.txt', 'val.txt'], help="k-shot or k-shot-10x (10x dev set)")

    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'], help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()

    path = os.path.join(args.data_dir, args.dataset)

    output_dir = os.path.join(path, args.mode)
    dataset = get_labels(path, args.data_file)

    for seed in args.seed:

        # Other datasets
        np.random.seed(seed)
        np.random.shuffle(dataset)

        # Set up dir
        k = args.k
        setting_dir = os.path.join(output_dir, f"{k}-{seed}")
        os.makedirs(setting_dir, exist_ok=True)

        # label_list = {}
        # for line in dataset:
        #     label = line['relation']
        #     if label not in label_list:
        #         label_list[label] = [line]
        #     else:
        #         label_list[label].append(line)
# train ---> 488   dev ---> 112     test ---> 79
        with open(os.path.join(setting_dir, "train.json"), "w") as f:
            # file_list = []
            # for label in label_list:
            #     for line in label_list[label][:k]:  # train中每一类取前k个数据
            # for data in dataset[:k]:
            data = dataset[:k]
            data = listToJson(data)
            f.write(data)   # 将line列表对象转为json字符串
            f.write('\n')

            f.close()





if __name__ == "__main__":
    main()

