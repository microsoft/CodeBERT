import os
import torch
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import CommentClsDataset, SimpleClsDataset
from sklearn.metrics import classification_report


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Start data file {data_file}.")
    if args.raw_input:
        dataset = SimpleClsDataset(tokenizer, pool, args, data_file)
    else:
        dataset = CommentClsDataset(tokenizer, pool, args, data_file)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_epoch_acc(args, eval_dataloader, model, tokenizer):
    # Start evaluating model
    logger.info("  " + "***** Running acc evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    local_rank = 0
    pred, gold = [], []
    with torch.no_grad():
        for step, examples in enumerate(tqdm(eval_dataloader), 1):
            if step == 1:
                ex = examples[0]
                logger.info(f"batch size: {len(examples)}")
                logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                logger.info(f"example target: {ex.y}")
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(local_rank)
            source_mask = source_ids.ne(tokenizer.pad_id)
            logits = model(
                cls=True,
                input_ids=source_ids,
                labels=None,
                attention_mask=source_mask
            )
            prediction = torch.argmax(logits, dim=-1).cpu().numpy()
            pred.extend(prediction)
            gold.extend([ex.y for ex in examples])
    logger.info("\n" + classification_report(gold, pred, digits=4))
    logger.info(f"Target positive percentage: {sum(gold) / len(gold)}")
    return


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.eval_batch_size)
    torch.cuda.set_device(local_rank)

    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)
    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    model.eval()
    eval_epoch_acc(args, dataloader, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Test finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
