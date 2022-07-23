import os, json
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
from utils import CommentGenDataset, SimpleGenDataset
from evaluator.smooth_bleu import bleu_fromstr


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
        dataset = SimpleGenDataset(tokenizer, pool, args, data_file)
    else:
        dataset = CommentGenDataset(tokenizer, pool, args, data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_epoch_bleu(args, eval_dataloader, model, tokenizer):
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []
    for step, examples in tqdm(enumerate(eval_dataloader, 1)):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(source_ids,
                            attention_mask=source_mask,
                            use_cache=True,
                            num_beams=args.beam_size,
                            early_stopping=True,
                            max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
        if args.break_cnt > 0 and len(pred_ids) >= args.break_cnt:
            break
    # [2:] to remove beginning '<s>' '<msg>'
    pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    valid_file = args.eval_file
    out_file = args.out_file
    outdics = []
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            outdics.append(json.loads(line))
            golds.append(outdics[-1]["msg"])
    outdics = outdics[:len(pred_nls)]
    golds = golds[:len(pred_nls)]
    with open(os.path.join(args.model_name_or_path, "preds.txt"), "w", encoding="utf-8") as f:
        for pred in pred_nls:
            f.write(pred.strip() + "\n")
    with open(os.path.join(args.model_name_or_path, "golds.txt"), "w", encoding="utf-8") as f:
        for gold in golds:
            f.write(gold.strip() + "\n")
    with open(out_file, "w", encoding="utf-8") as f:
        for i, outdic in enumerate(outdics):
            outdic["gen"] = pred_nls[i]
            f.write(json.dumps(outdic) + "\n")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    return bleu


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
    bleu = eval_epoch_bleu(args, dataloader, model, tokenizer)
    logger.warning(f"BLEU: {bleu}")

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
