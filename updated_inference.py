# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

import time
import os
import sys
from args import get_args
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer

from models.clip_utils import CLIP, compute_image_image_similarity_via_embeddings  # Ensure this exists
import json
import copy

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.log import Logger
from src.transformers import BartForTextInfill, BartTokenizer
from language_models.language_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset.ImgDataset import Imgdata, collate_img
from dataset.ImgDataset_img_return import Imgdata_img_return, collate_img_img_return

from utils.some_utils import set_seed, update_args_logger, PROMPT_ENSEMBLING
from utils.detect_utils import retrieve_concepts
from utils.generate_utils_ import Get_shuffle_score, filter_text


def compute_top_n_texts(image_embedding, memory_embeddings, memory_captions, top_n):
    """
    Compute the top-N textual descriptions for a given image embedding.
    Args:
        image_embedding: Embedding of the image (tensor).
        memory_embeddings: All memory embeddings (tensor).
        memory_captions: List of captions corresponding to memory embeddings.
        top_n: Number of top captions to retrieve.
    Returns:
        A list of top-N captions.
    """
    similarity_scores = torch.nn.functional.cosine_similarity(
        image_embedding, memory_embeddings, dim=-1
    )
    top_n_indices = torch.topk(similarity_scores, top_n).indices
    return [memory_captions[idx] for idx in top_n_indices]


class LLMRefiner:
    """
    A class to refine captions iteratively using an LLM.
    """
    def __init__(self, llm_model, tokenizer):
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def refine_captions(self, captions):
        """
        Use the LLM to refine a list of captions.
        Args:
            captions: List of captions to refine.
        Returns:
            Refined captions.
        """
        refined_captions = []
        for caption in captions:
            prompt = f"Improve the caption: '{caption}'."
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            with torch.no_grad():
                outputs = self.llm_model.generate(**inputs, max_new_tokens=50)
            refined_caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            refined_captions.append(refined_caption)
        return refined_captions


if __name__ == "__main__":
    args = get_args()
    set_seed(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    ## Update logger
    memory_id = args.memory_id
    test_datasets = args.img_path.split("/")[-1]
    lm_training_datasets = args.lm_model_path.split("/")[-1]
    save_file = f'MeaCap_{test_datasets}_memory_{memory_id}_lmTrainingCorpus_{lm_training_datasets}_{args.alpha}_{args.beta}_{args.gamma}_k{args.conzic_top_k}.json'
    log_file = f'MeaCap_{test_datasets}_memory_{memory_id}_lmTrainingCorpus_{lm_training_datasets}_{args.alpha}_{args.beta}_{args.gamma}_k{args.conzic_top_k}.log'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_path = os.path.join(args.output_path, log_file)
    logger = Logger(log_path)
    logger.logger.info(f'The log file is {log_path}')
    logger.logger.info(args)

    ## Load models
    tokenizer = BartTokenizer.from_pretrained(args.lm_model_path)
    lm_model = BartForTextInfill.from_pretrained(args.lm_model_path).to(device)
    vl_model = CLIP(args.vl_model).to(device)
    wte_model = SentenceTransformer(args.wte_model_path).to(device)
    parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint).eval().to(device)

    ## Dataset loading
    if args.use_memory:
        img_data = Imgdata_img_return(dir_path=args.img_path, match_model=vl_model)
        train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img_img_return, shuffle=False, drop_last=False)
    else:
        img_data = Imgdata(dir_path=args.img_path, match_model=vl_model)
        train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=False)

    ## Memory bank setup
    if args.use_memory:
        memory_caption_path = os.path.join(f"data/memory/{memory_id}", "memory_captions.json")
        memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
        memory_captions = json.load(open(memory_caption_path, 'r'))
        memory_clip_embeddings = torch.load(memory_clip_embedding_file)

    # Handle large memory banks
    if memory_id in ['cc3m', 'ss1m']:
        retrieve_on_CPU = True
        vl_model_retrieve = copy.deepcopy(vl_model).to(cpu_device)
        memory_clip_embeddings = memory_clip_embeddings.to(cpu_device)
    else:
        retrieve_on_CPU = False
        vl_model_retrieve = vl_model

    result_dict = {}
    for batch_idx, (batch_image_embeds, batch_name_list, batch_img_list) in enumerate(train_loader):
        start = time.time()
        logger.logger.info(f'{batch_idx + 1}/{len(train_loader)}, image name: {batch_name_list[0]}')

        # Top-k image retrieval
        top_k_image_paths = vl_model_retrieve.compute_image_image_similarity_via_embeddings(
            query_image_path=batch_img_list[0],
            candidate_image_paths=img_data.get_all_image_paths(),
            top_k=args.top_k
        )

        # Retrieve top-N captions
        all_retrieved_texts = []
        for image_path in top_k_image_paths:
            top_n_texts = compute_top_n_texts(
                image_embedding=vl_model.get_clip_embedding(image_path),
                memory_embeddings=memory_clip_embeddings,
                memory_captions=memory_captions,
                top_n=args.memory_caption_num
            )
            all_retrieved_texts.extend(top_n_texts)

        query_texts = compute_top_n_texts(
            image_embedding=batch_image_embeds,
            memory_embeddings=memory_clip_embeddings,
            memory_captions=memory_captions,
            top_n=args.memory_caption_num
        )
        all_retrieved_texts.extend(query_texts)

        # Combine and refine
        llm_refiner = LLMRefiner(llm_model=lm_model, tokenizer=tokenizer)
        refined_captions = llm_refiner.refine_captions(all_retrieved_texts)
        result_dict[os.path.splitext(batch_name_list[0])[0]] = refined_captions

        logger.logger.info(f'Completed processing image: {batch_name_list[0]} in {time.time() - start:.2f}s.')

    save_file = os.path.join(args.output_path, save_file)
    with open(save_file, 'w', encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
