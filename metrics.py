import json
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import torch
from PIL import Image
import clip
import os
from models.clip_utils import CLIP


def load_jsons():
    # Load MeaCap predictions
    with open('outputs/MeaCap_randtest2014_memory_coco_lmTrainingCorpus_CBART_COCO_0.1_0.8_0.2_k200.json', 'r') as f:
        meacap_preds = json.load(f)
    
    # Load ground truth annotations
    with open('data/coco/karpathy_test/annotations/dataset.json', 'r') as f:
        gt_data = json.load(f)
    
    return meacap_preds, gt_data

def prepare_data_for_metrics(meacap_preds, gt_data):
    # Create dictionaries for evaluation
    gts = {}
    res = {}
    
    # Create mapping of filename to ground truth captions
    gt_map = {}
    for img in gt_data['images']:
        filename = img['filename'].split('.')[0]  # Remove extension
        gt_map[filename] = [sent['raw'] for sent in img['sentences']]
    
    # Match predictions with ground truth
    for img_id, pred_caption in meacap_preds.items():
        img_key = img_id.split('.')[0]  # Remove extension if present
        if img_key in gt_map:
            gts[img_key] = gt_map[img_key]
            res[img_key] = [pred_caption]
    
    return gts, res

def calculate_metrics(gts, res):
    # Initialize tokenizer
    tokenizer = PTBTokenizer()
    
    # Tokenize ground truth and predictions
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    
    # Calculate BLEU scores (1-4)
    scorer = Bleu(n=4)
    bleu_scores, _ = scorer.compute_score(gts, res)
    
    # Calculate METEOR
    scorer = Meteor()
    meteor_score, _ = scorer.compute_score(gts, res)
    
    # Calculate CIDEr
    scorer = Cider()
    cider_score, _ = scorer.compute_score(gts, res)
    
    # Calculate SPICE
    scorer = Spice()
    spice_score, _ = scorer.compute_score(gts, res)
    
    return {
        'BLEU-1': bleu_scores[0],
        'BLEU-2': bleu_scores[1],
        'BLEU-3': bleu_scores[2],
        'BLEU-4': bleu_scores[3],
        'METEOR': meteor_score,
        'CIDEr': cider_score,
        'SPICE': spice_score
    }

def calculate_clip_score(meacap_preds):
    # Initialize CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIP(r'openai/clip-vit-base-patch32')
    clip_model.eval()
    clip_model.to(device)
    
    clip_scores = []
    
    for img_id, caption in meacap_preds.items():
        # Load image
        image_path = f'data/coco/karpathy_test/images/randtest2014/{img_id}.jpg'
        image = Image.open(image_path)
        
        # Calculate similarity
        with torch.no_grad():
            clip_score, _ = clip_model.compute_image_text_similarity_via_Image_text(
                image, 
                [caption]  # Pass caption as a list of one element
            )
            clip_scores.append(clip_score.item())

    
    return np.mean(clip_scores)

def main():
    # Load data
    meacap_preds, gt_data = load_jsons()
    
    # Prepare data for metrics
    gts, res = prepare_data_for_metrics(meacap_preds, gt_data)
    
    # # Calculate metrics
    metrics = calculate_metrics(gts, res)
    
    # Calculate CLIP score
    clip_score = calculate_clip_score(meacap_preds)
    metrics['CLIPScore'] = clip_score
    
    # Print results
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
