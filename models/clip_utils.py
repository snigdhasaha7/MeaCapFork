import torch
import requests
from torch import nn
from PIL import Image


class CLIP(nn.Module):
    def __init__(self, model_name):
        super(CLIP, self).__init__()
        # model name: e.g. openai/vl_models-vit-base-patch32
        print('Initializing CLIP model...')
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        from transformers import CLIPTokenizerFast
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.cuda_has_been_checked = False
        print('CLIP model initialized.')

    def check_cuda(self):
        self.cuda_available = next(self.model.parameters()).is_cuda
        self.device = next(self.model.parameters()).get_device()
        if self.cuda_available:
            print('Cuda is available.')
            print('Device is {}'.format(self.device))
        else:
            device = "cpu"
            print('Cuda is not available.')
            print('Device is {}'.format(self.device))

    def device_convert(self, tar_device):
        self.device = tar_device
        self.model.to(self.device)
        print(f'CLIP Model moved to {tar_device}!')

    @torch.no_grad()
    def compute_image_representation_from_image_path(self, image_path):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        # if self.cuda_available:
        pixel_values = pixel_values.to(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)  # [1 x embed_dim]
        return image_embeds

    def get_clip_embedding(self, image_path):
        """
        Compute the CLIP embedding for a given image.

        Args:
            image_path (str): Path to the image.

        Returns:
            torch.Tensor: The CLIP embedding of the image.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        return image_embedding

    def compute_image_representation_from_image_instance(self, image):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        # if self.cuda_available:
        pixel_values = pixel_values.to(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)  # [1 x embed_dim]
        return image_embeds

    def compute_frame_representation_from_tensor(self, pixel_values):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass

        # if self.cuda_available:
        pixel_values = pixel_values.to(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)  # [1 x embed_dim]
        return image_embeds

    def compute_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=50, truncation=True)
        # self.tokenizer.max_len_single_sentence + 2 = 77
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        # if self.cuda_available:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds
    
    def compute_image_image_similarity_via_embeddings(self, query_image_path, candidate_image_paths, top_k=5, embedding_path=None):
        """
        Compute similarity scores between a query image and candidate images.
        Use precomputed embeddings if available.

        Args:
            query_image_path (str): Path to the query image.
            candidate_image_paths (list): List of paths to candidate images.
            top_k (int): Number of top similar images to retrieve.
            embedding_path (str): Path to save/load candidate image embeddings.

        Returns:
            tuple: (top_k_embeddings, query_embedding)
        """
        # Compute query embedding
        query_image = Image.open(query_image_path)
        query_inputs = self.processor(images=query_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            query_embedding = self.model.get_image_features(**query_inputs)

        # Load or compute candidate embeddings
        if embedding_path and os.path.exists(embedding_path):
            print(f"Loading precomputed embeddings from {embedding_path}.")
            candidate_embeddings = torch.load(embedding_path)
        else:
            candidate_embeddings = []
            for image_path in candidate_image_paths:
                inputs = self.processor(images=Image.open(image_path), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs)
                candidate_embeddings.append(embedding)
            candidate_embeddings = torch.stack(candidate_embeddings)
            if embedding_path:
                torch.save(candidate_embeddings, embedding_path)
                print(f"Candidate embeddings saved to {embedding_path}.")

        # Calculate similarity
        similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, candidate_embeddings, dim=-1)
        top_k_indices = torch.topk(similarity_scores, top_k).indices
        top_k_embeddings = candidate_embeddings[top_k_indices]
        return top_k_embeddings, query_embedding

    def save_image_embeddings(self, image_paths, embedding_path):
        """
        Save image embeddings for a directory of images.

        Args:
            image_paths (list): List of image paths.
            embedding_path (str): Path to save embeddings.

        Returns:
            None
        """
        if os.path.exists(embedding_path):
            print(f"Embeddings already exist at {embedding_path}. Skipping computation.")
            return

        embeddings = []
        for image_path in image_paths:
            embedding = self.get_clip_embedding(image_path)
            embeddings.append(embedding)

        stacked_embeddings = torch.stack(embeddings)
        torch.save(stacked_embeddings, embedding_path)
        print(f"Embeddings saved to {embedding_path}.")

    def load_image_embeddings(self, embedding_path):
        """
        Load precomputed image embeddings.

        Args:
            embedding_path (str): Path to embeddings file.

        Returns:
            torch.Tensor: Loaded embeddings.
        """
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embeddings file not found at {embedding_path}.")
        return torch.load(embedding_path)
    
    def compute_scores(self, image_path, captions):
        """
        Compute CLIP similarity scores between an image and a list of captions.
        Args:
            image_path: Path to the query image.
            captions: List of textual captions.
        Returns:
            Tensor of similarity scores.
        """
        image_embedding = self.get_clip_embedding(image_path)
        text_embeddings = torch.stack([self.compute_text_representation([caption]).squeeze(0) for caption in captions])
        scores = torch.nn.functional.cosine_similarity(image_embedding, text_embeddings, dim=-1)
        return scores

    def compute_image_text_similarity_via_embeddings(self, image_embeds, text_embeds):
        '''
            image_embeds: batch x embed_dim
            text_embeds: batch x len(text_list) x embed_dim
        '''
        text_embeds = text_embeds.view(image_embeds.shape[0], -1, text_embeds.shape[-1])
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds.unsqueeze(-1)
        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds) * logit_scale
        logits_per_image = logits_per_text.squeeze(-1)
        return logits_per_image.softmax(dim=1), logits_per_image/logit_scale  # batch x len(text_list)

    def compute_image_text_similarity_via_raw_text(self, image_embeds, text_list):
        text_embeds = self.compute_text_representation(text_list)
        return self.compute_image_text_similarity_via_embeddings(image_embeds, text_embeds)

    def compute_image_text_similarity_via_Image_text(self, image, text_list):
        image_embeds = self.compute_image_representation_from_image_instance(image)
        text_embeds = self.compute_text_representation(text_list)
        return self.compute_image_text_similarity_via_embeddings(image_embeds, text_embeds)

    ### -------------------- functions for building index ---------------------- ###
    def compute_batch_index_image_features(self, image_list):
        '''
            # list of image instances
        '''
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        inputs = self.processor(images=image_list, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        # if self.cuda_available:
        pixel_values = pixel_values.to(self.device)
        visual_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = visual_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds) # [1 x embed_dim]
        return image_embeds # len(image_list) x embed_dim

    def compute_batch_index_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # text_list: a list of text
        #text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt",
            max_length=self.tokenizer.max_len_single_sentence + 2, truncation=True)
        input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']
        # if self.cuda_available:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds
        #logit_scale = self.model.logit_scale.exp()
        #text_embeds = text_embeds * logit_scale
        #return text_embeds

