import logging
import time
from abc import ABC

import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

from transformers_neuronx.opt.model import OPTForSampling

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class LLMHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(LLMHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the HF large model is loaded and
        partitioned into multiple stages each on one device using PiPPy.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        
        # settings for model compiliation and loading
        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        batch_size = ctx.model_yaml_config["handler"]["batch_size"]
        tp_degree = ctx.model_yaml_config["handler"]["tp_degree"]
        amp = ctx.model_yaml_config["handler"]["amp"]
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
        
        logger.info("Starting to compile the model")

        self.model = OPTForSampling.from_pretrained(model_dir, batch_size=batch_size, tp_degree=tp_degree, amp=amp)
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")


        self.max_length = ctx.model_yaml_config["handler"]["max_length"]



        self.initialized = True

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_ids_batch, attention_mask_batch = [], []
        for input_text in input_texts:
            input_ids, attention_mask = self.encode_input_text(input_text)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
        input_ids_batch = torch.cat(input_ids_batch, dim=0)
        attention_mask_batch = torch.cat(attention_mask_batch, dim=0)
        return input_ids_batch, attention_mask_batch

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        input_ids_batch, attention_mask_batch = input_batch
        input_ids_batch = input_ids_batch
        outputs = self.model.generate(
            input_ids_batch,
            attention_mask=attention_mask_batch,
            max_length=30,
        )

        inferences = [
            self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]
        logger.info("Generated text: %s", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output