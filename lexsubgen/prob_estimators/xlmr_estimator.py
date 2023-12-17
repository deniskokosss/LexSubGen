import json
import os
import string
from collections import defaultdict
from pathlib import Path
from string import punctuation
from typing import NoReturn, Dict, List, Tuple, Literal

import numpy as np
import torch
from fairseq import utils
from overrides import overrides
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM

from fairseq.models.roberta import XLMRModel, RobertaHubInterface, RobertaModel
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

from lexsubgen.prob_estimators import BaseProbEstimator
from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator


class XLMRProbEstimatorMultimasked(BaseProbEstimator):
    loaded = defaultdict(dict)

    def __init__(
        self,
        num_masks: int = 1,
        topk: int = 10,
        model_name: str = "xlmr.large",
        cuda_device: int = 0,
        verbose: bool = False,
        dynamic_pattern: str = '<M>',
        decoding_type: Literal['greedy', 'cwm'] = 'greedy'
    ):
        """
        Probability estimator based on the Roberta model.
        See Y. Liu et al. "RoBERTa: A Robustly Optimized
        BERT Pretraining Approach".

        Args:
            mask_type: the target word masking strategy.
            model_name: Roberta model name, see https://github.com/huggingface/transformers
            embedding_similarity: whether to compute BERT embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            use_attention_mask: whether to zero out attention on padding tokens
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            unk_word_embedding: how to handle words that are splitted into multiple subwords when computing
            embedding similarity
            verbose: whether to print misc information
        """
        super(XLMRProbEstimatorMultimasked, self).__init__(
            verbose=verbose,
        )
        self.model_name = model_name
        self.num_masks = num_masks
        self.topk = topk
        self.prev_word2id = {}
        self.dp = dynamic_pattern
        self.dec_type = decoding_type

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "xlmr",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_nums": self.num_masks
            }
        }

        self.register_model()

        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

    @property
    def tokenizer(self) -> SentencepieceBPE:
        """
        Model tokenizer.

        Returns:
            `transformers.RobertaTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["model"].bpe

    @property
    def model(self) -> RobertaHubInterface:
        return self.loaded[self.model_name]["model"]

    @property
    def parameters(self):
        parameters = f"{self.num_masks}{self.model_name}"

        return parameters

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in XLMRProbEstimatorMultimasked.loaded:

            if self.model_name in ['xlmr.large', 'xlmr.base']:
                model = torch.hub.load('pytorch/fairseq', self.model_name)
            else:
                p = Path(self.model_name)
                model = XLMRModel.from_pretrained(str(p.parent), checkpoint_file=p.name)
            model = model.to(self.device).eval()

            XLMRProbEstimatorMultimasked.loaded[self.model_name] = {
                "model": model,
            }
            XLMRProbEstimatorMultimasked.loaded[self.model_name]["ref_count"] = 1
        else:
            XLMRProbEstimatorMultimasked.loaded[self.model_name]["ref_count"] += 1

    def tokenize_around_target(
        self,
        tokens: List[str],
        target_idx: int,
        tokenizer: SentencepieceBPE = None,
    ):
        assert len(self.dp.split('<M>')) == 2
        left_dp, right_dp = self.dp.split('<M>')
        left_ctx = ' '.join(tokens[:target_idx]) + ' ' + left_dp
        if '<T>' in left_ctx:
            left_ctx = left_ctx.replace('<T>', tokens[target_idx])
        left_ctx_tokens = self.model.task.source_dictionary.encode_line(
            '<s> ' + tokenizer.encode(left_ctx), append_eos=False, add_if_not_exist=False,
        )

        right_ctx = right_dp + ' ' + ' '.join(tokens[target_idx + 1:])
        if '<T>' in right_ctx:
            right_ctx = right_ctx.replace('<T>', tokens[target_idx])
        right_ctx_tokens = self.model.task.source_dictionary.encode_line(
            tokenizer.encode(right_ctx), append_eos=True, add_if_not_exist=False,
        )

        target_start_idx = len(left_ctx_tokens)

        return left_ctx_tokens, right_ctx_tokens, target_start_idx

    def prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        num_masks: int,
        tokenizer: XLMRobertaTokenizer = None,
    ):
        """
        Prepares batch of contexts and target indexes into the form
        suitable for processing with BERT, e.g. tokenziation, addition of special tokens
        like [CLS] and [SEP], padding contexts to have the same size etc.

        Args:
            batch_of_tokens: list of contexts
            batch_of_target_ids: list of target word indexes
            num_masks: number of masks to replace target with
            tokenizer: tokenizer to use for word tokenization

        Returns:
            transformed contexts and target word indexes in these new contexts
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        roberta_batch_of_tokens, roberta_batch_of_target_ids = [], []
        max_seq_len = 0
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            tokenized = self.tokenize_around_target(tokens, target_idx, tokenizer)
            left_context, right_context, target_start = tokenized

            if self.dec_type == 'cwm':
                context = torch.cat([left_context,
                                 torch.as_tensor([self.model.task.mask_idx]),
                                 right_context])
            else:
                context = torch.cat([left_context,
                                 torch.as_tensor([self.model.task.mask_idx] * num_masks),
                                 right_context])

            if len(context) > 512:
                first_subtok = context[target_start]
                # Cropping maximum context around the target word
                left_idx = max(0, target_start - 256)
                right_idx = min(target_start + 256, len(context))
                context = context[left_idx: right_idx]
                target_start = target_start if target_start < 256 else 255
                assert first_subtok == context[target_start]

            max_seq_len = max(max_seq_len, len(context))

            roberta_batch_of_tokens.append(context)
            roberta_batch_of_target_ids.append(np.arange(target_start, target_start + num_masks))

        assert max_seq_len <= 512

        input_ids = torch.vstack([
            torch.cat([tokens, torch.Tensor([2] * (max_seq_len - len(tokens)))])
            for tokens in roberta_batch_of_tokens
        ])

        input_ids = input_ids.type(torch.long).to(self.device)
        roberta_batch_of_target_ids = torch.tensor(np.stack(roberta_batch_of_target_ids)).to(self.device)

        return input_ids, roberta_batch_of_target_ids

    def get_log_probs_at_mask(self, input_ids: torch.Tensor, mask_ids: torch.Tensor):
        features, extra = self.model.model.forward(
            input_ids.long().to(self.device), features_only=True, return_all_hiddens=False
        )
        features = features[:, mask_ids, :]
        features_at_mask = torch.diagonal(features).T
        logits = self.model.model.encoder.lm_head(features_at_mask)
        log_prob = torch.log(logits.softmax(dim=-1))

        return log_prob

    def decode_batch(self, batch_of_topk_tokens: torch.Tensor,) -> List[List[str]]:
        decoded = [
            [" ".join([self.model.task.source_dictionary[t.item()] for t in tokens])
             for tokens in batch_of_tokens
            ]
            for batch_of_tokens in batch_of_topk_tokens
        ]
        # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
        return decoded

    def _insert_mask(self, tokens, target_ids: torch.Tensor):
        bs, topk, seqlen = tokens.shape
        res = torch.zeros(bs, topk, seqlen + 1, device=self.device, dtype=torch.int32)
        for i, mask_id in enumerate(target_ids):
            res[i, :, :] = torch.cat([
                tokens[i, :, :mask_id],
                torch.full((topk, 1), self.model.task.mask_idx, device=self.device),
                tokens[i, :, mask_id:]
            ], dim=1)
        return res

    def predict_sentences(
            self, tokens_sentences: torch.Tensor, target_ids: torch.Tensor
    ) -> Tuple[np.array, List[List[str]]]:
        bs, seqlen = tokens_sentences.shape
        log_prob_over_dictionary = self.get_log_probs_at_mask(tokens_sentences[:, :], target_ids[:, 0])
        cur_input_tokens = tokens_sentences.unsqueeze(1).repeat([1, self.topk, 1]) # bs, topk, seqlen # 250k
        logits, cur_indexes = log_prob_over_dictionary.topk(k=self.topk, dim=-1) # topk ; topk
        logits = logits
        for i,mask_id in enumerate(target_ids[:, 0]):
            cur_input_tokens[i, :, mask_id] = cur_indexes[i]
        for i in range(1, self.num_masks):
            if self.dec_type == 'cwm':
                cur_input_tokens = self._insert_mask(cur_input_tokens, target_ids[:, i])
                seqlen += 1
            log_prob_over_dictionary = (
                self.get_log_probs_at_mask(
                    cur_input_tokens.view(bs * self.topk, seqlen),
                    target_ids[:, i].repeat_interleave(self.topk)
                )
            )
            log_prob_over_dictionary = log_prob_over_dictionary.view(bs, self.topk, -1)
            cur_logits, cur_indexes = log_prob_over_dictionary.max(dim=-1)

            for j, mask_id in enumerate(target_ids[:, i]):
                cur_input_tokens[j, :, mask_id] = cur_indexes[j]

            logits += cur_logits
        pred_words = self.decode_batch(torch.stack([cur_input_tokens[i, :, m] for i,m in enumerate(target_ids)]))
        return logits, pred_words

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes
            num_masks: number of masks to replace target with
        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
            word to index  mapping
        """
        log_probs = np.full((len(tokens_lists),  1 + self.topk*len(tokens_lists)), -np.inf)
        input_ids, target_ids = self.prepare_batch(tokens_lists, target_ids, self.num_masks)

        with torch.no_grad(), utils.model_eval(self.model):
            batch_logits, batch_pred_words = self.predict_sentences(
                input_ids, target_ids
            )
        # word2id = {'outofdictandsomereallylongstring': 0}
        word2id = {}
        max_id = -1
        for sentence_idx, topk_words in enumerate(batch_pred_words):
            for word_idx, word in enumerate(topk_words):
                if word not in word2id:
                    word2id[word] = max_id + 1
                    max_id += 1
                idx = word2id[word]
                log_probs[sentence_idx, idx] = batch_logits[sentence_idx, word_idx]

            # sum_probs = np.exp(batch_logits[sentence_idx, :].cpu().numpy().astype(np.float64)).sum()
            # log_probs[sentence_idx, 0] = np.log(1 - sum_probs)

        return log_probs[:, :max_id + 1], word2id

    def process_words(self, word2id: Dict[str, int]) -> Dict[str, List[int]]:
        merging_vocab = defaultdict(list)
        for k, v in word2id.items():
            if not k.startswith('\u2581'):
                k = '\u2581' + k
            k = k.split('\u2581')[1].replace(' ', '')
            k = k.translate(str.maketrans('', '', string.punctuation))
            if k:
                merging_vocab[k].append(v)
        return merging_vocab

    def update_vocab(
            self, merging_vocab: Dict[str, List[int]], log_probs: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        res_probs = []
        res_word2id = {}
        for i, (k, v) in enumerate(merging_vocab.items()):
            new_vector = log_probs[:, v].max(axis=1)
            res_probs.append(new_vector)
            res_word2id[k] = i
        res_probs_np = np.vstack(res_probs).T
        return res_probs_np, res_word2id

    def postprocess(
        self,
        log_probs: np.ndarray,
        word2id: Dict[str, int],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        merging_vocab = self.process_words(word2id)
        new_log_probs, new_word2id = self.update_vocab(merging_vocab, log_probs)
        return new_log_probs, new_word2id

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".

        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """

        log_probs, word2id = self.predict(tokens_lists, target_ids)
        log_probs, word2id = self.postprocess(log_probs, word2id)
        return log_probs, word2id


class XLMRProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        mask_type: str = "not_masked",
        model_name: str = "roberta-large",
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        use_attention_mask: bool = True,
        cuda_device: int = 0,
        sim_func: str = "dot-product",
        unk_word_embedding: str = "first_subtoken",
        filter_vocabulary_mode: str = "none",
        verbose: bool = False,
    ):
        """
        Probability estimator based on the Roberta model.
        See Y. Liu et al. "RoBERTa: A Robustly Optimized
        BERT Pretraining Approach".

        Args:
            mask_type: the target word masking strategy.
            model_name: Roberta model name, see https://github.com/huggingface/transformers
            embedding_similarity: whether to compute BERT embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            use_attention_mask: whether to zero out attention on padding tokens
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            unk_word_embedding: how to handle words that are splitted into multiple subwords when computing
            embedding similarity
            verbose: whether to print misc information
        """
        super(XLMRProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
        )
        self.mask_type = mask_type
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.unk_word_embedding = unk_word_embedding
        self.filter_vocabulary_mode = filter_vocabulary_mode
        self.prev_word2id = {}

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "xlmr",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_type": self.mask_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "unk_word_embedding": self.unk_word_embedding,
            }
        }

        self.register_model()

        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

    @property
    def tokenizer(self):
        """
        Model tokenizer.

        Returns:
            `transformers.RobertaTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]

    @property
    def parameters(self):
        parameters = f"{self.mask_type}{self.model_name}" \
                     f"{self.use_attention_mask}{self.filter_vocabulary_mode}"

        if self.embedding_similarity:
            parameters += f"embs{self.unk_word_embedding}{self.sim_func}"

        return parameters

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in XLMRProbEstimator.loaded:
            roberta_model = XLMRobertaForMaskedLM.from_pretrained(self.model_name)
            roberta_model.to(self.device).eval()
            roberta_tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
            roberta_word2id = XLMRProbEstimator.load_word2id(roberta_tokenizer)
            filter_word_ids = XLMRProbEstimator.load_filter_word_ids(
                roberta_word2id, punctuation
            )
            word_embeddings = (
                roberta_model.lm_head.decoder.weight.data.cpu().numpy()
            )

            norms = np.linalg.norm(word_embeddings, axis=-1, keepdims=True)
            normed_word_embeddings = word_embeddings / norms

            XLMRProbEstimator.loaded[self.model_name] = {
                "model": roberta_model,
                "tokenizer": roberta_tokenizer,
                "embeddings": word_embeddings,
                "normed_embeddings": normed_word_embeddings,
                "word2id": roberta_word2id,
                "filter_word_ids": filter_word_ids,
            }
            XLMRProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            XLMRProbEstimator.loaded[self.model_name]["ref_count"] += 1

    @property
    def normed_embeddings(self) -> np.ndarray:
        """
        Attribute that acquires model word normed_embeddings.

        Returns:
            2-D `numpy.ndarray` with rows representing word vectors.
        """
        return self.loaded[self.model_name]["normed_embeddings"]

    def get_emb_similarity(
        self, tokens_batch: List[List[str]], target_ids_batch: List[int],
    ) -> np.ndarray:
        """
        Computes similarity between each target word and substitutes
        according to their embedding vectors.

        Args:
            tokens_batch: list of contexts
            target_ids_batch: list of target word ids in the given contexts

        Returns:
            similarity scores between target words and
            words from the model vocabulary.
        """
        if self.sim_func == "dot-product":
            embeddings = self.embeddings
        else:
            embeddings = self.normed_embeddings

        target_word_embeddings = []
        for tokens, pos in zip(tokens_batch, target_ids_batch):
            tokenized = self.tokenize_around_target(tokens, pos, self.tokenizer)
            _, _, target_subtokens_ids = tokenized

            target_word_embeddings.append(
                self.get_target_embedding(target_subtokens_ids, embeddings)
            )

        target_word_embeddings = np.vstack(target_word_embeddings)
        emb_sim = np.matmul(target_word_embeddings, embeddings.T)

        return emb_sim / self.temperature

    def get_target_embedding(
        self,
        target_subtokens_ids: List[int],
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Returns an embedding that will be used if the given word is not presented
        in the vocabulary. The word is split into subwords and depending on the
        self.unk_word_embedding parameter the final embedding is built.

        Args:
            word: word for which the vector should be given
            target_subtokens_ids: vocabulary indexes of target word subtokens
            embeddings: roberta embeddings of target word subtokens

        Returns:
            embedding of the unknown word
        """
        if self.unk_word_embedding == "mean":
            return embeddings[target_subtokens_ids].mean(axis=0, keepdims=True)
        elif self.unk_word_embedding == "first_subtoken":
            return embeddings[target_subtokens_ids[0]]
        elif self.unk_word_embedding == "last_subtoken":
            return embeddings[target_subtokens_ids[-1]]
        else:
            raise ValueError(
                f"Incorrect value of unk_word_embedding: "
                f"{self.unk_word_embedding}"
            )

    @staticmethod
    def load_word2id(tokenizer: XLMRobertaTokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.

        Args:
            tokenizer: `transformers.RobertaTokenizer` tokenizer

        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id

    @staticmethod
    def load_filter_word_ids(word2id: Dict[str, int], filter_chars: str) -> List[int]:
        """
        Gathers words that should be filtered from the end distribution, e.g.
        punctuation.

        Args:
            word2id: model vocabulary
            filter_chars: words with this chars should be filtered from end distribution.

        Returns:
            Indexes of words to be filtered from the end distribution.
        """
        filter_word_ids = []
        set_filter_chars = set(filter_chars)
        for word, idx in word2id.items():
            if len(set(word) & set_filter_chars):
                filter_word_ids.append(idx)
        return filter_word_ids

    @property
    def filter_word_ids(self) -> List[int]:
        """
        Indexes of words to be filtered from the end distribution.

        Returns:
            list of indexes
        """
        return self.loaded[self.model_name]["filter_word_ids"]

    @staticmethod
    def tokenize_around_target(
        tokens: List[str],
        target_idx: int,
        tokenizer: XLMRobertaTokenizer = None,
    ):
        left_specsym_len = 1  # for BERT / ROBERTA there is 1 spec token before text
        input_text = ' '.join(tokens)
        tokenized_text = tokenizer.encode(' ' + input_text, add_special_tokens=True)

        left_ctx = ' '.join(tokens[:target_idx])
        target_start = left_specsym_len + len(tokenizer.encode(
            ' ' + left_ctx, add_special_tokens=False
        ))

        left_ctx_target = ' '.join(tokens[:target_idx + 1])
        target_subtokens_ids = tokenizer.encode(
            ' ' + left_ctx_target, add_special_tokens=False
        )[target_start - left_specsym_len:]

        return tokenized_text, target_start, target_subtokens_ids

    def prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: XLMRobertaTokenizer = None,
    ):
        """
        Prepares batch of contexts and target indexes into the form
        suitable for processing with BERT, e.g. tokenziation, addition of special tokens
        like [CLS] and [SEP], padding contexts to have the same size etc.

        Args:
            batch_of_tokens: list of contexts
            batch_of_target_ids: list of target word indexes
            tokenizer: tokenizer to use for word tokenization

        Returns:
            transformed contexts and target word indexes in these new contexts
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        roberta_batch_of_tokens, roberta_batch_of_target_ids = [], []
        max_seq_len = 0
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            tokenized = self.tokenize_around_target(tokens, target_idx, tokenizer)
            context, target_start, target_subtokens_ids = tokenized

            if self.mask_type == "masked":
                context = context[:target_start] + \
                          [tokenizer.mask_token_id] + \
                          context[target_start + len(target_subtokens_ids):]
            elif self.mask_type != "not_masked":
                raise ValueError(f"Unrecognised masking type {self.mask_type}.")

            if len(context) > 512:
                first_subtok = context[target_start]
                # Cropping maximum context around the target word
                left_idx = max(0, target_start - 256)
                right_idx = min(target_start + 256, len(context))
                context = context[left_idx: right_idx]
                target_start = target_start if target_start < 256 else 255
                assert first_subtok == context[target_start]

            max_seq_len = max(max_seq_len, len(context))

            roberta_batch_of_tokens.append(context)
            roberta_batch_of_target_ids.append(target_start)

        assert max_seq_len <= 512

        input_ids = np.vstack([
            tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
            for tokens in roberta_batch_of_tokens
        ])

        input_ids = torch.tensor(input_ids).to(self.device)

        return input_ids, roberta_batch_of_target_ids

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int],
    ) -> np.ndarray:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        input_ids, mod_target_ids = self.prepare_batch(tokens_lists, target_ids)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
            attention_mask = attention_mask.float().to(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = np.vstack([
                logits[idx, target_idx, :].cpu().numpy()
                for idx, target_idx in enumerate(mod_target_ids)
            ])
            return logits

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".

        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """

        if self.embedding_similarity:
            logits = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            logits = self.predict(tokens_lists, target_ids)

        return logits, self.word2id