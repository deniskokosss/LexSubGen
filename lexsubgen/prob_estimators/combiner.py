import logging
import os
from typing import List, Dict, Tuple

import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from overrides import overrides
from wordfreq import word_frequency

import fasttext
import fasttext.util

from lexsubgen.prob_estimators.base_estimator import BaseProbEstimator

def visualize_combination(log_probs_list, word2id, title):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    os.makedirs("vis", exist_ok=True)

    id2word = {v:k for k,v in word2id.items()}
    df = []
    for k, part in zip([0, 1], ['xlmr', 'fasttext']):
        for i in range(log_probs_list[k].shape[0]):
            for j in range(log_probs_list[k][i].shape[0]):
                df.append([i, id2word[j], j, part, log_probs_list[k][i][j],
                           log_probs_list[0][i][j] + log_probs_list[1][i][j]])
    df = pd.DataFrame(df, columns=['sentence', 'word', 'wid', 'part', 'prob', 'agg_prob'])
    fig,ax = plt.subplots(2, 2)
    fig.subplots_adjust(top=0.9)
    sns.histplot(data=df, x='prob', hue='part', ax=ax[0,0])

    sns.barplot(data=df[df.sentence == 0].nlargest(20, columns='agg_prob'), y='word', hue='part', x='prob', ax=ax[0,1])
    sns.barplot(data=df[(df.sentence == 0) & (df.part == 'xlmr')].nlargest(20, columns='prob'),
                y='word', hue='part', x='prob', ax=ax[1,0])
    sns.barplot(data=df[(df.sentence == 0) & (df.part == 'fasttext')].nlargest(20, columns='prob'),
                y='word', hue='part', x='prob', ax=ax[1,1])
    strlen = 70
    fig.suptitle("\n".join([title[strlen*i:strlen*i + strlen] for i in range(1 + len(title) // strlen)]))
    # plt.tight_layout()
    plt.savefig(f"vis/{title[:50]}.png", dpi=150)
    plt.close()



class Combiner(BaseProbEstimator):
    def __init__(self,
                 prob_estimators: List[BaseProbEstimator],
                 merge_vocab_type: str = 'intersect',
                 verbose: bool = False
    ):
        """
        Class that combines predictions from several probability estimators.

        Args:
            prob_estimators: list of probability estimators
            verbose: output verbosity
        """
        super(Combiner, self).__init__(verbose=verbose)
        self.prob_estimators = prob_estimators
        self.merge_vocab_type = merge_vocab_type

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines several log-probs into one distribution.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            log-probs with rows representing probability distribution over vocabulary.
        """
        raise NotImplementedError()

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes log-probs over vocabulary for a given instances.

        Args:
            tokens_lists: list of contexts.
            target_ids: list of target word ids.

        Returns:
            log-probs over vocabulary and respective vocabulary.
        """
        estimator_outputs = [
            estimator.get_log_probs(tokens_lists, target_ids)
            for estimator in self.prob_estimators
        ]
        estimator_log_probs = [log_probs for log_probs, _ in estimator_outputs]
        word2ids = [w2id for _, w2id in estimator_outputs]
        if self.merge_vocab_type == 'intersect':
            estimator_log_probs, word2id = self._intersect_vocabs(
                estimator_log_probs, word2ids
            )
        elif self.merge_vocab_type == 'union':
            estimator_log_probs, word2id = self._union_vocabs(
                estimator_log_probs, word2ids
            )
        else:
            raise NotImplementedError(f"{self.merge_vocab_type} merging method no implemented")
        combined_log_probs = self.combine(estimator_log_probs, word2id)
        return combined_log_probs, word2id

    @staticmethod
    def _intersect_vocabs(
        log_probs_list: List[np.ndarray], word2ids: List[Dict[str, int]]
    ):
        """
        Truncates vocabularies to their intersection and respectively
        truncates model distributions.

        Args:
            log_probs_list: list of log-probs that are given by several models.
            word2ids: list of model vocabularies

        Returns:
            truncated log-probs and vocabulary
        """

        if all(x == word2ids[0] for x in word2ids[1:]):
            # All word2ids elements are the same
            return log_probs_list, word2ids[0]

        common_vocab = set(word2ids[0].keys())
        for word2id in word2ids[1:]:
            common_vocab = common_vocab.intersection(set(word2id.keys()))

        cutted_word2id = {w: idx for w, idx in word2ids[0].items() if w in common_vocab}
        words = list(cutted_word2id.keys())
        new_word2id = {w: i for i, w in enumerate(words)}

        new_probs = []
        for log_probs, word2id in zip(log_probs_list, word2ids):
            idxs = np.array([word2id[w] for w in words])
            new_probs.append(log_probs[:, idxs])
        return new_probs, new_word2id

    @staticmethod
    def _union_vocabs(
        log_probs_list: List[np.ndarray], word2ids: List[Dict[str, int]]
    ):
        """
        Truncates vocabularies to their intersection and respectively
        truncates model distributions.

        Args:
            log_probs_list: list of log-probs that are given by several models.
            word2ids: list of model vocabularies

        Returns:
            truncated log-probs and vocabulary
        """

        if all(x == word2ids[0] for x in word2ids[1:]):
            # All word2ids elements are the same
            return log_probs_list, word2ids[0]
        new_word2id = {}

        common_vocab = set()
        for word2id in word2ids:
            common_vocab = common_vocab.union(word2id.keys())
        new_word2id = {k: i for i,k in enumerate(common_vocab)}

        new_probs = []
        for log_probs, word2id in zip(log_probs_list, word2ids):
            inverted_word2id = {v: k for k, v in word2id.items()}
            common_log_probs = np.full((log_probs_list[0].shape[0], len(common_vocab)), log_probs_list[0].min())
            idxs = np.array([new_word2id[inverted_word2id[i]] for i in range(log_probs.shape[1])])
            common_log_probs[:, idxs] = log_probs
            new_probs.append(common_log_probs)
        return new_probs, new_word2id


class AverageCombiner(Combiner):
    """
    Combiner that returns average over log-probs
    """

    @overrides
    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines several log-probs into one distribution.
        Takes average over log-probs.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing
            probability distribution over vocabulary.
        """
        mean_log_probs = np.mean(log_probs, axis=0)
        return mean_log_probs


class MaxCombiner(Combiner):
    """
    Combiner that returns max over log-probs
    """

    @overrides
    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines several log-probs into one distribution.
        Takes average over log-probs.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing
            probability distribution over vocabulary.
        """
        max_log_probs = np.max(log_probs, axis=0)
        return max_log_probs


class BcombCombiner(Combiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        k: float = 4.0,
        s: float = 1.05,
        beta: float = 0.0,
        lang: str = 'en',
        verbose: bool = False,
    ):
        """
        Combines models predictions with the log-probs that comes from
        embedding similarity scores according to the formula
        :math:`P(w|C, T) \\propto \\displaystyle \\frac{P(w|C)P(w|T)}{P(w)^\\beta}`,
        where :math:`\\beta` -- is a parameter controlling how we penalize frequent words and
        :math:`P(w) = \\displaystyle \\frac{1}{(k + \\text{rank}(w))^s}`.
        For more details see N. Arefyev et al. "Combining Lexical Substitutes in Neural Word Sense Induction".

        Args:
            prob_estimators: list of probability estimators to be combined
            k: value of parameter k in prior word distribution
            s: value of parameter s in prior word distribution
            beta: value of parameter beta
            verbose: whether to output misc information
        """
        super(BcombCombiner, self).__init__(
            prob_estimators=prob_estimators, verbose=verbose
        )
        self.k = k
        self.s = s
        self.lang = lang
        self.beta = beta
        self.bcomb_prior_log_prob = None
        self.prev_word2id = {}

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines model predictions with embeddings similarity scores.
        If three log-probs are given this method handles first two as
        forward and backward passes and the third one as embedding similarity scores,
        this type of computation is used, for example, with ELMo model.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing probability distribution over vocabulary.
        """
        if self.bcomb_prior_log_prob is None or self.prev_word2id != word2id:
            self.bcomb_prior_log_prob = self.get_prior_log_prob(word2id)
        self.prev_word2id = word2id
        log_probs, similarity_log_probs = self.get_parts(log_probs)
        assert len(word2id) == log_probs.shape[-1]
        if self.beta != 0:
            log_probs -= self.beta * self.bcomb_prior_log_prob
        log_probs += similarity_log_probs
        return log_probs

    def get_prior_log_prob(self, word2id: Dict[str, int]) -> np.ndarray:
        """
        Get prior word distribution log-probs.

        Args:
            word2id: vocabulary

        Returns:
            `numpy.ndarray` of prior log-probs
        """
        prior_prob = np.zeros(len(word2id), dtype=np.float32)
        for word, idx in word2id.items():
            prior_prob[idx] = word_frequency(word, self.lang)

        idxs = prior_prob.argsort()
        prior_prob[idxs] = np.arange(len(prior_prob), 0, -1) + self.k
        return -np.log(prior_prob)[np.newaxis, :] * self.s

    @staticmethod
    def get_parts(log_probs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split log-probs onto model predictions and similarity scores.
        If length of log_probs equals 3 than first two log-probs are
        considered as forward and backward passes (used with ELMo model).

        Args:
            log_probs: list of log-probs

        Returns:
            model predictions and similarity scores.
        """
        n_dists = len(log_probs)

        if n_dists == 3:
            fwd_log_probs, bwd_log_probs, sim_log_probs = log_probs
            log_probs = fwd_log_probs + bwd_log_probs
        elif n_dists == 2:
            log_probs, sim_log_probs = log_probs
        else:
            raise ValueError("Bcomb supports combination of 2 or 3 distributions!")
        return log_probs.copy(), sim_log_probs.copy()


class BCombFasttextCombiner(BcombCombiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        k: float = 4.0,
        s: float = 1.05,
        beta: float = 0.0,
        temperature: float = 1.0,
        lang: str = 'en',
        verbose: bool = False,
    ):
        """
        Combines models predictions with the log-probs that comes from
        embedding similarity scores according to the formula
        :math:`P(w|C, T) \\propto \\displaystyle \\frac{P(w|C)P(w|T)}{P(w)^\\beta}`,
        where :math:`\\beta` -- is a parameter controlling how we penalize frequent words and
        :math:`P(w) = \\displaystyle \\frac{1}{(k + \\text{rank}(w))^s}`.
        For more details see N. Arefyev et al. "Combining Lexical Substitutes in Neural Word Sense Induction".

        Args:
            prob_estimators: list of probability estimators to be combined
            k: value of parameter k in prior word distribution
            s: value of parameter s in prior word distribution
            beta: value of parameter beta
            verbose: whether to output misc information
        """
        super(BCombFasttextCombiner, self).__init__(
            prob_estimators=prob_estimators, verbose=verbose
        )
        self.k = k
        self.s = s
        self.lang = lang
        self.beta = beta
        self.temperature = temperature
        # cwd = os.getcwd()
        # os.makedirs('~/.ftcache', exist_ok=True)
        # logging.info(f"Changing wd to ~/.ftcache")
        # logging.info(os.getcwd())
        # os.chdir('~/.ftcache')
        fasttext.util.download_model(self.lang, if_exists='ignore')
        self.ft = fasttext.load_model(f'cc.{self.lang}.300.bin')
        # os.chdir(cwd)

    @overrides
    def get_log_probs(
            self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes log-probs over vocabulary for a given instances.

        Args:
            tokens_lists: list of contexts.
            target_ids: list of target word ids.

        Returns:
            log-probs over vocabulary and respective vocabulary.
        """
        assert len(self.prob_estimators) == 1, "Fasttext combiner must have exactly one prob_estimator"
        estimator_outputs = [
            estimator.get_log_probs(tokens_lists, target_ids)
            for estimator in self.prob_estimators
        ]

        estimator_log_probs = [np.log(softmax(log_probs, axis=-1) + 1e-10)
                               for log_probs, _ in estimator_outputs]
        word2ids = [w2id for _, w2id in estimator_outputs]

        estimator_log_probs.append(
            self.get_fasttext_similarity(
                word2ids[0], [tokens[idx] for tokens,idx in zip(tokens_lists, target_ids)], self.temperature
            )
        )
        word2ids.append(word2ids[0])

        if self.merge_vocab_type == 'intersect':
            estimator_log_probs, word2id = self._intersect_vocabs(
                estimator_log_probs, word2ids
            )
        elif self.merge_vocab_type == 'union':
            estimator_log_probs, word2id = self._union_vocabs(
                estimator_log_probs, word2ids
            )
        else:
            raise NotImplementedError(f"{self.merge_vocab_type} merging method no implemented")
        combined_log_probs = self.combine(estimator_log_probs, word2id)
        # ttl = tokens_lists[0][:]
        # ttl[target_ids[0]] = '<' + ttl[target_ids[0]] + '>'
        # visualize_combination(estimator_log_probs, word2id, " ".join(ttl))
        return combined_log_probs, word2id

    def get_fasttext_similarity(self, word2id: Dict[str, int], target_words: List[str], temp: float) -> np.ndarray:
        target_embs = np.stack(
            [self.ft.get_word_vector(t.replace('Ġ', '').replace('\u2581', '')) for t in target_words])
        # target_embs = target_embs / np.sqrt(np.sum(target_embs**2, axis=-1))[:, np.newaxis]
        word_embs = np.stack([self.ft.get_word_vector(t.replace('Ġ', '').replace('\u2581', ''))
                              for t, _ in sorted(word2id.items(), key=lambda x: x[1])])
        # word_embs = word_embs / np.sqrt(np.sum(word_embs**2, axis=-1))[:, np.newaxis]
        similarity = cosine_similarity(target_embs, word_embs)
        similarity = np.log(softmax(similarity.astype(np.float64) / temp, axis=-1)).astype(np.float32)
        self.fast_text_sim = similarity
        return similarity


class Bcomb3Combiner(BcombCombiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        k: float = 4.0,
        s: float = 1.05,
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        """
        Combines models predictions with the log-probs that comes from
        embedding similarity scores according to the formula
        :math:`P(w|C, T) \\propto \\displaystyle \\frac{P(w|C)P(w|T)}{P(w)^\\beta}`,
        where :math:`\\beta` equals to (n-1) where n -- number of estimators to combine and
        :math:`P(w) = \\displaystyle \\frac{1}{(k + \\text{rank}(w))^s}`.
        For more details see N. Arefyev et al. "Combining Lexical Substitutes in Neural Word Sense Induction".

        Args:
            prob_estimators: list of probability estimators to be combined
            k: value of parameter k in prior word distribution
            s: value of parameter s in prior word distribution
            verbose: whether to output misc information
        """
        super(Bcomb3Combiner, self).__init__(
            prob_estimators=prob_estimators, k=k, s=s, verbose=verbose
        )
        self.temperature = temperature

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines model predictions with embeddings similarity scores.
        If three log-probs are given this method handles first two as
        forward and backward passes and the third one as embedding similarity scores,
        this type of computation is used, for example, with ELMo model.

        Args:
            log_probs: list of log-probs from different models
            word2id: vocabulary

        Returns:
            `numpy.ndarray`, log-probs with rows representing probability distribution over vocabulary.
        """
        if self.bcomb_prior_log_prob is None or self.prev_word2id != word2id:
            self.bcomb_prior_log_prob = self.get_prior_log_prob(word2id)
        self.prev_word2id = word2id

        n_dists = len(log_probs)
        log_probs, similarity_log_probs = self.get_parts(log_probs)
        log_probs -= (n_dists - 1) * self.bcomb_prior_log_prob
        log_probs += similarity_log_probs / self.temperature
        return log_probs


class Bcomb3ZipfCombiner(BcombCombiner):
    """
        Bcomb3Combiner that uses Zipf's distribution as a prior
        word distribution. It's supposed that more frequent words
        are at the top of the vocabulary
    """

    def get_prior_log_prob(self, word2id):
        """
        Get Zipf's distribution of given size.

        Args:
            shape: size of the distribution

        Returns:
            Zipf's distribution of size `shape`
        """
        return -self.s * np.log(np.arange(len(word2id)) + self.k)[np.newaxis, :]


class BcombLmsCombiner(Combiner):
    def __init__(
        self,
        prob_estimators: List[BaseProbEstimator],
        alpha: float = 1.0,
        beta: float = 1.0,
        verbose: bool = False,
    ):
        """
        Combines two models distributions into one according to the formula:

        :math:`P(w|M_1, M_2) \\propto \\displaystyle \\frac{(P(w|M_1)P(w|M_2))^\\alpha}{P(w)^\\beta}` and
        :math:`P(w) = \\displaystyle \\frac{1}{1 + \\text{rank}(w)}` is a prior word distribution. It's supposed
        that words are sorted in vocabulary by their frequency -- more frequent words come first.

        Args:
            prob_estimators: list of probability estimators, supports only two estimators
            alpha: value of parameter alpha
            beta: value of parameter beta
            verbose: whether to print misc information
        """
        super(BcombLmsCombiner, self).__init__(
            prob_estimators=prob_estimators, verbose=verbose
        )
        self.alpha = alpha
        self.beta = beta

    def combine(
        self, log_probs: List[np.ndarray], word2id: Dict[str, int]
    ) -> np.ndarray:
        """
        Combines two model prediction into one distribution.
        Use it only for ELMo model, because its vocabulary contains words in sorted order
        Args:
            log_probs: list of log-probs from two models
            word2id: vocabulary

        Returns:
            log-probs with rows representing probability distribution over vocabulary.
        """
        assert len(log_probs) == 2
        fwd_log_probs, bwd_log_probs = log_probs
        log_probs = (fwd_log_probs + bwd_log_probs) * self.alpha

        x, y = log_probs.shape

        # [1.0, 0.5, 0.333, 0.25, ...]
        rank_w = 1.0 / (np.arange(1, y + 1))

        # Repeating rank_w x times
        # [
        #     [1.0, 0.5, 0.333, 0.25, ...],
        #     [1.0, 0.5, 0.333, 0.25, ...],
        #     [1.0, 0.5, 0.333, 0.25, ...],
        #     ...
        # ]
        zipf = rank_w[np.newaxis, :].repeat(x, axis=0)

        return log_probs - np.log(zipf) * self.beta
