from collections import OrderedDict
from typing import List, Dict, Tuple, Set, Union
from pathlib import Path
import tempfile
import subprocess
import re
import warnings

def compute_precision_recall_f1_topk(
    gold_substitutes: List[str],
    pred_substitutes: List[str],
    topk_list: List[int] = (1, 3, 10),
) -> Dict[str, float]:
    """
    Method for computing k-metrics for each k in the input 'topk_list'.

    Args:
        gold_substitutes: Gold substitutes provided by human annotators.
        pred_substitutes: Predicted substitutes.
        topk_list: List of integer numbers for metrics.
        For example, if 'topk_list' equal to [1, 3, 5], then there will calculating the following metrics:
            ['Precion@1', 'Recall@1', 'F1-score@1',
             'Precion@3', 'Recall@3', 'F1-score@3',
             'Precion@5', 'Recall@5', 'F1-score@5']

    Returns:
        Dictionary that maps k-values in input 'topk_list' to computed Precison@k, Recall@k, F1@k metrics.
    """
    k_metrics = OrderedDict()
    golds_set = set(gold_substitutes)
    for topk in topk_list:
        if topk > len(pred_substitutes) or topk <= 0:
            raise ValueError(f"Couldn't take top {topk} from {len(pred_substitutes)} substitues")

        topk_pred_substitutes = pred_substitutes[:topk]

        true_positives = sum(1 for s in topk_pred_substitutes if s in golds_set)
        precision, recall, f1_score = _precision_recall_f1_from_tp_tpfp_tpfn(
            true_positives,
            len(topk_pred_substitutes),
            len(gold_substitutes)
        )
        k_metrics[f"prec@{topk}"] = precision
        k_metrics[f"rec@{topk}"] = recall
        k_metrics[f"f1@{topk}"] = f1_score
    return k_metrics


def compute_precision_recall_f1_vocab(
    gold_substitutes: List[str],
    vocabulary: Union[Set[str], Dict[str, int]],
) -> Tuple[float, float, float]:
    """
    Method for computing basic metrics like Precision, Recall, F1-score on all Substitute Generator vocabulary.
    Args:
        gold_substitutes: Gold substitutes provided by human annotators.
        vocabulary: Vocabulary of the used Substitute Generator.

    Returns:
        Precision, Recall, F1 Score
    """
    true_positives = sum(1 for s in set(gold_substitutes) if s in vocabulary)
    precision, recall, f1_score = _precision_recall_f1_from_tp_tpfp_tpfn(
        true_positives,
        len(vocabulary),
        len(gold_substitutes)
    )
    return precision, recall, f1_score


def _precision_recall_f1_from_tp_tpfp_tpfn(
    tp: int, tpfp: int, tpfn: int
) -> Tuple[float, float, float]:
    """
    Computing precision, recall and f1 score
    Args:
        tp: number of true positives
        tpfp: number of true positives + false positives
        tpfn: number of true positives + false negatives

    Returns:
        Precision, Recall and F1 score
    """
    precision, recall, f1_score = 0.0, 0.0, 0.0
    if tpfp:
        precision = tp / tpfp
    if tpfn:
        recall = tp / tpfn
    if precision and recall:
        f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def oot_score(golds: Dict[str, int], substitutes: List[str]):
    """
    Method for computing Out-Of-Ten score

    Args:
        golds: Dictionary that maps gold word to its annotators number.
        substitutes: List of generated substitutes.
    Returns:
        score: Computed OOT score.
    """
    score = 0
    for subst in substitutes:
        if subst in golds:
            score += golds[subst]
    score = score / sum([value for value in golds.values()])
    return score


def dump_pred_substitutes_semeval2010task2(
    pred_substitutes: Dict[str, List[str]], dest_path: str,
):
    with open(f"{dest_path}.best", mode="w", encoding='utf-8') as f:
        for inst_id, substs in pred_substitutes.items():
            if len(substs) == 0:
                warnings.warn(f"Instance {inst_id} has 0 predicted substitutes")
                print(inst_id, "", sep=' :: ', file=f)
            else:
                print(inst_id, substs[0], sep=' :: ', file=f)

    with open(f"{dest_path}.oot", mode="w", encoding='utf-8') as f:
        for inst_id, substs in pred_substitutes.items():
            print(inst_id, "".join(f"{s};" for s in substs), sep=' ::: ', file=f)
    return


def dump_gold_substitutes_semeval2010task2(
    gold_substitutes: Dict[str, Dict[str, int]], dest_path: str,
):
    with open(dest_path, mode="w", encoding='utf-8') as f:
        for inst_id, subst2weight in gold_substitutes.items():
            substs_str = "".join(f"{s} {w};" for s, w in subst2weight.items())
            print(inst_id, substs_str, sep=' :: ', file=f)
    return


def extract_semeval2010task2_scores(
    results_path: str
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    with open(results_path, mode="r", encoding='utf-8') as f:
        read = f.read()
        ((rec, prec), (mrec, mprec)) = re.findall(
            r".*recall = (.*), .*precision = (.*).*", read
        )
        return (float(rec), float(prec)), (float(mrec), float(mprec))


def semeval2010task2_scores(
    gold_substitutes: Dict[str, Dict[str, int]],
    pred_substitutes: Dict[str, List[str]],
):
    SE2010T2_DIR = Path(__file__).parent
    scoring = SE2010T2_DIR / "cllsscore.pl"
    with tempfile.TemporaryDirectory() as tempdir:
        pred_substs = Path(tempdir) / f"substitutes.pred"
        gold_substs = Path(tempdir) / f"substitutes.gold"
        dump_pred_substitutes_semeval2010task2(pred_substitutes, dest_path=str(pred_substs))
        dump_gold_substitutes_semeval2010task2(gold_substitutes, dest_path=str(gold_substs))
        subprocess.run(
            ["perl", scoring, f"{pred_substs}.best", gold_substs],
            # capture_output=False
        )
        subprocess.run(
            ["perl", scoring, f"{pred_substs}.oot", gold_substs, "-t", "oot"],
            # capture_output=False
        )
        scores = OrderedDict()

        ((rec, prec), (mrec, mprec)) = extract_semeval2010task2_scores(
            f"{pred_substs}.best.results"
        )
        scores["se10t2-rec@1"], scores["se10t2-prec@1"] = rec, prec
        scores["se10t2-mrec@1"], scores["se10t2-mprec@1"] = mrec, mprec

        ((rec, prec), (mrec, mprec)) = extract_semeval2010task2_scores(
            f"{pred_substs}.oot.results"
        )
        scores["se10t2-rec@10"], scores["se10t2-prec@10"] = rec, prec
        scores["se10t2-mrec@10"], scores["se10t2-mprec@10"] = mrec, mprec
    return scores