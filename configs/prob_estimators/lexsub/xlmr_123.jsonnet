{
    class_name: "prob_estimators.combiner.MaxCombiner",
    merge_vocab_type: "union",
    prob_estimators: [
        {
            class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
            topk: 150,
            model_name: "xlmr.large",
            num_masks: 1
        },
        {
            class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
            topk: 150,
            model_name: "xlmr.large",
            num_masks: 2
        },
        {
            class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
            topk: 150,
            model_name: "xlmr.large",
            num_masks: 3
        },
    ],
}