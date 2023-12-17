local prob_estimator = import '../prob_estimators/lexsub/xlmr_123.jsonnet';
local post_processing = import '../subst_generators/post_processors/lemmatizer_name.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: {
        class_name: "prob_estimators.combiner.MaxCombiner",
        merge_vocab_type: "union",
        prob_estimators: [
            {
                class_name: "prob_estimators.combiner.AverageCombiner",
                merge_vocab_type: "union",
                prob_estimators: [
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: 10,
                        model_name: "xlmr.large",
                        num_masks: 1,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["left pattern"],
                            name: "dpl"
                        }
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: 10,
                        model_name: "xlmr.large",
                        num_masks: 1,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["right pattern"],
                            name: "dpr"
                        }
                    },
                ]
            },
            {
                class_name: "prob_estimators.combiner.AverageCombiner",
                merge_vocab_type: "union",
                prob_estimators: [
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: 10,
                        model_name: "xlmr.large",
                        num_masks: 2,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["left pattern"],
                            name: "dpl"
                        }
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: 10,
                        model_name: "xlmr.large",
                        num_masks: 2,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["right pattern"],
                            name: "dpr"
                        }
                    },
                ]
            },
            {
                class_name: "prob_estimators.combiner.AverageCombiner",
                merge_vocab_type: "union",
                prob_estimators: [
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: 10,
                        model_name: "xlmr.large",
                        num_masks: 3,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["left pattern"],
                            name: "dpl"
                        }
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: 10,
                        model_name: "xlmr.large",
                        num_masks: 3,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["right pattern"],
                            name: "dpr"
                        }
                    },
                ]
            },
        ],
    },
    post_processing: post_processing,
    top_k: 10,
}