local prob_estimator = import '../../prob_estimators/lexsub/xlmr_123.jsonnet';
local post_processing = import '../post_processors/spacy_old_sum.jsonnet';

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
                        topk: {
                            class_name: "Hyperparam",
                            values: [50],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 1,
                        dynamic_pattern: "<M> (or even <T>)"
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [50],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 1,
                        dynamic_pattern: "<T> (or even <M>)"
                    },
                ]
            },
            {
                class_name: "prob_estimators.combiner.AverageCombiner",
                merge_vocab_type: "union",
                prob_estimators: [
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [50],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 2,
                        dynamic_pattern: "<M> (or even <T>)"
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [50],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 2,
                        dynamic_pattern: "<T> (or even <M>)"
                    },
                ]
            },
            {
                class_name: "prob_estimators.combiner.AverageCombiner",
                merge_vocab_type: "union",
                prob_estimators: [
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [50],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 3,
                        dynamic_pattern: "<M> (or even <T>)"
                    },
                    {
                        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
                        topk: {
                            class_name: "Hyperparam",
                            values: [50],
                            name: "mask1_topk"
                        },
                        model_name: "xlmr.large",
                        num_masks: 3,
                        dynamic_pattern: {
                            class_name: "Hyperparam",
                            values: ["<T> (or even <M>)"],
                            name: "dp"
                        }
                    },
                ]
            },
        ],
    },
    post_processing: post_processing,
    top_k: 10,
}