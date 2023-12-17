local prob_estimator = import '../prob_estimators/lexsub/xlmr_123.jsonnet';
local post_processing = import '../subst_generators/post_processors/spacy_old_sum.jsonnet';
local dpl = "<T> or <M>"
local dpr = "<M> or <T>"

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
                        dynamic_pattern: dpl
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
                        dynamic_pattern: dpr
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
                        dynamic_pattern: dpl
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
                        dynamic_pattern: dpr
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
                        dynamic_pattern: dpl
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
                            values: [dpr],
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

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: {
        class_name: "prob_estimators.combiner.BCombFasttextCombiner",
        prob_estimators: [{
            class_name: "prob_estimators.combiner.MaxCombiner",
            merge_vocab_type: "union",
            prob_estimators: [{
                class_name: "prob_estimators.combiner.MaxCombiner",
                merge_vocab_type: "union",
                prob_estimators: [{
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
                            dynamic_pattern: "<M> or <T>"
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
                            dynamic_pattern: "<M> or <T>"
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
                            dynamic_pattern: "<M> or <T>"
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
                            dynamic_pattern: "<M> or <T>"
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
                            dynamic_pattern: "<M> or <T>"
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
                                values: ["<M> or <T>"],
                                name: "dp"
                            }
                        },
                    ]
                }]
            }],
        }],
        verbose: false,
        k: 4.0,
        s: {
            class_name: "Hyperparam",
            name: "s",
            values: [1.0],
        },
        temperature: {
            class_name: "Hyperparam",
            values: [0.01, 0.03, 0.05, 0.07, 0.1],
            name: "embs_temp"
        },
        beta: {
            class_name: "Hyperparam",
            name: "beta",
            values: [0.0],
        },
        lang: "en"
    },
    post_processing: post_processing,
    top_k: 10,
}