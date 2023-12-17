
local post_processing = import '../subst_generators/post_processors/lemmatizer_name.jsonnet';

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
            class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimatorMultimasked",
            topk: 60,
            model_name: "/mnt/d/science/summer-wsi/xlmr_large_cwm_multi/model.pt",
            num_masks: 3,
            decoding_type: {
                class_name: "Hyperparam",
                values: ["cwm"],
                name: "multitoken"
            }
        }],
        verbose: false,
        k: 4.0,
        s: 1.0,
        temperature: {
            class_name: "Hyperparam",
            values: [0.07],
            name: "embs_temp"
        },
        beta: 0.0,
        lang: "lang_template"
    },
    post_processing: post_processing,
    top_k: 10,
}