
local post_processing = import '../subst_generators/post_processors/spacy_old_sum.jsonnet';

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
            class_name: "prob_estimators.roberta_estimator.RobertaProbEstimator",
            mask_type: "masked",
            model_name: {
                        class_name: "Hyperparam",
                        values: ["roberta-large"],
                        name: "dpl"
            },
            embedding_similarity: false,
            temperature: 1.0,
            use_attention_mask: true,
            cuda_device: 0,
            verbose: false
        }],
        verbose: false,
        k: 4.0,
        s: 1.0,
        temperature: {
            class_name: "Hyperparam",
            values: [0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5],
            name: "embs_temp"
        },
        beta: 0.0,
        lang: "en"
    },
    post_processing: [
        {class_name: "post_processors.roberta_postproc.RobertaPostProcessor", strategy: "drop_subwords"},
    ] + post_processing,
    top_k: 10,
}