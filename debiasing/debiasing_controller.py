import logging

from controller import json_controller, calculation, augmentation_retrieval, specification_controller
from debiasing import bam, gbdd


def debiasing(methods, content, bar):
    logging.info("Debiasing-Engine: Started")
    # bar params: lower, uploaded, pca, space
    if content is None:
        return 'BAD REQUEST - NO BIAS SPEC JSON FOUND', 400
    if 'space' not in bar and 'uploaded' not in bar:
        return 'BAD REQUEST - NO SPACE SELECTED', 400
    space = 'fasttext'
    uploaded = 'false'
    lower = 'false'
    pca = 'true'
    lex = 'false'
    if 'space' in bar:
        space = bar['space']
    if 'uploaded' in bar:
        uploaded = bar['uploaded']
    if 'lower' in bar:
        lower = bar['lower']
    if 'pca' in bar:
        pca = bar['pca']
    if 'lex' in bar:
        lex = bar['lex']
    t1_list, t2_list, a1_list, a2_list, aug1_list, aug2_list = json_controller.json_to_debias_spec(content)
    if len(aug1_list) == 0:
        aug1_list, computed = augmentation_retrieval.retrieve_multiple_augmentations(t1_list)
    if len(aug2_list) == 0:
        aug2_list, computed = augmentation_retrieval.retrieve_multiple_augmentations(t2_list)
    if lower == 'true':
        t1_list = [x.lower() for x in t1_list]
        t2_list = [x.lower() for x in t2_list]
        a1_list = [x.lower() for x in a1_list]
        a2_list = [x.lower() for x in a2_list]
        aug1_list = [x.lower() for x in aug1_list]
        aug2_list = [x.lower() for x in aug2_list]
    equality_sets = []
    for t1 in aug1_list:
        for t2 in aug2_list:
            equality_sets.append((t1, t2))

    vocab, vecs = specification_controller.return_vocab_vecs(space, uploaded)
    if methods == 'full-bam':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list,
                                                                 a2_list)
        return vocab, new_vecs
    if methods == 'full-gbdd':

        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_gbdd(equality_sets, vocab, vecs, t1_list,
                                                                  t2_list, a1_list, a2_list)
        return vocab, new_vecs
    if methods == 'full-bamXgbdd':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_bam_gbdd(equality_sets, vocab, vecs, t1_list, t2_list,
                                                                      a1_list, a2_list)
        return vocab, new_vecs
    if methods == 'full-gbddXbam':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_gbdd_bam(equality_sets, vocab, vecs, t1_list, t2_list,
                                                                      a1_list, a2_list)
        return vocab, new_vecs

    t1, t2, a1, a2, aug1, aug2, not_found, deleted = specification_controller. \
        get_vectors_for_spec(space, lower, uploaded, t1_list, t2_list, a1_list, a2_list, aug1_list, aug2_list)
    vocab, vecs, lex_dict = {}, [], {}
    if lex != 'false':
        lex_dict = specification_controller.get_lex_dict(space, uploaded, lex)
        vocab, vecs = calculation.create_vocab_and_vecs(t1, t2, a1, a2, aug1, aug2, lex_dict)
    else:
        vocab, vecs = calculation.create_vocab_and_vecs(t1, t2, a1, a2, aug1, aug2)

    t1_deb, t2_deb, a1_deb, a2_deb, new_vecs = [], [], [], [], []
    logging.info("Debiasing-Engine: Specs loaded, starting computing")
    if methods == 'bam':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list,
                                                                 a2_list)
    if methods == 'gbdd':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_gbdd(equality_sets, vocab, vecs, t1_list,
                                                                  t2_list, a1_list, a2_list)
    if methods == 'bamXgbdd':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_bam_gbdd(equality_sets, vocab, vecs, t1_list, t2_list,
                                                                      a1_list, a2_list)
    if methods == 'gbddXbam':
        new_vecs, t1_deb, t2_deb, a1_deb, a2_deb = debiasing_gbdd_bam(equality_sets, vocab, vecs, t1_list, t2_list,
                                                                      a1_list, a2_list)

    if pca == 'true':
        biased_space = calculation.principal_componant_analysis2(vecs)
        debiased_space = calculation.principal_componant_analysis2(new_vecs)
        t1_pca_bias, t2_pca_bias, a1_pca_bias, a2_pca_bias = calculation.vocabs_to_dicts(vocab, biased_space, t1_list,
                                                                                         t2_list,
                                                                                         a1_list, a2_list)
        t1_pca_deb, t2_pca_deb, a1_pca_deb, a2_pca_deb = calculation.vocabs_to_dicts(vocab, debiased_space, t1_list,
                                                                                     t2_list,
                                                                                     a1_list, a2_list)
        if lex == 'false':
            response = json_controller.debiasing_json(space, lower, methods, pca, aug1_list, aug2_list, t1, t2, a1, a2,
                                                      t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted,
                                                      t1_pca_bias, t2_pca_bias, a1_pca_bias, a2_pca_bias,
                                                      t1_pca_deb, t2_pca_deb, a1_pca_deb, a2_pca_deb)
        else:
            if lex == 'simlex':
                lex_dict = calculation.vocab_to_dict(vocab, new_vecs, calculation.simlex_vocab)
            if lex == 'wordsim':
                lex_dict = calculation.vocab_to_dict(vocab, new_vecs, calculation.wordsim_vocab)
            response = json_controller.debiasing_json(space, lower, methods, pca, aug1_list, aug2_list, t1, t2, a1, a2,
                                                      t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted,
                                                      t1_pca_bias, t2_pca_bias, a1_pca_bias, a2_pca_bias,
                                                      t1_pca_deb, t2_pca_deb, a1_pca_deb, a2_pca_deb, lex_dict)
    else:
        if lex == 'false':
            response = json_controller.debiasing_json(space, lower, methods, pca, aug1_list, aug2_list, t1, t2, a1, a2,
                                                      t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted)
        else:
            response = json_controller.debiasing_json(space, lower, methods, pca, aug1_list, aug2_list, t1, t2, a1, a2,
                                                      t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted, lex_dict)

    logging.info("Debiasing-Engine: Finished")

    return response, 200


def debiasing_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    new_vecs, proj_mat = bam.debias_proc(equality_sets, vocab, vecs)
    t1, t2, a1, a2 = calculation.vocabs_to_dicts(vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return new_vecs, t1, t2, a1, a2,


def debiasing_gbdd(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    v_b = gbdd.get_bias_direction(equality_sets, vocab, vecs)
    new_vecs = gbdd.debias_direction_linear(v_b, vecs)
    t1, t2, a1, a2 = calculation.vocabs_to_dicts(vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return new_vecs, t1, t2, a1, a2


def debiasing_bam_gbdd(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    new_vocab, new_vecs = bam.debias_proc(equality_sets, vocab, vecs)
    v_b = gbdd.get_bias_direction(equality_sets, new_vecs, new_vocab)
    new_vocab, new_vecs = gbdd.debias_direction_linear(v_b, new_vecs)
    t1, t2, a1, a2 = calculation.vocabs_to_dicts(new_vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return new_vecs, t1, t2, a1, a2


def debiasing_gbdd_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    v_b = gbdd.get_bias_direction(equality_sets, vecs, vocab)
    new_vocab, new_vecs = gbdd.debias_direction_linear(v_b, vecs)
    new_vocab, new_vecs = bam.debias_proc(equality_sets, new_vocab, new_vecs)
    t1, t2, a1, a2 = calculation.vocabs_to_dicts(new_vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return new_vecs, t1, t2, a1, a2
