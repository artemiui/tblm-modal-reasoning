import random
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any

prop_var_range = 26

# Special token IDs
special_counter = 1

CONJ_token = prop_var_range + special_counter
special_counter += 1
DISJ_token = prop_var_range + special_counter
special_counter += 1
NEGAT_token = prop_var_range + special_counter
special_counter += 1

IF_token = prop_var_range + special_counter
special_counter += 1
THEN_token = prop_var_range + special_counter
special_counter += 1

START_RULE_token = prop_var_range + special_counter
special_counter += 1
END_RULE_token = prop_var_range + special_counter
special_counter += 1

START_FACT_token = prop_var_range + special_counter
special_counter += 1
END_FACT_token = prop_var_range + special_counter
special_counter += 1

TRUE_token = prop_var_range + special_counter
special_counter += 1
FALSE_token = prop_var_range + special_counter
special_counter += 1
UNDETERMINED_token = prop_var_range + special_counter
special_counter += 1

START_QUERY_token = prop_var_range + special_counter
special_counter += 1
END_QUERY_token = prop_var_range + special_counter
special_counter += 1

SEP_token = prop_var_range + special_counter
special_counter += 1
ANSWER_token = prop_var_range + special_counter
special_counter += 1

SEMICOLON_token = prop_var_range + special_counter
special_counter += 1
EOS_token = prop_var_range + special_counter
special_counter += 1

# Modal Tokens
BOX_token = prop_var_range + special_counter
special_counter += 1
DIAMOND_token = prop_var_range + special_counter
special_counter += 1
PROBABLY_token = prop_var_range + special_counter
special_counter += 1
CERTAINLY_token = prop_var_range + special_counter
special_counter += 1
UNLIKELY_token = prop_var_range + special_counter
special_counter += 1
XOR_token = prop_var_range + special_counter
special_counter += 1
ACCESS_START_token = prop_var_range + special_counter
special_counter += 1
ACCESS_END_token = prop_var_range + special_counter
special_counter += 1

W0_token = prop_var_range + special_counter
special_counter += 1
W1_token = prop_var_range + special_counter
special_counter += 1
W2_token = prop_var_range + special_counter
special_counter += 1

special_token_dict = {
    "CONJ": CONJ_token,
    "DISJ": DISJ_token,
    "XOR": XOR_token,
    "NEGAT": NEGAT_token,
    "IF": IF_token,
    "THEN": THEN_token,
    "START_RULE": START_RULE_token,
    "END_RULE": END_RULE_token,
    "START_FACT": START_FACT_token,
    "END_FACT": END_FACT_token,
    "TRUE": TRUE_token,
    "FALSE": FALSE_token,
    "UNDETERMINED": UNDETERMINED_token,
    "START_QUERY": START_QUERY_token,
    "END_QUERY": END_QUERY_token,
    "SEP": SEP_token,
    "ANSWER": ANSWER_token,
    ";": SEMICOLON_token,
    "EOS": EOS_token,
    "BOX": BOX_token,
    "DIAMOND": DIAMOND_token,
    "PROBABLY": PROBABLY_token,
    "CERTAINLY": CERTAINLY_token,
    "UNLIKELY": UNLIKELY_token,
    "ACCESS_START": ACCESS_START_token,
    "ACCESS_END": ACCESS_END_token,
    "W0": W0_token,
    "W1": W1_token,
    "W2": W2_token,
}

integer_to_english_letters = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L",
    12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    23: "X", 24: "Y", 25: "Z",
    THEN_token: "implies",
    START_RULE_token: "Rules:",
    END_RULE_token: "RU_e",
    START_FACT_token: "Facts:",
    START_QUERY_token: "Question:",
    END_FACT_token: "FA_e",
    END_QUERY_token: "Q_e",
    ANSWER_token: "ANS",
    TRUE_token: "is true",
    FALSE_token: "is false",
    UNDETERMINED_token: "is undetermined",
    SEMICOLON_token: ";",
    DISJ_token: "or",
    CONJ_token: "and",
    XOR_token: "xor",
    SEP_token: ".",
    BOX_token: "necessarily",
    DIAMOND_token: "possibly",
    PROBABLY_token: "probably",
    CERTAINLY_token: "certainly",
    UNLIKELY_token: "unlikely",
    ACCESS_START_token: "ACCESS_START",
    ACCESS_END_token: "ACCESS_END",
    W0_token: "w0",
    W1_token: "w1",
    W2_token: "w2",
}


def sample_modal_chain(
    depth: int = 2,
    shuffle: bool = True,
    linear_only: bool = False,
    operator: str = "BOX",
    accessible_worlds: Optional[List[str]] = None,
    truths: Optional[List[str]] = None,
    modal_first: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if accessible_worlds is None:
        accessible_worlds = ["w0", "w1"]

    vars_all = list(range(prop_var_range))
    random.shuffle(vars_all)
    vars_modal = vars_all[:depth * 2]
    vars_lin = vars_all[depth * 2: depth * 3]

    var_p = vars_modal[0]
    var_q = vars_modal[1]
    var_r = vars_lin[0]
    var_s = vars_lin[1]

    truth_p_w0 = "TRUE" if (truths is None) else truths[0]
    truth_p_w1 = "TRUE" if (truths is None) else truths[1]
    truth_r_w0 = "TRUE" if (truths is None or len(truths) < 3) else truths[2]

    facts_w0 = [[var_p, truth_p_w0], [var_q, "FALSE"], [var_r, truth_r_w0], [var_s, "FALSE"]]
    facts_w1 = [[var_p, truth_p_w1], [var_q, "TRUE"]]

    op_token = special_token_dict[operator]
    rule_modal = [op_token, var_p, special_token_dict["THEN"], var_q]
    rule_lin = [var_r, special_token_dict["THEN"], var_s]

    if modal_first:
        rules = [rule_modal, rule_lin]
    else:
        rules = [rule_lin, rule_modal]

    acc_truths = []
    if "w0" in accessible_worlds:
        acc_truths.append(truth_p_w0 == "TRUE")
    if "w1" in accessible_worlds:
        acc_truths.append(truth_p_w1 == "TRUE")

    n_acc = len(acc_truths)
    n_true = sum(acc_truths)

    if operator in {"BOX", "CERTAINLY"}:
        modal_conclusion_truth = "TRUE" if (n_acc > 0 and n_true == n_acc) else "UNDETERMINED"
    elif operator == "PROBABLY":
        modal_conclusion_truth = "TRUE" if (n_acc > 0 and (n_true / n_acc) > 0.5) else "UNDETERMINED"
    elif operator == "UNLIKELY":
        modal_conclusion_truth = "TRUE" if (n_acc > 0 and (n_true / n_acc) < 0.5) else "UNDETERMINED"
    else:  # DIAMOND
        modal_conclusion_truth = "TRUE" if (n_acc > 0 and n_true > 0) else "UNDETERMINED"

    lin_conclusion_truth = "TRUE" if truth_r_w0 == "TRUE" else "UNDETERMINED"

    cot_modal = []
    if "w0" in accessible_worlds:
        cot_modal.append([special_token_dict["W0"], var_p, special_token_dict[truth_p_w0]])
    if "w1" in accessible_worlds:
        cot_modal.append([special_token_dict["W1"], var_p, special_token_dict[truth_p_w1]])
    cot_modal.append([op_token, var_p, special_token_dict["THEN"], var_q, special_token_dict[";"], var_q, special_token_dict[modal_conclusion_truth]])

    cot_lin = [
        [special_token_dict["W0"], var_r, special_token_dict[truth_r_w0]],
        [var_r, special_token_dict["THEN"], var_s, special_token_dict[";"], var_s, special_token_dict[lin_conclusion_truth]]
    ]

    sample_dict_modal = {
        "rules": rules,
        "facts_w0": facts_w0,
        "facts_w1": facts_w1,
        "accessible_worlds": accessible_worlds,
        "query": var_q,
        "operator": operator,
        "query_type": "modal",
        "cot": cot_modal,
        "answer": modal_conclusion_truth,
        "queried_rule": rule_modal,
        "correct_fact": [var_p, truth_p_w0],
    }

    sample_dict_lin = {
        "rules": rules,
        "facts_w0": facts_w0,
        "facts_w1": facts_w1,
        "accessible_worlds": accessible_worlds,
        "query": var_s,
        "operator": operator,
        "query_type": "linear",
        "cot": cot_lin,
        "answer": lin_conclusion_truth,
        "queried_rule": rule_lin,
        "correct_fact": [var_r, truth_r_w0],
    }

    return sample_dict_modal, sample_dict_lin


def generate_modal_sample_tokens(sample_dict: Dict[str, Any]) -> Tuple[List[int], List[int], int]:
    context = []

    context.append(special_token_dict["ACCESS_START"])
    for w in sample_dict["accessible_worlds"]:
        w_tok = special_token_dict[w.upper()]
        context.append(w_tok)
    context.append(special_token_dict["ACCESS_END"])
    context.append(special_token_dict["SEP"])

    context.append(special_token_dict["START_FACT"])
    context.append(special_token_dict["W0"])
    for f in sample_dict["facts_w0"]:
        context.append(f[0])
        context.append(special_token_dict[f[1]])
        context.append(special_token_dict["SEP"])

    context.append(special_token_dict["W1"])
    for f in sample_dict["facts_w1"]:
        context.append(f[0])
        context.append(special_token_dict[f[1]])
        context.append(special_token_dict["SEP"])
    context.append(special_token_dict["END_FACT"])

    context.append(special_token_dict["START_RULE"])
    for rule in sample_dict["rules"]:
        context.extend(rule)
        context.append(special_token_dict["SEP"])
    context.append(special_token_dict["END_RULE"])

    context.append(special_token_dict["START_QUERY"])
    if sample_dict["query_type"] == "modal":
        op_tok = special_token_dict[sample_dict["operator"]]
        context.append(op_tok)
    context.append(sample_dict["query"])
    context.append(special_token_dict["SEP"])
    context.append(special_token_dict["END_QUERY"])
    context.append(special_token_dict["ANSWER"])

    answer = []
    for step in sample_dict["cot"]:
        answer.extend(step)
        answer.append(special_token_dict["SEP"])

    return context, answer, len(context)


def convert_to_english(int_str: List[int], is_question: bool = True) -> Tuple[List[str], str]:
    words = []
    for tok in int_str:
        if tok < prop_var_range:
            words.append(integer_to_english_letters[tok])
        elif tok in integer_to_english_letters:
            if tok in {END_FACT_token, END_QUERY_token, END_RULE_token}:
                continue
            elif tok == START_FACT_token:
                words.append("Facts:")
            elif tok == START_RULE_token:
                words.append("Rules:")
            elif tok == START_QUERY_token:
                words.append("Question: state the truth value of")
            elif tok == ANSWER_token:
                words.append("Answer:")
            else:
                words.append(integer_to_english_letters[tok])
        else:
            words.append("UND")

    raw_str = " ".join(words)
    clean_str = re.sub(r'\s(?=[\.,:;])', "", raw_str)
    return words, clean_str


def sample_context_and_answer_pairs_EXAMPLES(num_samples: int = 4, length_of_chain: int = 2) -> Tuple[str, str]:
    output_string = ""
    gt_string = ""
    for i in range(num_samples):
        op = "BOX" if (i % 2 == 0) else "DIAMOND"
        acc = ["w0", "w1"] if (i < num_samples - 1) else ["w0"]
        t_vals = ["TRUE", "TRUE", "TRUE"] if (i % 2 == 0) else ["FALSE", "TRUE", "FALSE"]
        s_modal, _ = sample_modal_chain(depth=length_of_chain, operator=op, accessible_worlds=acc, truths=t_vals)
        ctx, ans, _ = generate_modal_sample_tokens(s_modal)
        _, ctx_eng = convert_to_english(ctx)
        _, ans_eng = convert_to_english(ans, is_question=False)
        output_string += f"{ctx_eng} {ans_eng}\n"
        gt_string = ans_eng
    return output_string, gt_string


def generate_cot_question_query_based(length_of_chain: int = 2, num_cot_samples: int = 4) -> Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any]]:
    examples_str, _ = sample_context_and_answer_pairs_EXAMPLES(num_cot_samples, length_of_chain)
    s_modal, s_lin = sample_modal_chain(depth=length_of_chain, operator="BOX", accessible_worlds=["w0", "w1"], truths=["TRUE", "TRUE", "TRUE"])

    ctx_m, ans_m, _ = generate_modal_sample_tokens(s_modal)
    ctx_l, ans_l, _ = generate_modal_sample_tokens(s_lin)

    _, prompt_m = convert_to_english(ctx_m)
    _, prompt_l = convert_to_english(ctx_l)
    _, gt_m = convert_to_english(ans_m, is_question=False)
    _, gt_l = convert_to_english(ans_l, is_question=False)

    full_prompt_m = f"{examples_str}\n{prompt_m}"
    full_prompt_l = f"{examples_str}\n{prompt_l}"

    info_m = {"queried_rule": s_modal["queried_rule"], "correct_fact": s_modal["correct_fact"], "operator": "BOX"}
    info_l = {"queried_rule": s_lin["queried_rule"], "correct_fact": s_lin["correct_fact"], "operator": "BOX"}

    return full_prompt_m, gt_m, full_prompt_l, gt_l, info_m, info_l


def generate_cot_question_operator_based(length_of_chain: int = 2, num_cot_samples: int = 4) -> Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any]]:
    examples_str, _ = sample_context_and_answer_pairs_EXAMPLES(num_cot_samples, length_of_chain)
    s_box, _ = sample_modal_chain(depth=length_of_chain, operator="BOX", accessible_worlds=["w0", "w1"], truths=["FALSE", "TRUE", "FALSE"])
    s_dia, _ = sample_modal_chain(depth=length_of_chain, operator="DIAMOND", accessible_worlds=["w0", "w1"], truths=["FALSE", "TRUE", "FALSE"])

    ctx_b, ans_b, _ = generate_modal_sample_tokens(s_box)
    ctx_d, ans_d, _ = generate_modal_sample_tokens(s_dia)

    _, prompt_b = convert_to_english(ctx_b)
    _, prompt_d = convert_to_english(ctx_d)
    _, gt_b = convert_to_english(ans_b, is_question=False)
    _, gt_d = convert_to_english(ans_d, is_question=False)

    return f"{examples_str}\n{prompt_b}", gt_b, f"{examples_str}\n{prompt_d}", gt_d, s_box, s_dia


def generate_cot_question_accessibility_based(length_of_chain: int = 2, num_cot_samples: int = 4) -> Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any]]:
    examples_str, _ = sample_context_and_answer_pairs_EXAMPLES(num_cot_samples, length_of_chain)
    s_acc2, _ = sample_modal_chain(depth=length_of_chain, operator="DIAMOND", accessible_worlds=["w0", "w1"], truths=["FALSE", "TRUE", "FALSE"])
    s_acc1, _ = sample_modal_chain(depth=length_of_chain, operator="DIAMOND", accessible_worlds=["w0"], truths=["FALSE", "TRUE", "FALSE"])

    ctx_2, ans_2, _ = generate_modal_sample_tokens(s_acc2)
    ctx_1, ans_1, _ = generate_modal_sample_tokens(s_acc1)

    _, prompt_2 = convert_to_english(ctx_2)
    _, prompt_1 = convert_to_english(ctx_1)
    _, gt_2 = convert_to_english(ans_2, is_question=False)
    _, gt_1 = convert_to_english(ans_1, is_question=False)

    return f"{examples_str}\n{prompt_2}", gt_2, f"{examples_str}\n{prompt_1}", gt_1, s_acc2, s_acc1


def generate_cot_question_graded_operator_based(length_of_chain: int = 2, num_cot_samples: int = 4) -> Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any]]:
    """Generate counterfactual pair contrasting 'probably' (majority) vs 'certainly' (all)."""
    examples_str, _ = sample_context_and_answer_pairs_EXAMPLES(num_cot_samples, length_of_chain)
    s_prob, _ = sample_modal_chain(depth=length_of_chain, operator="PROBABLY", accessible_worlds=["w0", "w1"], truths=["TRUE", "FALSE", "FALSE"])
    s_cert, _ = sample_modal_chain(depth=length_of_chain, operator="CERTAINLY", accessible_worlds=["w0", "w1"], truths=["TRUE", "FALSE", "FALSE"])

    ctx_p, ans_p, _ = generate_modal_sample_tokens(s_prob)
    ctx_c, ans_c, _ = generate_modal_sample_tokens(s_cert)

    _, prompt_p = convert_to_english(ctx_p)
    _, prompt_c = convert_to_english(ctx_c)
    _, gt_p = convert_to_english(ans_p, is_question=False)
    _, gt_c = convert_to_english(ans_c, is_question=False)

    return f"{examples_str}\n{prompt_p}", gt_p, f"{examples_str}\n{prompt_c}", gt_c, s_prob, s_cert


def generate_cot_question_connective_based(length_of_chain: int = 2, num_cot_samples: int = 4) -> Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any]]:
    """Generate counterfactual pair contrasting disjunctive 'or' vs conjunctive 'and' rules."""
    examples_str, _ = sample_context_and_answer_pairs_EXAMPLES(num_cot_samples, length_of_chain)
    s_or, _ = sample_modal_chain(depth=length_of_chain, operator="BOX", accessible_worlds=["w0", "w1"], truths=["TRUE", "FALSE", "FALSE"])
    s_and, _ = sample_modal_chain(depth=length_of_chain, operator="BOX", accessible_worlds=["w0", "w1"], truths=["TRUE", "FALSE", "FALSE"])

    ctx_o, ans_o, _ = generate_modal_sample_tokens(s_or)
    ctx_a, ans_a, _ = generate_modal_sample_tokens(s_and)

    _, prompt_o = convert_to_english(ctx_o)
    _, prompt_a = convert_to_english(ctx_a)
    _, gt_o = convert_to_english(ans_o, is_question=False)
    _, gt_a = convert_to_english(ans_a, is_question=False)

    return f"{examples_str}\n{prompt_o}", gt_o, f"{examples_str}\n{prompt_a}", gt_a, s_or, s_and
