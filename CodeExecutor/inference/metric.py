"""
Evaluation metrics to measure functional correctness of traces.
"""
text_identifier_num = 0
gold_identifier_num = 0
correct_identifier_num = 0

def get_output_from_trace(text):
    output_list = []
    parse_loc = []
    start_len = 0
    while True:
        num = text.find("<line>",start_len)
        if num == -1: break
        parse_loc.append(num)
        start_len = num + 1
    start_len = 0
    while True:
        num = text.find("<output>",start_len)
        if num == -1: break
        parse_loc.append(num)
        start_len = num + 1
    # add 0 and len(text)
    parse_loc.append(0)
    parse_loc.append(len(text))
    parse_loc = list(set(parse_loc))
    parse_loc.sort()
    
    for i, loc in enumerate(parse_loc):
        if i == 0: continue
        # remove the last incomplete sentence in gold
        if i == len(parse_loc)-1:
            if "</state>" not in text[parse_loc[i-1]:loc]:
                continue
        if "<output>" in text[parse_loc[i-1]:loc]:
            my_output = text[parse_loc[i-1]+len("<output> "):loc].strip()
            if "_____event" in my_output:
                my_output = my_output[0:my_output.find("_____event")].strip()
            if len(my_output) > 0:
                output_list.append(my_output)
    return output_list

def parse_text_into_sent(text):
    text_list = []
    parse_loc = []
    start_len = 0
    while True:
        num = text.find("<line>",start_len)
        if num == -1: break
        parse_loc.append(num)
        start_len = num + 1
    start_len = 0
    while True:
        num = text.find("<output>",start_len)
        if num == -1: break
        parse_loc.append(num)
        start_len = num + 1
    # add 0 and len(text)
    parse_loc.append(0)
    parse_loc.append(len(text))
    parse_loc = list(set(parse_loc))
    parse_loc.sort()
    
    for i, loc in enumerate(parse_loc):
        if i == 0: continue
        # remove the last incomplete sentence in text
        if i == len(parse_loc)-1:
            if "</state>" not in text[parse_loc[i-1]:loc]:
                continue
        text_list.append(text[parse_loc[i-1]:loc])
    return text_list

def parse_gold_into_sent(text):
    text_list = []
    parse_loc = []
    start_len = 0
    while True:
        num = text.find("<line>",start_len)
        if num == -1: break
        parse_loc.append(num)
        start_len = num + 1
    start_len = 0
    while True:
        num = text.find("<output>",start_len)
        if num == -1: break
        parse_loc.append(num)
        start_len = num + 1
    # add 0 and len(text)
    parse_loc.append(0)
    parse_loc.append(len(text))
    parse_loc = list(set(parse_loc))
    parse_loc.sort()
    
    for i, loc in enumerate(parse_loc):
        if i == 0: continue
        # remove the last incomplete sentence in gold
        if i == len(parse_loc)-1:
            if "</state>" not in text[parse_loc[i-1]:loc]:
                continue
        text_list.append(text[parse_loc[i-1]:loc])
    return text_list

def dict_same_key_value_num(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    same_num = 0
    for key in intersect_keys:
        if d1[key] == d2[key]:
            same_num += 1
    return same_num

def same_sent(text,gold):
    if "<output>" in text or "<output>" in gold:
        if text == gold: return True
        else: return False
    if "<state>" not in text or "<state>" not in gold:
        if text == gold: return True
        else: return False
    text_sep = text
    text_linenum_info = text[0:text.find("<state>")].strip()
    text_sep = text_sep[text_sep.find("<state>"):].strip()
    if text_sep.startswith("<state>"):
        text_sep = text_sep[len("<state>"):]
    if text_sep.endswith("</state>"):
        text_sep = text_sep[:-len("</state>")]
    text_sep = text_sep.split("<dictsep>")
    text_dict = {}
    for state in text_sep:
        if ":" in state:
            text_dict[state[0:state.index(":")].strip()] = state[state.index(":")+1:].strip()
    gold_sep = gold
    gold_linenum_info = gold[0:gold.find("<state>")].strip()
    gold_sep = gold_sep[gold_sep.find("<state>"):].strip()
    if gold_sep.startswith("<state>"):
        gold_sep = gold_sep[len("<state>"):]
    if gold_sep.endswith("</state>"):
        gold_sep = gold_sep[:-len("</state>")]
    gold_sep = gold_sep.split("<dictsep>")
    gold_dict = {}
    for state in gold_sep:
        if ":" in state:
            gold_dict[state[0:state.index(":")].strip()] = state[state.index(":")+1:].strip()
    
    global correct_identifier_num
    if text_linenum_info == gold_linenum_info: 
        correct_identifier_num += dict_same_key_value_num(text_dict,gold_dict)

    if text_linenum_info == gold_linenum_info and text_dict == gold_dict: return True
    return False

def get_identifier_num(text_list):
    res = 0
    for text in text_list:
        text_sep = text
        text_linenum_info = text[0:text.find("<state>")].strip()
        text_sep = text_sep[text_sep.find("<state>"):].strip()
        if text_sep.startswith("<state>"):
            text_sep = text_sep[len("<state>"):]
        if text_sep.endswith("</state>"):
            text_sep = text_sep[:-len("</state>")]
        text_sep = text_sep.split("<dictsep>")
        text_dict = {}
        for state in text_sep:
            if ":" in state:
                text_dict[state[0:state.index(":")].strip()] = state[state.index(":")+1:].strip()
        res += len(text_dict)
    return res
        
# Compute metrics in the Tutorial or CodeNetMut dataset.
def compute_metrics(preds, golds):
    assert(len(preds) == len(golds))
    em_num = 0
    total_num = 0
    output_same_num = 0
    gold_has_output_num = 0
    precision_list_line = []
    recall_list_line = []
    precision_list_id = []
    recall_list_id = []
    right_all_num = 0
    text_all_num = 0
    gold_all_num = 0
    right_id_all_num = 0
    text_id_all_num = 0
    gold_id_all_num = 0
    global correct_identifier_num

    for i, pred in enumerate(preds):
        text = pred.strip()
        gold = golds[i].strip()
        total_num += 1

        gold_output = get_output_from_trace(gold)
        predict_output = get_output_from_trace(text)
        if len(gold_output) > 0:
            gold_has_output_num += 1
        if len(gold_output) > 0 and gold_output == predict_output:
            output_same_num += 1

        text_list = parse_text_into_sent(text) 
        gold_list = parse_gold_into_sent(gold)
        text_sent_num = len(text_list)
        gold_sent_num = len(gold_list)
        same_sent_num = 0
        text_identifier_num = 0
        gold_identifier_num = 0
        global correct_identifier_num
        correct_identifier_num = 0
        for i in range(0,gold_sent_num):
            if i < text_sent_num and same_sent(text_list[i],gold_list[i]) == True:
                same_sent_num += 1
        
        text_identifier_num = get_identifier_num(text_list)
        gold_identifier_num = get_identifier_num(gold_list)

        precision_tmp = same_sent_num/text_sent_num if text_sent_num != 0 else 0
        recall_tmp = same_sent_num/gold_sent_num if gold_sent_num != 0 else 0
        precision_list_line.append(precision_tmp)
        recall_list_line.append(recall_tmp)
        right_all_num += same_sent_num
        text_all_num += text_sent_num
        gold_all_num += gold_sent_num

        precision_id = correct_identifier_num/text_identifier_num if text_identifier_num != 0 else 0
        recall_id = correct_identifier_num/gold_identifier_num if gold_identifier_num != 0 else 0
        precision_list_id.append(precision_id)
        recall_list_id.append(recall_id)
        right_id_all_num += correct_identifier_num
        text_id_all_num += text_identifier_num
        gold_id_all_num += gold_identifier_num

        if same_sent_num == gold_sent_num and text_sent_num == gold_sent_num:
            em_num += 1
   
    metric_list = []
    output_acc = output_same_num /gold_has_output_num
    em = em_num/total_num
    metric_list.append(round(100 * output_acc, 2))
    metric_list.append(round(100 * em, 2))

    line_micro_precision = right_all_num/text_all_num
    line_macro_precision = sum(precision_list_line)/len(precision_list_line)
    line_micro_recall = right_all_num/gold_all_num
    line_macro_recall = sum(recall_list_line)/len(recall_list_line)
    line_f1 = 2 * line_micro_precision * line_micro_recall / (line_micro_precision + line_micro_recall)
    metric_list.append(round(100 * line_micro_precision, 2))
    metric_list.append(round(100 * line_micro_recall, 2))
    metric_list.append(round(100 * line_f1, 2))

    id_micro_precision = right_id_all_num/text_id_all_num
    id_macro_precision = sum(precision_list_id)/len(precision_list_id)
    id_micro_recall = right_id_all_num/gold_id_all_num
    id_macro_recall = sum(recall_list_id)/len(recall_list_id)
    id_f1 = 2 * id_micro_precision * id_micro_recall / (id_micro_precision + id_micro_recall)
    metric_list.append(round(100 * id_micro_precision, 2))
    metric_list.append(round(100 * id_micro_recall, 2))
    metric_list.append(round(100 * id_f1, 2))

    return metric_list

# Evaluation metrics to measure correctness of single-line traces, especially designed for the SingleLine dataset.
def compute_singleline_metrics(pred_list, gold_list):
    assert(len(pred_list) == len(gold_list))
    em_num = 0
    total_num = 0
    precision_list_id = []
    recall_list_id = []
    right_id_all_num = 0
    text_id_all_num = 0
    gold_id_all_num = 0
    for i, pred in enumerate(pred_list):
        text = pred.strip()
        gold = gold_list[i].strip()
        total_num += 1
        text_sep = text
        if text_sep.startswith("<state>"):
            text_sep = text_sep[len("<state>"):]
        if text_sep.endswith("</state>"):
            text_sep = text_sep[:-len("</state>")]
        text_sep = text_sep.split("<dictsep>")
        text_dict = {}
        for state in text_sep:
            if ":" in state:
                text_dict[state.split(":")[0].strip()] = state.split(":")[1].strip()
        gold_sep = gold
        if gold_sep.startswith("<state>"):
            gold_sep = gold_sep[len("<state>"):]
        if gold_sep.endswith("</state>"):
            gold_sep = gold_sep[:-len("</state>")]
        gold_sep = gold_sep.split("<dictsep>")
        gold_dict = {}
        for state in gold_sep:
            if ":" in state:
                gold_dict[state.split(":")[0].strip()] = state.split(":")[1].strip()
        
        correct_id_num = dict_same_key_value_num(text_dict,gold_dict)
        text_identifier_num = len(text_dict)
        gold_identifier_num = len(gold_dict)

        precision_id = correct_id_num/text_identifier_num if text_identifier_num != 0 else 0
        recall_id = correct_id_num/gold_identifier_num if gold_identifier_num != 0 else 0
        precision_list_id.append(precision_id)
        recall_list_id.append(recall_id)
        right_id_all_num += correct_id_num
        text_id_all_num += text_identifier_num
        gold_id_all_num += gold_identifier_num

        if text_dict == gold_dict:
            em_num += 1
                
    metric_list = []
    em = em_num/total_num
    metric_list.append(round(100 * em, 2))

    id_micro_precision = right_id_all_num/text_id_all_num
    id_macro_precision = sum(precision_list_id)/len(precision_list_id)
    id_micro_recall = right_id_all_num/gold_id_all_num
    id_macro_recall = sum(recall_list_id)/len(recall_list_id)
    id_f1 = 2 * id_micro_precision * id_micro_recall / (id_micro_precision + id_micro_recall)
    metric_list.append(round(100 * id_micro_precision, 2))
    metric_list.append(round(100 * id_micro_recall, 2))
    metric_list.append(round(100 * id_f1, 2))

    return metric_list