from sklearn.metrics import precision_score, recall_score, f1_score


def eval_binary(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'uar': recall_score(y_true, y_pred, average='macro')
    }