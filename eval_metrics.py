import evaluate
import numpy as np

perplexity_metric = evaluate.load("perplexity", module_type="metric")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return perplexity_metric.compute(predictions=predictions, references=labels)
