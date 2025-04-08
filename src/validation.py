'''Validation method'''
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score
import torch
from tqdm import tqdm

def validation(model, validds, loss, device,):
    all_embeddings = []
    all_labels = []
    run_val_loss_cls = 0
    model = model.to(device)
    model.eval()
    for sample, labels, _ in tqdm(validds):
        with torch.no_grad():
            labels = labels.to(device, non_blocking=True)
            embeddings = model(sample.to(device, non_blocking=True))

            loss_cls = loss(embeddings, labels) if loss is not None else None
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
        run_val_loss_cls += loss_cls.item() if loss is not None else 0
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Считаем метрики
    acc = get_metrics(all_embeddings, all_labels)
    acc['loss'] = run_val_loss_cls / len(validds)
    torch.cuda.empty_cache()
    return acc

def get_metrics(output, target):
    """
    Рассчитывает метрики average precision, precision, recall, F1-score и accuracy для многоклассовой классификации.

    :param output: Тензор (batch, n_cls) с логитами модели.
    :param target: Тензор (batch,) с истинными метками классов.
    :return: Словарь с метриками.
    """
    batch_size, n_cls = output.shape

    # Получаем предсказанные классы (argmax по логитам)
    pred_probs = F.softmax(output, dim=1).cpu().numpy()  # Преобразуем логиты в вероятности
    pred = pred_probs.argmax(axis=1)
    target = target.cpu().numpy()

    # Вычисляем метрики
    metrics = {
        "accuracy": accuracy_score(target, pred),
        "precision": precision_score(target, pred, average='weighted', zero_division=0),
        "recall": recall_score(target, pred, average='weighted', zero_division=0),
        "f1_score": f1_score(target, pred, average='weighted', zero_division=0),
    }

    # Для AP нужен формат one-vs-all, преобразуем метки в бинарные (one-hot encoding)
    target_one_hot = F.one_hot(torch.tensor(target), num_classes=n_cls).numpy()
    
    try:
        metrics["average_precision"] = average_precision_score(target_one_hot, pred_probs, average=None)
    except ValueError:
        metrics["average_precision"] = 0.0  # Если не удалось вычислить

    return metrics

if __name__ == '__main__':
    pass