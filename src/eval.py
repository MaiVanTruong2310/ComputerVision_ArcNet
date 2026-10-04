# ==============================================================================
# Tên file: eval.py
# Công dụng: Các hàm đánh giá mô hình nhận diện khuôn mặt trên tập validation (.bin).
#            Hỗ trợ tính toán độ chính xác Accuracy, sai lệch tiêu chuẩn Std, 
#            vẽ đường cong ROC, tính toán tỷ lệ lỗi bằng nhau EER (Equal Error Rate),
#            và thực hiện kỹ thuật Test-Time Augmentation (TTA - lật ảnh) để tối ưu độ chính xác.
# ==============================================================================
import torch
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    # Cosine similarity
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold (Cross Validation)
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds)
    
    # Calculate EER (Equal Error Rate)
    # EER is where FPR == 1 - TPR (or where FPR == FNR)
    # FNR = 1 - TPR
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    
    return accuracy.mean(), accuracy.std(), best_thresholds.mean(), eer, tpr, fpr

@torch.no_grad()
def perform_val(model, device, val_data, issame_list, batch_size=128, return_full=False):
    model.eval()
    embeddings = []
    
    # Test-Time Augmentation (TTA - Tăng cường lúc test)
    for i in range(0, len(val_data), batch_size):
        batch = torch.stack(val_data[i:i+batch_size]).to(device)
        # 1. Thực hiện lật ngang batch ảnh hiện tại (chiều số 3 là chiều Width [B, C, H, W])
        batch_flip = torch.flip(batch, [3])

        # 2. Trích xuất đặc trưng của batch ảnh gốc
        embed = model(batch)

        # 3. Trích xuất đặc trưng của batch ảnh đã lật ngang
        embed_flip = model(batch_flip)

        # 4. Cộng gộp 2 vector đặc trưng này lại với nhau
        combined = embed + embed_flip

        # 5. Thực hiện chuẩn hóa L2 (L2 Normalization) để đưa vector về độ dài bằng 1 trên mặt cầu
        combined = torch.nn.functional.normalize(combined)
        
        embeddings.append(combined.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    acc_mean, acc_std, threshold, eer, tpr, fpr = evaluate(embeddings, issame_list)
    
    if return_full:
        return acc_mean, acc_std, threshold, eer, tpr, fpr
    return acc_mean, acc_std, threshold
