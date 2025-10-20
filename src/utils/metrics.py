import torch

def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_nse(preds, labels, null_val=torch.tensor(float('nan'))):
    """
    Nash–Sutcliffe Efficiency as a loss: 1 - NSE
    """
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    obs_mean = torch.sum(labels * mask) / torch.sum(mask)
    numerator = torch.sum(((labels - preds) ** 2) * mask)
    denominator = torch.sum(((labels - obs_mean) ** 2) * mask)
    nse = 1 - numerator / (denominator + 1e-6)
    return nse


def masked_corr_loss(preds, labels, null_val=torch.tensor(float('nan'))):
    """
    Pearson correlation loss: 1 - r
    """
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    preds_mean = torch.sum(preds * mask) / torch.sum(mask)
    labels_mean = torch.sum(labels * mask) / torch.sum(mask)

    cov = torch.sum(((preds - preds_mean) * (labels - labels_mean)) * mask)
    preds_var = torch.sum((preds - preds_mean) ** 2 * mask)
    labels_var = torch.sum((labels - labels_mean) ** 2 * mask)

    corr = cov / (torch.sqrt(preds_var * labels_var) + 1e-6)
    return corr

def masked_modified_nse(preds, labels, null_val=torch.tensor(float('nan'))):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    labels_mean = torch.sum(labels * mask) / torch.sum(mask)

    numerator = torch.sum(((preds - labels) ** 2) * mask)
    denominator = torch.sum(((torch.abs(labels - labels_mean) + torch.abs(preds - labels_mean)) ** 2) * mask)

    mnse = 1 - numerator / (denominator + 1e-6)
    return mnse

def masked_kge(preds, labels, null_val=torch.tensor(float('nan'))):
    """
    Kling–Gupta Efficiency
    """
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    y_mean = torch.sum(labels * mask) / torch.sum(mask)
    y_std = torch.sqrt(torch.sum(((labels - y_mean) ** 2) * mask) / torch.sum(mask))

    p_mean = torch.sum(preds * mask) / torch.sum(mask)
    p_std = torch.sqrt(torch.sum(((preds - p_mean) ** 2) * mask) / torch.sum(mask))

    # Pearson correlation
    r_num = torch.sum((labels - y_mean) * (preds - p_mean) * mask)
    r_den = torch.sqrt(torch.sum(((labels - y_mean) ** 2) * mask) * torch.sum(((preds - p_mean) ** 2) * mask))
    r = r_num / (r_den + 1e-6)

    beta = p_mean / (y_mean + 1e-6)
    gamma = p_std / (y_std + 1e-6)

    return 1 - torch.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    nse = masked_nse(preds, labels, null_val).item()
    corr = masked_corr_loss(preds, labels, null_val).item()
    mNse = masked_modified_nse(preds, labels, null_val).item()
    kge = masked_kge(preds, labels, null_val).item()
    return mae, mape, rmse, nse, corr, mNse, kge