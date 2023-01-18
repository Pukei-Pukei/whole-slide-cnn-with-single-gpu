import os
from math import ceil

import torch
from timm.utils.agc import adaptive_clip_grad
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def extract_feature(model, phase, device, patch_batch_list, batch_size, config):
    iterator = tqdm(patch_batch_list, total=int(len(patch_batch_list)), leave=False)
    iterator.set_description('Extracting')

    if phase == 'train':
        model.train()
    elif phase == 'val':
        model.eval()
    else:
        raise Exception(f'Wrong phase argument : {phase}')

    H = config.RESIZE[0] // config.PATCH_SIZE
    W = config.RESIZE[1] // config.PATCH_SIZE

    feature_map = torch.zeros(batch_size, config.NUM_FEATURES, H, W)

    for patch_batch, coords, labels in iterator:
        patch_batch = patch_batch.to(device, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(patch_batch)
            for idx, (img_idx, i, j) in enumerate(coords):
                feature_map[img_idx, :, i, j] = outputs[idx]

    return feature_map


def train_extractor(model, gradient, optimizer, device, patch_batch_list, grad_divisor, config):
    iterator = tqdm(patch_batch_list, total=int(len(patch_batch_list)), leave=False)
    iterator.set_description('Training_Extractor')

    model.train()

    for idx, (patch_batch, coords, labels) in enumerate(iterator):
        patch_batch = patch_batch.to(device, non_blocking=True)

        grad_batch = []
        for img_idx, i, j in coords:
            grad_batch.append(gradient[img_idx, :, i, j])
        grad_batch = torch.stack(grad_batch) / grad_divisor

        outputs = model(patch_batch)
        outputs.backward(grad_batch)

    adaptive_clip_grad(model.parameters(),  clip_factor=config.CLIP_FACTOR)
    optimizer.step()
    optimizer.zero_grad()


def train_classifier_one_batch(model, loss_fn, optimizer, device, feature_map, labels):
    model.train()

    feature_map, labels = feature_map.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    feature_map.requires_grad = True

    pred = model(feature_map)
    loss = loss_fn(pred, labels)
            
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return feature_map.grad, loss.item(), pred


def validate_classifier_one_batch(model, loss_fn, device, feature_map, labels):
    model.eval()

    feature_map, labels = feature_map.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    with torch.no_grad():
        pred = model(feature_map)
        loss = loss_fn(pred, labels)

    return loss.item(), pred


def train_one_epoch(extractor, classifier, loss_fn, ext_optimizer, clf_optimizer, ext_scheduler, clf_scheduler, device, train_loader, config, history={'train': [], 'val': []}):
    if not history['train']:
        iter = 0
    else:
        iter = history['train'][-1]['iter']

    loss_list = []
    pred_list = []
    target_list = []
    loss_ema = 0
    f1score_recent = 0
    iterator = tqdm(train_loader, total=int(len(train_loader)))

    for i, (patch_batch_list, labels) in enumerate(iterator):
        feature_map = extract_feature(extractor, 'train', device, patch_batch_list, len(labels), config)

        gradient, loss, pred = train_classifier_one_batch(classifier, loss_fn, clf_optimizer, device, feature_map, labels)

        if not config.TRAIN_ONLY_CLASSIFIER:
            grad_divisor = (config.RESIZE[0] // config.PATCH_SIZE) * (config.RESIZE[1] // config.PATCH_SIZE)
            train_extractor(extractor, gradient, ext_optimizer, device, patch_batch_list, grad_divisor, config)
            ext_scheduler.step()
        clf_scheduler.step()

        loss_list.append(loss)
        if config.DATA.IS_MULTI_LABEL:
            threshold = 0.5
            pred_list.extend(((torch.sigmoid(pred) >= threshold).to(torch.int)).tolist())
            average = 'samples'
        else:
            pred_list.extend(torch.argmax(pred, dim=1).tolist())
            average = 'macro'
        target_list.extend(labels.tolist())
        
        if i % config.SCORE_UPDATE_FREQ == 0:
            window = config.SCORE_WINDOW * config.BATCH_SIZE
            if len(target_list) > window:
                loss_recent = sum(loss_list[-config.SCORE_WINDOW:]) / config.SCORE_WINDOW
                acc_recent = accuracy_score(target_list[-window:], pred_list[-window:])
                f1score_recent = f1_score(target_list[-window:], pred_list[-window:], average=average)
            else:
                loss_recent = sum(loss_list) / len(loss_list)
                acc_recent = accuracy_score(target_list, pred_list)
                f1score_recent = f1_score(target_list, pred_list, average=average)

            hist_dict = {'iter': iter + i + 1, 'loss_recent': loss_recent, 'acc_recent': acc_recent, 'f1score_recent': f1score_recent}
            history['train'].append(hist_dict)

        iterator.set_postfix(hist_dict)

        if config.SAVE_ITER_FREQ is not None and (i+1) % config.SAVE_ITER_FREQ == 0:
            save_dict = {'extractor_state_dict': extractor.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'ext_optimizer_state_dict': ext_optimizer.state_dict(),
                        'clf_optimizer_state_dict': clf_optimizer.state_dict(),
                        'ext_scheduler_state_dict': ext_scheduler.state_dict(),
                        'clf_scheduler_state_dict': clf_scheduler.state_dict(),
                        'iter': iter,
                        'history': history,
                        'config': config}
            torch.save(save_dict,
                       os.path.join(config.DATA.PATH.RECORDER, f"last_iter__train_save.pth"))

    loss = sum(loss_list) / len(loss_list)
    acc = accuracy_score(target_list, pred_list)
    f1score = f1_score(target_list, pred_list, average=average)

    hist_dict = {'loss': loss, 'acc': acc, 'f1score': f1score}
    history['train'][-1].update(hist_dict)

    iterator.set_postfix(history['train'][-1])
    iterator.display()

    return history


def validate(extractor, classifier, loss_fn, device, val_loader, config, history={'train': [], 'val': []}):
    if not history['train']:
        iter = 0
    else:
        iter = history['train'][-1]['iter']

    loss_list = []
    pred_list = []
    target_list = []
    iterator = tqdm(val_loader, total=int(len(val_loader)))

    for i, (patch_batch_list, labels) in enumerate(iterator):
        feature_map = extract_feature(extractor, 'val', device, patch_batch_list, len(labels), config)

        loss, pred = validate_classifier_one_batch(classifier, loss_fn, device, feature_map, labels)

        loss_list.append(loss)
        if config.DATA.IS_MULTI_LABEL:
            threshold = 0.5
            pred_list.extend(((torch.sigmoid(pred) >= threshold).to(torch.int)).tolist())
            average = 'samples'
        else:
            pred_list.extend(torch.argmax(pred, dim=1).tolist())
            average = 'macro'
        target_list.extend(labels.tolist())
        
        if i % config. SCORE_UPDATE_FREQ == 0:
            loss_mean = sum(loss_list) / len(loss_list)
            acc = accuracy_score(target_list, pred_list)
            f1score = f1_score(target_list, pred_list, average=average)

        hist_dict = {'VALIDATION': iter, 'loss': loss_mean, 'acc': acc, 'f1score': f1score}

        iterator.set_postfix(hist_dict)

    loss_mean = sum(loss_list) / len(loss_list)
    acc = accuracy_score(target_list, pred_list)
    f1score = f1_score(target_list, pred_list, average=average)

    hist_dict = {'iter': iter, 'loss': loss_mean, 'acc': acc, 'f1score': f1score}

    iterator.set_postfix({'VALIDATION': iter, 'loss': loss_mean, 'acc': acc, 'f1score': f1score})
    iterator.display()

    history['val'].append(hist_dict)

    return history, target_list, pred_list


def inference(extractor, classifier, device, test_loader, config):
    pred_list = []
    iterator = tqdm(test_loader, total=int(len(test_loader)))

    for i, (patch_batch_list, labels) in enumerate(iterator):
        feature_map = extract_feature(extractor, 'val', device, patch_batch_list, len(labels), config)

        classifier.eval()
        feature_map, labels = feature_map.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = classifier(feature_map)

        if config.INFER_LOGIT:
            pred_list.extend(pred.tolist())
        else:
            if config.DATA.IS_MULTI_LABEL:
                threshold = 0.5
                pred_list.extend(((torch.sigmoid(pred) >= threshold).to(torch.int)).tolist())
            else:
                pred_list.extend(torch.argmax(pred, dim=1).tolist())

    return pred_list