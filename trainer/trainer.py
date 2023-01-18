import os

import torch
import pandas as pd
from tqdm.auto import tqdm

from .util import *
from dataloader import CustomDataset, CustomDataset_csv


def train_model(extractor,
                classifier,
                loss_fn,
                ext_optimizer,
                clf_optimizer,
                ext_scheduler,
                clf_scheduler,
                device,
                loaders,
                config):
    best_score = config.BEST_SCORE
    resume = config.RESUME_PATH
    history = {'train': [], 'val': []}

    if resume:
        checkpoint = torch.load(resume)
        extractor.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

        if config.LOAD_HISTORY:
            history = checkpoint['history']
            # best_score = checkpoint['history']['val'][-1]['f1score'] 
        if config.LOAD_OPTIMIZER:
            ext_optimizer.load_state_dict(checkpoint['ext_optimizer_state_dict'])
            clf_optimizer.load_state_dict(checkpoint['clf_optimizer_state_dict'])
        if config.LOAD_SCHEDULER:
            ext_scheduler.load_state_dict(checkpoint['ext_scheduler_state_dict'])
            clf_scheduler.load_state_dict(checkpoint['clf_scheduler_state_dict'])
        else:
            for param_group in ext_optimizer.param_groups:
                param_group['lr'] = config.EXT_LR
            for param_group in clf_optimizer.param_groups:
                param_group['lr'] = config.CLF_LR

        del checkpoint

    iterator = tqdm(range(1, config.EPOCHS + 1))

    for i in iterator:
        history = train_one_epoch(extractor, classifier, loss_fn, ext_optimizer, clf_optimizer, ext_scheduler, clf_scheduler, device, loaders['train'], config, history)

        if loaders['val'] is not None:
            history, _, _ = validate(extractor, classifier, loss_fn, device, loaders['val'], config, history)

        save_dict = {'extractor_state_dict': extractor.state_dict(),
                     'classifier_state_dict': classifier.state_dict(),
                     'ext_optimizer_state_dict': ext_optimizer.state_dict(),
                     'clf_optimizer_state_dict': clf_optimizer.state_dict(),
                     'ext_scheduler_state_dict': ext_scheduler.state_dict(),
                     'clf_scheduler_state_dict': clf_scheduler.state_dict(),
                     'iter': history['train'][-1]['iter'],
                     'history': history,
                     'config': config}

        if loaders['val'] is not None:
            f1score = history['val'][-1]['f1score']
            best_iter = history['val'][-1]['iter']
            if best_score < f1score:
                best_score = f1score

                torch.save(save_dict,
                        os.path.join(config.DATA.PATH.RECORDER, "best.pth"))

                with open(os.path.join(config.DATA.PATH.RECORDER, "best.txt"), "w") as f:
                    f.write(f"iter\t{best_iter:08}\nf1score\t{f1score:.4f}")

                iterator.set_postfix(best_score=best_score, best_iter=best_iter)
        else:
            f1score = history['train'][-1]['f1score']
            best_iter = history['train'][-1]['iter']
            if best_score < f1score:
                best_score = f1score
                iterator.set_postfix(best_score=best_score, best_iter=best_iter)

        if i % config.SAVE_FREQ == 0 or i == config.EPOCHS - 1:
            torch.save(save_dict,
                       os.path.join(config.DATA.PATH.RECORDER, f"iter{best_iter:08}_f1score{f1score*1e4:.0f}.pth"))


def eval_model(extractor,
                classifier,
                loss_fn,
                ext_optimizer,
                clf_optimizer,
                ext_scheduler,
                clf_scheduler,
                device,
                loaders,
                config):
    resume = config.RESUME_PATH
    history = {'train': [], 'val': []}

    if resume:
        checkpoint = torch.load(resume)
        extractor.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        del checkpoint

    history, target_list, pred_list = validate(extractor, classifier, loss_fn, device, loaders['val'], config, history)

    save_dict = {'history': history, 'target_list': target_list, 'pred_list': pred_list}
    torch.torch.save(save_dict, os.path.join(config.DATA.PATH.RECORDER, "eval_result.pth"))


def infer_model(extractor,
                classifier,
                loss_fn,
                ext_optimizer,
                clf_optimizer,
                ext_scheduler,
                clf_scheduler,
                device,
                loaders,
                config):
    resume = config.RESUME_PATH

    if resume:
        checkpoint = torch.load(resume)
        extractor.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        del checkpoint

    pred_list = inference(extractor, classifier, device, loaders['test'], config)
    df = pd.read_csv(config.DATA.PATH.TEST_CSV)

    if config.INFER_LOGIT:
        for i in range(config.NUM_CLASSES):
            df[i] = [v[i] for v in pred_list]
    else:
        dataset = CustomDataset_csv(config, 'test') if config.DATA.READ_TYPE == 'csv' else CustomDataset(config, 'test')
        classes = dataset.classes
        pred_temp = []
        if config.DATA.IS_MULTI_LABEL:
            for pred in pred_list:
                temp = []
                for i, v in enumerate(pred):
                    if v == 1: temp.append(classes[i])
                pred_temp.append(' '.join(temp))
        else:
            for pred in pred_list:
                pred_temp.append(classes[pred])

        pred_list = pred_temp
        df[config.DATA.LABEL_COL] = pred_list
    
    df.to_csv(os.path.join(config.DATA.PATH.RECORDER, 'submission.csv'), index=False)