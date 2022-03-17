"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, QANet, BiDAFBASE
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    word_vectors_new = util.torch_from_json("data/word_emb_new.json")
    character_vectors = util.torch_from_json(args.char_emb_file)

    # Get model
    log.info('Building model...')

    # bidaf
    # model_1 = BiDAFBASE(word_vectors=word_vectors,
    #                     hidden_size=args.hidden_size)
    # model_1 = nn.DataParallel(model_1, gpu_ids)
    # log.info(f'Loading checkpoint from {args.load_path} for model 2...')
    # model_1 = util.load_model(model_1, "save/train/baseline/best.pth.tar", gpu_ids,
    #                           return_step=False)
    # model_1 = model_1.to(device)
    # model_1.eval()


    #bidaf + char.emb
    # model_2 = BiDAF(word_vectors=word_vectors,
    #               hidden_size=args.hidden_size,
    #               character_vectors=character_vectors,
    #               char_channel_size=args.char_channel_size,
    #               char_channel_width=args.char_channel_width)
    #
    # model_2 = nn.DataParallel(model_2, gpu_ids)
    # log.info(f'Loading checkpoint from {args.load_path} for model 2...')
    # model_2 = util.load_model(model_2, "save/train/char_emb/best.pth.tar", gpu_ids,
    #                           return_step=False)
    # model_2 = model_2.to(device)
    # model_2.eval()

    # # bidaf + char.emb + data aug
    # model_7 = BiDAF(word_vectors=word_vectors_new,
    #                 hidden_size=args.hidden_size,
    #                 character_vectors=character_vectors,
    #                 char_channel_size=args.char_channel_size,
    #                 char_channel_width=args.char_channel_width)
    #
    # model_7 = nn.DataParallel(model_7, gpu_ids)
    # log.info(f'Loading checkpoint from {args.load_path} for model 2...')
    # model_7 = util.load_model(model_7, "aug.best.pth.tar", gpu_ids,
    #                           return_step=False)
    # model_7 = model_7.to(device)
    # model_7.eval()
    #
    # #qanet
    # model_3 = QANet(word_vectors=word_vectors,
    #               character_vectors=character_vectors,
    #               hidden_size=args.hidden_size,
    #               char_channel_size=args.char_channel_size,
    #               char_channel_width=args.char_channel_width)
    #
    # model_3 = nn.DataParallel(model_3, gpu_ids)
    # log.info(f'Loading checkpoint from {args.load_path} for model...')
    # model_3 = util.load_model(model_3, "save/train/qanet/best.pth.tar", gpu_ids, return_step=False)
    # model_3 = model_3.to(device)
    # model_3.eval()
    # #
    # # # qanet 4 head
    # model_4 = QANet(word_vectors=word_vectors,
    #                 character_vectors=character_vectors,
    #                 hidden_size=128,
    #                 char_channel_size=args.char_channel_size,
    #                 char_channel_width=args.char_channel_width)
    #
    # model_4 = nn.DataParallel(model_4, gpu_ids)
    # log.info(f'Loading checkpoint from {args.load_path} for model 2...')
    # model_4 = util.load_model(model_4, "save/train/4head128size-01/best.pth.tar", gpu_ids,
    #                           return_step=False)
    # model_4 = model_4.to(device)
    # model_4.eval()
    # #
    # # # qanet 8 head
    # model_5 = QANet(word_vectors=word_vectors,
    #                 character_vectors=character_vectors,
    #                 hidden_size=128,
    #                 char_channel_size=args.char_channel_size,
    #                 char_channel_width=args.char_channel_width)
    #
    # model_5 = nn.DataParallel(model_5, gpu_ids)
    # log.info(f'Loading checkpoint from {args.load_path} for model 2...')
    # model_5 = util.load_model(model_5, "8head.best.pth.tar", gpu_ids,
    #                           return_step=False)
    # model_5 = model_5.to(device)
    # model_5.eval()

    # # # qanet gru
    model_6 = QANet(word_vectors=word_vectors,
                    character_vectors=character_vectors,
                    hidden_size=args.hidden_size,
                    char_channel_size=args.char_channel_size,
                    char_channel_width=args.char_channel_width)

    model_6 = nn.DataParallel(model_6, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path} for model 2...')
    model_6 = util.load_model(model_6, "gru.best.pth.tar", gpu_ids,
                              return_step=False)
    model_6 = model_6.to(device)
    model_6.eval()


    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    # record_file = "./data/dev_other.npz"
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    # pred_dict_0 = {}
    # pred_dict_0_2 = {}
    # pred_dict_2_4 = {}
    # pred_dict_4_6 = {}
    # pred_dict_6 = {}
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    # eval_file = "./data/other_eval.json"
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model_6(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            # p1_2, p2_2 = model_2(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            # p1_3, p2_3 = model_3(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            # p1_4, p2_4 = model_4(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            # p1_5, p2_5 = model_5(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            # p1_6, p2_6 = model_6(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            # p1_7, p2_7 = model_7(cw_idxs, qw_idxs, cc_idxs, qc_idxs)

            # p1_2, p2_2 = p1_2.exp(), p2_2.exp()
            # p1_3, p2_3 = p1_3.exp(), p2_3.exp()
            # p1_4, p2_4 = p1_4.exp(), p2_4.exp()
            # p1_5, p2_5 = p1_5.exp(), p2_5.exp()
            # p1_6, p2_6 = p1_6.exp(), p2_6.exp()
            # p1_7, p2_7 = p1_7.exp(), p2_7.exp()

            # p1_avg = (p1_2 + p1_3 + p1_4 + p1_5 + p1_6 + p1_7)/7
            # p2_avg = (p2_2 + p2_3 + p2_4 + p2_5 + p2_6 + p2_7)/7
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)
            # starts, ends = util.discretize(p1_avg, p2_avg, args.max_ans_len, args.use_squad_v2)


            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            # ids_0 = []
            # starts_0 = []
            # ends_0 = []
            #
            # ids_0_2 = []
            # starts_0_2 = []
            # ends_0_2 = []
            #
            # ids_2_4 = []
            # starts_2_4 = []
            # ends_2_4 = []
            #
            # ids_4_6 = []
            # starts_4_6 = []
            # ends_4_6 = []
            #
            # ids_6 = []
            # starts_6 = []
            # ends_6 = []
            #
            # for i in range(len(ids)):
            #     id = ids[i]
            #     start = starts[i]
            #     end = ends[i]
            #     if end == 0 and start == 0:
            #         ids_0.append(id.item())
            #         starts_0.append(start)
            #         ends_0.append(end)
            #     elif end - start <= 2:
            #         ids_0_2.append(id.item())
            #         starts_0_2.append(start)
            #         ends_0_2.append(end)
            #     elif end - start <= 4:
            #         ids_2_4.append(id.item())
            #         starts_2_4.append(start)
            #         ends_2_4.append(end)
            #     elif end - start <= 6:
            #         ids_4_6.append(id.item())
            #         starts_4_6.append(start)
            #         ends_4_6.append(end)
            #     else:
            #         ids_6.append(id.item())
            #         starts_6.append(start)
            #         ends_6.append(end)
            #
            # idx2pred, uuid2pred = util.convert_tokens(gold_dict,
            #                                           ids_0,
            #                                           starts_0,
            #                                           ends_0,
            #                                           args.use_squad_v2)
            # pred_dict_0.update(idx2pred)
            # sub_dict.update(uuid2pred)
            #
            # idx2pred, uuid2pred = util.convert_tokens(gold_dict,
            #                                           ids_0_2,
            #                                           starts_0_2,
            #                                           ends_0_2,
            #                                           args.use_squad_v2)
            # pred_dict_0_2.update(idx2pred)
            # sub_dict.update(uuid2pred)
            #
            # idx2pred, uuid2pred = util.convert_tokens(gold_dict,
            #                                           ids_2_4,
            #                                           starts_2_4,
            #                                           ends_2_4,
            #                                           args.use_squad_v2)
            # pred_dict_2_4.update(idx2pred)
            # sub_dict.update(uuid2pred)
            #
            # idx2pred, uuid2pred = util.convert_tokens(gold_dict,
            #                                           ids_4_6,
            #                                           starts_4_6,
            #                                           ends_4_6,
            #                                           args.use_squad_v2)
            # pred_dict_4_6.update(idx2pred)
            # sub_dict.update(uuid2pred)
            #
            # idx2pred, uuid2pred = util.convert_tokens(gold_dict,
            #                                           ids_6,
            #                                           starts_6,
            #                                           ends_6,
            #                                           args.use_squad_v2)
            # pred_dict_6.update(idx2pred)
            # sub_dict.update(uuid2pred)
            #


            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)



    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        # print("no answer")
        # print(len(pred_dict_0))
        # results = util.eval_dicts(gold_dict, pred_dict_0, args.use_squad_v2)
        # results_list = [('F1', results['F1']),
        #                 ('EM', results['EM'])]
        # if args.use_squad_v2:
        #     results_list.append(('AvNA', results['AvNA']))
        # results = OrderedDict(results_list)
        #
        # # Log to console
        # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        # log.info(f'{args.split.title()} {results_str}')
        #
        #
        # print("answer length 0-2")
        # print(len(pred_dict_0_2))
        # results = util.eval_dicts(gold_dict, pred_dict_0_2, args.use_squad_v2)
        # results_list = [('F1', results['F1']),
        #                 ('EM', results['EM'])]
        # if args.use_squad_v2:
        #     results_list.append(('AvNA', results['AvNA']))
        # results = OrderedDict(results_list)
        #
        # # Log to console
        # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        # log.info(f'{args.split.title()} {results_str}')
        #
        # print("answer length 2-4")
        # print(len(pred_dict_2_4))
        # results = util.eval_dicts(gold_dict, pred_dict_2_4, args.use_squad_v2)
        # results_list = [('F1', results['F1']),
        #                 ('EM', results['EM'])]
        # if args.use_squad_v2:
        #     results_list.append(('AvNA', results['AvNA']))
        # results = OrderedDict(results_list)
        #
        # # Log to console
        # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        # log.info(f'{args.split.title()} {results_str}')
        #
        # print("answer length 4-6")
        # print(len(pred_dict_4_6))
        # results = util.eval_dicts(gold_dict, pred_dict_4_6, args.use_squad_v2)
        # results_list = [('F1', results['F1']),
        #                 ('EM', results['EM'])]
        # if args.use_squad_v2:
        #     results_list.append(('AvNA', results['AvNA']))
        # results = OrderedDict(results_list)
        #
        # # Log to console
        # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        # log.info(f'{args.split.title()} {results_str}')
        #
        # print("answer length >6")
        # print(len(pred_dict_6))
        # results = util.eval_dicts(gold_dict, pred_dict_6, args.use_squad_v2)
        # results_list = [('F1', results['F1']),
        #                 ('EM', results['EM'])]
        # if args.use_squad_v2:
        #     results_list.append(('AvNA', results['AvNA']))
        # results = OrderedDict(results_list)
        #
        # # Log to console
        # results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        # log.info(f'{args.split.title()} {results_str}')



        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
