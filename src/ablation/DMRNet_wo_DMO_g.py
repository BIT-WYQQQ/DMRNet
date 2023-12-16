import torch
import argparse
import numpy as np
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch, pad_num_replace
from models import DMRNet_wo_DMO_g
from util import llprint, sequence_output_process, ddi_rate_score, get_n_params, output_flatten
from recommend import eval, test

import sys
sys.path.append("..")

torch.manual_seed(1203)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'DMRNet_wo_DMO_g'
resume_path = ''

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--feature_size', type=int, default=8, help='feature dimension size')
parser.add_argument('--dmo_alpha', type=int, default=1e-7, help='dmo memory weight')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')

args = parser.parse_args()

def main(args):
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'
    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    dmo_adj_path = '../data/dmo_adj_final.pkl'
    device = torch.device('cuda')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    dmo_adj = dill.load(open(dmo_adj_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for dis_med in adm[0]:
                med_count[dis_med] += 1

    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_dis_med = sorted(data[i][j][0], key=lambda x: med_count[x])
            data[i][j][0] = cur_dis_med

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point + eval_len:]

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_eval)
    test_dataset = mimic_data(data_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch, shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch, shuffle=True,  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch, shuffle=True, pin_memory=True)

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model = DMRNet_wo_DMO_g(voc_size, ehr_adj, ddi_adj, dmo_adj, emb_dim=args.emb_dim, feature_size=args.feature_size, dmo_alpha=args.dmo_alpha, device=device)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        smm_record, ja, prauc, precision, recall, f1, med_num = test(model, test_dataloader, voc_size, device, TOKENS, ddi_adj, args)
        result = []
        for _ in range(10):
            data_num = len(ja)
            final_length = int(0.8 * data_num)
            idx_list = list(range(data_num))
            random.shuffle(idx_list)
            idx_list = idx_list[:final_length]
            avg_ja = np.mean([ja[i] for i in idx_list])
            avg_prauc = np.mean([prauc[i] for i in idx_list])
            avg_precision = np.mean([precision[i] for i in idx_list])
            avg_recall = np.mean([recall[i] for i in idx_list])
            avg_f1 = np.mean([f1[i] for i in idx_list])
            avg_med = np.mean([med_num[i] for i in idx_list])
            cur_smm_record = [smm_record[i] for i in idx_list]
            ddi_rate = ddi_rate_score(cur_smm_record, path='../../data/ddi_A_final.pkl')
            result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
            llprint(
                '\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                    ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)
        print('test time: {}'.format(time.time() - tic))
        return

    start_epoch = 0
    if resume_path != '':
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        start_epoch = 0
    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 100
    for epoch in range(start_epoch, EPOCH):
        tic = time.time()
        print('\nepoch {} --------------------------'.format(epoch))

        model.train()
        for idx, data in enumerate(train_dataloader):
            dis_med, adm_med, adm_type, adm_route, adm_time, pre_med, pre_type, pre_route, pre_time, diag, proc, \
            dis_mask, adm_mask, pre_mask, diag_mask, proc_mask, dis_length_matrix, seq_length = data

            diag = pad_num_replace(diag, -1, DIAG_PAD_TOKEN).to(device)
            proc = pad_num_replace(proc, -1, DIAG_PAD_TOKEN).to(device)
            dis_med = dis_med.to(device)
            adm_med = adm_med.to(device)
            adm_type = adm_type.to(device)
            adm_route = adm_route.to(device)
            adm_time = adm_time.to(device)
            pre_med = pre_med.to(device)
            pre_type = pre_type.to(device)
            pre_route = pre_route.to(device)
            pre_time = pre_time.to(device)
            dis_mask = dis_mask.to(device)
            adm_mask = adm_mask.to(device)
            pre_mask = pre_mask.to(device)
            diag_mask = diag_mask.to(device)
            proc_mask = proc_mask.to(device)
            output_logits = model(dis_med, adm_med, adm_type, adm_route, adm_time, pre_med, pre_type, pre_route,
                                  pre_time, diag, proc, dis_mask, adm_mask, pre_mask, diag_mask, proc_mask)
            labels, predictions = output_flatten(dis_med, output_logits, seq_length, dis_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
            loss = F.nll_loss(predictions, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            llprint('\rtraining step: {} / {}'.format(idx, len(train_dataloader)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, voc_size, device, TOKENS, args)
        print('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

        dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main(args)