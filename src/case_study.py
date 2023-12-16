import dill


out_list_path = './out_list.pkl'
out_list_gt_path = './out_list_gt.pkl'
voc_path = '../data/voc_final.pkl'
data_path = '../data/records_final.pkl'
out_list = dill.load(open(out_list_path, 'rb'))
out_list_gt = dill.load(open(out_list_gt_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
data = dill.load(open(data_path, 'rb'))
for patient in data[:5]:
    for visit in patient:
        print(visit[0])
print(out_list)
# print(out_list_gt)
for patient in out_list_gt:
    for i in range(len(patient)):
        patient[i] = patient[i].tolist()
print(out_list_gt)
for patient in out_list_gt:
    for visit in patient:
        for i in range(len(visit)):
            visit[i] = voc['med_voc'].idx2word[visit[i]]

print(out_list_gt)

