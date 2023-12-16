from torch.utils import data
import torch


class mimic_data(data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def pad_batch(batch):
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)
    diag_length_matrix = []
    proc_length_matrix = []
    adm_length_matrix = []
    pre_length_matrix = []
    dis_length_matrix = []
    diag_max = 0
    proc_max = 0
    med_max = 0
    for data in batch:
        diag_buf, proc_buf, dis_buf, adm_buf, pre_buf = [], [], [], [], []
        for idx, seq in enumerate(data):
            dis_buf.append(len(seq[0]))
            adm_buf.append(len(seq[1]))
            pre_buf.append(len(seq[5]))
            diag_buf.append(len(seq[9]))
            proc_buf.append(len(seq[10]))
            med_max = max(med_max, len(seq[0]))
            med_max = max(med_max, len(seq[1]))
            med_max = max(med_max, len(seq[5]))
            diag_max = max(diag_max, len(seq[9]))
            proc_max = max(proc_max, len(seq[10]))
        diag_length_matrix.append(diag_buf)
        proc_length_matrix.append(proc_buf)
        dis_length_matrix.append(dis_buf)
        adm_length_matrix.append(adm_buf)
        pre_length_matrix.append(pre_buf)

    dis_mask_matrix = torch.full((batch_size, max_seq, med_max), -1e9)
    for i in range(batch_size):
        for j in range(len(dis_length_matrix[i])):
            dis_mask_matrix[i, j, :dis_length_matrix[i][j]] = 0.

    adm_mask_matrix = torch.full((batch_size, max_seq, med_max), -1e9)
    for i in range(batch_size):
        for j in range(len(adm_length_matrix[i])):
            adm_mask_matrix[i, j, :adm_length_matrix[i][j]] = 0.

    pre_mask_matrix = torch.full((batch_size, max_seq, med_max), -1e9)
    for i in range(batch_size):
        for j in range(len(pre_length_matrix[i])):
            pre_mask_matrix[i, j, :pre_length_matrix[i][j]] = 0.

    diag_mask_matrix = torch.full((batch_size, max_seq, diag_max), -1e9)
    for i in range(batch_size):
        for j in range(len(diag_length_matrix[i])):
            diag_mask_matrix[i, j, :diag_length_matrix[i][j]] = 0.

    proc_mask_matrix = torch.full((batch_size, max_seq, proc_max), -1e9)
    for i in range(batch_size):
        for j in range(len(proc_length_matrix[i])):
            proc_mask_matrix[i, j, :proc_length_matrix[i][j]] = 0.

    dis_tensor = torch.full((batch_size, max_seq, med_max), 0)
    adm_tensor = torch.full((batch_size, max_seq, med_max), 0)
    adm_type_tensor = torch.full((batch_size, max_seq, med_max), -1)
    adm_route_tensor = torch.full((batch_size, max_seq, med_max), -1)
    adm_time_tensor = torch.full((batch_size, max_seq, med_max), -1)
    pre_tensor = torch.full((batch_size, max_seq, med_max), 0)
    pre_type_tensor = torch.full((batch_size, max_seq, med_max), -1)
    pre_route_tensor = torch.full((batch_size, max_seq, med_max), -1)
    pre_time_tensor = torch.full((batch_size, max_seq, med_max), -1)
    diag_tensor = torch.full((batch_size, max_seq, diag_max), -1)
    proc_tensor = torch.full((batch_size, max_seq, proc_max), -1)

    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            dis_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            adm_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            adm_type_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])
            adm_route_tensor[b_id, s_id, :len(adm[3])] = torch.tensor(adm[3])
            adm_time_tensor[b_id, s_id, :len(adm[4])] = torch.tensor(adm[4])
            pre_tensor[b_id, s_id, :len(adm[5])] = torch.tensor(adm[5])
            pre_type_tensor[b_id, s_id, :len(adm[6])] = torch.tensor(adm[6])
            pre_route_tensor[b_id, s_id, :len(adm[7])] = torch.tensor(adm[7])
            pre_time_tensor[b_id, s_id, :len(adm[8])] = torch.tensor(adm[8])
            diag_tensor[b_id, s_id, :len(adm[9])] = torch.tensor(adm[9])
            proc_tensor[b_id, s_id, :len(adm[10])] = torch.tensor(adm[10])

    return dis_tensor, adm_tensor, adm_type_tensor, adm_route_tensor, adm_time_tensor, pre_tensor, pre_type_tensor, \
           pre_route_tensor, pre_time_tensor, diag_tensor, proc_tensor, dis_mask_matrix, adm_mask_matrix, \
           pre_mask_matrix, diag_mask_matrix, proc_mask_matrix, dis_length_matrix, seq_length


def pad_num_replace(tensor, src_num, target_num):
    # replace_tensor = torch.full_like(tensor, target_num)
    return torch.where(tensor==src_num, target_num, tensor)