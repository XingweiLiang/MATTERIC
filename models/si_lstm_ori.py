import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.encoder import EncoderLayer


class JointModelAttention(nn.Module):
    def __init__(self, dh1, dh2, attn_dropout=0.2):
        """
        Args:
            dh1: model1 dimension
            dh2: model2 dimension

        """
        super(JointModelAttention, self).__init__()
        self.dh1 = dh1
        self.dh2 = dh2

        # 1, D
        self.Wq = nn.Parameter(torch.ones(self.dh1).unsqueeze(0))
        self.Wk = nn.Parameter(torch.ones(self.dh2).unsqueeze(0))
        # self.Wv = nn.Parameter(torch.ones(self.dh).unsqueeze(0))
        
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x_1, x_2):
        """
        Args:
            x_1: model 1, [B, D]
            x_2: model 2, [B, D]

        Returns:
            模态1到模态2的JointModelAttention的结果, [B, D]
        """
        # x: B, D
        x_1 = x_1.unsqueeze(-1) # B, D, 1
        x_2 = x_2.unsqueeze(-1) # B, D, 1

        Q = torch.matmul(x_1, self.Wq)   # B, D, D
        K = torch.matmul(x_2, self.Wk)   # B, D, D
        V = x_2 # B, D, 1

        attn = F.softmax(torch.matmul(Q / (self.dh1 ** 0.5), K) , dim=-1)  # B, D, D
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).squeeze(-1)  # B, D

        return output


class CrossAttention(nn.Module):
    def __init__(self, dh1, dh2, dk, dv, attn_dropout=0.2):
        super(CrossAttention, self).__init__()
        self.dk = dk
        self.dv = dv

        self.Wq = nn.Parameter(torch.ones(dh1, self.dk))  # D1 * Dk
        self.Wk = nn.Parameter(torch.ones(dh2, self.dk))  # D2 * Dk
        self.Wv = nn.Parameter(torch.ones(dh2, self.dv))  # D2 * Dv
        
        self.dropout = nn.Dropout(attn_dropout)

        self.layer_norm = nn.LayerNorm(dh2, eps=1e-6)

    def forward(self, x_1, x_2):
        # x: L, B, D
        residual = x_1
        x_1 = x_1.permute(1, 0, 2) # B, L1, D1
        x_2 = x_2.permute(1, 0, 2) # B, L2, D2

        Q = torch.matmul(x_1, self.Wq)   # B, L1, Dk
        K = torch.matmul(x_2, self.Wk)   # B, L2, Dk
        V = torch.matmul(x_2, self.Wv)   # B, L2, Dv

        attn = F.softmax(torch.matmul(Q / (self.dk ** 0.5), K.transpose(1, 2)) , dim=-1)  # B, L1, L2
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).permute(1, 0, 2)  # L1, B, Dv

        # add & norm
        output += residual
        output = self.layer_norm(output)

        return output


class SILSTM_Cell(nn.Module):
    def __init__(self, cell_size, in_size, jma_dim, speaker_dim):
        """
        Args:
            cell_size: LSTM hidden layer dimension
            in_size: the dimension of the input feature
            jma_dim: joint model attention dimension
            speaker_dim: speaker dimension
        """
        super(SILSTM_Cell, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4*self.cell_size)
        self.U = nn.Linear(cell_size, 4*self.cell_size)
        self.V = nn.Linear(jma_dim, 4*self.cell_size)
        self.S = nn.Linear(speaker_dim, 4*self.cell_size)

    def forward(self, x, ct_1, ht_1, zt, st):
        """
        Args:
            x: the input of the mode
            ct_1: cell memory at time t - 1
            ht_1: hidden meory at time t - 1
            zt: joint model attention at time t
            st: the state information of the current speaker at time t
        """
        input_affine = self.W(x)        # N, 4*dim
        output_affine = self.U(ht_1)     # N, 4*dim
        jma_affine = self.V(zt)     # N, 4*dim
        speaker_affine=self.S(st)

        sums = input_affine + output_affine + jma_affine + speaker_affine  # N, 4*dim
        
        # biases are already part of W and U and V
        f_t = torch.sigmoid(sums[:, :self.cell_size])   # N, 128
        i_t = torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])   # N, 128
        o_t = torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])     # N, 128
        ch_t = torch.tanh(sums[:, 3*self.cell_size:])       # N, 128
        c_t = f_t * ct_1 + i_t * ch_t
        h_t = torch.tanh(c_t) * o_t

        return c_t, h_t


class _SILSTM(nn.Module):
    def __init__(self, dh_l, dh_a, d_l, d_a, dropout=0.5) -> None:
        """
        Args:
            dh_l: hidden layer dimension of SILSTM (text modal)
            dh_a: Hidden layer dimension of SILSTM (audio modal)
            d_l: Input dimension of text modal
            d_a: Input dimension of audio modal
        """
        super(_SILSTM, self).__init__()
        self.crossatt_l2a = JointModelAttention(dh_l, dh_a)
        self.crossatt_a2l = JointModelAttention(dh_a, dh_l)
        self.dh_l, self.dh_a = dh_l, dh_a
        self.dh_q = dh_l
        self.d_l, self.d_a = d_l, d_a
        self.speaker_size = 4 * self.dh_l
        self.dh_s = 256

        self.silstm_l = SILSTM_Cell(self.dh_l, self.d_l, self.dh_l, self.dh_s)
        self.silstm_a = SILSTM_Cell(self.dh_a, self.d_a, self.dh_l, self.dh_s)

        self.gru_s = nn.GRUCell(self.d_l + self.d_a, self.dh_s)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_l, x_a, qmask):
        """
        Args:
            x: text dim + audio dim, [T, B, D_t + D_a]
            x_l: text model dimension, [T, B, D]
            x_a: audio model dimension, [T, B, D]
            qmask: used to distinguish the speaker, [T, B, 2]
        
        Returns:
            h_l: the output of SILSTM (text modal), [T, B, D]
            h_a: the output of SILSTM (audio modal), [T, B, D]
            h_sp: all speaker status, [T, B, 2]
        """
        N = x.shape[1]
        T = x.shape[0]
        h_l = torch.zeros(N, self.dh_l).to(x.device)
        h_a = torch.zeros(N, self.dh_a).to(x.device)
        
        c_l = torch.zeros(N, self.dh_l).to(x.device)
        c_a = torch.zeros(N, self.dh_a).to(x.device)

        z_l = torch.zeros(N, self.dh_l).to(x.device)
        z_a = torch.zeros(N, self.dh_a).to(x.device)
        q = torch.zeros(qmask.size()[1], qmask.size()[2], self.dh_s).to(x.device) # batch, party, D_speaker

        h_l_lst, h_a_lst, h_sp_lst = [], [], []
        for i in range(T):
            # U = torch.cat((x_l[i], x_a[i]), dim=1)
            U = x[i]
            # 选择当前说话人
            qm_idx = torch.argmax(qmask[i], 1)
            # 上一时刻speaker 和 listener 状态
            qs_0, ql_0 = self._select_parties(q, qm_idx)  # B, D
            # 更新speaker状态
            h_s_ = self.dropout(self.gru_s(U, qs_0))
            # 更新listener状态
            h_l_ = ql_0
            # 更新q
            q_s = h_s_.unsqueeze(1).expand(-1, qmask[i].size()[1], -1)
            q_l = h_l_.unsqueeze(1).expand(-1, qmask[i].size()[1], -1)  # N, 2, D
            qmask_ = qmask[i].unsqueeze(2) #qmask -> batch, party      N, 2, 1
            q = q_l * (1 - qmask_) + q_s * qmask_      # N, 2, D

            # current time step
            c_l, h_l = self.silstm_l(x_l[i], *(c_l, h_l, z_l, h_s_))  # B, D
            h_l = self.dropout(h_l)
            c_a, h_a = self.silstm_a(x_a[i], *(c_a, h_a, z_a, h_s_))  # B, D
            h_a = self.dropout(h_a)

            z_l = self.crossatt_l2a(c_l, c_a)  # B, D
            z_a = self.crossatt_a2l(c_a, c_l)   # B, D

            h_l_lst.append(h_l.unsqueeze(0))
            h_a_lst.append(h_a.unsqueeze(0))
            h_sp_lst.append(h_s_.unsqueeze(0))

        h_l = torch.cat(h_l_lst, dim=0)
        h_a = torch.cat(h_a_lst, dim=0)
        h_sp = torch.cat(h_sp_lst, dim=0)

        return h_l, h_a, h_sp

    def _select_parties(self, X, indices):
        """
        select current speaker
        Args:
            X: all speaker status
            indices: current speaker's index
        """
        qs_0, ql_0 = [], []
        for idx, j in zip(indices, X):
            qs_0.append(j[idx].unsqueeze(0))
            ql_0.append(j[1 - idx].unsqueeze(0))
        qs_0 = torch.cat(qs_0, 0)
        ql_0 = torch.cat(ql_0, 0)
        return qs_0, ql_0


class SILSTM(nn.Module):
    def __init__(self, n_classes, dataset):
        """
        Args:
            n_classes: num of classification
            dataset: dataset used
        """
        super(SILSTM, self).__init__()
        if dataset == 'IEMOCAP':
            self.dl_in, self.da_in = 1024, 100
        elif dataset == 'MELD':
            self.dl_in, self.da_in = 1024, 300
        else:
            raise
        # 文本/语音模态降维后维度
        self.d_l, self.d_a = 128, 128
        # 文本/语音模态各隐藏层维度
        self.dh_l, self.dh_a = 256, 256
        self.total_h_dim = self.dh_l + self.dh_a

        self.linear_in_l = nn.Linear(self.dl_in, self.d_l)
        self.linear_in_a = nn.Linear(self.da_in, self.d_a)
        self.shlstm_cell_f = _SILSTM(self.dh_l, self.dh_a, self.d_l, self.d_a)
        self.shlstm_cell_b = _SILSTM(self.dh_l, self.dh_a, self.d_l, self.d_a)

        output_dim = n_classes
        final_out = 2 * (self.total_h_dim + self.d_l)
        h_out = 256
        out_dropout = 0.2
        self.fc = nn.Sequential(
            nn.Linear(self.d_l, final_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.d_a, final_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout)
        )

        self.nn_out = nn.Sequential(
            nn.Linear(final_out, h_out), 
            nn.ReLU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(h_out, output_dim))

        self.dropout_rec = nn.Dropout(0.5)

        # encoder
        d_inner, n_head, d_k, d_v = 40, 8, 40, 40
        self.encoders_l, self.encoders_a = nn.ModuleList(), nn.ModuleList()
        self.encoders_num = 2
        for _ in range(self.encoders_num):
            self.encoders_l.append(EncoderLayer(self.d_l, d_inner, n_head, d_k, d_v))
            self.encoders_a.append(EncoderLayer(self.d_a, d_inner, n_head, d_k, d_v))

        # cross attention
        self.crossatt_l2a = CrossAttention(self.d_l, self.d_a, self.d_a, self.d_a)
        self.crossatt_a2l = CrossAttention(self.d_a, self.d_l, self.d_l, self.d_l)

        self.p = nn.Parameter(torch.ones(2))

        print("SILSTM is initialized ……")

    def forward(self, x, qmask, umask):
        """
        Args:
            x: [T, B, D_t + D_a] 
            qmask: used to distinguish the speaker
            umask: used to record of the utterance 
        """
        # x: T, B, D
        x_l = x[:, :, :self.dl_in].to(x.device).permute(1, 0, 2)
        x_a = x[:, :, self.dl_in: self.dl_in + self.da_in].to(x.device).permute(1, 0, 2)
        x_l = self.linear_in_l(x_l)
        x_a = self.linear_in_a(x_a)
        x = torch.cat([x_l, x_a], dim=2).permute(1, 0, 2)  

        x_en_l, x_en_a = x_l, x_a
        for encoder_l, encoder_a in zip(self.encoders_l, self.encoders_a):
            x_l, _ = encoder_l(x_en_l)
            x_a, _ = encoder_a(x_en_a)

            x_en_l = x_l
            x_en_a = x_a

        x_l = x_l.permute(1, 0, 2)  
        x_a = x_a.permute(1, 0, 2)

        # 正向
        hf_l, hf_a, hf_sp = self.shlstm_cell_f(x, x_l, x_a, qmask)    # L, B, D
        hf_l = self.dropout_rec(hf_l)
        hf_a = self.dropout_rec(hf_a)
        
        # 反向
        rev_x = self._reverse_seq(x, umask) 
        rev_x_l = self._reverse_seq(x_l, umask) 
        rev_x_a = self._reverse_seq(x_a, umask) 
        rev_qmask = self._reverse_seq(qmask, umask)  #倒置本句说话人

        hb_l, hb_a, hb_sp = self.shlstm_cell_b(rev_x, rev_x_l, rev_x_a, rev_qmask)  #反方向rnn输出
        hb_l = self.dropout_rec(self._reverse_seq(hb_l, umask))
        hb_a = self.dropout_rec(self._reverse_seq(hb_a, umask))

        h_l = torch.cat([hf_l, hb_l], dim=-1)   # L, B, 2D
        h_a = torch.cat([hf_a, hb_a], dim=-1)   # L, B, 2D

        attn1 = self.crossatt_l2a(x_l, x_a)   # L, B, D
        attn2 = self.crossatt_a2l(x_a, x_l)   # L, B, D

        w1 = torch.exp(self.p[0]) / torch.sum(torch.exp(self.p))
        w2 = torch.exp(self.p[1]) / torch.sum(torch.exp(self.p))

        resid_l = self.fc(x_l)
        resid_a = self.fc1(x_a)
        l = torch.cat([h_l, attn2], dim=2) # L, B, 3D
        a = torch.cat([h_a, attn1], dim=2) # L, B, 3D
        output = F.log_softmax(self.nn_out(torch.cat([w1 * l, w2 * a], dim=-1) + resid_a + resid_l), 2)

        output = output.permute(1, 0, 2)
        output = output.reshape(-1, output.size()[-1])
        return output

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()
        #即每一轮各对话句子数list

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0]) #在第0维翻转（把每一轮各对话进行翻转）
            xfs.append(xf)

        return pad_sequence(xfs)#（后面补零填充）