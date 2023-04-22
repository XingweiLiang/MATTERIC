import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.init import xavier_normal_

from models.encoder import EncoderLayer


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

    def forward(self, x, ct_1, ht_1, st):
        """
        Args:
            x: the input of the mode
            ct_1: cell memory at time t - 1
            ht_1: hidden meory at time t - 1
            st: the state information of the current speaker at time t
        """
        input_affine = self.W(x)        # N, 4*dim
        output_affine = self.U(ht_1)     # N, 4*dim
        speaker_affine=self.S(st)

        sums = input_affine + output_affine + speaker_affine  # N, 4*dim
        
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
            c_l, h_l = self.silstm_l(x_l[i], *(c_l, h_l, h_s_))  # B, D
            h_l = self.dropout(h_l)
            c_a, h_a = self.silstm_a(x_a[i], *(c_a, h_a, h_s_))  # B, D
            h_a = self.dropout(h_a)

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
    def __init__(self, n_classes, dataset, fusion_type, mult_task):
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
        self.fusion_type = fusion_type.upper()
        self.mult_task = mult_task
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
        if self.fusion_type == 'LMF' or self.fusion_type == 'TFN':
            final_out = 4 * self.total_h_dim
        else:
            final_out = 2 * self.total_h_dim
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

        self.dropout_rec = nn.Dropout(0.5)

        # encoder
        d_inner, n_head, d_k, d_v = 40, 8, 40, 40
        self.encoders_l, self.encoders_a = nn.ModuleList(), nn.ModuleList()
        self.encoders_num = 2
        for _ in range(self.encoders_num):
            self.encoders_l.append(EncoderLayer(self.d_l, d_inner, n_head, d_k, d_v))
            self.encoders_a.append(EncoderLayer(self.d_a, d_inner, n_head, d_k, d_v))

        if self.fusion_type == 'LMF':
            self.rank = 4
            self.audio_factor = nn.Parameter(torch.Tensor(self.rank, self.total_h_dim + 1, final_out))
            self.text_factor = nn.Parameter(torch.Tensor(self.rank, self.total_h_dim + 1, final_out))
            self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
            self.fusion_bias = nn.Parameter(torch.Tensor(1, final_out))
            # init teh factors
            xavier_normal_(self.audio_factor)
            xavier_normal_(self.text_factor)
            # xavier_normal_(self.fusion_weights)
            self.fusion_bias.data.fill_(0)
        elif self.fusion_type == 'WF':
            self.p = nn.Parameter(torch.ones(2))
        elif self.fusion_type == 'TFN':
            self.fushion_dropout = nn.Dropout(0.5)
            self.fushion_layer = nn.Linear((self.total_h_dim + 1) * (self.total_h_dim + 1), final_out)
        elif self.fusion_type == 'LFDNN':
            pass

        for m in self.mult_task:
            if m == 'M':
                self.nn_out = nn.Sequential(
                    nn.Linear(final_out, h_out), 
                    nn.ReLU(), 
                    nn.Dropout(out_dropout), 
                    nn.Linear(h_out, output_dim))
            elif m == 'A':
                self.a_out = nn.Sequential(
                    nn.Linear(2 * self.dh_a, h_out), 
                    nn.ReLU(), 
                    nn.Dropout(out_dropout), 
                    nn.Linear(h_out, output_dim))
            elif m == 'T':
                self.l_out = nn.Sequential(
                    nn.Linear(2 * self.dh_l, h_out), 
                    nn.ReLU(), 
                    nn.Dropout(out_dropout), 
                    nn.Linear(h_out, output_dim))


        print(f"SILSTM is initialized, fusion type is {self.fusion_type} ……")

    def forward(self, x, qmask, umask):
        """
        Args:
            x: [T, B, D_t + D_a] 
            qmask: used to distinguish the speaker
            umask: used to record of the utterance 
        """
        length, batch, dim = x.size()
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

        resid_l = self.fc(x_l)
        resid_a = self.fc1(x_a)

        if self.fusion_type == 'LMF' or self.fusion_type == 'TFN':
            add_one = torch.ones(size=[length, batch, 1], requires_grad=False).type_as(h_a).to(h_a.device)
            _audio_h = torch.cat((add_one, h_a), dim=-1)  # L, B, 3D + 1
            _text_h = torch.cat((add_one, h_l), dim=-1)   # L, B, 3D + 1

        if self.fusion_type == 'LMF':
            fusion_audio = torch.matmul(_audio_h.unsqueeze(1), self.audio_factor)   # L, 2, B, D
            fusion_text = torch.matmul(_text_h.unsqueeze(1), self.text_factor)      # L, 2, B, D
            # fusion
            fusion_tensor = fusion_audio * fusion_text      # L, 2, B, D
            fusion_tensor = torch.matmul(self.fusion_weights, fusion_tensor.permute(0, 2, 1, 3)).squeeze() + self.fusion_bias   # L, B, D
        elif self.fusion_type == 'TFN':
            fusion_tensor = torch.matmul(_audio_h.unsqueeze(3), _text_h.unsqueeze(2))   # L, B, Da+1, Dt+1
            fusion_tensor = self.fushion_dropout(fusion_tensor.reshape(length, batch, -1))   # B, L, D
            fusion_tensor = self.fushion_layer(fusion_tensor)
        elif self.fusion_type == 'WF':
            w1 = torch.exp(self.p[0]) / torch.sum(torch.exp(self.p))
            w2 = torch.exp(self.p[1]) / torch.sum(torch.exp(self.p))
            fusion_tensor = torch.cat([w1 * h_l, w2 * h_a], dim=-1)
        elif self.fusion_type == 'LFDNN':
            fusion_tensor = torch.cat([h_l, h_a], dim=-1) 
        
        # output
        output_text = ''
        output_audio = ''
        for m in self.mult_task:
            if m == 'M':
                output_fusion = F.log_softmax(self.nn_out(fusion_tensor + resid_a + resid_l), 2)
                output_fusion = output_fusion.permute(1, 0, 2)
                output_fusion = output_fusion.reshape(-1, output_fusion.size()[-1])
            elif m == 'A':
                output_audio = F.log_softmax(self.a_out(h_a), 2)
                output_audio = output_audio.permute(1, 0, 2)
                output_audio = output_audio.reshape(-1, output_audio.size()[-1])
            elif m == 'T':
                output_text = F.log_softmax(self.l_out(h_l), 2)
                output_text = output_text.permute(1, 0, 2)
                output_text = output_text.reshape(-1, output_text.size()[-1])

        output = {
            'M': output_fusion,
            'T': output_text,
            'A': output_audio,
        }
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