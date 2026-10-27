import numpy as np
from utils.utils import *
import torch.nn.functional as F
from model.transformer.Beam import Beam
from utils.utils import *

from model.transformer.Transformers import Encoder_video, Encoder_video_umt
from model.Decoder import Decoder


class HybirdNet(nn.Module):
    '''
        A sequence to sequence model with attention mechanism.
    '''

    def pos_emb_generation(word_labels):
        '''
            Generate the position embedding input for Transformers.
        '''

        seq = list(range(1, word_labels.shape[1] + 1))
        tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
        binary_mask = (word_labels != 0).long()

        return tgt_pos * binary_mask

    def __init__(
            self,
            n_cap_vocab, n_cms_vocab, cap_max_seq, cms_max_seq_int, cms_max_seq_eff, cms_max_seq_att,
            vis_emb=2048, i3d_emb=1024, audio_emb=1024, d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, rnn_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1,umt_emb=768,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()

        # set RNN layers at 1 or 2 yield better performance.
        self.encoder = Encoder_video(d_vison=vis_emb,  d_i3d=i3d_emb, d_audio=audio_emb, d_model=d_model)
        self.encoder_umt = Encoder_video_umt(d_vison=vis_emb, d_i3d=i3d_emb, d_audio=audio_emb, d_model=d_model, d_umt=umt_emb)

        self.decoder = Decoder(
            n_tgt_vocab=n_cap_vocab, len_max_seq=cap_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder_int_T = Decoder(
            n_tgt_vocab=n_cap_vocab, len_max_seq=cap_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder_eff_T = Decoder(
            n_tgt_vocab=n_cap_vocab, len_max_seq=cap_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder_att_T = Decoder(
            n_tgt_vocab=n_cap_vocab, len_max_seq=cap_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.cms_decoder_int = Decoder(
            n_tgt_vocab=n_cms_vocab, len_max_seq=cms_max_seq_int,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.cms_decoder_eff = Decoder(
            n_tgt_vocab=n_cms_vocab, len_max_seq=cms_max_seq_eff,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.cms_decoder_att = Decoder(
            n_tgt_vocab=n_cms_vocab, len_max_seq=cms_max_seq_att,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.cap_word_prj = nn.Linear(d_model, n_cap_vocab, bias=False)
        self.cms_word_prj = nn.Linear(d_model, n_cms_vocab, bias=False)

        nn.init.xavier_normal_(self.cap_word_prj.weight)
        nn.init.xavier_normal_(self.cms_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, ' \
            'the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.cap_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.cms_decoder_eff.tgt_word_emb.weight = self.cms_decoder_att.tgt_word_emb.weight
            self.cms_decoder_int.tgt_word_emb.weight = self.cms_decoder_eff.tgt_word_emb.weight
            self.cms_word_prj.weight = self.cms_decoder_int.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, vis_feat, i3d, aud, tgt_seq, tgt_pos,
                cms_seq_int, cms_pos_int, cms_seq_eff, cms_pos_eff, cms_seq_att, cms_pos_att, umt_tensor,vis):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        cms_seq_int, cms_pos_int = cms_seq_int[:, :-1], cms_pos_int[:, :-1]
        cms_seq_eff, cms_pos_eff = cms_seq_eff[:, :-1], cms_pos_eff[:, :-1]
        cms_seq_att, cms_pos_att = cms_seq_att[:, :-1], cms_pos_att[:, :-1]


        #enc_output = self.encoder(vis_feat, i3d, aud)
        enc_output = self.encoder_umt(vis_feat, i3d, aud, umt_tensor)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, enc_output, enc_output)
        seq_logit = self.cap_word_prj(dec_output) * self.x_logit_scale

        # Concatenate visual and caption encoding
        
        if vis == False:
            enc_output[:] = 0
        
        cat_output = torch.cat((enc_output, dec_output), dim=1)



        ################
        dec_output_int_T, *_ = self.decoder_int_T(cms_seq_int, cms_pos_int, enc_output, enc_output)
        dec_output_eff_T, *_ = self.decoder_eff_T(cms_seq_eff, cms_pos_eff, enc_output, enc_output)
        dec_output_att_T, *_ = self.decoder_att_T(cms_seq_att, cms_pos_att, enc_output, enc_output)
        cat_output_int = torch.cat((enc_output, dec_output, dec_output_eff_T, dec_output_att_T), dim=1)
        cat_output_eff = torch.cat((enc_output, dec_output, dec_output_int_T, dec_output_att_T),
                                   dim=1)
        cat_output_att = torch.cat((enc_output, dec_output, dec_output_int_T, dec_output_eff_T),
                                   dim=1)
        ################


        ################ teacher model
        #dec_output_int_T, *_ = self.decoder_int_T(cms_seq_int, cms_pos_int, enc_output, enc_output)
        #dec_output_eff_T, *_ = self.decoder_eff_T(cms_seq_int, cms_pos_int, enc_output, enc_output)
        #dec_output_att_T, *_ = self.decoder_att_T(cms_seq_int, cms_pos_int, enc_output, enc_output)
        #cat_output = torch.cat((enc_output, dec_output, dec_output_int_T, dec_output_eff_T, dec_output_att_T), dim=1)
        ################


        cms_dec_output_int, *_ = self.cms_decoder_int(cms_seq_int, cms_pos_int, cat_output_int, cat_output_int)
        cms_dec_output_eff, *_ = self.cms_decoder_eff(cms_seq_eff, cms_pos_eff, cat_output_eff, cat_output_eff)
        cms_dec_output_att, *_ = self.cms_decoder_att(cms_seq_att, cms_pos_att, cat_output_att, cat_output_att)

        cms_logit_int = self.cms_word_prj(cms_dec_output_int) * self.x_logit_scale
        cms_logit_eff = self.cms_word_prj(cms_dec_output_eff) * self.x_logit_scale
        cms_logit_att = self.cms_word_prj(cms_dec_output_att) * self.x_logit_scale



        return seq_logit.view(-1, seq_logit.size(2)), cms_logit_int.view(-1, cms_logit_int.size(2)), cms_logit_eff.view(
            -1, cms_logit_eff.size(2)), cms_logit_att.view(-1, cms_logit_att.size(2))
