''' This module will handle the text generation with beam search. '''
import numpy as np
from utils.utils import *
import torch.nn.functional as F
from model.transformer.Beam import Beam



def pos_emb_generation(word_labels):
    '''
        Generate the position embedding input for Transformers.
    '''

    seq = list(range(1, word_labels.shape[1] + 1))
    tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
    binary_mask = (word_labels != 0).long()

    return tgt_pos*binary_mask

def translate_batch_cap2cms(model, src_emb, cap_label, i3d, audio,opt, umt_tensor):
    ''' Translation work in one batch '''

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).cuda()

        active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):        ########### different in attention or not
            if mode == 'cap':
                dec_output, *_ = model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            elif mode == 'int':
                if opt['cms'] == 'int':
                    dec_output, *_ = model.cms_decoder_int(dec_seq, dec_pos, src_seq, enc_output)
                elif opt['cms'] == 'eff':
                    dec_output, *_ = model.cms_decoder_eff(dec_seq, dec_pos, src_seq, enc_output)
                else:
                    dec_output, *_ = model.cms_decoder_att(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)


        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # <--------------------------------------------- Decode Video ------------------------------------------------->
        # src_seq = src_emb.cuda()    # [b,40,2048]
        # src_enc, vis_sen_output = model.encoder(src_seq)

        src_seq = src_emb.cuda()
        #src_enc = model.encoder(src_seq, i3d, audio)
        src_enc = model.encoder_umt(src_seq, i3d, audio, umt_tensor)

        video_encoding = src_enc  # [b,20+10+10,d_model]

        # <--------------------------------------------- Decode CAP --------------------------------------------------->
        cap_pos = pos_emb_generation(cap_label)
        cap_label, cap_pos = cap_label[:, :-1], cap_pos[:, :-1]

        cap_dec_output, *_ = model.decoder(cap_label, cap_pos, video_encoding, video_encoding)  # [b,seq_cap,d_model]

        # Concatenate visual and caption encoding
        cat_encoding = torch.cat((video_encoding, cap_dec_output), 1)


        # <--------------------------------------------- Decode CMS --------------------------------------------------->
        # Repeat data for beam search for CMS
        n_bm = 2
        n_inst, len_s, d_h = cat_encoding.size()  # [b,40+seq_cap,d_model]
        src_enc = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
        src_seq = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1)

        # Prepare beams
        inst_dec_beams = [Beam(n_bm, device='cuda') for _ in range(n_inst)]

        # Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))  # [0,1,2,3...]
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)  # {0:0,1:1,...}

        if opt['cms'] == 'int':
            len_dec_seq_total = opt['int_max_len']
        elif opt['cms'] == 'eff':
            len_dec_seq_total = opt['eff_max_len']
        else:
            len_dec_seq_total = opt['att_max_len']

        for len_dec_seq in range(1, len_dec_seq_total - 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mode='int')

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        cms_batch_hyp, cms_batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

    return cms_batch_hyp


def translate_batch_cap(model, src_emb, i3d, audio, umt_tensor, opt):
    ''' Translation work in one batch '''

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).cuda()

        active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            if mode == 'cap':
                dec_output, dec_slf_attn_list, dec_enc_attn_list = model.decoder\
                    (dec_seq, dec_pos, src_seq, enc_output, return_attns=True)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            elif mode == 'int':
                dec_output, dec_slf_attn_list, dec_enc_attn_list = model.cms_decoder\
                    (dec_seq, dec_pos, src_seq, enc_output, return_attns=True)
                # print(dec_enc_attn_list[-1][0])
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():

        # 40rnn
        src_seq = src_emb.cuda()
        #src_enc = model.encoder(src_seq, i3d, audio,)
        src_enc = model.encoder_umt(src_seq, i3d, audio, umt_tensor,)
        src_seq = src_enc



        n_bm = 2
        n_inst, len_s, d_h = src_enc.size()
        src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
        src_seq = src_seq.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1)
        # Prepare beams
        inst_dec_beams = [Beam(n_bm, device='cuda') for _ in range(n_inst)]
        # Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        # <---------------------------------------------Decode CAP ---------------------------------------------------->
        for len_dec_seq in range(1, 28 + 1):


            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mode='cap')

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

    return batch_hyp


def translate_batch(model, src_emb, opt):
    ''' Translation work in one batch '''

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).cuda()

        active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            if mode == 'cap':
                dec_output, dec_slf_attn_list, dec_enc_attn_list = model.decoder\
                    (dec_seq, dec_pos, src_seq, enc_output, return_attns=True)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            elif mode == 'int':
                dec_output, dec_slf_attn_list, dec_enc_attn_list = model.cms_decoder\
                    (dec_seq, dec_pos, src_seq, enc_output, return_attns=True)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # Encode
        src_seq = src_emb.cuda()
        src_enc, *_ = model.encoder(src_seq)
        video_encoding = src_enc

        # Reepat data for beam search
        n_bm = 5
        n_inst, len_s, d_h = src_enc.size()
        src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
        src_seq = src_seq.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1)

        # Prepare beams
        inst_dec_beams = [Beam(n_bm, device='cuda') for _ in range(n_inst)]

        # Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # <---------------------------------------------Decode CAP ---------------------------------------------------->
        for len_dec_seq in range(1, 28 + 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mode='cap')

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)

        # <---------------------------------------------Decode CMS ---------------------------------------------------->
        cms_batch_hyp = []
        for cap_idx in range(n_bm):
            [_[0].insert(0, 2) for _ in batch_hyp]  # Start with <BOS> symbol
            dec_seq = np.zeros((opt['batch_size'], opt['cap_max_len']))
            for idx, seq in enumerate(batch_hyp):
                dec_seq[idx, :len(seq[cap_idx])] = seq[cap_idx]
            dec_seq = torch.as_tensor(dec_seq).cuda().long()
            dec_pos = pos_emb_generation(dec_seq).long()
            dec_output_, *_ = model.decoder(dec_seq[:, :-1], dec_pos[:, :-1], src_emb.cuda(), video_encoding)

            # Concatenate visual-captioning encodings
            cat_encoding = torch.cat((video_encoding, dec_output_), 1)

            # Repeat data for beam search for CMS
            n_inst, len_s, d_h = cat_encoding.size()
            src_enc = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            src_seq = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1)

            # Prepare beams
            inst_dec_beams = [Beam(n_bm, device='cuda') for _ in range(n_inst)]

            # Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            for len_dec_seq in range(1, opt['eff_max_len'] - 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mode='int')

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

            cms_hyp, cms_batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 3)
            cms_batch_hyp.append(cms_hyp[0])

    return batch_hyp, cms_batch_hyp





def translate_batch_cms(model, src_emb, cap_label, i3d, audio,opt, cms_labels_int, cms_labels_eff, cms_labels_att, umt_tensor):
    ''' Translation work in one batch '''

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).cuda()

        active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            if mode == 'cap':
                dec_output, *_ = model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            elif mode == 'int':
                if opt['cms'] == 'int':
                    dec_output, *_ = model.cms_decoder_int(dec_seq, dec_pos, src_seq, enc_output)
                elif opt['cms'] == 'eff':
                    dec_output, *_ = model.cms_decoder_eff(dec_seq, dec_pos, src_seq, enc_output)
                else:
                    dec_output, *_ = model.cms_decoder_att(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h

                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)


        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)


        return active_inst_idx_list

    def beam_decode_step_NOKD(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos


        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            if mode == 'cap':
                dec_output, *_ = model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)



            elif mode == 'int':
                if opt['cms'] == 'int':
                    dec_output, *_ = model.decoder_int_T(dec_seq, dec_pos, src_seq, enc_output)
                elif opt['cms'] == 'eff':
                    dec_output, *_ = model.decoder_eff_T(dec_seq, dec_pos, src_seq, enc_output)
                else:
                    dec_output, *_ = model.decoder_att_T(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h

                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)

                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)


        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)


        return active_inst_idx_list


    def beam_decode_step_teacher(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            if mode == 'cap':
                dec_output, *_ = model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            elif mode == 'int':
                if opt['cms'] == 'int':
                    dec_output, *_ = model.cms_decoder_int(dec_seq, dec_pos, src_seq, enc_output)
                elif opt['cms'] == 'eff':
                    dec_output, *_ = model.cms_decoder_eff(dec_seq, dec_pos, src_seq, enc_output)
                else:
                    dec_output, *_ = model.cms_decoder_att(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            if mode == 'teacher_int':
                #dec_output, *_ = model.cms_decoder_att(dec_seq, dec_pos, src_seq, enc_output)
                #dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h

                dec_output_int_T, *_ = model.decoder_int_T(dec_seq, dec_pos, src_seq, enc_output)
                dec_output_final = dec_output_int_T
                dec_output_int_T = dec_output_int_T[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output_int_T), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                #return dec_output_int_T

            if mode == 'teacher_eff':
                dec_output_eff_T, *_ = model.decoder_eff_T(dec_seq, dec_pos, src_seq, enc_output)
                dec_output_final = dec_output_eff_T
                dec_output_eff_T = dec_output_eff_T[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output_eff_T), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                #return dec_output_eff_T
            if mode == 'teacher_att':
                dec_output_att_T, *_ = model.decoder_att_T(dec_seq, dec_pos, src_seq, enc_output)
                dec_output_final = dec_output_att_T
                dec_output_att_T = dec_output_att_T[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output_att_T), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                #return dec_output_att_T


            return word_prob, dec_output_final

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob, dec_output_final = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)


        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list, dec_output_final

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    def remove_trailing_duplicates_and_convert_to_list(tensor):
        result_list = []
        for row in tensor:

            unique_elements = []
            seen = set()
            #
            for element in reversed(row.tolist()):
                if element not in seen:
                    seen.add(element)
                    unique_elements.append(element)

            unique_elements.reverse()
            result_list.append(unique_elements)

        while len(result_list) < 20:
            result_list.append([])

        transformed_data = [[[item for item in sublist]] for sublist in result_list[:20]]

        return transformed_data

    def transform_tensor_to_list(tensor):

        tensor_cpu = tensor.cpu()
        transformed_data = []

        for row in tensor_cpu:

            non_zero_elements = row[row != 0].tolist()


            if not non_zero_elements:
                transformed_data.append([[]])
            else:
                transformed_data.append([non_zero_elements])

        return transformed_data

    ##########################

    with torch.no_grad():
        # <--------------------------------------------- Decode Video ------------------------------------------------->
        # src_seq = src_emb.cuda()    # [b,40,2048]
        # src_enc, vis_sen_output = model.encoder(src_seq)

        src_seq = src_emb.cuda()
        #src_enc = model.encoder(src_seq, i3d, audio)
        src_enc = model.encoder_umt(src_seq, i3d, audio, umt_tensor)

        video_encoding = src_enc  # [b,20+10+10,d_model]

        # <--------------------------------------------- Decode CAP --------------------------------------------------->
        cap_pos = pos_emb_generation(cap_label)
        cap_label, cap_pos = cap_label[:, :-1], cap_pos[:, :-1]

        cap_dec_output, *_ = model.decoder(cap_label, cap_pos, video_encoding, video_encoding)  # [b,seq_cap,d_model]

        # Concatenate visual and caption encoding
        cat_encoding = torch.cat((video_encoding, cap_dec_output), 1)




        # <--------------------------------------------- Decode teacher model --------------------------------------------------->

        cms_pos_int = pos_emb_generation(cms_labels_int)
        cms_pos_eff = pos_emb_generation(cms_labels_eff)
        cms_pos_att = pos_emb_generation(cms_labels_att)

        cms_labels_int, cms_pos_int = cms_labels_int[:, :-1], cms_pos_int[:, :-1]
        cms_labels_eff, cms_pos_eff = cms_labels_eff[:, :-1], cms_pos_eff[:, :-1]
        cms_labels_att, cms_pos_att = cms_labels_att[:, :-1], cms_pos_att[:, :-1]

        dec_output_int_T, *_ = model.decoder_int_T(cms_labels_int, cms_pos_int, video_encoding, video_encoding)
        dec_output_eff_T, *_ = model.decoder_eff_T(cms_labels_eff, cms_pos_eff, video_encoding, video_encoding)
        dec_output_att_T, *_ = model.decoder_att_T(cms_labels_att, cms_pos_att, video_encoding, video_encoding)
        cat_encoding = torch.cat((video_encoding, cap_dec_output, dec_output_int_T, dec_output_eff_T, dec_output_att_T), dim=1)
        cat_encoding_int = torch.cat((video_encoding, cap_dec_output, dec_output_eff_T, dec_output_att_T),
                                 dim=1)
        cat_encoding_eff = torch.cat((video_encoding, cap_dec_output, dec_output_int_T, dec_output_att_T),
                                 dim=1)
        cat_encoding_att = torch.cat((video_encoding, cap_dec_output, dec_output_int_T, dec_output_eff_T),
                                 dim=1)
        if opt['cms'] == 'int':
            cat_encoding = cat_encoding_int
        elif opt['cms'] == 'eff':
            cat_encoding = cat_encoding_eff
        else:
            cat_encoding = cat_encoding_att



        ############################################  use ground-truth for test in teacher model    <---------------------------------------------------------------------->

        cms_dec_output_int, *_ = model.cms_decoder_int(cms_labels_int, cms_pos_int, cat_encoding_int, cat_encoding_int)
        cms_dec_output_eff, *_ = model.cms_decoder_eff(cms_labels_eff, cms_pos_eff, cat_encoding_eff, cat_encoding_eff)
        cms_dec_output_att, *_ = model.cms_decoder_att(cms_labels_att, cms_pos_att, cat_encoding_att, cat_encoding_att)

        cms_logit_int = model.cms_word_prj(cms_dec_output_int) #* model.x_logit_scale
        cms_logit_eff = model.cms_word_prj(cms_dec_output_eff)  # * model.x_logit_scale
        cms_logit_att = model.cms_word_prj(cms_dec_output_att)  # * model.x_logit_scale

        if opt['cms'] == 'int':
            word_prob_yyy = F.log_softmax(cms_logit_int, dim=1)
        if opt['cms'] == 'eff':
            word_prob_yyy = F.log_softmax(cms_logit_eff, dim=1)
        if opt['cms'] == 'att':
            word_prob_yyy = F.log_softmax(cms_logit_att, dim=1)
        ############################################


        # <--------------------------------------------- Decode CMS --------------------------------------------------->
        # Repeat data for beam search for CMS
        n_bm = 2
        n_inst, len_s, d_h = cat_encoding.size()  # [b,40+seq_cap,d_model]
        src_enc = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)     #input
        src_seq = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1)      #input

        # Prepare beams
        inst_dec_beams = [Beam(n_bm, device='cuda') for _ in range(n_inst)]         #output

        # Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))  # [0,1,2,3...]
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)  # {0:0,1:1,...}

        if opt['cms'] == 'int':
            len_dec_seq_total = opt['int_max_len']
        elif opt['cms'] == 'eff':
            len_dec_seq_total = opt['eff_max_len']
        else:
            len_dec_seq_total = opt['att_max_len']


        for len_dec_seq in range(1, len_dec_seq_total - 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mode='int')

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        cms_batch_hyp, cms_batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)




        # <--------------------------------------------- Decode teacher model with ground-truth --------------------------------------------------->

        prob_prev = torch.exp(word_prob_yyy)
        sampleLogprobs, it = torch.max(prob_prev, 2)


        result = remove_trailing_duplicates_and_convert_to_list(it)

        result_gt = transform_tensor_to_list(cms_labels_att[:,1:])



    return cms_batch_hyp

