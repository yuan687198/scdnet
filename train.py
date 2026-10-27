''' Training Scropt for V2C captioning task. '''

import os
import numpy as np
from opts import *
from utils.utils import *
import torch.optim as optim
from model.Model import HybirdNet as Model
from model.Model_T import HybirdNet as Model_T
from torch.utils.data import DataLoader
from utils.dataloader import VideoDataset
from model.transformer.Optim import ScheduledOptim
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.feature_extraction.text import TfidfVectorizer  
  

def gumbel_softmax(logits, tau=1, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = F.softmax((logits + gumbel_noise) / tau, dim=-1)

    if hard:
        y_hard = (y == y.max(dim=-1, keepdim=True)[0]).float()
        y = (y_hard - y).detach() + y  
    return y



def train(loader, model, optimizer, opt, cap_vocab, cms_vocab, model_T, optimizer_T):

    model_T.train()
    model.train()

    for epoch in range(1, opt['epochs']+1):
        iteration = 0
        cap_n_correct_total = 0
        cms_int_n_correct_total = 0
        cms_eff_n_correct_total = 0
        cms_att_n_correct_total = 0
        n_word_total = 0
        cms_int_n_word_total = 0
        cms_eff_n_word_total = 0
        cms_att_n_word_total = 0
        cap_train_loss_total = 0
        cms_int_train_loss_total = 0
        cms_eff_train_loss_total = 0
        cms_att_train_loss_total = 0
        cap_n_correct_total_T = 0
        cms_int_n_correct_total_T = 0
        cms_eff_n_correct_total_T = 0
        cms_att_n_correct_total_T = 0
        n_word_total_T = 0
        cms_int_n_word_total_T = 0
        cms_eff_n_word_total_T = 0
        cms_att_n_word_total_T = 0
        cap_train_loss_total_T = 0
        cms_int_train_loss_total_T = 0
        cms_eff_train_loss_total_T = 0
        cms_att_train_loss_total_T = 0

        for data in loader:
            eval_id = 0

            res_greedy = {}
            res_generate = {}
            gts = {}

            res_greedy_int = {}
            res_generate_int = {}
            gts_int = {}
            res_greedy_eff = {}
            res_generate_eff = {}
            gts_eff = {}
            res_greedy_att = {}
            res_generate_att = {}
            gts_att = {}

            torch.cuda.synchronize()

            cms_labels_int = data['int_labels']   
            cms_labels_eff = data['eff_labels']   
            cms_labels_att = data['att_labels']    

            batchsize_ymq = 60
            cosine_sim = 0


            load_torch = torch.load('saved_tensors/' + str(data['video_ids'][0]) + '.pt').cuda()
            for j in range(len(data['video_ids'])-1):
                load_torch1 = torch.load('saved_tensors/' + str(data['video_ids'][j+1]) + '.pt').cuda()
                load_torch = torch.cat((load_torch,load_torch1),0).cuda()        

            umt_tensor = load_torch.repeat(1, 5, 1).cuda()


            if opt['cuda']:
                fc_feats = data['fc_feats'].cuda()
                i3d = data['i3d'].cuda()
                audio = data['audio'].cuda()
                cap_labels = data['cap_labels'].cuda()
                cms_labels_int = cms_labels_int.cuda()
                cms_labels_eff = cms_labels_eff.cuda()
                cms_labels_att = cms_labels_att.cuda()

            optimizer.zero_grad()
            cap_pos = pos_emb_generation(cap_labels)             
            cms_pos_int = pos_emb_generation(cms_labels_int)     
            cms_pos_eff = pos_emb_generation(cms_labels_eff)     
            cms_pos_att = pos_emb_generation(cms_labels_att)     


            if epoch > 200 :

                model.eval()
                with torch.no_grad():  # greedy
                    cap_probs_r, cms_int_probs_r, cms_eff_probs_r, cms_att_probs_r = model(fc_feats, i3d, audio,
                                                                                           cap_labels,
                                                                                           cap_pos, cms_labels_int,
                                                                                           cms_pos_int,
                                                                                           cms_labels_eff, cms_pos_eff,
                                                                                           cms_labels_att, cms_pos_att, umt_tensor,vis=True)
                    sampleLogprobs_r, it_r = torch.max(cap_probs_r, 1)
                    it_r = it_r.view(-1).long()

                    sampleLogprobs_r_int, it_r_int = torch.max(cms_int_probs_r, 1)
                    it_r_int = it_r_int.view(-1).long()
                    sampleLogprobs_r_eff, it_r_eff = torch.max(cms_eff_probs_r, 1)
                    it_r_eff = it_r_eff.view(-1).long()
                    sampleLogprobs_r_att, it_r_att = torch.max(cms_att_probs_r, 1)
                    it_r_att = it_r_att.view(-1).long()

                model.train()

                # generate
                cap_probs_v, cms_int_probs_v, cms_eff_probs_v, cms_att_probs_v = model(fc_feats, i3d, audio, cap_labels,
                                                                               cap_pos, cms_labels_int, cms_pos_int,
                                                                               cms_labels_eff, cms_pos_eff,
                                                                               cms_labels_att, cms_pos_att, umt_tensor,vis=True)
                
                cap_probs, cms_int_probs, cms_eff_probs, cms_att_probs = model(fc_feats, i3d, audio, cap_labels,
                                                                               cap_pos, cms_labels_int, cms_pos_int,
                                                                               cms_labels_eff, cms_pos_eff,
                                                                               cms_labels_att, cms_pos_att, umt_tensor,vis=False)
                
                cap_probs_h = cap_probs.detach()
                cms_int_probs_h = cms_int_probs.detach()
                cms_eff_probs_h = cms_eff_probs.detach()
                cms_att_probs_h = cms_att_probs.detach()


                epsilon = 1e-10

                prob_prev = torch.exp(cap_probs_h) + epsilon
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = cap_probs_h.gather(1, it)
                prob_prev_int = torch.exp(cms_int_probs_h) + epsilon
                it_int = torch.multinomial(prob_prev_int, 1)
                sampleLogprobs_int = cms_int_probs_h.gather(1, it_int)
                prob_prev_eff = torch.exp(cms_eff_probs_h) + epsilon
                it_eff = torch.multinomial(prob_prev_eff, 1)
                sampleLogprobs_eff = cms_eff_probs_h.gather(1, it_eff)
                prob_prev_att = torch.exp(cms_att_probs_h) + epsilon
                it_att = torch.multinomial(prob_prev_att, 1)
                sampleLogprobs_att = cms_att_probs_h.gather(1, it_att)

                cap_probs_T, cms_int_probs_T, cms_eff_probs_T, cms_att_probs_T = model_T(fc_feats, i3d, audio,
                                                                                         cap_labels,
                                                                                         cap_pos, cms_labels_int,
                                                                                         cms_pos_int,
                                                                                         cms_labels_eff,
                                                                                         cms_pos_eff,
                                                                                         cms_labels_att,
                                                                                         cms_pos_att,
                                                                                         umt_tensor, vis=True)

                if it.shape[0] % batchsize_ymq == 0:

                    batch_size = fc_feats.shape[0]
                    it = it.clone().view(batch_size, -1)
                    it_r = it_r.clone().view(batch_size, -1)
                    it_int = it_int.clone().view(batch_size, -1)
                    it_r_int = it_r_int.clone().view(batch_size, -1)
                    it_eff = it_eff.clone().view(batch_size, -1)
                    it_r_eff = it_r_eff.clone().view(batch_size, -1)
                    it_att = it_att.clone().view(batch_size, -1)
                    it_r_att = it_r_att.clone().view(batch_size, -1)


                    for random_id in range(cap_labels.shape[0]):
                        # if EOS in cms_batch_hyp[random_id][0]:
                        #    stop_id = cms_batch_hyp[random_id][0].index(EOS)
                        # else:
                        #    stop_id = -1

                        cms = list_to_sentence([cap_vocab[str(word.cpu().numpy())] for word in
                                                cap_labels[random_id, 1:] if word != 0][0:-1])
                        cms_greedy = list_to_sentence([cap_vocab[str(word.cpu().numpy())] for word in
                                                       it_r[random_id, 1:] if word != 0][0:-1])
                        cms_generate = list_to_sentence([cap_vocab[str(word.cpu().numpy())] for word in
                                                         it[random_id, 1:] if word != 0][0:-1])
                        gts[eval_id] = [cms]
                        res_greedy[eval_id] = [cms_greedy]
                        res_generate[eval_id] = [cms_generate]

                        cms_int = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                    cms_labels_int[random_id, 1:] if word != 0][0:-1])
                        cms_greedy_int = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                           it_r_int[random_id, 1:] if word != 0][0:-1])
                        cms_generate_int = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                             it_int[random_id, 1:] if word != 0][0:-1])
                        gts_int[eval_id] = [cms_int]
                        res_greedy_int[eval_id] = [cms_greedy_int]
                        res_generate_int[eval_id] = [cms_generate_int]

                        cms_eff = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                    cms_labels_eff[random_id, 1:] if word != 0][0:-1])
                        cms_greedy_eff = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                           it_r_eff[random_id, 1:] if word != 0][0:-1])
                        cms_generate_eff = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                             it_eff[random_id, 1:] if word != 0][0:-1])
                        gts_eff[eval_id] = [cms_eff]
                        res_greedy_eff[eval_id] = [cms_greedy_eff]
                        res_generate_eff[eval_id] = [cms_generate_eff]

                        cms_att = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                    cms_labels_att[random_id, 1:] if word != 0][0:-1])
                        cms_greedy_att = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                           it_r_att[random_id, 1:] if word != 0][0:-1])
                        cms_generate_att = list_to_sentence([cms_vocab[str(word.cpu().numpy())] for word in
                                                             it_att[random_id, 1:] if word != 0][0:-1])
                        gts_att[eval_id] = [cms_att]
                        res_greedy_att[eval_id] = [cms_greedy_att]
                        res_generate_att[eval_id] = [cms_generate_att]
                        eval_id += 1


                    avg_bleu_score, bleu_scores_greedy = Bleu(1).compute_score(gts, res_greedy)
                    bleu_scores_greedy = np.array(bleu_scores_greedy[0])
                    avg_bleu_score, bleu_scores_generate = Bleu(1).compute_score(gts, res_generate)
                    bleu_scores_generate = np.array(bleu_scores_generate[0])
                    rewards = bleu_scores_generate - bleu_scores_greedy
                    rewards = np.repeat(rewards[:, np.newaxis], it.shape[1], 1)  
                    rewards = torch.from_numpy(rewards).float()
                    rewards = rewards.cuda()
                    avg_bleu_score_int, bleu_scores_greedy_int = Bleu(1).compute_score(gts_int, res_greedy_int)
                    bleu_scores_greedy_int = np.array(bleu_scores_greedy_int[0])
                    avg_bleu_score_int, bleu_scores_generate_int = Bleu(1).compute_score(gts_int, res_generate_int)
                    bleu_scores_generate_int = np.array(bleu_scores_generate_int[0])
                    rewards_int = bleu_scores_generate_int - bleu_scores_greedy_int
                    rewards_int = np.repeat(rewards_int[:, np.newaxis], it_int.shape[1],
                                            1)  
                    rewards_int = torch.from_numpy(rewards_int).float()
                    rewards_int = rewards_int.cuda()
                    avg_bleu_score_eff, bleu_scores_greedy_eff = Bleu(1).compute_score(gts_eff, res_greedy_eff)
                    bleu_scores_greedy_eff = np.array(bleu_scores_greedy_eff[0])
                    avg_bleu_score_eff, bleu_scores_generate_eff = Bleu(1).compute_score(gts_eff, res_generate_eff)
                    bleu_scores_generate_eff = np.array(bleu_scores_generate_eff[0])
                    rewards_eff = bleu_scores_generate_eff - bleu_scores_greedy_eff
                    rewards_eff = np.repeat(rewards_eff[:, np.newaxis], it_eff.shape[1],
                                            1)  
                    rewards_eff = torch.from_numpy(rewards_eff).float()
                    rewards_eff = rewards_eff.cuda()

                    avg_bleu_score_att, bleu_scores_greedy_att = Bleu(1).compute_score(gts_att, res_greedy_att)
                    bleu_scores_greedy_att = np.array(bleu_scores_greedy_att[0])
                    avg_bleu_score_att, bleu_scores_generate_att = Bleu(1).compute_score(gts_att, res_generate_att)
                    bleu_scores_generate_att = np.array(bleu_scores_generate_att[0])
                    rewards_att = bleu_scores_generate_att - bleu_scores_greedy_att
                    rewards_att = np.repeat(rewards_att[:, np.newaxis], it_att.shape[1],
                                            1)  
                    rewards_att = torch.from_numpy(rewards_att).float()
                    rewards_att = rewards_att.cuda()
                                                       
                    vectorizer = TfidfVectorizer().fit_transform([gts[0][0], res_greedy[0][0]])
                    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
                    cosine_sim_tensor = torch.from_numpy(-0.01*cosine_sim)
                    cosine_sim_tensor = cosine_sim_tensor.cuda()

                    vectorizer = TfidfVectorizer().fit_transform([gts[0][0], res_greedy[0][0]])
                    cosine_sim_int = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
                    cosine_sim_int_tensor = torch.from_numpy(-0.01*cosine_sim_int)
                    cosine_sim_int_tensor = cosine_sim_int_tensor.cuda()

                    vectorizer = TfidfVectorizer().fit_transform([gts[0][0], res_greedy[0][0]])
                    cosine_sim_eff = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
                    cosine_sim_eff_tensor = torch.from_numpy(-0.01*cosine_sim_eff)
                    cosine_sim_eff_tensor = cosine_sim_eff_tensor.cuda()

                    vectorizer = TfidfVectorizer().fit_transform([gts[0][0], res_greedy[0][0]])
                    cosine_sim_att = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
                    cosine_sim_att_tensor = torch.from_numpy(-0.01*cosine_sim_att)
                    cosine_sim_att_tensor = cosine_sim_att_tensor.cuda()
                 
                    int_copy = cosine_sim_int_tensor.expand_as(rewards_int)
                    
                    eff_copy = cosine_sim_eff_tensor.expand_as(rewards_eff)

                    att_copy = cosine_sim_att_tensor.expand_as(rewards_att)




                    cap_loss, cap_n_correct = cal_performance_reward(cap_probs_v, cap_labels[:, 1:], rewards,
                                                                     smoothing=True
                                                                     )  
                    cms_int_loss, cms_int_n_correct = cal_performance_reward(cms_int_probs_v, cms_labels_int[:, 1:],
                                                                             rewards_int,
                                                                             smoothing=True)  
                    cms_eff_loss, cms_eff_n_correct = cal_performance_reward(cms_eff_probs_v, cms_labels_eff[:, 1:],
                                                                             rewards_eff,
                                                                             smoothing=True)  
                    cms_att_loss, cms_att_n_correct = cal_performance_reward(cms_att_probs_v, cms_labels_att[:, 1:],
                                                                             rewards_att,
                                                                             smoothing=True)  

                    cap_loss_T, cap_n_correct_T = cal_performance_reward(cap_probs_T, cap_labels[:, 1:], rewards,
                                                                     smoothing=True
                                                                     )  
                    cms_int_loss_T, cms_int_n_correct_T = cal_performance_reward(cms_int_probs_T, cms_labels_int[:, 1:],
                                                                             rewards_int,
                                                                             smoothing=True)  
                    cms_eff_loss_T, cms_eff_n_correct_T = cal_performance_reward(cms_eff_probs_T, cms_labels_eff[:, 1:],
                                                                             rewards_eff,
                                                                             smoothing=True)  
                    cms_att_loss_T, cms_att_n_correct_T = cal_performance_reward(cms_att_probs_T, cms_labels_att[:, 1:],
                                                                             rewards_att,
                                                                             smoothing=True)  

                else:
                    cap_loss, cap_n_correct = cal_performance(cap_probs_v, cap_labels[:, 1:],
                                                              smoothing=True)  
                    cms_int_loss, cms_int_n_correct = cal_performance(cms_int_probs_v, cms_labels_int[:, 1:],
                                                                      smoothing=True)  
                    cms_eff_loss, cms_eff_n_correct = cal_performance(cms_eff_probs_v, cms_labels_eff[:, 1:],
                                                                      smoothing=True)  
                    cms_att_loss, cms_att_n_correct = cal_performance(cms_att_probs_v, cms_labels_att[:, 1:],
                                                                      smoothing=True)  

                    infoce_loss = 0

                    cap_loss_T, cap_n_correct_T = cal_performance(cap_probs_T, cap_labels[:, 1:], smoothing=True)
                    cms_int_loss_T, cms_int_n_correct_T = cal_performance(cms_int_probs_T, cms_labels_int[:, 1:],
                                                                          smoothing=True)
                    cms_eff_loss_T, cms_eff_n_correct_T = cal_performance(cms_eff_probs_T, cms_labels_eff[:, 1:],
                                                                          smoothing=True)
                    cms_att_loss_T, cms_att_n_correct_T = cal_performance(cms_att_probs_T, cms_labels_att[:, 1:],
                                                                          smoothing=True)


            else:

                cap_probs_v, cms_int_probs_v, cms_eff_probs_v, cms_att_probs_v = model(fc_feats, i3d, audio, cap_labels,
                                                                               cap_pos, cms_labels_int, cms_pos_int,
                                                                               cms_labels_eff, cms_pos_eff,
                                                                               cms_labels_att, cms_pos_att,umt_tensor, vis=True)

                cap_loss, cap_n_correct = cal_performance(cap_probs_v, cap_labels[:, 1:], smoothing=True)
                cms_int_loss, cms_int_n_correct = cal_performance(cms_int_probs_v, cms_labels_int[:, 1:], smoothing=True)
                cms_eff_loss, cms_eff_n_correct = cal_performance(cms_eff_probs_v, cms_labels_eff[:, 1:], smoothing=True)
                cms_att_loss, cms_att_n_correct = cal_performance(cms_att_probs_v, cms_labels_att[:, 1:], smoothing=True)

                infoce_loss = 0
                cos_sim_mean = 0

                cap_probs_T, cms_int_probs_T, cms_eff_probs_T, cms_att_probs_T = model_T(fc_feats, i3d, audio,
                                                                                         cap_labels,
                                                                                         cap_pos, cms_labels_int,
                                                                                         cms_pos_int,
                                                                                         cms_labels_eff, cms_pos_eff,
                                                                                         cms_labels_att, cms_pos_att,
                                                                                         umt_tensor, vis=True)

                cap_loss_T, cap_n_correct_T = cal_performance(cap_probs_T, cap_labels[:, 1:], smoothing=True)
                cms_int_loss_T, cms_int_n_correct_T = cal_performance(cms_int_probs_T, cms_labels_int[:, 1:],
                                                                      smoothing=True)
                cms_eff_loss_T, cms_eff_n_correct_T = cal_performance(cms_eff_probs_T, cms_labels_eff[:, 1:],
                                                                      smoothing=True)
                cms_att_loss_T, cms_att_n_correct_T = cal_performance(cms_att_probs_T, cms_labels_att[:, 1:],
                                                                      smoothing=True)

            epsilon = 1e-10
            p_int = cms_int_probs_T.softmax(dim=1)   
            q_int = cms_int_probs_v.softmax(dim=1)
            p_int = p_int.clone() + epsilon
            q_int = q_int.clone() + epsilon
            log_p_int = torch.log(p_int)
            kl_div_int = F.kl_div(log_p_int, q_int, reduction='batchmean')

            p_eff = cms_eff_probs_T.softmax(dim=1)
            q_eff = cms_eff_probs_v.softmax(dim=1)
            p_eff = p_eff.clone() + epsilon
            q_eff = q_eff.clone() + epsilon
            log_p_eff = torch.log(p_eff)
            kl_div_eff = F.kl_div(log_p_eff, q_eff, reduction='batchmean')

            p_att = cms_att_probs_T.softmax(dim=1)
            q_att = cms_att_probs_v.softmax(dim=1)
            p_att = p_att.clone() + epsilon
            q_att = q_att.clone() + epsilon
            log_p_att = torch.log(p_att)
            kl_div_att = F.kl_div(log_p_att, q_att, reduction='batchmean')


            # compute the token prediction Acc.
            non_pad_mask = cap_labels[:, 1:].ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()

            cms_int_non_pad_mask = cms_labels_int[:, 1:].ne(Constants.PAD)
            cms_int_n_word = cms_int_non_pad_mask.sum().item()

            cms_eff_non_pad_mask = cms_labels_eff[:, 1:].ne(Constants.PAD)
            cms_eff_n_word = cms_eff_non_pad_mask.sum().item()

            cms_att_non_pad_mask = cms_labels_att[:, 1:].ne(Constants.PAD)
            cms_att_n_word = cms_att_non_pad_mask.sum().item()


            cap_loss /= n_word
            cms_int_loss /= cms_int_n_word
            cms_eff_loss /= cms_eff_n_word
            cms_att_loss /= cms_att_n_word


            loss = 1*cap_loss + 5*cms_int_loss + 5*cms_eff_loss + 5*cms_att_loss + 0.005*(kl_div_int.item() + kl_div_eff.item() + kl_div_att.item()) #+  0*infoce_loss #+ 1000*cos_sim_mean


            loss.backward()
            optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)

            # update parameters
            cap_train_loss = cap_loss.item()
            cms_int_train_loss = cms_int_loss.item()
            cms_eff_train_loss = cms_eff_loss.item()
            cms_att_train_loss = cms_att_loss.item()
            

            cap_loss_T /= n_word
            cms_int_loss_T /= cms_int_n_word
            cms_eff_loss_T /= cms_eff_n_word
            cms_att_loss_T /= cms_att_n_word
            loss_T = cap_loss_T + cms_int_loss_T + cms_eff_loss_T + cms_att_loss_T
            loss_T.backward()#retain_graph=True)
            optimizer_T.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_T.parameters()), 1)

            cap_train_loss_T = cap_loss_T.item()
            cms_int_train_loss_T = cms_int_loss_T.item()
            cms_eff_train_loss_T = cms_eff_loss_T.item()
            cms_att_train_loss_T = cms_att_loss_T.item()
            

            if opt['cuda']: torch.cuda.synchronize()

            iteration += 1
            cms_int_n_correct_total_T += cms_int_n_correct_T
            cms_eff_n_correct_total_T += cms_eff_n_correct_T
            cms_att_n_correct_total_T += cms_att_n_correct_T

            cms_int_n_word_total += cms_int_n_word
            cms_eff_n_word_total += cms_eff_n_word
            cms_att_n_word_total += cms_att_n_word

            cms_int_train_loss_total_T += cms_int_train_loss_T
            cms_eff_train_loss_total_T += cms_eff_train_loss_T
            cms_att_train_loss_total_T += cms_att_train_loss_T

            if opt['cuda']: torch.cuda.synchronize()

            if iteration % opt['print_loss_every'] ==0:
                print('iter %d (epoch %d), cap_train_loss = %.6f, cms_int_train_loss_T = %.6f, cms_eff_train_loss_T = %.6f, cms_att_train_loss_T = %.6f,'
                      ' current step = %d, current lr = %.3E, cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f'
                      % (iteration, epoch, cap_train_loss, cms_int_train_loss_T, cms_eff_train_loss_T, cms_att_train_loss_T, optimizer.n_current_steps,
                         optimizer._optimizer.param_groups[0]['lr'],
                         cap_n_correct/n_word, cms_int_n_correct_T/cms_int_n_word, cms_eff_n_correct_T/cms_eff_n_word, cms_att_n_correct_T/cms_att_n_word))   #the generate accuracy of the word in the generated caption


                with open(opt['info_path'], 'a') as f:
                    f.write('model_%d, cap_loss: %.6f, cms_int_train_loss_T = %.6f, cms_eff_train_loss_T = %.6f, cms_att_train_loss_T = %.6f,\n'
                            % (epoch, cap_train_loss / iteration, cms_int_train_loss_T / iteration,
                               cms_eff_train_loss_T / iteration,
                               cms_att_train_loss_T / iteration))

            if opt['cuda']: torch.cuda.synchronize()
            cap_n_correct_total += cap_n_correct
            cms_int_n_correct_total += cms_int_n_correct
            cms_eff_n_correct_total += cms_eff_n_correct
            cms_att_n_correct_total += cms_att_n_correct
            n_word_total += n_word
            cms_int_n_word_total += cms_int_n_word
            cms_eff_n_word_total += cms_eff_n_word
            cms_att_n_word_total += cms_att_n_word
            cap_train_loss_total += cap_train_loss
            cms_int_train_loss_total += cms_int_train_loss
            cms_eff_train_loss_total += cms_eff_train_loss
            cms_att_train_loss_total += cms_att_train_loss

            if iteration % opt['print_loss_every'] ==0:
                print('iter %d (epoch %d), cap_train_loss = %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,'
                      ' current step = %d, current lr = %.3E, cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f'
                      % (iteration, epoch, cap_train_loss, cms_int_train_loss, cms_eff_train_loss, cms_att_train_loss, optimizer.n_current_steps,
                         optimizer._optimizer.param_groups[0]['lr'],
                         cap_n_correct/n_word, cms_int_n_correct/cms_int_n_word, cms_eff_n_correct/cms_eff_n_word, cms_att_n_correct/cms_att_n_word))   #the generate accuracy of the word in the generated caption

                with open(opt['info_path'], 'a') as f:
                    f.write('model_%d, cap_loss: %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,\n'
                            % (epoch, cap_train_loss / iteration, cms_int_train_loss / iteration,
                               cms_eff_train_loss / iteration,
                               cms_att_train_loss / iteration))


        print('model_%d, cap_loss: %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,'
              'cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f\n'
            % (epoch, cap_train_loss_total / iteration, cms_int_train_loss_total / iteration, cms_eff_train_loss_total / iteration, cms_att_train_loss_total / iteration,
               cap_n_correct_total/n_word_total, cms_int_n_correct_total/cms_int_n_word_total, cms_eff_n_correct_total/cms_eff_n_word_total, cms_att_n_correct_total/cms_att_n_word_total))


        if epoch % opt['save_checkpoint_every'] == 0:

            # save the checkpoint
            model_path = os.path.join(opt['output_dir'],
                                      '{}_{}.pth'
                                      .format(opt['output_dir'].split('/')[-1], epoch))

            if torch.__version__ == '1.3.1':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

            print('model saved to %s' % model_path)
            with open(opt['model_info_path'], 'a') as f:
                f.write('model_%d, cap_loss: %.6f, cms_int_train_loss = %.6f, cms_eff_train_loss = %.6f, cms_att_train_loss = %.6f,'
                        'cap_acc = %.3f, cms_int_acc = %.3f, cms_eff_acc = %.3f, cms_att_acc = %.3f\n'
                        % (epoch, cap_train_loss_total / iteration, cms_int_train_loss_total / iteration, cms_eff_train_loss_total / iteration, cms_att_train_loss_total / iteration,
                           cap_n_correct_total/n_word_total, cms_int_n_correct_total/cms_int_n_word_total, cms_eff_n_correct_total/cms_eff_n_word_total, cms_att_n_correct_total/cms_att_n_word_total))



class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, queries, positive_keys, negative_keys):

        positive_similarities = (queries * positive_keys).sum(dim=-1) / self.temperature  

        negative_similarities = (queries[:, None, :] * negative_keys).sum(
            dim=-1) / self.temperature  

        similarities = torch.cat([positive_similarities.unsqueeze(-1), negative_similarities],
                                 dim=-1)  

        labels = torch.zeros(similarities.shape[0], device=queries.device, dtype=torch.long)

        loss = F.cross_entropy(similarities, labels)
        return loss


def main(opt):
    # load and define dataloader
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)

    opt['cms_vocab_size'] = dataset.get_cms_vocab_size()
    opt['cap_vocab_size'] = dataset.get_cap_vocab_size()


    cms_int_text_length = opt['int_max_len']
    cms_eff_text_length = opt['eff_max_len']
    cms_att_text_length = opt['att_max_len']

    # model initialization.
    model = Model(
        dataset.get_cap_vocab_size(),
        dataset.get_cms_vocab_size(),
        cap_max_seq=opt['cap_max_len'],
        cms_max_seq_int=cms_int_text_length,
        cms_max_seq_eff=cms_eff_text_length,
        cms_max_seq_att=cms_att_text_length,
        tgt_emb_prj_weight_sharing=False,
        vis_emb=opt['dim_vis_feat'],
        rnn_layers=opt['rnn_layer'],
        d_k=opt['dim_head'],
        d_v=opt['dim_head'],
        d_model=opt['dim_model'],
        d_word_vec=opt['dim_word'],
        d_inner=opt['dim_inner'],
        n_layers=opt['num_layer'],
        n_head=opt['num_head'],
        dropout=opt['dropout'])

    # number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of learnable parameters are {}'.format(params))

    if opt['cuda']: model = model.cuda()

    # resume from previous checkpoint if indicated
    if opt['load_checkpoint'] and opt['resume']:
        cap_state_dict = torch.load(opt['load_checkpoint'])
        model_dict = model.state_dict()
        model_dict.update(cap_state_dict)
        model.load_state_dict(model_dict)

    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09), 512, opt['warm_up_steps'])

    opt['init_lr'] = round(optimizer.init_lr, 3)

    # create checkpoint output directory
    dir = opt['output_dir']
    if not os.path.exists(dir): os.makedirs(dir)

    # save the model snapshot to local
    info_path = os.path.join(dir, 'iteration_info_log.log')
    print('model architecture saved to {} \n {}'.format(info_path, str(model)))
    with open(info_path, 'a') as f:
        f.write(str(model))
        f.write('\n')
        f.write(str(params))
        f.write('\n')

    model_T = Model_T(
        dataset.get_cap_vocab_size(),
        dataset.get_cms_vocab_size(),
        cap_max_seq=opt['cap_max_len'],
        cms_max_seq_int=cms_int_text_length,
        cms_max_seq_eff=cms_eff_text_length,
        cms_max_seq_att=cms_att_text_length,
        tgt_emb_prj_weight_sharing=False,
        vis_emb=opt['dim_vis_feat'],
        rnn_layers=opt['rnn_layer'],
        d_k=opt['dim_head'],
        d_v=opt['dim_head'],
        d_model=opt['dim_model'],
        d_word_vec=opt['dim_word'],
        d_inner=opt['dim_inner'],
        n_layers=opt['num_layer'],
        n_head=opt['num_head'],
        dropout=opt['dropout'])

    # number of parameters
    model_T_parameters = filter(lambda p: p.requires_grad, model_T.parameters())
    params = sum([np.prod(p.size()) for p in model_T_parameters])
    print('number of learnable parameters are {}'.format(params))

    if opt['cuda']: model_T = model_T.cuda()

    # resume from previous checkpoint if indicated
    if opt['load_checkpoint'] and opt['resume']:
        cap_state_dict = torch.load(opt['load_checkpoint'])
        model_T_dict = model_T.state_dict()
        model_T_dict.update(cap_state_dict)
        model_T.load_state_dict(model_T_dict)

    optimizer_T = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model_T.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09), 512, opt['warm_up_steps'])

    opt['init_lr_T'] = round(optimizer_T.init_lr, 3)

    # create checkpoint output directory
    dir = opt['output_dir']
    if not os.path.exists(dir): os.makedirs(dir)

    # save the model snapshot to local
    info_path = os.path.join(dir, 'iteration_info_log.log')
    print('model architecture saved to {} \n {}'.format(info_path, str(model_T)))
    with open(info_path, 'a') as f:
        f.write(str(model_T))
        f.write('\n')
        f.write(str(params))
        f.write('\n')

    # log file directory
    opt['info_path'] = info_path
    opt['model_info_path'] = os.path.join(opt['output_dir'],
                                          'checkpoint_loss_log.log')

    train(dataloader, model, optimizer, opt, dataset.get_cap_vocab(), dataset.get_cms_vocab(), model_T, optimizer_T)

if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    main(opt)