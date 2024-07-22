import torch
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder
from time import process_time_ns
import numpy as np
import os

@torch.no_grad()
def beam_speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id, max_len : int , gamma : int = 4, width : int = 8, 
                         num_beams: int = 8, min_num_beams: int = 1,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    #print(prefix)
    #xxx = input()
    if pad_token_id is None:
        pad_token_id = eos_token_id

    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    acc_len = []
    acc_rate = []

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    assert prefix.shape[0] == 1, "input batch size must be 1"

    approx_time = 0
    target_time = 0
    sample_time = 0
    target_call_times = 0
    approx_call_times = 0
    d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times,
                'target_model_time': 0,
                'target_pre_cache_time': 0,
                'target_post_prob_time': 0
            }
    num_beams_list = []

    if approx_model.config.is_encoder_decoder == True:
        encoder_outputs = approx_model.get_encoder()(
                    prefix, return_dict=True
                    )
        for key, val in encoder_outputs.items():
            if key != 'last_hidden_state':
                del encdoer_outputs[key]
        output_prefix = torch.LongTensor([[pad_token_id]]).to(prefix.device)
        init_len = 1
        T = max_len
    else:
        output_prefix = prefix
        init_len = seq_len

    start_t = process_time_ns()


#    with tqdm(total=T, desc="speculative sampling") as pbar:
    first_input = True
    candidates = []
    try:
        while output_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = output_prefix.shape[1]

            # generate x of size width * (prefix_len+gamma)
            tt = process_time_ns()

            if approx_model.config.is_encoder_decoder:
                if output_prefix.size(0) == 1:
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0:1]
                else:
                    shape = encoder_outputs.last_hidden_state.shape
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0:1].repeat(output_prefix.size(0),
                                                                                                      *([1] * (len(shape)-1)))

                ret = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       encoder_outputs = encoder_outputs,
                       return_intermediate_results = True,
                       ret_seq_scores = True
                       )
            else:
                ret = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       return_intermediate_results = True,
                       output_scores = True,
                       ret_seq_scores = True
                       )
            out, all_seq, all_beam_idx, all_next_token, all_prob, all_input_idx = ret[0],ret[1],ret[2],ret[3],ret[4], ret[5]
            """
            for x in all_seq:
                print(x.size())
            xxx = input()
            for x in all_beam_idx:
                print(x.size())
            xxx = input()
            for x in all_next_token:
                print(x.size())
            xxx = input()
            for x in all_prob:
                print(x.size())
            xxx = input('end')
            """

            #TODO change x, padding
            max_len = all_seq[-1].size(1)
            x = [F.pad(seq, (0,max_len-seq.size(1),0,0), 'constant', pad_token_id) for seq in all_seq]
            att_mask = [F.pad(torch.ones_like(seq), (0,max_len-seq.size(1),0,0), 'constant', pad_token_id) for seq in all_seq]

            x = torch.concat(x, dim=0)
            att_mask = torch.concat(att_mask, dim=0)

            #x = out['sequences'] # width * (prefix_len+gamma)
            #q, seq_q = out['scores'] # tuples of gamma * (width * vocab) ?
 
            inc_len = x.shape[1] - prefix_len
            approx_call_times += 1
            approx_time += process_time_ns() - tt

            #print(x[:4])
            #print('=====================')

            tt = process_time_ns()
            all_input_idx = torch.concat(all_input_idx, dim=0)
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1, attention_mask = att_mask, copy_cache_index = all_input_idx)
            else:
                _ = target_model_cache.generate(
                        prefix.repeat_interleave(x.size(0), dim=0),
                        1, 
                        attention_mask = att_mask,
                        decoder_input_ids = x,
                        copy_cache_index = all_input_idx)
            target_call_times += 1
            p = target_model_cache._prob_history
            vocab_size = p.size(-1)
            
            target_time += process_time_ns() - tt

            
            """ compute acc rate """
            """
            for w in range(width):
                cur_target_p = 0
                for i in range(gamma):
                    if prefix_len + i >= x.size(1):
                        break
                    j = x[w, prefix_len+i]
                    cur_target_p += torch.log(p[w, prefix_len + i - 1, j])
                    cur_draft_p = seq_q[w, i]
                    acc_rate.append((torch.exp(cur_target_p)/cur_draft_p).item())
                    if acc_rate[-1] > 1:
                        acc_rate[-1] = 1
            """

            """ verification process """
            tt = process_time_ns()
            
            if first_input == True:
                cur_valid_beam = torch.zeros_like(all_beam_idx[0])
                cur_valid_beam[0] = 1
                cur_valid_beam = cur_valid_beam.bool()
                beam_scores = torch.zeros_like(all_prob[0])
            else:
                cur_valid_beam = torch.ones_like(all_beam_idx[0]).bool()

            n = prefix_len - 1

            max_l = 0
            start = 0
            for i in range(inc_len):
                #print('beam scores')
                #print(beam_scores)
                #print(cur_valid_beam)
                #TODO check accept/reject decision, since we set approx and target model the same, most of decision should be accept
                end = start + num_beams
                cur_beam_idx = all_beam_idx[i]
                #print(cur_beam_idx)
                # get sampled distribution of the small model
                q_scores = all_prob[i]
                #print(q_scores)
                # speacial treatment for i==0
                if first_input:
                    cur_beam_idx[:] = 0
                    q_scores = q_scores * num_beams
                    first_input = False

                # shift cur_beam_idx by cur_valid_beam
                shift = torch.cumsum(cur_valid_beam.long(),dim=0)-1
                shift_beam_idx = shift[cur_beam_idx]

                # Step 1 get sampling distribution of the large model
                #print('prefix_len+i')
                #print(prefix_len+i-1,i)
                #print(x[0,prefix_len+i-1])
                cur_p = p[start:end, prefix_len+i-1].squeeze() # shape: batch_size * V 
                #print(cur_p[0].nonzero())
                #print(cur_p[1].nonzero())
                cur_p = cur_p[cur_valid_beam] # shape: num_valid_beam * V
                #print(cur_p.nonzero())
                #print('===============')
                #print(prefix)
                #print(x[0])
                #xxx = input()

                from_valid_beam = cur_valid_beam[cur_beam_idx]
                #print(from_valid_beam)
                if from_valid_beam.any():
                    p_next_token_scores = beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
                    p_next_token_scores = torch.softmax(p_next_token_scores.view(-1), dim=0)
                    shift_beam_idx = torch.clamp(shift_beam_idx, min=0)
                    cur_sample_idx = shift_beam_idx * vocab_size + all_next_token[i]
                    #TODO update beam score
                    """
                    pickle.dump(p_next_token_scores, open('tmp.pkl','wb'))
                    print(p_next_token_scores.size())
                    print(cur_sample_idx)
                    print(cur_valid_beam)
                    print(cur_beam_idx)
                    print(shift)
                    print(shift_beam_idx)
                    print('========================')
                    """
                    p_scores = torch.gather(p_next_token_scores, dim=0, index=cur_sample_idx)
                    mask = from_valid_beam.logical_not()
                    p_scores[mask] = 0


                    r = torch.rand(1, device = p.device)-1e-5
                    #p_scores = q_scores
                    accept = (p_scores/(q_scores+1e-5)) > r
#                    if i == inc_len - 1:
#                        accept[:] = False
                    """
                    if torch.logical_not(accept).any():
                        print('wait')
                        print(p_scores)
                        print(q_scores)
                        print(p_scores/q_scores)
                        print(r)
                        print('--------------------')
                        #xxx = input()
                    """
                 #   print('p_scores')
                 #   print(p_scores)
                 #   print(p_scores/q_scores)
             
                else:
                    # non of the sample is from valid beams, all reject
                    accept = from_valid_beam

                #accept[:] = True
                #p_scores = q_scores
                acc_cnt, acc_r = accept.float().sum().item(), accept.float().mean().item()
                acc_rate.append(acc_r)
                #print(accept)

                if acc_cnt >= min_num_beams:
                    num_beams_list.append(acc_cnt)
                    # Step 5 update cur_valid_beam
                    cur_valid_beam = accept
                  #  print(p_scores)
                    beam_scores = p_scores.log() 
                     
                    #print('update beam scores (inside)')
                    #print(beam_scores)
                    #print(cur_valid_beam)
                    
                    
                    n += 1
                    max_l += 1
                    start = end
                else:
                    # if all draft are rejected, terminate and start re-sample
                    num_beams_list.append(num_beams)
                    break

            #TODO re-sample based on cur_valid_beam and p
            end=start + num_beams
            acc_len.append(max_l)
            
            if max_l == inc_len: # all accept

                cur_p = p[start:end, -1]
                cur_p = cur_p[cur_valid_beam]
                p_next_token_scores = beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
                op = p_next_token_scores 
                p_next_token_scores = norm_logits(p_next_token_scores.view(1,-1), temperature = temperature, top_k = top_k, top_p = top_p).squeeze()

                """    
                mask = p_next_token_scores.isnan().view(op.size(0),op.size(1)) 
                print(mask.float().sum())
                print(op.size())
                print(beam_scores[cur_valid_beam])

                print(p_next_token_scores.isinf().any())
                print(p_next_token_scores.isnan().any())
                print(op[mask])
                print(cur_p[mask])
                print(cur_p[mask].log())
                print(beam_scores[cur_valid_beam][:,None].expand_as(cur_p)[mask])
                """

                try:
                    t = sample(p_next_token_scores, num_samples = num_beams)
                except:
                    t = sample(torch.softmax(op.view(-1), dim=0), num_samples = num_beams)

                beam_idx = torch.div(t, vocab_size, rounding_mode='floor')
                beam_idx = beam_idx.long()
                token = t % vocab_size
                token = token[:,None]
                beam_scores = p_next_token_scores[t].log().squeeze()
                """
                print('update beam scores')
                print(p_next_token_scores.nonzero())
                print(t)
                print(p_next_token_scores[t])
                print(beam_scores)
                """


                choice = cur_valid_beam.nonzero()[beam_idx].squeeze()
                choice = start + choice
                output_prefix = x[choice, :n+1]
                #beam_cnt = 0
                #for i in range(num_beams):
                #    if cur_valid_beam[i] == True:
                #        beam_cnt += 1
                #    if beam_idx == beam_cnt - 1:
                #        output_prefix = x[start+i:start+i+1, :n + 1]
                #        choice = start + i
                #        break
                """
                print(cur_valid_beam)
                print(beam_idx)
                print(start)
                print(choice)
                print(output_prefix.size())
                print(token.size())
                """
                #TODO for debug
                output_prefix = torch.concat([output_prefix, token], dim=1)
                target_model_cache.rollback(n+2, choice)
            else:
                cur_p = p[start:end, n]
                #print(cur_p.sum(dim=1))
                cur_p = cur_p[cur_valid_beam]
                p_next_token_scores = beam_scores[cur_valid_beam][:,None].expand_as(cur_p) + cur_p.log()
                op = p_next_token_scores 
                #p_next_token_scores = torch.softmax(p_next_token_scores.view(-1), dim=0)
                p_next_token_scores = norm_logits(p_next_token_scores.view(1,-1), temperature = temperature, top_k = top_k, top_p = top_p).squeeze()

                """
                mask = p_next_token_scores.isnan().view(op.size(0),op.size(1)) 
                print(mask.float().sum())
                print(op.size())
                print(beam_scores[cur_valid_beam])

                print(p_next_token_scores.isinf().any())
                print(p_next_token_scores.isnan().any())
                print(op[mask])
                print(cur_p[mask])
                print(cur_p[mask].log())
                print(beam_scores[cur_valid_beam][:,None].expand_as(cur_p)[mask])
                """
                #TODO minus q
                try:
                    t = sample(p_next_token_scores, num_samples = num_beams)
                except:
                    t = sample(torch.softmax(op.view(-1), dim=0), num_samples = num_beams)
                beam_idx = torch.div(t, vocab_size, rounding_mode='floor')
                beam_idx = beam_idx.long()
                token = t % vocab_size
                token = token[:,None]
                choice = cur_valid_beam.nonzero()[beam_idx].squeeze()
                choice = start + choice
                output_prefix = x[choice, :n+1]
                """
                print(cur_valid_beam)
                print(beam_idx)
                print(start)
                print(choice)
                print(output_prefix.size())
                print(token.size())
                """
                beam_scores = p_next_token_scores[t].log().squeeze()
                """
                print('update beam scores')
                print(p_next_token_scores.nonzero())
                print(t)
                print(p_next_token_scores[t])
                print(beam_scores)
                """
                #beam_cnt = 0
                #for i in range(num_beams):
                #    if cur_valid_beam[i] == True:
                #        beam_cnt += 1
                #    if beam_idx == beam_cnt - 1:
                #        output_prefix = x[start+i:start+i+1, :n + 1]
                #        choice = start+i
                #        break
                output_prefix = torch.concat([output_prefix, token], dim=1)
                target_model_cache.rollback(n+1, choice)


            if max_l == inc_len:
                last_beam_idx = all_beam_idx[-1] 
#                print(all_beam_idx)
#                print(choice)
#                print(last_beam_idx[choice%num_beams])
#                xxx = input()
                approx_model_cache.beam_rollback(max_l, last_beam_idx[choice%num_beams])
            else:
                approx_model_cache.beam_rollback(max_l, choice%num_beams)

            cur_valid_beam = torch.ones_like(all_beam_idx[0]).bool()


            # check each sequence for eos_token, if there is, save it as one of the candidates, continue the search
            mask = (output_prefix == eos_token_id)
            end_cnt = 0
            for i in range(mask.size(0)):
                if mask[i].int().sum() > ori_eos_cnt: #encounter eos
                    end_cnt += 1
                    row_mask = torch.cumsum(mask[i].float(), dim=0)
                    row_mask = (row_mask < ori_eos_cnt+1)
                    end = row_mask.int().sum()
                    if end < mask.size(1):
                        row_mask[end] = True
                    output_candidate = output_prefix[i][row_mask] 
                    #print(output_prefix[i].size())
                    #print(mask[i].size())
                    #print(output_candidate.size())

                    cdd_score = beam_scores[i]/(output_candidate.size(-1) - init_len)
                    cur_valid_beam[i] = False
                    candidates.append((output_candidate, cdd_score))
            if end_cnt >= mask.size(0):
                break
            



            sample_time += process_time_ns() - tt
    except Exception as e:
        print(e)
        raise RuntimeError('')

    for i in range(output_prefix.size(0)):
        output_candidate = output_prefix[i]
        cdd_score = beam_scores[i]/(output_candidate.size(-1)-init_len)
        candidates.append((output_candidate, cdd_score))

    best_score = -10000
    for cdd, cdd_score in candidates:
        if cdd_score > best_score:
            output_prefix = cdd
            best_score = cdd_score
#    print(output_prefix.size())


    if approx_model.config.is_encoder_decoder:
        output_prefix = torch.cat((prefix, output_prefix[None,:]), dim=1)
    else:
        output_prefix = output_prefix[None,:]


    if verbose:
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times,
                'num_beams_list': num_beams_list,
                'target_model_time': target_model_cache.forward_time_dict['_model_time'],
                'target_pre_cache_time': target_model_cache.forward_time_dict['prepare_cache_time'],
                'target_post_prob_time': target_model_cache.forward_time_dict['norm_prob_time'],
            }
        return output_prefix, d
    else:
        return output_prefix


@torch.no_grad()
def mjsd_speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id, max_len : int , 
                         gamma : int = 4, width : int = 8, num_beams: int = 8, accept_thres: float = 0.1,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    if pad_token_id is None:
        pad_token_id = eos_token_id

    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    acc_len = []
    acc_rate = []

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    assert prefix.shape[0] == 1, "input batch size must be 1"

    approx_time = 0
    target_time = 0
    sample_time = 0
    target_call_times = 0
    approx_call_times = 0

    if approx_model.config.is_encoder_decoder == True:
        encoder_outputs = approx_model.get_encoder()(
                    prefix, return_dict=True
                    )
        for key, val in encoder_outputs.items():
            if key != 'last_hidden_state':
                del encdoer_outputs[key]
        output_prefix = torch.LongTensor([[pad_token_id]]).to(prefix.device)
        T = max_len
    else:
        output_prefix = prefix

    start_t = process_time_ns()


#    with tqdm(total=T, desc="speculative sampling") as pbar:
    if True:
#    try:
        while output_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = output_prefix.shape[1]

            # generate x of size width * (prefix_len+gamma)
            tt = process_time_ns()
            #num_beams = max(4, width)
            if approx_model.config.is_encoder_decoder:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0:1]

                out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       encoder_outputs = encoder_outputs,
                       ret_seq_scores = True
                       )
            else:
                out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       ret_seq_scores = True
                       )


            x = out['sequences'] # width * (prefix_len+gamma)
            q, seq_q = out['scores'] # tuples of gamma * (width * vocab) ?
 
            inc_len = x.shape[1] - prefix_len
            approx_call_times += 1
            approx_time += process_time_ns() - tt

            tt = process_time_ns()
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(
                        prefix.repeat_interleave(width, dim=0),
                        1, 
                        decoder_input_ids = x)
            target_call_times += 1
            p = target_model_cache._prob_history
            target_time += process_time_ns() - tt


            for w in range(width):
                cur_target_p = 0
                for i in range(gamma):
                    if prefix_len + i >= x.size(1):
                        break
                    j = x[w, prefix_len+i]
                    cur_target_p += torch.log(p[w, prefix_len + i - 1, j])
                    cur_draft_p = seq_q[w, i]
                    acc_rate.append((torch.exp(cur_target_p)/cur_draft_p).item())
                    if acc_rate[-1] > 1:
                        acc_rate[-1] = 1



            tt = process_time_ns()
            is_all_accept = False
            n = prefix_len - 1
            max_n = prefix_len - 1
            max_l = 0
            choice = 0

            """
            for w in range(width):
                cur_n = prefix_len - 1
                cur_l = 0
                cur_all_accept = True

                cur_target_p = 0
                for i in range(inc_len):
                    if prefix_len + i >= x.size(1):
                        break
                    if random_seed:
                        torch.manual_seed(random_seed)
                    r = torch.rand(1, device = p.device)
                    #r = 0.5
                    j = x[w, prefix_len + i]

                    cur_target_p += torch.log(p[w, prefix_len + i - 1, j])
                    cur_draft_p = seq_q[w, i]
               
                    if r < torch.min(torch.tensor([1], device=q.device), torch.exp(cur_target_p)/cur_draft_p):
                        cur_l += 1
                    # accept, and update n
                        cur_n += 1
                    else:
                        # reject
                        cur_all_accept = False
                        break
                if cur_l > max_l:
                    assert cur_n > max_n, f"cur_n {cur_n}, max_n {max_n}"
                    max_n = cur_n
                    max_l = cur_l
                    choice = w
                    if cur_all_accept == True:
                        is_all_accept = True
                        break
            """
            """
            Let's try accept the longest sequences
            """
            
            for w in range(width):
                cur_n = prefix_len - 1
                cur_l = 0
                cur_all_accept = True

                cur_target_p = 0
                for i in range(inc_len):
                    if prefix_len + i >= x.size(1):
                        break
                    if random_seed:
                        torch.manual_seed(random_seed)
                    #r = torch.rand(1, device = p.device)
                    r = accept_thres
                    j = x[w, prefix_len + i]

                    cur_target_p += torch.log(p[w, prefix_len + i - 1, j])
                    cur_draft_p = seq_q[w, i]

                    cur_l += 1
                    cur_n += 1

                    if r >= 1: # always reject
                        continue
               
                    if r <= torch.min(torch.tensor([1], device=q.device), torch.exp(cur_target_p)/cur_draft_p) or r < 0: # r < 0 means always accept
                        if cur_l > max_l:
                            max_n = cur_n
                            max_l = cur_l
                            choice = w
                    else:
                        continue
                if max_l == inc_len:
                    is_all_accept = True
                    break
            

            acc_len.append(max_l)

            n = max_n
         
            output_prefix = x[choice:choice+1, :n + 1]
        
            approx_model_cache.rollback(n+1, choice)
 
            if is_all_accept:
                t = sample(p[choice:choice+1, -1, :])
                target_model_cache.rollback(n+2, choice)

            else:
                #print(torch.sum(p[choice,n,:]).item(), torch.sum(q[choice,max_l,:]).item())
                #tmp = p[choice,n,:] - q[choice,max_l,:]
                #print(torch.sum((tmp>0).float()).item(), torch.sum((tmp<=0).float()).item())
                #print(torch.sum((p[choice,n,:]==q[choice,max_l,:]).float()).item())

                #t = sample(max_fn(p[choice:choice+1, n, :] - q[choice:choice+1, max_l, :]))
                t = sample(max_fn(p[choice:choice+1, n, :]))

                target_model_cache.rollback(n+1, choice)
           
            output_prefix = torch.cat((output_prefix, t), dim=1)
            #pbar.update(n - pbar.n)
            mask = (output_prefix == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                output_prefix = output_prefix[mask][None,:] 
                break




            sample_time += process_time_ns() - tt
#    except Exception as e:
#        print(e)

    if approx_model.config.is_encoder_decoder:
        output_prefix = torch.cat((prefix, output_prefix), dim=1)

    if verbose:
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times
            }
        return output_prefix, d
    else:
        return output_prefix


@torch.no_grad()
def multi_speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id, max_len : int ,
                         gamma : int = 4, width : int = 8, num_beams = None, strategy : str = "beam", 
                         acc_rate_head = None, acc_rate_thres = 0.4, 
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    acc_len = []
    acc_rate = []

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    """ 
    print(prefix.size())
    out = approx_model_cache.beam_sample_with_kv_cache(prefix, gamma=gamma, num_beams=width, 
            top_k=top_k, top_p=top_p,
            num_return_sequences=width,
          )
    print(out['sequences'].size())
    xxx = input('')
    prefix = out['sequences'][0:1]
    approx_model_cache.rollback(prefix.size(1), 0)
    out = approx_model_cache.beam_sample_with_kv_cache(prefix, gamma=gamma, num_beams=width, 
            top_k=top_k, top_p=top_p,
            num_return_sequences=width)
    print(out['sequences'].size())
    xxx = input('')
    return None, None
    """

    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    approx_time = 0
    target_time = 0
    sample_time = 0
    target_call_times = 0
    approx_call_times = 0

    start_t = process_time_ns()

    if approx_model.config.is_encoder_decoder == True:
        encoder_outputs = approx_model.get_encoder()(
                    prefix, return_dict=True
                    )
        for key, val in encoder_outputs.items():
            if key != 'last_hidden_state':
                del encdoer_outputs[key]
        output_prefix = torch.LongTensor([[pad_token_id]]).to(prefix.device)
        T = max_len
    else:
        output_prefix = prefix
    model_kwargs = {}



#    with tqdm(total=T, desc="speculative sampling") as pbar:
#    if True:
    try:
        while output_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = output_prefix.shape[1]

            # generate x of size width * (prefix_len+gamma)
            tt = process_time_ns()
            if strategy == "beam":
                #out = approx_model.generate(x, num_beams=width, 
                #        output_scores=True, return_dict_in_generate=True, 
                #        num_return_sequences=width,
                #        top_k = top_k,
                #        top_p = top_p,
                #        do_sample=True,
                #        max_new_tokens = gamma)
                if num_beams is None:
                    num_beams = max(4, width)

                if approx_model.config.is_encoder_decoder:
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0:1]

                    out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       encoder_outputs = encoder_outputs
                       )
                else:
                    out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       )

                x = out['sequences'] # width * (prefix_len+gamma)

                q = out['scores'] # tuples of gamma * (width * vocab) ?
                #q = torch.stack(q, dim=1) # width * gamma * vocab
                #for w in range(width):
                #   for i in range(q.shape[1]):
                       #q[w:w+1, i, :] = norm_logits(q[w:w+1, i, :],
                       #                  temperature, top_k, top_p)
                #       q[w:w+1, i, :] = F.softmax(q[w:w+1, i, :], dim=1)
            elif strategy == 'acc_beam':
                assert (acc_rate_head is not None)
                out = approx_model_cache.beam_sample_with_kv_cache(output_prefix, 
                     gamma=gamma, 
                     num_beams=width, 
                     top_k=top_k, top_p=top_p,
                     num_return_sequences=width,
                     return_dict_in_generate = True,
                     output_scores = True,
                     acc_rate_head = acc_rate_head,
                     acc_rate_thres = acc_rate_thres,
                     **model_kwargs
                     )

                x = out['sequences'] # width * (prefix_len+gamma)

                q = out['scores'] # tuples of gamma * (width * vocab) ?

            elif strategy == "diverse":
                raise NotImplementedError
                #out = approx_model.generate(x, num_beams=width, num_beam_groups=4, 
                #        diversity_penalty=0.1,
                #        output_scores=True, return_dict_in_generate=True, 
                #        top_k = top_k,
                #        top_p = top_p,
                #        do_sample=False,
                #        num_return_sequences=width,
                #        max_new_tokens = gamma)

                #x = out['sequences'] # width * (prefix_len+gamma)
                #q = out['scores'] # tuples of gamma * (width * vocab) ?
                #q = torch.stack(q, dim=1) # width * gamma * vocab
                #for w in range(width):
                #   for i in range(q.shape[1]):
                       #q[w:w+1, i, :] = norm_logits(q[w:w+1, i, :],
                       #                  temperature, top_k, top_p)
                #       q[w:w+1, i, :] = F.softmax(q[w:w+1, i, :], dim=1)


            elif strategy == 'iid':
                if approx_model.config.is_encoder_decoder == False:
                    out = approx_model_cache.generate(output_prefix,
                        gamma,
                        multi = width,
                        strategy = 'iid')
                else:
                    out = approx_model_cache.generate(prefix,
                        gamma,
                        decoder_input_ids = output_prefix,
                        multi = width,
                        strategy = 'iid')

                q = approx_model_cache._prob_history[:,prefix_len-1:,:]
                x = out

            else:
                raise RuntimeError("Strategy not implemented "+strategy)

            inc_len = x.shape[1] - prefix_len
            approx_call_times += 1



            approx_time += process_time_ns() - tt

            tt = process_time_ns()
   #         p = target_model(x).logits 
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(
                        prefix.repeat_interleave(width, dim=0),
                        1, 
                        decoder_input_ids = x)


            target_call_times += 1
            p = target_model_cache._prob_history
            #for w in range(width):
            #    for i in range(p.shape[1]):
            #        p[w:w+1, i, :] = norm_logits(p[w:w+1, i, :],
            #                temperature, top_k, top_p)
            target_time += process_time_ns() - tt
            """
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            """
            for w in range(width):
                for i in range(gamma):
                    if prefix_len + i >= x.size(1):
                        break
                    j = x[w, prefix_len+i]
                    acc_rate.append((p[w, prefix_len + i - 1, j] / q[w, i, j]).item())
                    if acc_rate[-1] > 1:
                        acc_rate[-1] = 1
                    if q[w,i,j] == 0:
                        acc_rate[-1] = 0
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            tt = process_time_ns()
            
            is_all_accept = False
            n = prefix_len - 1
            max_n = prefix_len - 1
            max_l = 0
            choice = 0
            for w in range(width):
                cur_n = prefix_len - 1
                cur_l = 0
                cur_all_accept = True
                for i in range(inc_len):
                    if prefix_len + i >= x.size(1):
                        break
                    if random_seed:
                        torch.manual_seed(random_seed)
                    r = torch.rand(1, device = p.device)
                    j = x[w, prefix_len + i]

               
                    if r < torch.min(torch.tensor([1], device=q.device), p[w, prefix_len + i - 1, j] / q[w, i, j]):
                        cur_l += 1
                    # accept, and update n
                        cur_n += 1
                    else:
                        # reject
                        cur_all_accept = False
                        break
                if cur_l > max_l:
                    assert cur_n > max_n, f"cur_n {cur_n}, max_n {max_n}"
                    max_n = cur_n
                    max_l = cur_l
                    choice = w
                    if cur_all_accept == True:
                        is_all_accept = True
                        break
            acc_len.append(max_l)

            n = max_n
         
            output_prefix = x[choice:choice+1, :n + 1]
        
            approx_model_cache.rollback(n+1, choice)
           
            if is_all_accept:
                t = sample(p[choice:choice+1, -1, :])
                target_model_cache.rollback(n+2, choice)

            else:
                #print(torch.sum(p[choice,n,:]).item(), torch.sum(q[choice,max_l,:]).item())
                #tmp = p[choice,n,:] - q[choice,max_l,:]
                #print(torch.sum((tmp>0).float()).item(), torch.sum((tmp<=0).float()).item())
                #print(torch.sum((p[choice,n,:]==q[choice,max_l,:]).float()).item())
                new_p = max_fn(p[choice:choice+1, n, :] - q[choice:choice+1, max_l, :])

                try:
                    t = sample(new_p)
                except Exception as e:
                    # it seems it is possible to sample x where p = 0 and q = 0
                    t = sample(p[choice:choice+1, n, :])
                    #print(new_p.sum())
                    #print((p[choice:choice+1, n, :] - q[choice:choice+1, max_l, :]).max())
                    #print(p[choice:choice+1, n, :].sum())
                    #print(q[choice:choice+1, max_l, :].sum())
                    #j = x[choice, n+1]
                    #print(p[choice, n, j], q[choice, max_l, j])
                    #raise RuntimeError(f'{e}')


                target_model_cache.rollback(n+1, choice)
           
            output_prefix = torch.cat((output_prefix, t), dim=1)
            #pbar.update(n - pbar.n)
            sample_time += process_time_ns() - tt

            mask = (output_prefix == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                output_prefix = output_prefix[mask][None,:] 
                break
    except Exception as e:
        print(e)



    
    if approx_model.config.is_encoder_decoder:
        output_prefix = torch.cat((prefix, output_prefix), dim=1)

    if verbose:
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times
            }
        return output_prefix, d
    else:
        return output_prefix

@torch.no_grad()
def BiLD_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         gamma, eos_token_id,  
                         pad_token_id, fallback_thres, rollback_thres,
                         max_len : int ,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    approx_time = 0
    target_time = 0
    sample_time = 0
    approx_call_times = 0
    target_call_times = 0

    start_t = process_time_ns()

    acc_rate = []
    acc_len = []

    if pad_token_id is None:
        pad_token_id = eos_token_id
    decoder_input_ids = torch.LongTensor([[pad_token_id]]).to(prefix.device)

    if approx_model.config.is_encoder_decoder == False:
        last_check = seq_len - 1
    else:
        last_check = 0

    try:
        while prefix.shape[1] + decoder_input_ids.shape[1] - 1 < T:
            #print('loop')
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            tt = process_time_ns()
          
            #TODO for debug
#            approx_model_cache.reset_cache()
            if approx_model.config.is_encoder_decoder == False:
                x = approx_model_cache.generate(prefix, 1)
                prefix_len = prefix.shape[1]
            else:
                x = approx_model_cache.generate(prefix, 1, decoder_input_ids = decoder_input_ids)
                prefix_len = decoder_input_ids.shape[1]

            q = approx_model_cache._prob_history[:,prefix_len-1:,:]
   
            approx_call_times += 1
        
            approx_time += process_time_ns() - tt
            tt = process_time_ns()

            if torch.max(q[:,-1,:]) < fallback_thres or x.size(1)-last_check-1 >= gamma:
                # use large model to check
                ttt = process_time_ns()
        
                if target_model.config.is_encoder_decoder == False:
                    _ = target_model_cache.generate(x, 1)
                else:
                    _ = target_model_cache.generate(prefix, 1, decoder_input_ids = x)
                target_call_times += 1
 
                target_time += process_time_ns() - ttt
                p = target_model_cache._prob_history
                n = x.size(1) - 1
                l = 0
                for i in range(last_check, x.size(1)-1):
                    j = x[:, i+1]
                    if -p[:, i, j].log() > rollback_thres:
                        n = i
                        break
                    l += 1
                acc_len.append(l)
                if approx_model.config.is_encoder_decoder == False:
                    prefix = x[:, :n+1]
                else:
                    decoder_input_ids = x[:, :n+1]
                
                approx_model_cache.rollback(n+1)
                t = sample(p[:, n, :])
                target_model_cache.rollback(n+1)
                last_check = n+1

                if approx_model.config.is_encoder_decoder == False:
                    prefix = torch.cat((prefix, t), dim=1)
                else:
                    decoder_input_ids = torch.cat((decoder_input_ids, t), dim=1)


            else: # continue
                if approx_model.config.is_encoder_decoder:
                    decoder_input_ids = x
                else:
                    prefix = x

            if approx_model.config.is_encoder_decoder:
                out = decoder_input_ids
            else:
                out = prefix

            mask = (out == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                out = out[mask][None,:] 
                break



            sample_time += process_time_ns() - tt
    except Exception as e:
        print(e)

    if approx_model.config.is_encoder_decoder == True:
        out = torch.cat((prefix, out), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"Acc rate: {np.mean(acc_rate)}")
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times

            }
        return out, d
    else:
        return out



@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id,
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    approx_time = 0
    target_time = 0
    sample_time = 0
    approx_call_times = 0
    target_call_times = 0

    start_t = process_time_ns()

    acc_rate = []
    acc_len = []

    if pad_token_id is None:
        pad_token_id = eos_token_id
    decoder_input_ids = torch.LongTensor([[pad_token_id]]).to(prefix.device)

    try:
        while prefix.shape[1] + decoder_input_ids.shape[1] - 1 < T:
            #print('loop')
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            tt = process_time_ns()

            # TODO for debug
#            approx_model_cache.reset_cache()
        
            if approx_model.config.is_encoder_decoder == False:
                x = approx_model_cache.generate(prefix, gamma)
                prefix_len = prefix.shape[1]
            else:
                x = approx_model_cache.generate(prefix, gamma, decoder_input_ids = decoder_input_ids)
                prefix_len = decoder_input_ids.shape[1]

            approx_call_times += 1
        
            approx_time += process_time_ns() - tt
            tt = process_time_ns()
        
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(prefix, 1, decoder_input_ids = x)
            target_call_times += 1

            target_time += process_time_ns() - tt
            tt = process_time_ns()
        
            n = prefix_len + gamma - 1

            for i in range(gamma):
                j = x[:, prefix_len + i]
                acc_rate.append(((target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j])).item())
                if acc_rate[-1] > 1:
                    acc_rate[-1] = 1
        

            l = 0
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = device)
                j = x[:, prefix_len + i]
            
                if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    # reject
                    n = prefix_len + i - 1
                    break
            
                if verbose:
                    print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

                accepted_count += 1
                l += 1
            acc_len.append(l)
        
            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            if approx_model.config.is_encoder_decoder == False:
                prefix = x[:, :n + 1]
            else:
                decoder_input_ids = x[:, :n+1]
        
            approx_model_cache.rollback(n+1)
            #print('after roll back')
            #print(approx_model_cache._prob_history.size())
            assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
            if n < prefix_len + gamma - 1:
                # reject someone, sample from the pos n
                try:
                    t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
                except:
                    t = sample(max_fn(target_model_cache._prob_history[:, n, :]))

                if verbose:
                    print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
                resample_count += 1
                target_model_cache.rollback(n+1)
            else:
                 # all approx model decoding accepted
                assert n == target_model_cache._prob_history.shape[1] - 1
                t = sample(target_model_cache._prob_history[:, -1, :])
                if verbose:
                    print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
                target_sample_count += 1
                target_model_cache.rollback(n+2)
        
        
            if approx_model.config.is_encoder_decoder == False:
                prefix = torch.cat((prefix, t), dim=1)
                out = prefix
            else:
                decoder_input_ids = torch.cat((decoder_input_ids, t), dim=1)
                out = decoder_input_ids

            mask = (out == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                out = out[mask][None,:] 
                break

            sample_time += process_time_ns() - tt
    except Exception as e:
        print(e)

        #print(f'n={n}, l={l}')

    if approx_model.config.is_encoder_decoder == True:
        out = torch.cat((prefix, out), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"Acc rate: {np.mean(acc_rate)}")
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times,
                'target_model_time': target_model_cache.forward_time_dict['_model_time'],
                'target_pre_cache_time': target_model_cache.forward_time_dict['prepare_cache_time'],
                'target_post_prob_time': target_model_cache.forward_time_dict['norm_prob_time'],
            }
        return out, d
    else:
        return out


@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None, details : bool = False) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    approx_time = 0
    target_time = 0
    sample_time = 0


    acc_rate = []
    acc_len = []


    #with tqdm(total=T, desc="speculative sampling") as pbar:
    if True:
        while prefix.shape[1] < T:
            tt = process_time_ns()
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            approx_time += process_time_ns() - tt
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            tt = process_time_ns()
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            target_time += process_time_ns() - tt

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            tt = process_time_ns()
            
            is_all_accept = True
            n = prefix_len - 1
            l = 0
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                acc_rate.append(torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]).item())
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                    l += 1
                else:
                    # reject
                    try:
                        t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    except Exception as e:
                        print(e)
                        print('reject, r, t: ', r, p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j])
                        print(prefix_len+i-1, n)
                        print(torch.sum(p[:,n,:]), torch.sum(q[:,n,:]))
                        print(torch.sum(((p[:,n,:]-q[:,n,:])>0).float()))
                        raise RuntimeError(f'{e}')

                    is_all_accept = False
                    break
            acc_len.append(l)
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            sample_time += process_time_ns() - tt
#            pbar.update(n - pbar.n)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate)
            }
        return prefix, d
    else:
        return prefix


