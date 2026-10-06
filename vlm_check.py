"""CPU-only official tokenizer/processor and constrained digit-loss checks.

Run with PYTHONPATH=.vlm-deps using the proposal Python environment. No weights.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, __version__

from stage0 import digit_log_mass

ROOT=Path(__file__).resolve().parent


def constrained_log_mass(prediction_logits, target_token_ids, digit_ids):
    """Score four true-prefix digit predictions, normalized over ten legal tokens.

    Caller gathers causal model logits at target positions minus one. Fixed
    punctuation and all prompt tokens have no loss. Returns one log cell mass.
    """
    if prediction_logits.ndim!=2 or prediction_logits.shape[0]!=4 or target_token_ids.shape!=(4,) or len(digit_ids)!=10 or len(set(digit_ids))!=10:
        raise ValueError('Four digit predictions and ten unique legal tokens are required.')
    ids=torch.tensor(digit_ids,device=prediction_logits.device)
    match=target_token_ids[:,None]==ids[None,:]
    if not torch.all(match.sum(-1)==1):raise ValueError('Target is not a legal digit token.')
    legal=prediction_logits.index_select(-1,ids).float()
    return legal.log_softmax(-1)[torch.arange(4,device=legal.device),match.long().argmax(-1)].sum()


def prepare(processor, entry):
    text=entry['conversations'][0]['value'].replace('<image>',processor.image_token)
    prefix=processor.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    response=entry['conversations'][1]['value']
    images=[]
    try:
        for path in entry['images']:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        batch=processor(text=prefix+response,images=images,return_tensors='pt',
                        crop_to_patches=False,min_patches=1,max_patches=1,truncation=False)
    finally:
        for image in images:
            image.close()
    response_ids=processor.tokenizer.encode(response,add_special_tokens=False)
    ids=batch['input_ids'][0]
    if ids[-len(response_ids):].tolist()!=response_ids:
        raise ValueError('Assistant boundary changes tokenization; inspect before applying loss.')
    encoded_digits=[processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(10)]
    if any(len(tokens)!=1 for tokens in encoded_digits):
        raise ValueError('Each decimal digit must be exactly one token.')
    digit_ids=[tokens[0] for tokens in encoded_digits]
    if len(set(digit_ids))!=10:raise ValueError('Digit tokens must be unique.')
    positions=[len(ids)-len(response_ids)+i for i,t in enumerate(response_ids) if t in digit_ids]
    if len(positions)!=4:raise ValueError('Expected exactly four response digits.')
    return batch,positions,digit_ids


def main():
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(ROOT/'outputs/internvl-metadata',local_files_only=True,trust_remote_code=False)
    tokenizer=processor.tokenizer
    digit_ids=[]
    for i in range(10):
        ids=tokenizer.encode(str(i),add_special_tokens=False);assert len(ids)==1;digit_ids.append(ids[0])
    # Exhaustively check the 10,000 legal response strings without model execution.
    for x in range(100):
        for y in range(100):
            response=f'[({x:02d}, {y:02d})]'
            ids=tokenizer.encode(response,add_special_tokens=False)
            found=[digit_ids.index(i) for i in ids if i in digit_ids]
            assert found==[x//10,x%10,y//10,y%10]
    data={v:json.loads((ROOT/f'outputs/vlm/{v}_train.json').read_text()) for v in ['d1','d4','r4']}
    samples=[0,next(i for i,r in enumerate(data['d4']) if len(r['images'])==4),
             max(range(len(data['d4'])),key=lambda i:len(data['d4'][i]['conversations'][0]['value']))]
    results=[]
    for i in sorted(set(samples)):
        token_counts={}
        for variant in ['d1','d4','r4']:
            entry=data[variant][i];batch,positions,legal=prepare(processor,entry)
            ids=batch['input_ids'][0]
            visual=int((ids==processor.image_token_id).sum())
            assert visual==256*len(entry['images'])
            assert batch['pixel_values'].shape[0]==len(entry['images'])
            token_counts[variant]=visual
            results.append({'row':i,'variant':variant,'images':len(entry['images']),
                            'input_tokens':len(ids),'visual_tokens':visual,'digit_positions':positions,
                            'causal_logit_positions':[p-1 for p in positions],
                            'pixel_values_shape':list(batch['pixel_values'].shape)})
        assert token_counts['d4']==token_counts['r4']
    # Synthetic full-vocabulary predictions verify loss, score and gradient masking.
    torch.manual_seed(4)
    logits=torch.randn(4,len(tokenizer),requires_grad=True)
    target=torch.tensor([digit_ids[d] for d in [3,2,6,7]])
    ll=constrained_log_mass(logits,target,digit_ids)
    expected=digit_log_mass(logits.detach().numpy()[:,digit_ids],[32,67])
    np.testing.assert_allclose(float(ll.detach()),expected,atol=1e-6)
    (-ll).backward();mask=torch.ones(len(tokenizer),dtype=torch.bool);mask[digit_ids]=False
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad[:,mask])==0
    result={'status':'CPU processor/tokenizer/loss checks passed; model forward/backward untested',
            'transformers':__version__,'digit_ids':digit_ids,'response_pairs_checked':10000,
            'processor_samples':results,'patch_policy':'One 448x448 patch per image; crop_to_patches=False',
            'synthetic_loss_score_error_nats':abs(float(ll.detach())-expected),
            'limitations':['No model weights loaded; no VLM likelihood or gradient computed.',
                           'Processor checks cover selected examples, not all multimodal batches.',
                           'Constrained loss helper must be wired to causal model logits, not default SFT loss.',
                           'New video LoRA initialization and GPU resource measurements remain pending.']}
    (ROOT/'outputs/vlm/interface-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
