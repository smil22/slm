import torch
import components

def get_batch(split_type,block_size,batch_size,training_data,validation_data=None,device='cpu'):
    """This function aims to split training or validation data in batches according to block size and 
    batch size."""
    
    if validation_data is None:
        data = training_data
    else:
        data = training_data if split_type == 'train' else validation_data
    starting_index = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starting_index])
    y = torch.stack([data[i+1:i+block_size+1] for i in starting_index])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad() # this decorator assures that the function will not be able to change the model's parameters
def estimate_loss(model,block_size,batch_size,training_iters,training_data,validation_data=None,device='cpu'):
    """This function trains and evaluates the model in order to estimate the loss on a few batches of data."""

    out = {}
    model.eval()
    if validation_data is None:
        losses = torch.zeros(training_iters)
        for k in range(training_iters):
            X,Y = get_batch('train',block_size,batch_size,training_data,validation_data,device)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out['train'] = losses.mean()
        model.train()
    else:
        for split_type in ['train', 'val']:
            losses = torch.zeros(training_iters)
            for k in range(training_iters):
                X,Y = get_batch(split_type,block_size,batch_size,training_data,validation_data,device)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split_type] = losses.mean()
        model.train()
    return out

def generate_dataset(vocabulary,sequence_number,sequence_length,pattern):
    """This function generates a dataset of sequences of characters that contains a given pattern."""

    data = open("./data/dataset.txt","w")
    vocab_len = len(vocabulary)
    indices = torch.arange(vocab_len)
    with open("./data/dataset.txt","a") as data:
        for i in range(sequence_number):
            sequence = ""
            while True:
                sequence_current_length = len(sequence)
                if sequence_current_length == sequence_length:
                    break
                choice = torch.randint(0, vocab_len, (1,))
                token = vocabulary[choice]
                if token == pattern[0] and sequence_length-sequence_current_length < len(pattern): # not incomplete pattern at the end
                    pattern_free = indices[indices != choice]
                    choice = pattern_free[torch.randint(0, vocab_len-1, (1,))]
                    token = vocabulary[choice]
                sequence += token
    
                if token == pattern[0] and sequence_length-sequence_current_length >= len(pattern): # pattern insertion in the sequence
                    sequence += pattern[1:]
            sequence += '\n'
            data.write(sequence)

def evaluate(model,test_data,vocabulary,block_size,pattern,device):
    """"This function uses the test data to evaluate the model's ability to learn properly the pattern inserted in the training data."""

    pattern_number = 0
    model_success = 0
    n = len(test_data)
    pattern_length = len(pattern)
    encoded_pattern = components.token_encoder_v1(vocabulary,pattern) # encodes the pattern
    for i in range(n):
        #print(test_data[i:i+pattern_length].tolist(),encoded_pattern[0].tolist())
        if test_data[i:i+pattern_length].tolist() == encoded_pattern:
            pattern_number += 1
            # get the model's prediction
            x = torch.tensor([[encoded_pattern[0]]], dtype=torch.long, device=device) # changes the starting token in a pytorch tensor
            y = model.generate(x,block_size,max_new_tokens=pattern_length-1)[0].tolist()
            model_answer = components.token_decoder_v1(vocabulary,y)
            if model_answer == pattern:
                model_success += 1
    score = model_success/pattern_number
    return score
