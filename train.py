# This is the file to execute.

# This model is a transformer that uses self-attention mechanism to predict the next character in a sequence. Its unique goal is to 
# produce a new text that is similar to what it has been trained on.

import torch
import components
import tools
import pathlib
#import re
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters of the model

batch_size = 64 # independent data sequences number to process in parallel (forward and backward pass)
block_size = 8 # maximum number of tokens (characters) that the model can consider for a predicition: context length

max_iters = 2000+1 # maximum number of iterations for training
train_eval_steps = 10 # number of steps before printing the training state
learning_rate = 1e-5 # learning rate for the optimizer
training_iters = 1000 # number of iterations for training data batches

embedding_dimension = 128 # dimension of the vectors that represent the tokens -------- embedding_dimension = attention_head_number * attention_head_size
attention_head_number = 4 # number of attention heads used for the masked self-attention block
decoder_block_number = 1 # number of decoder blocks set for the model
dropout_rate = 0.1 # rate of neurons that will be deactivated during training

# Adam optimizer tuning
weight_decay = 0 # for L2 regularization of the optimizer
l1_coef = 1e-2 # for L1 regularization for the model training
#betas=(0.8, 0.999)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # the script will use the GPU if it is available

torch.manual_seed(1337) # to assure that randomization is reproducible

# Data generation
import pathlib, tools
vocabulary = ['a','b','c','d']
vocab_len = len(vocabulary)
pattern = 'ab' # pattern that the model will have to learn
file_path = pathlib.Path('./data/dataset.txt')
if file_path.is_file() is False:
    sequence_length = 5
    sequence_number = vocab_len**sequence_length # total number of sequences to generate
    #sequence_number = int(1e3) # number of sequences to generate
    tools.generate_dataset(vocabulary,sequence_number,sequence_length,pattern)

# Data gathering
with open('./data/dataset.txt','r',encoding='utf-8') as data: # opens the file that contains the text to be trained on and gets its content
    text = data.read()

characters = sorted(list(set(text))) # gets all the characters present in the text and sorts them (alphabetically)
vocabulary_size = len(characters) # gets the number of unique characters in the text

# Data encoding
data = torch.tensor(components.token_encoder_v1(characters,text), dtype=torch.long)

# Data splitting
training_percent = 0.8 # percentage of the data that will be used for training
limit = int(training_percent*len(data)) # calculates the index that will be used to split the data into training and validation sets
training_data = data[:limit] # gets the training data
test_data = data[limit:] # gets the test data
validation_data = None
# remaining_data = data[limit:]
# validation_percent = 0.5 # percentage of the remaining data that will be used for tests
# limit = int(validation_percent*len(remaining_data))
# validation_data = remaining_data[:limit] # gets the validation data
# test_data = remaining_data[limit:] # gets the test data

# Model creation
model = components.SLModel(vocabulary_size,embedding_dimension,block_size,attention_head_number,decoder_block_number,embedding_dimension,dropout_rate) # creates our model
model = model.to(device) # sends the model to the device (CPU or GPU)
parameters_number = sum(p.numel() for p in model.parameters()) # counts the number of parameters in the model

# Training phase
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # optimizer used for training

results = open("./output/logs.txt","w")
with open("./output/logs.txt","a") as results:
    if parameters_number > 1e6:
        results.write("Parameters number: {0:2f} M\n\n".format(parameters_number/1e6))
    else:
        results.write("Parameters number: {0:2d} \n\n".format(parameters_number))

    # Lists to store iteration numbers, scores and training losses
    iterations = np.array([])
    scores = np.array([])
    training_scores = np.array([])
    training_losses = np.array([])
    if validation_data is not None:
        val_losses = np.array([])
    losses_iterations = np.array([])
    patience = 3 # number of iterations before stopping the training if the model does not improve (early stopping)
    
    for iter in range(max_iters):
        # evaluate the loss once in a while
        print(iter+1)
        if iter % train_eval_steps == 0:
            losses = tools.estimate_loss(model,block_size,batch_size,training_iters,training_data,validation_data,device)

            # if validation_data is not None: # early stopping
            #     if losses['val'] < losses['train']:
            #         patience -= 1
            #         if patience == 0:
            #             break
            #     else:
            #         patience = 3

            losses_iterations = np.append(losses_iterations,iter)
            training_losses = np.append(training_losses,losses['train'])
            if validation_data is not None:
                val_losses = np.append(val_losses,losses['val'])

            if validation_data is None:
                results.write("Step: {0:2d}\tTraining loss: {1:2.2f}\n".format(iter,losses['train']))
            else:
                results.write("Step: {0:2d}\tTraining loss: {1:2.2f}\tValidation loss: {2:2.2f}\n".format(iter,losses['train'],losses['val']))

        # sample a batch of data
        xb,yb = tools.get_batch('train',block_size,batch_size,training_data,validation_data,device)
        
        logits, loss = model(xb,yb)

        # Add L1 regularization term
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        # Complete loss with regularization
        loss = loss + l1_coef * l1_reg

        # model's performance evaluation
        iterations = np.append(iterations,iter)
        score = tools.evaluate(model,test_data,characters,block_size,pattern,device)
        scores = np.append(scores,score)
        training_score = tools.evaluate(model,training_data,characters,block_size,pattern,device)
        training_scores = np.append(training_scores,training_score)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# Model saving
#torch.save(model.state_dict(),"./output/model.pth")

# Predictions of the model from the starting token
starting_token = 'a' # token from which the model will start generating the text
starting_index = torch.tensor([components.token_encoder_v1(characters,starting_token)], dtype=torch.long, device=device) # gets the index of the starting token

predictions = open("./output/predictions.txt","w")
with open("./output/predictions.txt","a") as predictions:
   predictions.write(components.token_decoder_v1(characters,model.generate(starting_index,block_size,max_new_tokens=500)[0].tolist()))

# Model evaluation on the test data
file_path = pathlib.Path('./output/perfs_history.txt')
if file_path.is_file():
    with open("./output/perfs_history.txt","a") as perfs:
        score = tools.evaluate(model,test_data,characters,block_size,pattern,device)
        perfs.write("{0:2d}\t\t{1:2d}\t\t{2:2d}\t\t{3:2d}\t\t{4:2d}\t\t{5:2d}\t\t\t{6:2.1f}\t\t{7:2.2f}\n".format(max_iters,decoder_block_number, \
        block_size,batch_size,embedding_dimension,attention_head_number,dropout_rate,score*100))
else:
    perfs = open("./output/perfs_history.txt","w")
    with open("./output/perfs_history.txt","a") as perfs:
        perfs.write("Iterations\tBlocks\t\tContext\t\tBatches\t\td_model\t\tatt_head\t\tDropout\t\tScores(%)\n")
        score = tools.evaluate(model,test_data,characters,block_size,pattern,device)
        perfs.write("{0:2d}\t\t{1:2d}\t\t{2:2d}\t\t{3:2d}\t\t{4:2d}\t\t{5:2d}\t\t\t{6:2.1f}\t\t{7:2.2f}\n".format(max_iters,decoder_block_number, \
        block_size,batch_size,embedding_dimension,attention_head_number,dropout_rate,score*100))

# Graph on the performances
# file_path = './output/performances.txt'


# with open(file_path, 'r') as file:
#     for line in file:
#         match = re.search(r"(\d+)\s+\d+\s+([\d.]+)", line)
#         if match:
#             iteration = int(match.group(1))
#             score = float(match.group(2))
#             iterations.append(iteration)
#             scores.append(score)

# for i in range(len(scores)):
#     scores[i] = scores[i]/100

#training_losses = (training_losses-training_losses.min())/(training_losses.max()-training_losses.min()) # normalizes the training losses
#val_losses = (val_losses-val_losses.min())/(val_losses.max()-val_losses.min()) # normalizes the validation losses

plt.figure()
plt.title('Model performances rates')
plt.plot(iterations, scores)
plt.plot(iterations, training_scores)
plt.xlabel('Iterations')
plt.ylabel('Measures')
plt.legend(['Test acc.','Train acc.'])
plt.savefig('./output/performance_plot.jpg', format='jpg', dpi=300)
plt.close()
#plt.show()

plt.figure(figsize=(12, 6))
plt.title('Model losses')
plt.semilogy(losses_iterations, training_losses)
if validation_data is not None:
    plt.semilogy(losses_iterations, val_losses)
plt.xlabel('Iterations')
plt.ylabel('Losses')
if validation_data is not None:
    plt.legend(['Training loss','Val. loss'])
else:
    plt.legend(['Training loss'])
plt.savefig('./output/loss_plot.jpg', format='jpg', dpi=300)
plt.close()