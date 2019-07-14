import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch import nn, optim, autograd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from metrics import Metrics
from dataset import MyDataset
from utils import *
from hparams import *


class PriorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(GRU_DIM, FC_DIM),
                                  nn.BatchNorm1d(FC_DIM),
                                  nn.Tanh(),
                                  nn.Linear(FC_DIM, FC_DIM),
                                  nn.BatchNorm1d(FC_DIM),
                                  nn.Tanh())
        self.mu_layer = nn.Linear(FC_DIM, FC_DIM)
        self.logvar_layer = nn.Linear(FC_DIM, FC_DIM)
        
    def forward(self, x):
        h = self.core(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
        
class RecognitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(3*GRU_DIM, FC_DIM),
                                  nn.BatchNorm1d(FC_DIM),
                                  nn.Tanh(),
                                  nn.Linear(FC_DIM, FC_DIM),
                                  nn.BatchNorm1d(FC_DIM),
                                  nn.Tanh())
        self.mu_layer = nn.Linear(FC_DIM, FC_DIM)
        self.logvar_layer = nn.Linear(FC_DIM, FC_DIM)
        
    def forward(self, x):
        h = self.core(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

class DialogWAE(nn.Module):
    def __init__(self, train_dir, test_dir, word2vec_dir,
                 batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.epoch = 1
        self.device = device
        
        self.data = MyDataset(train_dir)
        self.test_data = MyDataset(test_dir, self.data.vocab)
        
        self.test_data_loader = DataLoader(self.test_data, 1, shuffle=True,
                                           collate_fn = self.data.collector)
        
        self.data_loader = DataLoader(self.data, self.batch_size, shuffle=True, 
                                      collate_fn = self.data.collector)
        
        self.embedding = self.init_word2vec(word2vec_dir)
        self.metrics = Metrics(self.embedding.weight.data.numpy())
        
        self.UEnc = nn.GRU(EMBEDDING, GRU_DIM,
                           bidirectional=True, 
                           batch_first = True)
        
        self.CEnc = nn.GRU(GRU_DIM*2 + 2, GRU_DIM,
                           batch_first = True)
        
        self.Q = nn.Sequential(nn.Linear(FC_DIM, FC_DIM),
                               nn.BatchNorm1d(FC_DIM),
                               nn.ReLU(),
                               nn.Linear(FC_DIM, FC_DIM),
                               nn.BatchNorm1d(FC_DIM),
                               nn.ReLU(),
                               nn.Linear(FC_DIM, LATENT_DIM))
        
        self.G = nn.Sequential(nn.Linear(FC_DIM, FC_DIM),
                               nn.BatchNorm1d(FC_DIM),
                               nn.ReLU(),
                               nn.Linear(FC_DIM, FC_DIM),
                               nn.BatchNorm1d(FC_DIM),
                               nn.ReLU(),
                               nn.Linear(FC_DIM, LATENT_DIM))
        
        self.D = nn.Sequential(nn.Linear(FC_DIM + GRU_DIM, FC_DIM_D),
                               nn.BatchNorm1d(FC_DIM_D),
#                              nn.ReLU(),
                               nn.LeakyReLU(0.2),
                               nn.Linear(FC_DIM_D, FC_DIM_D),
                               nn.BatchNorm1d(FC_DIM_D),
                               nn.LeakyReLU(0.2),
#                              nn.ReLU(),
                               nn.Linear(FC_DIM_D, 1))
        self.PriNet = PriorNet()
        self.RecNet = RecognitionNet()
        self.Dec = nn.GRU(EMBEDDING, LATENT_DIM + GRU_DIM,
                          batch_first = True)
        self.dec_projector = nn.Linear(LATENT_DIM + GRU_DIM, VOCAB_SIZE)
        self.init_weights()
        nn.init.uniform_(self.dec_projector.weight, -0.1, 0.1)

        self.optimizer_gen = optim.RMSprop(list(self.Q.parameters()) \
                                           +list(self.G.parameters()) \
                                           +list(self.PriNet.parameters()) \
                                           +list(self.RecNet.parameters()), LR_GEN)
        
        self.optimizer_discriminator = optim.RMSprop(self.D.parameters(), LR_DISC)
        
        self.encoder = nn.ModuleList([self.embedding,
                                 self.UEnc,
                                 self.CEnc])
        self.decoder = nn.ModuleList([self.embedding,
                                 self.Dec,
                                 self.dec_projector])
        
        self.optimizer_reconstruction = optim.SGD(list(self.encoder.parameters()) \
                                                  +list(self.RecNet.parameters()) \
                                                  +list(self.Q.parameters()) \
                                                  +list(self.decoder.parameters()),
                                                  LR_RECON)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer_reconstruction,
                                                  10, 0.6)


        
    def init_weights(self):
        for layer in self.modules():
            if type(layer) == nn.Linear:
                nn.init.uniform_(layer.weight, -INIT_SCALE, INIT_SCALE)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRU:
                for w in layer.parameters(): 
                    if w.dim()>1:
                        nn.init.orthogonal_(w)
    
    def init_word2vec(self, word2vec_dir):
        if word2vec_dir:
            SIZE = 1.2 * 10**6
            found = 0
            word2vec_pretrained = torch.randn(VOCAB_SIZE,200)
            word2vec_pretrained[self.data.PAD] = torch.zeros(200)
            with open(word2vec_dir, 'r') as f:
                for entry in f:
                    entry = entry.strip().split(" ")
                    if entry[0] in self.data.vocab['word_to_num']:
                        idx = self.data.vocab['word_to_num'][entry[0]]
                        word2vec_pretrained[idx] = torch.tensor(list(map(float, entry[1:])))
                        found += 1
                    if found == VOCAB_SIZE:
                        break
                print("Done Initializing Word2Vec (found %d/%d)" %(found, VOCAB_SIZE))
            return nn.Embedding.from_pretrained(word2vec_pretrained, freeze=False)
        else:
            return nn.Embedding(VOCAB_SIZE, 200, padding_idx = self.data.PAD)

    def forward(self, X, Xlen):
        c = self.encode_c(X, Xlen)

        eps_prior = self.sample_prior(c)
        z_prior = self.G(eps_prior)
        
        init_state = torch.unsqueeze(torch.cat((z_prior, c), 1), 0)
            
        return self.decode(None, init_state)
        
        
    
    def discriminator_loss(self, z_prior, z_posterior, c):
        prior_input = torch.cat((z_prior, c), -1)
        posterior_input = torch.cat((z_posterior, c), -1)
        
        d_prior = self.D(prior_input)
        d_posterior = self.D(posterior_input)
        
        loss = torch.mean(d_posterior) - torch.mean(d_prior)
        return loss
    
    def gradient_penalty(self, z_prior, z_posterior, c):
        batch_size = c.size(0)
        alpha = torch.rand((batch_size, 1)).to(self.device)
        
        alpha = alpha.expand(z_prior.size())
        interpolates = alpha * z_prior.data + (1-alpha) * z_posterior.data
        interpolates.requires_grad = True
        
        d_interpolates = torch.mean(self.D(torch.cat((interpolates, c), -1)))
        
        grad_outputs = torch.FloatTensor([1]).to(self.device)
        grad = autograd.grad(d_interpolates, interpolates,
                            grad_outputs = grad_outputs,
                            only_inputs = True, create_graph = True,
                            retain_graph=True)[0]
        grad_norm = torch.norm(grad, p = 2, dim = 1)
        
        penalty = (grad_norm - 1) ** 2
        return torch.mean(penalty)
    
    def reconstruction_loss(self, pred, target):
        target = target.contiguous().view(-1)
        mask = [idx for idx, val in enumerate(target) if val != self.data.PAD]
        mask = torch.tensor(mask).to(self.device)
        pred = F.log_softmax(pred, -1).view(-1, VOCAB_SIZE)
        pred = torch.index_select(pred, 0, mask)
        target = torch.index_select(target, 0, mask)
        loss = F.nll_loss(pred, target)
        return loss
    
    def combine_context(self, X, lengths, X_raw):
        batch_size = len(lengths)
        context_window = max(lengths)
        LISTENER_VECTOR = [1,0]
        SPEAKER_VECTOR = [0,1]
        speaker_first = [SPEAKER_VECTOR if i%2==0 else LISTENER_VECTOR
                         for i in range(context_window)]
        listener_first = [LISTENER_VECTOR if i%2==0 else SPEAKER_VECTOR
                          for i in range(context_window)]
        floors = []        
        new_X = list()
        offset = 0
        for length in lengths:
            segment = X[offset:offset+length]
            floor = speaker_first if length%2==0 else listener_first
            if X_raw[offset][1] == self.data.SOD:
                floor[0] = LISTENER_VECTOR
            floors.append(floor)
            segment_len, dim = segment.shape
            segment_padded = torch.cat((segment,
                                      torch.randn(context_window - segment_len, dim)\
                                        .to(self.device)))
            new_X.append(segment_padded)
            offset += length
        X = torch.stack(new_X, 0)
        floors = torch.tensor(floors, dtype=torch.float).to(self.device)
        
        return torch.cat((X, floors), 2)
        
    def encode_x(self, Y, Ylen):
        Ylen_sorted, ids = torch.sort(Ylen, descending=True)
        _, ids_reverse = torch.sort(ids, descending=False)
        Y_sorted = torch.index_select(Y, 0, ids)
        Y_sorted = Y_sorted[:,1:]
        Ylen_sorted -= 1
        Y_embed = self.embedding(Y_sorted)
        #Dropout
        Y_embed = F.dropout(Y_embed, p=0.5, training=self.encoder.training)
        Y_packed = pack_padded_sequence(Y_embed,
                                       Ylen_sorted,
                                       batch_first = True)
        
        _, res = self.UEnc(Y_packed)
        Y_encoded = torch.cat([res[0], res[1]], dim=1)
        Y_ordered = torch.index_select(Y_encoded, 0, ids_reverse)
        return Y_ordered
        
    def encode_c(self, X, Xlen):
        original_size = X.size()
        batch_size = original_size[0]
        
        real_utts = torch.tensor([i for i,x in enumerate(Xlen) if x > 0]).to(self.device)
        X_real = torch.index_select(X.view(-1, original_size[-1]),
                                   0,
                                   real_utts)
        Xlen_real = torch.index_select(Xlen, 0, real_utts)

        Xlen_sorted, ids = torch.sort(Xlen_real, descending=True)
        _, ids_reverse = torch.sort(ids, descending=False)
        X_sorted = torch.index_select(X_real, 0, ids)
        X_sorted = X_sorted[:,1:]
        Xlen_sorted -= 1
        
        X_embed = self.embedding(X_sorted)
        #Dropout
        X_embed = F.dropout(X_embed, p=0.5, training=self.encoder.training)
        X_packed = pack_padded_sequence(X_embed,
                                        Xlen_sorted,
                                        batch_first=True)
        _, res = self.UEnc(X_packed)
        X_encoded = torch.cat([res[0], res[1]], dim=1)
        X_ordered = torch.index_select(X_encoded, 0, ids_reverse)

        context_lengths = torch.sum(Xlen.view(original_size[:-1])>0, 1)
        X_contexts = self.combine_context(X_ordered, context_lengths, X_real) 
        
        #Dropout
        X_contexts = F.dropout(X_contexts, p=0.25, training=self.encoder.training)


        Clen_sorted, ids = torch.sort(context_lengths, descending=True)
        _, ids_reverse = torch.sort(ids, descending=False)
        C_sorted = torch.index_select(X_contexts, 0, ids)


        C_packed = pack_padded_sequence(C_sorted,
                                        Clen_sorted, 
                                        batch_first = True)
        _, C_encoded = self.CEnc(C_packed)
        
        C_ordered = torch.index_select(torch.squeeze(C_encoded, 0),
                                      0,
                                      ids_reverse)
        return C_ordered
    
    def sample_prior(self, c):
        batch_size = c.size(0)
        mu, logvar = self.PriNet(c)
        stddev = torch.exp(0.5*logvar)
        noise = torch.randn((batch_size, LATENT_DIM)).to(self.device)
        return stddev * noise + mu
        
    def sample_posterior(self, x, c):
        batch_size = c.size(0)
        xc = torch.cat((x,c), 1)
        mu, logvar = self.RecNet(xc)
        stddev = torch.exp(0.5*logvar)
        noise = torch.randn((batch_size, LATENT_DIM)).to(self.device)
        return stddev * noise + mu
    
    def decode(self, decoder_input, init_state):
        if decoder_input:
            res, _ = self.Dec(decoder_input, init_state)
            return res
        else:
            # Dynamic decoding
            batch_size = init_state.size(1)
            decoder_input = torch.full((batch_size, 1), self.data.SOS,
                                      dtype=torch.long).to(self.device)
            h = init_state
            decoder_output = list()
            decoder_output_lengths = torch.zeros(batch_size).to(self.device)
            for i in range(MAX_UTT):
                decoder_input = self.embedding(decoder_input)
                out, h = self.Dec(decoder_input, h)
                out = self.dec_projector(out)
                
                pred = torch.argmax(out, -1)
                decoder_output.append(out)
                
                ended = pred == self.data.EOS
                running = decoder_output_lengths == 0
                new_ended = ended.view(-1) * running
                ids = [idx for idx, v in enumerate(new_ended) if v > 0]
                decoder_output_lengths[ids] = i + 1
                decoder_input = pred
                
            not_ended = decoder_output_lengths == 0
            ids = [idx for idx, v in enumerate(not_ended) if v > 0]
            decoder_output_lengths[ids] = MAX_UTT
            
            decoder_output = torch.cat(decoder_output, 1)
            return decoder_output, decoder_output_lengths
            
    
    def disc_step(self, batch):
        self.optimizer_discriminator.zero_grad()
        X, Xlen, Y, Ylen = batch
        X = X.to(self.device)
        Xlen = Xlen.to(self.device)
        Y = Y.to(self.device)
        Ylen = Ylen.to(self.device)
        
        c = self.encode_c(X, Xlen)
        x = self.encode_x(Y, Ylen)
        
        eps_prior = self.sample_prior(c)
        eps_posterior = self.sample_posterior(x, c)

        z_prior = self.G(eps_prior)
        z_posterior = self.Q(eps_posterior)
        
        disc_loss = self.discriminator_loss(z_prior.detach(), z_posterior.detach(), c.detach())
#         disc_loss.backward()
        grad_penalty = self.gradient_penalty(z_prior, z_posterior, c.detach())
        loss = disc_loss + LAMBDA_D * grad_penalty
        
        loss.backward()
        self.optimizer_discriminator.step()
        return loss, disc_loss
    
    def fit(self, epochs, test_every):
        loss_recon_train = []
        loss_disc_train = []
        loss_disc_with_grad_train = []
        loss_gen_train = []
        bleus0, bleus1 = [], []
        for _ in range(epochs):
            self.train()
            loss_recon_epoch = []
            data = iter(self.data_loader)
            batch = next(data, None)
            while batch:
                #setup
                X, Xlen, Y, Ylen = batch
                X = X.to(self.device)
                Xlen = Xlen.to(self.device)
                Y = Y.to(self.device)
                Ylen = Ylen.to(self.device)
                
                #optimize reconstruction
                self.optimizer_reconstruction.zero_grad()
                self.encoder.train()
                self.decoder.train()

                c = self.encode_c(X, Xlen)
                x = self.encode_x(Y, Ylen)
                
                eps_posterior = self.sample_posterior(x, c)
                z_posterior = self.Q(eps_posterior)
                
                init_state = torch.unsqueeze(torch.cat((z_posterior, c), 1), 0)
                
                targets = Y[:,:-1]
                targetslen = Ylen - 1
                targetslen_sorted, ids = torch.sort(targetslen, descending=True)
                _, ids_reverse = torch.sort(ids, descending=False)
                targets_sorted = torch.index_select(targets, 0, ids)
                targets_embed = self.embedding(targets_sorted)
                #Dropout
                targets_embed = F.dropout(targets_embed, p=0.5,
                                          training=self.decoder.training)
                decoder_input = pack_padded_sequence(targets_embed,
                                                    targetslen_sorted,
                                                    batch_first = True)
                
                pred_packed = self.decode(decoder_input, init_state)
                pred, _ = pad_packed_sequence(pred_packed,
                                             batch_first=True)
                pred = torch.index_select(pred, 0, ids_reverse)
                pred = self.dec_projector(pred)
                reconstruction_loss = self.reconstruction_loss(pred, Y[:,1:])
                
                loss_recon_epoch.append(reconstruction_loss.item())
                reconstruction_loss.backward()
                clip_grad_norm_(list(self.encoder.parameters())
                                +list(self.decoder.parameters()), MAX_NORM)
                self.optimizer_reconstruction.step()
                #optimize generator
                self.encoder.eval()
                for p in self.D.parameters():
                    p.requires_grad = False
                self.optimizer_gen.zero_grad()

                c = self.encode_c(X, Xlen)
                x = self.encode_x(Y, Ylen)
                
                eps_prior = self.sample_prior(c.detach())
                eps_posterior = self.sample_posterior(x.detach(), c.detach())
                
                z_prior = self.G(eps_prior)
                z_posterior = self.Q(eps_posterior)
                
                gen_loss = -1 * self.discriminator_loss(z_prior, z_posterior, c.detach())
                loss_gen_train.append(gen_loss.item())
                gen_loss.backward()

                self.optimizer_gen.step()
                for p in self.D.parameters():
                    p.requires_grad = True

                #optimize discriminator
                self.encoder.eval()
                self.D.train()
                for _ in range(N_CRITIC):
                    loss_with_grad, loss = self.disc_step(batch)
                    batch = next(data, None)
                    if not batch:
                        break
                loss_disc_with_grad_train.append(loss_with_grad.item())
                loss_disc_train.append(loss.item())
                
            self.lr_decay.step()
                
            loss_recon_train.append(mean(loss_recon_epoch))
            print("\n[Epcoh %d] ----------Mean Loss: %f----------"%(self.epoch, loss_recon_train[-1]))
            if self.epoch % test_every == 0:
                bleu0, bleu1 = self.test()
                bleus0.append(bleu0)
                bleus1.append(bleu1)
                if len(bleus0) > 1:
                    plt.plot(bleus0, label='bleus0')
                    plt.plot(bleus1, label='bleus1')
                    plt.title(label='Test BLEU')
                    plt.legend()
                    plt.show()
            ckpt_name = 'checkpoints/model' + str(self.epoch) + '.pkl'
            self.epoch += 1
            torch.save(self, ckpt_name)
            copyfile(ckpt_name, 'checkpoints/model.pkl')
            
            print("*** [Train]")
            print("*** Context")
            for utt in X[0]:
                x = to_string(self.data.to_text(utt.tolist()))
                if len(x) > 0:
                    print(x)
            print("*** Response")
            x = torch.argmax(pred, -1)
            print('target:', to_string(self.data.to_text(targets[0,1:].tolist())))
            print('predic:', to_string(self.data.to_text(x[0].tolist())))
            if self.epoch > 2:
                plt.figure(figsize=(20,10))
                plt.subplot(221)
                plt.plot(loss_disc_train, label='train')
                plt.title(label='Discriminator')
                plt.legend()
                plt.subplot(222)
                plt.plot(loss_gen_train, label='train')
                plt.title(label='Generator')
                plt.legend()
                plt.subplot(223)
                plt.plot(loss_disc_with_grad_train, label='train')
                plt.title(label='Discriminator With Grad Penalty')
                plt.legend()
                plt.subplot(224)
                plt.plot(loss_recon_train, label='train')
                plt.title(label='Reconstruction')
                plt.legend()
                plt.show()
            
    def test(self):
        self.eval()
        self.encoder.eval()
        self.decoder.eval()
        bleu0 = []
        bleu1 = []
        for sample in iter(self.test_data_loader):
            x, xlen, y, ylen = sample
            x = x.to(self.device)
            xlen = xlen.to(self.device)
            pred, predlen = self(x, xlen)
            pred = torch.argmax(pred, -1)
            pred = self.data.to_text(pred[0].tolist())
            predlen = int(predlen.item())
            target = self.data.to_text(y[0].tolist())[1:]
            bleu = self.metrics.sim_bleu(pred, target)
            bleu0.append(bleu[0])
            bleu1.append(bleu[1])
#         print("*** [Test]")
#         print("*** Context")
#         for utt in x[0]:
#             x = to_string(self.data.to_text(utt.tolist()))
#             if len(x) > 0:
#                 print(x)
#         print("*** Response")
#         x = torch.argmax(pred, -1)
#         print('target:', to_string(self.data.to_text(target[0].tolist())))
#         print('predic:', to_string(self.data.to_text(x[0].tolist())))
        return mean(bleu0), mean(bleu1)
