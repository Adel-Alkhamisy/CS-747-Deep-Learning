import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.init as init
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type="rnn", n_layers=1):
        super(RNN, self).__init__()
        """
        Initialize the RNN model.
        
        You should create:
        - An Embedding object which will learn a mapping from tensors
        of dimension input_size to embedding of dimension hidden_size.
        - Your RNN network which takes the embedding as input (use models
        in torch.nn). This network should have input size hidden_size and
        output size hidden_size.
        - A linear layer of dimension hidden_size x output_size which
        will predict output scores.

        Inputs:
        - input_size: Dimension of individual element in input sequence to model
        - hidden_size: Hidden layer dimension of RNN model
        - output_size: Dimension of individual element in output sequence from model
        - model_type: RNN network type can be "rnn" (for basic rnn), "gru", or "lstm"
        - n_layers: number of layers in your RNN network
        """
        
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.encoder = nn.Embedding(input_size, hidden_size)
        if model_type == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        elif model_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        else:
            raise ValueError("Invalid model_type provided")

        self.decoder = nn.Linear(hidden_size, output_size)
        #Apply Xavier initialization to the weights of the RNN layers
        for i in range(n_layers):
            weight_ih = getattr(self.rnn, f'weight_ih_l{i}')
            weight_hh = getattr(self.rnn, f'weight_hh_l{i}')
            init.xavier_uniform_(weight_ih)
            init.xavier_uniform_(weight_hh)
        self.decoder = nn.utils.spectral_norm(nn.Linear(hidden_size, output_size))
        #########       END      ##########
        


    def forward(self, input, hidden):
        """
        Forward pass through RNN model. Use your Embedding object to create 
        an embedded input to your RNN network. You should then use the 
        linear layer to get an output of self.output_size. 

        Inputs:
        - input: the input data tensor to your model of dimension (batch_size)
        - hidden: the hidden state tensor of dimension (n_layers x batch_size x hidden_size) 

        Returns:
        - output: the output of your linear layer
        - hidden: the output of the RNN network before your linear layer (hidden state)
        """
        
        # output = None
        # hidden = None
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        batch_size = input.size(0)
        #print("batch_size=", batch_size)
        encoded = self.encoder(input)
        #print("encoded=",encoded)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        ##########       END      ##########
        return output, hidden

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden states to all 0s during training.
        
        Hidden states should be initilized to dimension (n_layers x batch_size x hidden_size) 

        Inputs:
        - batch_size: batch size

        Returns:
        - hidden: initialized hidden values for input to forward function
        """
        
        hidden = None
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        noise_std=0.01
        if self.model_type == 'lstm':
            h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            
            # Add Gaussian noise
            h_noise = torch.randn_like(h) * noise_std
            c_noise = torch.randn_like(c) * noise_std
            
            hidden = (h + h_noise, c + c_noise)
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            # Add Gaussian noise
            noise = torch.randn_like(hidden) * noise_std
            hidden = hidden + noise
        ##########       END      ##########

        return hidden

class AWD_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.5):
        super(AWD_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.lstm(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size, device=None):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))



