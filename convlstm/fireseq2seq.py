import torch
from torch import nn
from typing import Tuple

class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int], 
        padding: Tuple[int, int], 
        frame_size: Tuple[int, int], 
        bias: bool,
        device="cuda"
    ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        out_channels: int
            Number of channels of the output/hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        padding: (int, int)
            Padding of the convolutional kernel.
        frame_size: (int, int)
            Size of the input frame (h x w).
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.device = device
        
        # does all convolutional operations involving x_t and h_{t-1} (i.e,. Wx * x + Wh * h)
        # there are 4 of these operations, for the i(nput) gate, f(orget) gate, o(utput) gate, and c(ell) state,
        # which is why the number of out channels is 4 * self.out_channels
        # rather than individually doing 8 separate convolutions and having to learn 8 sets of weights
        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels * 4,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        ).to(device)
        
        # Weights for hadamard product operations with c_t
        base_tensor = torch.Tensor(self.out_channels, *frame_size).to(device)
        # for input gate
        self.Wci = nn.Parameter(base_tensor)
        # for forget gate
        self.Wcf = nn.Parameter(base_tensor)
        # for output gate
        self.Wco = nn.Parameter(base_tensor)
        
        
    def forward(
        self, 
        h_prev: torch.Tensor, 
        c_prev: torch.Tensor, 
        X: torch.Tensor
    ):
        """
        Forward pass through ConvLSTMCell
        
        Parameters
        ----------
        h_prev: torch.Tensor
            The hidden state of the previous cell.
        c_prev: torch.Tensor
            The cell state of the previous cell.
        x: torch.Tensor
            The input to the cell - the current timestep.
        """
        
        h_prev = h_prev.to(self.device)
        c_prev = c_prev.to(self.device)
        X = X.to(self.device)
                
        # X shape: [batch, in_channels, height, width]
        # h_prev shape: [batch, out_channels, height, width]
        conv_out = self.conv(torch.cat([X, h_prev], dim=1))
        
        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_out, chunks=4, dim=1)
        # input gate, i_conv = Wxi * X + Whi * h_prev
        i = torch.sigmoid(i_conv + self.Wci * c_prev)
        # forget gate, f_conv = Wxf * X + Whf * h_prev
        f = torch.sigmoid(f_conv + self.Wcf * c_prev)
        # cell state, c_conv = Wxc * X + Whc * h_prev
        c = f * c_prev + i * torch.tanh(c_conv)
        # output gate, o_conv = Wxo * X + Who * h_prev
        o = torch.sigmoid(o_conv + self.Wco * c)
        # hidden state - passed to the next cell
        h = o * torch.tanh(c)
        # return hidden state and cell state to be passed to the next cell
        return h, c

class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int], 
        padding: Tuple[int, int], 
        frame_size: Tuple[int, int], 
        bias: bool,
        device="cuda"
    ):
        super(ConvLSTM, self).__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_channels = out_channels
        
        self.cell = ConvLSTMCell(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            frame_size=frame_size,
            bias=bias
        ).to(self.device)
    
    def forward(self, X):
        """
        Makes a forward pass through each timestep of the input sequence
        
        Parameters
        ----------
        X: torch.Tensor
            The input batch of size (batch_size, channels, seq_length, height, width)
        """
        batch_size, _, seq_length, height, width = X.size()
                
        # initialize output
        o = torch.zeros(
            batch_size, 
            self.out_channels, 
            seq_length, # only include seq_length dim for output
            height, 
            width, 
            device=self.device
        )
        
        # initialize hidden state
        h = torch.zeros(
            batch_size, 
            self.out_channels, 
            height, 
            width, 
            device=self.device
        )
        
        # initialize cell state
        c = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width,
            device=self.device
        )
        
        for t in range(seq_length):
            # X[:,:,t] => the batch of 2d grids for the particular timestep of x
            h, c = self.cell(h, c, X[:,:,t])
            o[:,:,t] = h
        
        return o


class FireSeq2Seq(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frame_size: Tuple[int, int],
        bias: bool,
        num_kernels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
        num_layers=2,
        activation="tanh",
        device="cuda",
    ):
        super(FireSeq2Seq, self).__init__()
        
        self.device = device
        
        self.sequential = nn.Sequential().to(device)
        
        self.sequential.add_module(
            "ConvLSTM_1",
            ConvLSTM(
                in_channels=in_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                frame_size=frame_size,
                bias=bias,
                device=device
            )
        )
        
        self.sequential.add_module(
            "BatchNorm_1",
            nn.BatchNorm3d(num_features=num_kernels).to(device)
        )
        
        for i in range(2, num_layers + 1):
            self.sequential.add_module(
                f"ConvLSTM_{i}",
                ConvLSTM(
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    frame_size=frame_size,
                    bias=bias,
                    device=self.device
                )
            )
            
            self.sequential.add_module(
                f"BatchNorm_{i}",
                nn.BatchNorm3d(num_features=num_kernels).to(device)
            )
        
        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=padding
        ).to(device)
        
    
    def forward(self, X):
        X = X.to(self.device)
        output = self.sequential(X)
        _, _, seq_length, _, _ = output.size()
        pred = torch.stack([
            self.conv(output[:, :, time_step])
            for time_step in range(seq_length)
        ], dim=2)
        return pred