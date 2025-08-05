# Conditional UNet for Polygon Colorization

This repository contains a PyTorch implementation of a conditional U-Net model built from scratch. The model is designed to perform a specific image-to-image translation task: taking a grayscale image of a polygon and a textual color prompt as input, it generates an RGB image of the polygon filled with the specified color.

## Key Features

- **U-Net from Scratch**: The entire model architecture is implemented from the ground up in PyTorch.
- **Conditional Generation**: The model is conditioned on color names using learnable embeddings.
- **FiLM Conditioning**: Utilizes Feature-wise Linear Modulation (FiLM) layers to inject color information effectively into the decoder.
- **Experiment Tracking**: Fully integrated with Weights & Biases for logging metrics, hyperparameters, and image samples.
- **Reproducible Notebook**: The entire workflow, from setup and training to inference, is contained in a single Google Colab notebook.

## Results & Inference

The model successfully learns to color various polygons based on the provided color names.

*<img width="462" height="716" alt="image" src="https://github.com/user-attachments/assets/a71896ef-e18f-4fca-97b8-1dc11e0332b1" />*


## Setup & Usage

This project is designed to be run in a Google Colab environment to ensure dependency management and GPU access.

### Prerequisites

- Python 3.x
- PyTorch
- Torchvision
- Weights & Biases (wandb)
- TorchMetrics

### Running the Project

The entire project is contained within a single `.ipynb` notebook.

1. **Upload Data**: Upload the `dataset.zip` file to your Google Colab session.

2. **Run the Setup Cell**: The first cell in the notebook handles the installation of all required libraries. It will automatically restart the Colab runtime after completion.

```bash
# This command installs compatible versions of torch and other libraries
!pip install -q torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q wandb torchmetrics sympy==1.11
```

3. **Run the Main Cell**: After the runtime restarts, run the main cell again. This will:
   - Unzip the dataset.
   - Define the `PolygonDataset` and `ConditionedUNet` classes.
   - Start the training process, logging everything to Weights & Biases.
   - Save the trained model weights as `conditioned_unet.pth`.
   - Run the inference section to display test results.

## Architecture Deep Dive

### U-Net Structure

The model is based on the classic U-Net architecture, which consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables precise localization.

- **Encoder**: Composed of 4 blocks. Each block contains two `Conv2d -> BatchNorm2d -> ReLU` layers followed by a `MaxPool2d` operation for downsampling.
- **Bottleneck**: A single block at the bottom of the "U" that connects the encoder and decoder.
- **Decoder**: Composed of 4 blocks. Each block starts with an up-sampling `ConvTranspose2d` layer, concatenates the result with the corresponding feature map from the encoder via a skip connection, and then passes it through two `Conv2d -> BatchNorm2d -> ReLU` layers.

### Conditioning with FiLM

The key to making the U-Net color-aware is the conditioning mechanism. We use Feature-wise Linear Modulation (FiLM).

1. **Color Embedding**: The input color name (e.g., "red") is converted into a learnable 32-dimensional vector using an `nn.Embedding` layer.
2. **Modulation**: This single color vector is fed into a small linear layer within each decoder block to generate two new vectors: a scaling factor `gamma` and a shifting factor `beta`.
3. **Application**: These `gamma` and `beta` vectors are then applied element-wise to the feature maps in the decoder, effectively "styling" the output with the correct color information.

This is more effective than simple concatenation as it allows the color information to dynamically control the entire feature map at each stage of the decoding process.

<details>
<summary><b>Click to see the ConditionedUNet PyTorch Code</b></summary>

```python
class FiLMLayer(nn.Module):
    def __init__(self, channels, embedding_dim):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, channels * 2)
    
    def forward(self, x, embedding):
        gamma_beta = self.layer(embedding).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma * x + beta

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        return self.conv_block(x)

class ConditionedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, channels=(64, 128, 256, 512), num_colors=10, embedding_dim=32):
        super().__init__()
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.encoders = nn.ModuleList()
        for i in range(len(channels)):
            self.encoders.append(UNetBlock(in_channels if i == 0 else channels[i-1], channels[i]))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(channels[-1], channels[-1] * 2)
        self.decoders, self.film_layers = nn.ModuleList(), nn.ModuleList()
        rev_channels = channels[::-1]
        for i in range(len(rev_channels)):
            self.decoders.append(nn.ConvTranspose2d(rev_channels[i] * 2, rev_channels[i], 2, 2))
            self.decoders.append(UNetBlock(rev_channels[i] * 2, rev_channels[i]))
            self.film_layers.append(FiLMLayer(rev_channels[i], embedding_dim))
        self.final_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x, color_idx):
        embedding = self.color_embedding(color_idx)
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[i+1](concat_x)
            x = self.film_layers[i//2](x, embedding)
        return torch.sigmoid(self.final_conv(x))
```

</details>

## Experiment Tracking & Insights

### Final Metrics

The model was trained for 100 epochs, achieving the following performance on the validation set:

- **Final Validation Loss**: ~0.0013
- **Final Validation SSIM**: ~0.996
- **Final Validation PSNR**: ~45.4 dB

### Training Curves

*<img width="1765" height="766" alt="image" src="https://github.com/user-attachments/assets/f0094dbc-e26c-4cc4-830c-519e871d5257" />*


### Key Learnings

- **FiLM Efficacy**: The FiLM conditioning mechanism was highly effective, allowing the model to quickly learn the association between abstract color embeddings and their corresponding RGB values.
- **Loss Function**: L1Loss proved to be a good choice, resulting in sharp boundaries and avoiding the blurriness often associated with MSELoss in image generation tasks.
- **Failure Modes**: The primary failure mode observed was a slight rounding of very sharp corners on complex polygons like stars. This could likely be mitigated by training on a larger, more diverse dataset of synthetic shapes or by using a higher input resolution.
