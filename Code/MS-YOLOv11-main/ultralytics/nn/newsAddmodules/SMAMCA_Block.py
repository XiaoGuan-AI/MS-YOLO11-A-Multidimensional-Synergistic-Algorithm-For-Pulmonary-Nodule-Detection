
import torch
import torch.nn as nn
from MCA import C3k2_MCA
from SMABlock import SMAFormerBlock


class SMAMCA_Block(nn.Module):
    def __init__(self, channels, heads=8, dropout=0.1, forward_expansion=2):
        super(SMAMCA_Block, self).__init__()
        self.mca = C3k2_MCA(channels, channels)
        self.sma = SMAFormerBlock(channels, heads, dropout, forward_expansion)

        #Sewing method: Fusion suture (included)
        self.conv_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        # Extract features through the MCA module
        mca_out = self.mca(x)

        # Extract features through SMAFormerBlock module
        sma_out = self.sma(x)

        # Feature Fusion (Included)
        fused = torch.cat([mca_out, sma_out], dim=1)
        fused = self.conv_fuse(fused)
        fused = self.norm(fused)
        output = self.activation(fused)

        return output


# Test module
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)
    model = SMAMCA_Block(64)
    output_tensor = model(input_tensor)
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")
