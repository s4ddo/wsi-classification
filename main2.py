import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
import math 
from h5_dataset import H5FeatureBagDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def get_vram_gb():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved

class PolarMapping(nn.Module):
    """
    Converts standard Cartesian feature vectors into Polar (Hyperspherical) coordinates.
    Returns the magnitude (radius) and the unit vector (representing the angles).
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # 1. Calculate Radius (Magnitude: How strong the core data is)
        radius = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # 2. Calculate Direction (Angles: Mapped to a fixed predictable circular grid)
        direction = x / (radius + self.eps)
        
        return radius, direction

class PolarMLA(nn.Module):
    # Added `dropout=0.1` to the arguments
    def __init__(self, dim, num_heads, latent_dim, dropout=0.1): 
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False) 
        self.kv_down = nn.Linear(dim, latent_dim, bias=False)
        self.kv_up = nn.Linear(latent_dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.polar_map = PolarMapping()

        # Save the probability as a float for the SDPA function
        self.attn_dropout_p = dropout 
        # Standard dropout layer for the final projection
        self.proj_drop = nn.Dropout(dropout) 

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_latent = self.kv_down(x)
        kv = self.kv_up(kv_latent)
        k, v = kv.chunk(2, dim=-1)
        
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Polar Mapping
        _, q_direction = self.polar_map(q)
        _, k_direction = self.polar_map(k)
        
        # USE NATIVE SDPA
        out = F.scaled_dot_product_attention(
            q_direction, 
            k_direction, 
            v, 
            dropout_p=self.attn_dropout_p if self.training else 0.0, # Apply dropout here!
            scale=10.0 
        )
        
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Apply the projection dropout before returning
        return self.proj_drop(self.out_proj(out))

# class PolarMLA(nn.Module):
#     def __init__(self, dim, num_heads, latent_dim):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
        
#         self.q_proj = nn.Linear(dim, dim, bias=False) # Bias removed for cleaner polar mapping
#         self.kv_down = nn.Linear(dim, latent_dim, bias=False)
#         self.kv_up = nn.Linear(latent_dim, dim * 2, bias=False)
#         self.out_proj = nn.Linear(dim, dim)
        
#         self.polar_map = PolarMapping()

#     def forward(self, x):
#             B, N, C = x.shape
            
#             q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
#             kv_latent = self.kv_down(x)
#             kv = self.kv_up(kv_latent)
#             k, v = kv.chunk(2, dim=-1)
            
#             k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#             v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

#             # Apply Polar Mapping
#             _, q_direction = self.polar_map(q)
#             _, k_direction = self.polar_map(k)
            
#             # USE NATIVE SDPA INSTEAD OF CUSTOM MATMUL
#             # PyTorch 2.0+ will automatically use FlashAttention here, saving massive VRAM.
#             # We simulate your * 10.0 temp scaling by modifying the scale parameter.
            
#             # Standard scale is 1 / sqrt(d). You wanted a fixed 10.0
#             # Wait until SDPA supports custom scale (PyTorch 2.1+), otherwise just use it raw.
#             out = F.scaled_dot_product_attention(
#                 q_direction, 
#                 k_direction, 
#                 v, 
#                 scale=10.0 # Only works in PyTorch 2.1+
#             )
            
#             out = out.transpose(1, 2).reshape(B, N, C)
#             return self.out_proj(out)

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x): 
        return self.net(x)

# Mixture of experts block 
class DeepSeekMoE(nn.Module):
    def __init__(self, dim, num_shared, num_routed, top_k, hidden_dim):
        super().__init__()
        self.top_k = top_k
        self.shared_experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_shared)])
        self.routed_experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_routed)])
        self.router = nn.Linear(dim, num_routed, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        route_logits = self.router(x_flat)
        route_probs = F.softmax(route_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(route_probs, self.top_k, dim=-1)
        
        routed_out = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.routed_experts):
            mask = (topk_indices == i)
            if mask.any():
                idx_tokens, idx_k = torch.where(mask)
                expert_in = x_flat[idx_tokens]
                expert_out = expert(expert_in) * topk_probs[idx_tokens, idx_k].unsqueeze(-1)
                routed_out.index_add_(0, idx_tokens, expert_out)
                
        out = shared_out + routed_out
        return out.view(B, N, C)

class PolarViTBlock(nn.Module):
    # Added dropout=0.3 as a default argument at the end
    def __init__(self, dim, num_heads, latent_dim, num_shared, num_routed, top_k, hidden_dim, dropout=0.3):
        super().__init__()
        
        # Pass the dropout value down into your PolarMLA layer
        self.attn = PolarMLA(dim, num_heads, latent_dim, dropout=dropout)
        
        # Initialize the MoE
        self.moe = DeepSeekMoE(dim, num_shared, num_routed, top_k, hidden_dim)
        
        # Initialize the dropout layer for the residual connections
        self.drop_path = nn.Dropout(dropout)

    def forward(self, x):
        # Apply dropout to the outputs of attention and MoE before adding to the residual stream
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.moe(x))
        return x

class SpatialEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Assuming dim is even, we split it for X and Y
        self.half_dim = dim // 2
        
    def forward(self, coords):
        # coords: [B, N, 2]
        x_coords = coords[:, :, 0:1]
        y_coords = coords[:, :, 1:2]
        
        # Create frequency bands
        div_term = torch.exp(torch.arange(0, self.half_dim, 2, device=coords.device).float() * (-math.log(10000.0) / self.half_dim))
        
        # Calculate sine/cosine for X
        pos_x_sin = torch.sin(x_coords * div_term)
        pos_x_cos = torch.cos(x_coords * div_term)
        pos_x = torch.cat([pos_x_sin, pos_x_cos], dim=-1)
        
        # Calculate sine/cosine for Y
        pos_y_sin = torch.sin(y_coords * div_term)
        pos_y_cos = torch.cos(y_coords * div_term)
        pos_y = torch.cat([pos_y_sin, pos_y_cos], dim=-1)
        
        # Concatenate X and Y encodings
        return torch.cat([pos_x, pos_y], dim=-1)



class DeepSeekSpatialViT(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2, dim=128, depth=4, 
                 num_heads=4, latent_dim=64, num_shared=1, num_routed=4, top_k=2):
        super().__init__()
        
        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim) # Dynamic positional embedding generator

        self.attention_pool = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

        self.blocks = nn.ModuleList([
            PolarViTBlock(
                dim=dim, 
                num_heads=num_heads, 
                latent_dim=latent_dim, 
                num_shared=num_shared, 
                num_routed=num_routed, 
                top_k=top_k, 
                hidden_dim=dim * 2, 
                dropout=0.3 # <--- Explicitly pass the dropout here!
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, coords):
        x = self.feature_proj(x)
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens 
        
        for block in self.blocks:
            # Checkpoint each block instead of running it normally
            x = checkpoint(block, x, use_reentrant=False)
        x = self.norm(x) # Shape: [B, N, dim]

        # Attention Pooling over the N sequence length
        A = self.attention_pool(x) # [B, N, 1]
        A = F.softmax(A, dim=1) # [B, N, 1]
        M = torch.sum(A * x, dim=1) # [B, dim]

        return self.head(M)

# 3. Training & Evaluation Loops
def train(trainloader, input_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Initialize the spatially-aware MIL model
    # model = DeepSeekSpatialViT(input_dim=input_dim, num_classes=num_classes).to(device)
    model = DeepSeekSpatialViT(
        input_dim=1280, 
        num_classes=num_classes, 
        dim=1280,          # Increased width
        depth=6,          # Increased depth
        num_heads=20,      # Increased parallel attention
        latent_dim=256,   # Scaled latent bottleneck
        num_shared=1, 
        num_routed=4, 
        top_k=2
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) # CHANGED FROM 1E-4

    scaler = torch.cuda.amp.GradScaler()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        accumulation_steps = 16 
        optimizer.zero_grad()
        
        for i, batch in enumerate(trainloader):
            # Extract features, coordinates, and labels
            inputs = batch["input"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["label"].to(device)

            
        # Cast operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs, coords)
                loss = criterion(outputs, labels)
                scaled_loss = loss / accumulation_steps
                
            # Scale the loss and call backward
            scaler.scale(scaled_loss).backward()
            # adding gradient accumulation
            # Only step the optimizer every `accumulation_steps`
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            
            # Print stats every 10 steps (since batch size is 1)
            if i % 10 == 9:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 10:.4f}")
                alloc_gb, res_gb = get_vram_gb()
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}]")
                print(f"Loss: {running_loss / 10:.4f}")
                print(f"VRAM Allocated: {alloc_gb:.2f} GB | Reserved: {res_gb:.2f} GB")
                running_loss = 0.0
                
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Epoch {epoch+1} completed. Peak VRAM used: {peak_vram:.2f} GB")

    print("Finished Training!")
    return model, device

def test(model, testloader, device):
    print("Evaluating on test set...")
    model.eval() 
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in testloader:
            inputs = batch["input"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(inputs, coords)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test dataset: {accuracy:.2f}%')
    else:
        print("No samples in test set.")

# 4. Main Execution Setup
if __name__ == "__main__":
    csv_path = "/home/tapotheker/dp2/data/combined_tcga_amc.csv"
    features_dir = "/home/tapotheker/dp2/data"

    print("Initializing Dataset...")
    dataset = H5FeatureBagDataset(
        csv_path=csv_path,
        features_dir=features_dir,
        label_col_name="msi_sensor_disc"
    )

    if len(dataset) == 0:
        print("No valid data found. Check your paths and CSV file.")
        exit()

    # Determine dynamic number of classes based on the dataset mapping
    num_classes = len(dataset.label_map) if dataset.label_map else 2 
    print(f"Detected {num_classes} classes.")

    # # testing on a significantly smaller dataset 25 percent
    # small_size = int(0.25 * len(dataset))
    # ignored_size = len(dataset) - small_size

    # # Split it, and use '_' to discard the 75% we don't want
    # small_dataset, _ = random_split(dataset, [small_size, ignored_size])
    # print(f"Subsampled dataset to {len(small_dataset)} items.")

    # # 2. Split the small dataset into 80% Train, 20% Test
    # train_size = int(0.8 * len(small_dataset))
    # test_size = len(small_dataset) - train_size

    # train_dataset, test_dataset = random_split(small_dataset, [train_size, test_size])
    # print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


    # Split dataset into 80% Train, 20% Test (for the complete dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # IMPORTANT: batch_size is set to 1 to handle variable sequence lengths (N)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 1. Train the model (Assuming HDF5 features are size 1280)
    trained_model, compute_device = train(train_loader, input_dim=1280, num_classes=num_classes) 
    
    # 2. Test it
    test(trained_model, test_loader, compute_device)
     
    # 3. Save it
    torch.save(trained_model.state_dict(), "msi_sensor_disc09-04.pth")
    print("msi_sensor_disc08-04.pth")