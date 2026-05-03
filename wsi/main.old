import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time

# Import the custom dataset class from your provided file
from h5_dataset import H5FeatureBagDataset

# 1. Multi-Head Latent Attention & MoE Blocks 
class SimplifiedMLA(nn.Module):
    def __init__(self, dim, num_heads, latent_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_down = nn.Linear(dim, latent_dim) 
        self.kv_up = nn.Linear(latent_dim, dim * 2) 
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_latent = self.kv_down(x)
        kv = self.kv_up(kv_latent)
        k, v = kv.chunk(2, dim=-1)
        
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

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

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, num_shared, num_routed, top_k, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimplifiedMLA(dim, num_heads, latent_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.moe = DeepSeekMoE(dim, num_shared, num_routed, top_k, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

# 2. Spatially-Aware MIL Transformer Architecture
class SpatialEncoding(nn.Module):
    """Projects 2D coordinates (X, Y) into the transformer's hidden dimension."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, coords):
        # Scale down coordinates to prevent massive values from overwhelming the network
        # (Assuming typical WSI coordinates are in the tens of thousands of pixels)
        normalized_coords = coords / 10000.0 
        return self.proj(normalized_coords)

class DeepSeekSpatialViT(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2, dim=128, depth=4, 
                 num_heads=4, latent_dim=64, num_shared=1, num_routed=4, top_k=2):
        super().__init__()
        
        self.feature_proj = nn.Linear(input_dim, dim)
        self.pos_embed = SpatialEncoding(dim) # Dynamic positional embedding generator
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, latent_dim, num_shared, num_routed, top_k, dim * 2)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, coords):
        """
        Args:
            x (torch.Tensor): Feature bags of shape [B, N, Input_Dim]
            coords (torch.Tensor): Coordinates of shape [B, N, 2]
        """
        B, N, _ = x.shape
        
        # 1. Project 1280D image features to hidden dimension
        x = self.feature_proj(x) # -> [B, N, dim]
        
        # 2. Generate and add spatial positional embeddings
        spatial_tokens = self.pos_embed(coords)
        x = x + spatial_tokens 
        
        # 3. Prepend CLS token for classification
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> [B, N+1, dim]

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        
        # Classification using only the CLS token output
        return self.head(x[:, 0])

# 3. Training & Evaluation Loops
def train(trainloader, input_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Initialize the spatially-aware MIL model
    model = DeepSeekSpatialViT(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(trainloader):
            # Extract features, coordinates, and labels
            inputs = batch["input"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            
            # Pass BOTH features and coordinates to the model
            outputs = model(inputs, coords)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Print stats every 10 steps (since batch size is 1)
            if i % 10 == 9:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0
                
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds.")

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
        label_col_name="tmb_combined"
    )

    if len(dataset) == 0:
        print("No valid data found. Check your paths and CSV file.")
        exit()

    # Determine dynamic number of classes based on the dataset mapping
    num_classes = len(dataset.label_map) if dataset.label_map else 2 
    print(f"Detected {num_classes} classes.")

    # testing on a significantly smaller dataset 25 percent
    small_size = int(0.25 * len(dataset))
    ignored_size = len(dataset) - small_size

    # Split it, and use '_' to discard the 75% we don't want
    small_dataset, _ = random_split(dataset, [small_size, ignored_size])
    print(f"Subsampled dataset to {len(small_dataset)} items.")

    # 2. Split the small dataset into 80% Train, 20% Test
    train_size = int(0.8 * len(small_dataset))
    test_size = len(small_dataset) - train_size

    train_dataset, test_dataset = random_split(small_dataset, [train_size, test_size])
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


    # # Split dataset into 80% Train, 20% Test (for the complete dataset)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # IMPORTANT: batch_size is set to 1 to handle variable sequence lengths (N)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 1. Train the model (Assuming HDF5 features are size 1280)
    trained_model, compute_device = train(train_loader, input_dim=1280, num_classes=num_classes) 
    
    # 2. Test it
    test(trained_model, test_loader, compute_device)
     
    # 3. Save it
    torch.save(trained_model.state_dict(), "deepseek_spatial_mil.pth")
    print("Model weights saved to deepseek_spatial_mil.pth")