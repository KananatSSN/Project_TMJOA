import torch
import torch.nn as nn
import torch.nn.functional as F

# Block 5: 3D ResNet Architecture with MedicalNet-style Pre-training
class DropBlock3D(nn.Module):
    """3D DropBlock for spatial regularization in medical volumes"""
    def __init__(self, drop_rate=0.1, block_size=5):
        super(DropBlock3D, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
        
    def forward(self, x):
        if not self.training:
            return x
            
        # Calculate gamma (probability to set seed)
        gamma = self.drop_rate / (self.block_size ** 3)
        
        # Generate mask
        mask_shape = (x.shape[0], x.shape[1], 
                     x.shape[2] - self.block_size + 1,
                     x.shape[3] - self.block_size + 1,
                     x.shape[4] - self.block_size + 1)
        
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
        
        # Expand mask to block size
        mask = F.pad(mask, [self.block_size//2] * 6, value=0)
        mask = F.max_pool3d(mask, kernel_size=self.block_size, 
                           stride=1, padding=self.block_size//2)
        
        # Apply mask
        mask = 1 - mask
        normalize_factor = mask.numel() / mask.sum()
        
        return x * mask * normalize_factor

class SpatialDropout3D(nn.Module):
    """3D Spatial Dropout for feature map regularization"""
    def __init__(self, p=0.2):
        super(SpatialDropout3D, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training:
            return x
            
        # Generate noise for each channel
        noise = torch.bernoulli(torch.full((x.size(0), x.size(1), 1, 1, 1), 
                                         1 - self.p, device=x.device))
        return x * noise / (1 - self.p)

class StochasticDepth(nn.Module):
    """Stochastic depth for very deep 3D networks"""
    def __init__(self, survival_prob=0.8):
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob
        
    def forward(self, x, residual):
        if not self.training:
            return x + residual
            
        if torch.rand(1).item() < self.survival_prob:
            return x + residual
        else:
            return x

class FocalLoss(nn.Module):
    """Unified Focal Loss for class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LayerNorm3D(nn.Module):
    """3D Layer Normalization for small batch sizes"""
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm3D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape weight and bias for broadcasting
        weight = self.weight.view(1, -1, 1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1, 1)
        
        return x_norm * weight + bias
class BasicBlock3D(nn.Module):
    """3D BasicBlock for ResNet"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 dropout_rate=0.1, use_dropblock=True):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
        # Advanced regularization
        self.dropout = SpatialDropout3D(dropout_rate) if dropout_rate > 0 else None
        self.dropblock = DropBlock3D(drop_rate=0.1, block_size=5) if use_dropblock else None
        self.stochastic_depth = StochasticDepth(survival_prob=0.8)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        if self.dropblock:
            out = self.dropblock(out)
            
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Apply stochastic depth
        out = self.stochastic_depth(out, residual)
        out = self.relu(out)
        
        return out

class Bottleneck3D(nn.Module):
    """3D Bottleneck block for deeper ResNets"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dropout_rate=0.1, use_dropblock=True):
        super(Bottleneck3D, self).__init__()
        
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Advanced regularization
        self.dropout = SpatialDropout3D(dropout_rate) if dropout_rate > 0 else None
        self.dropblock = DropBlock3D(drop_rate=0.1, block_size=5) if use_dropblock else None
        self.stochastic_depth = StochasticDepth(survival_prob=0.8)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        if self.dropblock:
            out = self.dropblock(out)
            
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = self.stochastic_depth(out, residual)
        out = self.relu(out)
        
        return out

class ResNet3D(nn.Module):
    """3D ResNet for TMJ classification with medical-specific modifications"""
    
    def __init__(self, block, layers, num_classes=2, input_channels=1,
                 dropout_rate=0.3, use_dropblock=True, use_layer_norm=False):
        super(ResNet3D, self).__init__()
        
        self.inplanes = 64
        self.dropout_rate = dropout_rate
        self.use_dropblock = use_dropblock
        self.use_layer_norm = use_layer_norm
        
        # Initial convolution (larger kernel for 3D medical data)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        
        if use_layer_norm:
            self.bn1 = LayerNorm3D(64)
        else:
            self.bn1 = nn.BatchNorm3d(64)
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.use_layer_norm:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                             kernel_size=1, stride=stride, bias=False),
                    LayerNorm3D(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                           self.dropout_rate, self.use_dropblock))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate,
                               use_dropblock=self.use_dropblock))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using medical imaging best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, LayerNorm3D)):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """Extract features for ensemble or visualization"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features

def create_medical_resnet3d(arch='resnet18', num_classes=2, pretrained_path=None):
    """Create 3D ResNet optimized for medical imaging"""
    
    if arch == 'resnet18':
        model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes)
    elif arch == 'resnet34':
        model = ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes)
    elif arch == 'resnet50':
        model = ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes=num_classes)
    elif arch == 'resnet101':
        model = ResNet3D(Bottleneck3D, [3, 4, 23, 3], num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Load MedicalNet pre-trained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load weights, ignoring classifier
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in new_state_dict.items() 
                           if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
            print(f"Loaded {len(filtered_dict)} pre-trained layers")
            
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Training from scratch...")
    
    return model