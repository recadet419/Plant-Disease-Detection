from torch import nn
from transformers import ViTModel
from utils import *
from timm import create_model
import torch
from torchsummary import summary
from torchinfo import summary




class EfficientNetV2B3(nn.Module):
    def __init__(self):
        super(EfficientNetV2B3, self).__init__()
        self.effnet = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)

        for param in self.effnet.parameters():
            param.requires_grad = False

        self.effnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, FEATURES)
        )

    def forward(self, x):
        return self.effnet(x)


class ViT(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pret0rained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None


# https://arxiv.org/abs/1608.06993
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = create_model("densenet121", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, FEATURES)
        )

    def forward(self, x):
        return self.model(x)


class MobileNetV3_large(nn.Module):
    def __init__(self):
        super(MobileNetV3_large, self).__init__()
        self.model = create_model("mobilenetv3_large_100", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, FEATURES)
        )

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=FEATURES):
        super(VGG16, self).__init__()
        self.model = create_model("vgg16", pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        for param in self.model.head.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = create_model("resnet50.a1_in1k", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, FEATURES)
        )

    def forward(self, x):
        return self.model(x)


# initial hybrid model EfficientNetV2B3+ViT
class EfficientNetV2B3ViT(nn.Module):  
    def __init__(self, num_labels=FEATURES):
        super(EfficientNetV2B3ViT, self).__init__()

        # Part of EfficientNet
        self.effnet = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
        self.effnet.classifier = nn.Identity() 
        for param in self.effnet.parameters():
            param.requires_grad = False  

        # Part of ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False  

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1536 + 512, num_labels)  

    def forward(self, x):
        # Extract features using EfficientNet
        effnet_output = self.effnet.forward_features(x)
        effnet_output = torch.flatten(effnet_output, start_dim=2)  
        effnet_output = effnet_output.mean(dim=2)  

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)  

        combined = torch.cat((effnet_output, vit_output), dim=1)

        combined = self.dropout(combined)

        output = self.classifier(combined)  
        return output


class DenseNet121ViT(nn.Module):
    def __init__(self):
        super(DenseNet121ViT, self).__init__()
        self.densenet = create_model("densenet121", pretrained=True)
        self.densenet.classifier = nn.Identity()
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024 + 512, FEATURES)

    def forward(self, x):
        # Extract features using DenseNet121
        densenet_output = self.densenet.forward_features(x)
        densenet_output = torch.flatten(densenet_output, start_dim=2)
        densenet_output = densenet_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        combined = torch.cat((densenet_output, vit_output), dim=1)

        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output


class MobileNetV3ViT(nn.Module):
    def __init__(self):
        super(MobileNetV3ViT, self).__init__()
        self.mobilenetv3 = create_model("mobilenetv3_large_100", pretrained=True)
        self.mobilenetv3.classifier = nn.Identity()
        for param in self.mobilenetv3.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(960 + 512, FEATURES)

    def forward(self, x):
        # Extract features using MobileNetV3
        mobilenetv3_output = self.mobilenetv3.forward_features(x)
        mobilenetv3_output = torch.flatten(mobilenetv3_output, start_dim=2)
        mobilenetv3_output = mobilenetv3_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        combined = torch.cat((mobilenetv3_output, vit_output), dim=1)

        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output


class VGG16ViT(nn.Module):
    def __init__(self):
        super(VGG16ViT, self).__init__()
        self.vgg16 = create_model("vgg16", pretrained=True)
        self.vgg16.classifier = nn.Identity()
        for param in self.vgg16.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512 + 512, FEATURES)

    def forward(self, x):
        # Extract features using VGG16
        vgg16_output = self.vgg16.forward_features(x)
        vgg16_output = torch.flatten(vgg16_output, start_dim=2)
        vgg16_output = vgg16_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        combined = torch.cat((vgg16_output, vit_output), dim=1)

        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output


class ResNet50ViT(nn.Module):
    def __init__(self):
        super(ResNet50ViT, self).__init__()
        self.resnet50 = create_model("resnet50.a1_in1k", pretrained=True)
        self.resnet50.fc = nn.Identity()
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2048 + 512, FEATURES)

    def forward(self, x):
        # Extract features using ResNet50
        resnet50_output = self.resnet50.forward_features(x)
        resnet50_output = torch.flatten(resnet50_output, start_dim=2)
        resnet50_output = resnet50_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        combined = torch.cat((resnet50_output, vit_output), dim=1)

        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output


# ==================== my methods (Rect) ============================


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=FEATURES):
        super(ConvNeXt, self).__init__()
        # Pretrained ConvNeXt Base on ImageNet-22k finetuned on 1k
        self.model = create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classification head
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)



class ConvNeXtViT(nn.Module):
    def __init__(self, num_classes=FEATURES):
        super(ConvNeXtViT, self).__init__()

        # -------------------- ConvNeXt backbone --------------------
        self.convnext = create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True)
        self.convnext.head.fc = nn.Identity()  
        for param in self.convnext.parameters():
            param.requires_grad = False  

        # -------------------- ViT backbone --------------------
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)  
        for param in self.vit.parameters():
            param.requires_grad = False 

        # -------------------- Fusion --------------------
        convnext_out_dim = 1024  # convnext_base output dim after global pooling
        vit_out_dim = 512
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(convnext_out_dim + vit_out_dim, num_classes)

    def forward(self, x):
        # ConvNeXt feature extraction
        convnext_features = self.convnext.forward_features(x)
        convnext_features = torch.flatten(convnext_features, start_dim=2)  
        convnext_features = convnext_features.mean(dim=2) 

        # ViT feature extraction
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]  
        vit_output = self.vit_linear(vit_output) 

        # Concatenate features
        combined = torch.cat((convnext_features, vit_output), dim=1)  

        # Classification
        combined = self.dropout(combined)
        output = self.classifier(combined)

        return output


# ---------- Simple GCN layer (no PyG dependencies) ----------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, A_hat):
        return A_hat @ self.lin(x)


class GraphBlock(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.gcn1 = GCNLayer(dim, hidden)
        self.gcn2 = GCNLayer(hidden, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_hat):
        # PreNorm + residual
        h = self.norm1(x)
        h = self.act(self.gcn1(h, A_hat))
        h = self.dropout(h)
        h = self.gcn2(h, A_hat)
        x = x + self.dropout(h)

        # MLP head (post message passing) + residual
        h2 = self.norm2(x)
        h2 = self.act(h2)
        h2 = self.dropout(h2)
        return x + h2


# ---------- The Hybrid: ConvNeXt + GNN ----------
class ConvNeXtGNN(nn.Module):
    def __init__(self, num_classes=FEATURES, k=8, gnn_blocks=2, dropout=0.2):
        super().__init__()
        # ConvNeXt backbone (pretrained, frozen)
        self.backbone = create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True)
        self.backbone.head.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # figure out channel dim from backbone
        self.proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # GNN stack working on node features (tokens)
        # use feature dim from backbone (set at runtime after a dry run or assume 1024)
        self.node_dim = 1024
        hidden = 512

        blocks = []
        for _ in range(gnn_blocks):
            blocks.append(GraphBlock(dim=self.node_dim, hidden=hidden, dropout=dropout))
        self.gnn = nn.ModuleList(blocks)

        # readout + classifier
        self.readout = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(self.node_dim, num_classes)

        # graph params
        self.k = k

    @torch.no_grad()
    def _features(self, x):
        f = self.backbone.forward_features(x)
        return f

    def forward(self, x, labels=None):
        B = x.size(0)
        f = self._features(x)                 
        B, C, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2) 
        tokens = self.proj(tokens)           

        outs = []
        losses = []
        for b in range(B):
            xb = tokens[b]                  
            A_hat = knn_adjacency(xb, k=self.k) 

            hb = xb
            for block in self.gnn:
                hb = block(hb, A_hat)      

            hb = self.readout(hb)
            graph_feat = hb.mean(dim=0)      
            outs.append(graph_feat)

        out = torch.stack(outs, dim=0)        
        out = self.dropout(out)
        logits = self.classifier(out)        

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return logits, loss
        return logits

# ------------ Hybrid: EfficientNetV2 + GNN ----------
class EfficientNetGNN(nn.Module):
    def __init__(self, num_classes=FEATURES, k=8, gnn_blocks=2, dropout=0.2):
        super().__init__()

        # Backbone: EfficientNetV2-B3 (you can swap to L/XL)
        self.backbone = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
        self.backbone.classifier = nn.Identity() 
        for p in self.backbone.parameters():
            p.requires_grad = False

        # From timm: forward_features -> (B, C, H, W), typically C=1536 for B3
        self.node_dim = 1536
        hidden = 512

        self.gnn = nn.ModuleList([
            GraphBlock(dim=self.node_dim, hidden=hidden, dropout=dropout)
            for _ in range(gnn_blocks)
        ])

        self.readout = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(self.node_dim, num_classes)

        self.k = k
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _features(self, x):
        return self.backbone.forward_features(x)   

    def forward(self, x, labels=None):
        B = x.size(0)
        f = self._features(x)                      
        B, C, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2)       

        outs = []
        for b in range(B):
            xb = tokens[b]                          
            A_hat = knn_adjacency(xb, k=self.k)     

            hb = xb
            for block in self.gnn:
                hb = block(hb, A_hat)

            hb = self.readout(hb)
            graph_feat = hb.mean(dim=0)            
            outs.append(graph_feat)

        out = torch.stack(outs, dim=0)              
        out = self.dropout(out)
        logits = self.classifier(out)               

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return logits, loss
        return logits