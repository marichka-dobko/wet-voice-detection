import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import torchmetrics


class ResNet18Model(pl.LightningModule):

    def __init__(self, class_weights, pretrained=True, in_channels=3, num_classes=16, lr=3e-4, freeze=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor(class_weights))

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        x, y = batch

        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)

        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)

        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):

        x, y = batch
        preds = self.model(x)
        self.test_acc(torch.argmax(preds, dim=1), y)

        self.log('test_acc', self.test_acc, on_epoch=True)

        
class MobileNetv2Model(pl.LightningModule):

    def __init__(self, pretrained=True, in_channels=3, num_classes=16, lr=1e-4, freeze=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        model = models.mobilenet_v2(pretrained=True)

        original_input_layer = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels,
                                         original_input_layer.out_channels,
                                         kernel_size=original_input_layer.kernel_size,
                                         stride=original_input_layer.stride,
                                         padding=original_input_layer.padding,
                                         bias=False)

        model.classifier[1] = nn.Linear(model.last_channel, 1)
        
        self.model = model
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
#         weights = torch.tensor([0.5])
#         self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        
        return [optimizer] #, [scheduler]

    def training_step(self, batch, batch_idx):

        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(torch.argmax(preds, dim=1), y)
        self.train_acc(torch.sigmoid(preds), y)

        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)

        logits = self.model(x).squeeze()  # Ensure it's the correct shape
        loss = self.loss_fn(logits, y.float())  # Ensure y is float for BCEWithLogitsLoss

        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5  # Convert logits to probabilities and then to binary predictions
        correct = preds.eq(y.view_as(preds)).sum().item()  # Count correct predictions
        accuracy = correct / len(y)  # Compute accuracy

        # Log loss and accuracy
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
    
        logits = self.model(x)
        loss = self.loss_fn(logits, y.unsqueeze(1).float()) 

        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5  
        correct = preds.eq(y.view_as(preds)).sum().item()  
        accuracy = correct / len(y)  # Compute accuracy

        # Log loss and accuracy
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', accuracy, on_epoch=True)


    def test_step(self, batch, batch_idx):

        x, y = batch
        logits = self.model(x).squeeze()  
        loss = self.loss_fn(logits, y.float()) 

        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5  
        correct = preds.eq(y.view_as(preds)).sum().item()  
        accuracy = correct / len(y)  

        self.log('test_acc', self.test_acc, on_epoch=True)

