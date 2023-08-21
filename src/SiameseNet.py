'''
    Siamese network.
'''
########## IMPORTs START ##########

# processing
import cv2

# nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")

# data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# utils
import utils

###### IMPORTs - END ##############

###
###
###

###### CONSTANTs - START ##########

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=512),
        A.CenterCrop(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

###### CONSTANTs - END ############

###
###
###

###### CLASSEs - START ##########

class SiameseNetworkTask(pl.LightningModule):
    def __init__(self, 
                embedding_net,          # embedding network
                _optimizer='sgd',       # optimizer
                lr=0.01,                # learning rate
                momentum=0.99,          # momentum
                margin=2,               # margin loss
                weight_decay=0.0001     # weight decay
            ):
        super(SiameseNetworkTask, self).__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_net
        # loss definition
        self.criterion = ContrastiveLoss(margin)
        self._optimizer = _optimizer
                    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        if self._optimizer == 'sgd':
            return optim.SGD(self.embedding_net.parameters(), 
                                self.hparams.lr, 
                                momentum=self.hparams.momentum)
        elif self._optimizer == 'adam':
            return optim.Adam(self.embedding_net.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        else:
            utils.p_error_n_exit('Optimizer not supported: {}'.format(self._optimizer))
    
    def training_step(self, batch, batch_idx):
        # take into account the I_i and I_j images and the label l_ij
        I_i, I_j, l_ij, *_ = batch
        
        # compute the embeddings
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)

        # loss computation
        l, _ = self.criterion(phi_i, phi_j, l_ij)
        
        self.log('train/loss', l)
        return l
    
    def validation_step(self, batch, batch_idx):
        I_i, I_j, l_ij, *_ = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        l, _ = self.criterion(phi_i, phi_j, l_ij)
        self.log('valid/loss', l)
        
        if batch_idx==0:
            self.logger.experiment.add_embedding(phi_i, batch[3], I_i, global_step=self.global_step)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2):
        super(ContrastiveLoss, self).__init__()
        self.m = m

    def forward(self, phi_i, phi_j, l_ij):
        d = F.pairwise_distance(phi_i, phi_j)
        l = 0.5 * (1 - l_ij.float()) * torch.pow(d, 2) + \
            0.5 * l_ij.float() * torch.pow( torch.clamp( self.m - d, min = 0) , 2)
        return l.mean(), d
    
###### CLASSEs - END ############

###
###
###

###### FUNCTIONs - START ##########

def compute_embedding(image_filepath, model, device):
    """
    Function to compute the embedding of an image.
    :param image_filepath: path of the image
    :param model: model to use
    :param device: device to use
    :return: embedding
    """
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = test_transforms(image=image)["image"]

    # get embedding
    x = image.unsqueeze(0).to(device)
    embedding = model(x)
    embedding = embedding.detach().cpu()

    return embedding

###### FUNCTIONs - END ############

###
###
###

###### COMMENTs ###################