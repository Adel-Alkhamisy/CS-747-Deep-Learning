import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
#     N = logits_real.size(0)
#     true_labels = torch.ones(N, 1).to(logits_real.device)
#     false_labels = torch.zeros(N, 1).to(logits_fake.device)

#     loss_real = bce_loss(logits_real, true_labels)
#     loss_fake = bce_loss(logits_fake, false_labels)
#     loss = (loss_real + loss_fake) / 2

    """
    Computes the discriminator loss.
    """
    N_real = logits_real.size(0)
    N_fake = logits_fake.size(0)

    true_labels_real = torch.ones(N_real, 1).to(logits_real.device)
    false_labels_fake = torch.zeros(N_fake, 1).to(logits_fake.device)

    loss_real = bce_loss(logits_real, true_labels_real)
    loss_fake = bce_loss(logits_fake, false_labels_fake)
    
    loss = (loss_real + loss_fake) / 2
    
    ##########       END      ##########
    
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
#     N = logits_fake.size(0)
#     true_labels = torch.ones(N, 1).to(logits_fake.device)
    
#     loss = bce_loss(logits_fake, true_labels)
    """
    Computes the generator loss.
    """
    N_fake = logits_fake.size(0)
    true_labels_fake = torch.ones(N_fake, 1).to(logits_fake.device)

    loss = bce_loss(logits_fake, true_labels_fake)
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    loss_real = 0.5 * torch.mean((scores_real - 1)**2)
    loss_fake = 0.5 * torch.mean(scores_fake**2)
    loss = (loss_real + loss_fake)

    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = 0.5 * torch.mean((scores_fake - 1)**2)
    
    ##########       END      ##########
    
    return loss
def wgan_discriminator_loss(real_preds, fake_preds):
    return fake_preds.mean() - real_preds.mean()


def wgan_generator_loss(fake_preds):
    return -fake_preds.mean()