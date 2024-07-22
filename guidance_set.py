import torch
from torch import nn
import torch.nn.functional as F

def get_loss_dim(concept_enc, concept_dims, concept_values):
    """
    Args:
        concept_enc (callable): Concept encoder function, returning (B, D) where D is the total number of dimensions.
        concept_dims (list[int]): List of indices corresponding to the concept dimensions of interest.
        concept_values (torch.Tensor): Tensor of shape (B, len(concept_dims)) containing the target values for each concept dimension for each sample in the batch.

    Returns:
        callable: A loss function that calculates the mean squared error loss over the specified concept dimensions for each sample in the batch.
    """
    def loss_dim(x):
        c_pre = concept_enc(x)  # (B, D), B is batch size, D is number of dimensions
        # Select the relevant concept dimensions
        c_pre_selected = c_pre[:, concept_dims]  # (B, len(concept_dims))
        # Calculate MSE loss for the selected dimensions
        loss = F.mse_loss(c_pre_selected, concept_values, reduction='none')
        return loss.mean(dim=1)  # Reduce mean across the concept dimensions for each sample, result is (B,)
    return loss_dim

def get_loss_smooth(n_diag=2, threshold=0.95):
    """
    Returns a loss function that calculates the mean squared error (MSE) loss over a target similarity matrix.

    The target similarity matrix is structured such that the first and second off-diagonals are set to 1, while the
    main diagonal and other off-diagonal elements are ignored. This is useful for tasks where the similarity between
    neighboring samples in the batch is to be maximized.

    Returns:
        callable: A loss function that computes the MSE loss between the computed similarity matrix and the target
        similarity matrix for a given batch of inputs.
    """
    def loss_smooth(x):
        N = x.size(0)
        assert n_diag < N, f"n_diag must be less than the batch size N={N}"

        # Compute similarity matrix
        sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

        # Initialize target similarity matrix with NaNs to ignore in loss calculation
        target_sim = torch.full((N, N), 0.0, device=x.device)

        # Set target similarities on the upper triangular of the first n off-diagonals to 1
        for i in range(1, n_diag + 1):
            target_sim += torch.diag(torch.full((N - i,), 1., device=x.device), i)
            # target_sim += torch.diag(torch.full((N - i,), 1.0, device=x.device), -i) 

        # Mask out NaN values in target_sim for loss calculation
        valid_mask = target_sim != 0
        eps = 1e-6
        valid_mask[sim>=(threshold+eps)] = False
        # in case of no valid mask
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        loss = F.mse_loss(sim[valid_mask], target_sim[valid_mask])
        # loss = F.mse_loss(sim.view(-1), target_sim.view(-1))
        return loss.mean()

    return loss_smooth

def get_loss_similarity(h_target, threshold=0.95):
    """
    Returns a loss function that calculates the mean squared error (MSE) loss for the cosine similarity between each input
    and a target embedding, but ignores those where the similarity exceeds a specified threshold.

    Args:
        h_target (torch.Tensor): Target embedding tensor of shape (1, D), where D is the dimensionality of the embeddings.
        threshold (float): Similarity threshold above which no loss is computed.

    Returns:
        callable: A loss function that computes the MSE loss for inputs similar to the target below a certain threshold.
    """
    def loss_similarity(x):
        # Calculate the cosine similarity between the batch x and the target h_target
        sim = F.cosine_similarity(x, h_target.repeat(x.size(0), 1), dim=1)
        
        # Apply threshold: Only consider embeddings with a similarity below the threshold
        mask = sim < threshold
        
        # If all embeddings exceed the threshold, return zero loss
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Compute MSE loss for the selected embeddings below the threshold
        target_sim_values = torch.ones_like(sim[mask])
        loss = F.mse_loss(sim[mask], target_sim_values)
        
        return loss.mean()

    return loss_similarity


def get_loss_probability(concept_enc, prob, refs):
    """
    Returns a function that computes the cross-entropy loss 
        for the probability of choosing ref1 as the more similar image.
    
    Args:
        concept_enc (callable): Concept encoder function, returning (B, D) 
            where D is the total number of dimensions.
        prob (torch.Tensor): Tensor of shape (B,) 
            containing the target probabilities for each sample in the batch.
        refs (torch.Tensor): (2, D) Tensor of the clip embedding of the reference image 1&2.
    
    Returns:
        callable: A loss function that computes the cross-entropy loss based on the probability of choosing ref1.
    """
    def loss_probability(x):
        """
        Computes the cross-entropy loss for the given input tensor based on the probability of choosing ref1.
        
        Args:
            x (torch.Tensor): A tensor of shape (B, D) where B is the batch size.
        
        Returns:
            torch.Tensor: The cross-entropy loss.
        """
        c_pre = concept_enc(x)  # (B, D)
        c_refs = concept_enc(refs)  # (2, D)
        similarity = c_pre @ c_refs.T # (B, 2)
        prob_pre = similarity.softmax(dim=-1)
        prob_pre1 = prob_pre[:, 0]  # (B, )
        eps = 1e-6
        # Compute cross-entropy loss between prob and prob_pre1
        # loss = F.binary_cross_entropy(prob_pre1, prob)
        loss = - (prob * (prob_pre1+eps).log() + (1-prob) * (1-prob_pre1+eps).log())
        return loss.mean()

    return loss_probability


def get_loss_triplet_entropy(concept_enc, refs):
    def loss_triplet_entropy(x):
        c_pre = concept_enc(x)  # (B, D)
        c_refs = concept_enc(refs)  # (2, D)
        # create c (B, 3, D)
        c = torch.cat([c_pre.unsqueeze(1), c_refs.unsqueeze(0).repeat(x.size(0), 1, 1)], dim=1)
        similarity = torch.einsum('bnd,bmd->bnm', c, c) # (B, 3, 3)
        similarity = torch.triu(similarity, diagonal=1) # (B, 3, 3)
        similarity_123 = similarity[similarity.triu(diagonal=1) != 0].view(-1, 3) # (B, 3)
        prob_triplet = similarity_123.softmax(dim=-1) # (B, 3)
        # entropy of triplet 
        eps = 1e-6
        entropy = - (prob_triplet * (prob_triplet+eps).log()).sum(dim=-1)
        loss = - entropy
        return loss.mean()
    return loss_triplet_entropy

def fns_collector(fns, scales):
    """
    Combines multiple functions with corresponding scales into a single loss function.
    
    Args:
    fns (list[callable]): List of function objects, each accepting the same type of input.
    scales (list[float]): List of scaling factors for each function in `fns`.
    
    Returns:
    callable: A combined function that computes the scaled sum of individual functions.
    """
    def loss_func(x):
        # Compute the weighted sum of functions
        loss = sum(scale * fn(x) for scale, fn in zip(scales, fns))
        return loss.mean()

    return loss_func