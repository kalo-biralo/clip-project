import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import os
import torchvision.transforms as T


class ImageEncoder(nn.Module):
    """
    A PyTorch module to encode images to fixed-size vectors using a ResNet model.

    This class uses a pretrained ResNet model to encode input images into fixed-size vectors.
    The final fully connected layer of the ResNet model is removed, and the remaining layers
    are used to produce the image embeddings.

    Attributes:
    -----------
    model : nn.Sequential
        The ResNet model without the final fully connected layer.
    num_features : int
        The number of features in the output of the ResNet model.

    Methods:
    --------
    forward(x):
        Encodes the input images and returns the feature vectors.

    Parameters:
    -----------
    model_name : str, optional
        The name of the ResNet model to use (default is 'resnet50').
        Supported models are 'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'.
    """

    def __init__(self, model_name="resnet50"):
        super().__init__()

        # Load the specified ResNet model with pretrained weights
        if model_name == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet101":
            self.model = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1
            )
        elif model_name == "resnet152":
            self.model = models.resnet152(
                weights=models.ResNet152_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        self.num_features = (
            self.model.fc.in_features
        )  # Number of features in the output

        # Remove the final fully connected layer from the ResNet model
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Freeze the ResNet model parameters to prevent them from being updated during training
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Encodes the input images and returns the feature vectors.

        Parameters:
        -----------
        x : torch.Tensor
            The input images of shape (batch_size, n_channels, height, width).

        Returns:
        --------
        torch.Tensor
            The feature vectors for the input images of shape (batch_size, num_features).
        """
        return self.model(x).squeeze()


class TextEncoder(nn.Module):
    """
    This class uses a pretrained DistilBERT model to encode input text sequences into fixed-size vectors.
    The embedding of the [CLS] token (first token) is used as the sequence representation.

    Attributes:
    -----------
    model : DistilBertModel
        The DistilBERT model used for encoding text sequences.
    tokenizer : DistilBertTokenizer
        The tokenizer used for converting text to input IDs and attention masks.

    Methods:
    --------
    forward(texts, max_length=64, device='cpu'):
        Encodes the input text sequences and returns the [CLS] token embeddings.
    encode_text(texts, max_length=64, device='cpu'):
        Tokenizes and encodes the input text sequences.

    Parameters:
    -----------
    pretrained_model_name : str, optional
        The name of the pretrained DistilBERT model to use (default is 'distilbert-base-uncased').
    """

    def __init__(self, pretrained_model_name="distilbert-base-uncased"):
        super().__init__()
        # Load pretrained DistilBERT model and tokenizer
        self.model = DistilBertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)

        # Freeze DistilBERT parameters to prevent them from being updated during training
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts, max_length=64, device="cpu"):
        """
        Encodes the input text sequences and returns the [CLS] token embeddings.

        Parameters:
        -----------
        texts : list of str
            A list of text sequences to encode.
        max_length : int, optional
            The maximum length of the input sequences (default is 64).
        device : str, optional
            The device to run the model on, either 'cpu' or 'cuda' (default is 'cpu').

        Returns:
        --------
        torch.Tensor
            The embeddings of the [CLS] token for each input sequence.
        """
        # Tokenize and encode the input text sequences
        input_ids, attention_mask = self.encode_text(texts, max_length, device=device)
        # Pass the inputs through the DistilBERT model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return the embeddings of the [CLS] token (first token)
        return outputs.last_hidden_state[:, 0, :]

    def encode_text(self, texts, max_length=64, device="cpu"):
        """
        Tokenizes and encodes the input text sequences.

        Parameters:
        -----------
        texts : list of str
            A list of text sequences to encode.
        max_length : int, optional
            The maximum length of the input sequences (default is 64).
        device : str, optional
            The device to run the model on, either 'cpu' or 'cuda' (default is 'cpu').

        Returns:
        --------
        torch.Tensor, torch.Tensor
            The input IDs and attention masks for the encoded text sequences.
        """
        # Tokenize and encode the text sequences, padding and truncating as necessary
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        return input_ids, attention_mask


class ProjectionHead(nn.Module):
    """
    Projection layer to project image and text embeddings into the same dimension

    Attributes:
    -----------
    projection : nn.Linear
        Linear layer to project embeddings to a lower-dimensional space.
    gelu : nn.GELU
        GELU activation function.
    fc : nn.Linear
        Fully connected linear layer for further projection.
    dropout : nn.Dropout
        Dropout layer for regularization.
    layer_norm : nn.LayerNorm
        Layer normalization to stabilize the output.

    Methods:
    --------
    forward(x):
        Applies the projection head to the input tensor `x`.

    Parameters:
    -----------
    embedding_dim : int
        Dimensionality of the input embeddings.
    projection_dim : int
        Dimensionality of the projected space.
    dropout : float
        Dropout rate for regularization.

    Returns:
    --------
    torch.Tensor
        The output tensor after applying the projection head.
    """

    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """
        Applies the projection head to the input tensor `x`.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, embedding_dim).

        Returns:
        --------
        torch.Tensor
            The output tensor of shape (batch_size, projection_dim) after applying
            linear projection, GELU activation, another linear layer, dropout,
            residual connection, and layer normalization.
        """
        # Apply linear projection
        projected = self.projection(x)
        # Apply GELU activation
        x = self.gelu(projected)
        # Apply another linear layer
        x = self.fc(x)
        # Apply dropout
        x = self.dropout(x)
        # Add residual connection
        x = x + projected
        # Apply layer normalization
        x = self.layer_norm(x)
        return x


class CLIP(nn.Module):
    """
    A PyTorch module for Contrastive Language-Image Pretraining (CLIP).

    This class combines an image encoder and a text encoder to jointly learn
    image and text representations. It includes projection heads for both
    modalities and a learnable temperature parameter to scale the logits.

    Attributes:
    -----------
    image_encoder : nn.Module
        The image encoder module to encode images into feature vectors.
    text_encoder : nn.Module
        The text encoder module to encode text into feature vectors.
    image_projection_head : ProjectionHead
        The projection head for image embeddings.
    text_projection_head : ProjectionHead
        The projection head for text embeddings.
    temperature : nn.Parameter
        A learnable parameter to scale the logits.

    Methods:
    --------
    forward(images, texts, device):
        Computes the contrastive loss between image and text embeddings.

    Parameters:
    -----------
    image_encoder : nn.Module
        An instance of an image encoder module.
    text_encoder : nn.Module
        An instance of a text encoder module.
    embed_dim : int, optional
        The dimensionality of the projected embeddings (default is 256).
    dropout : float, optional
        The dropout rate for the projection heads (default is 0.1).
    """

    def __init__(self, device, pretrained):
        super(CLIP, self).__init__()

        embed_dim = 768
        dropout = 0.2

        self.device = device
        self.image_encoder = ImageEncoder(model_name="resnet50").to(device)
        self.text_encoder = TextEncoder().to(device)

        # Projection heads for image and text embeddings
        self.image_projection_head = ProjectionHead(
            self.image_encoder.num_features, embed_dim, dropout
        )
        self.text_projection_head = ProjectionHead(
            self.text_encoder.model.config.hidden_size, embed_dim, dropout
        )

        # Learnable temperature parameter for scaling the logits
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if pretrained:
            self.load_weights()

    def forward(self, images, texts):
        """
        Computes the contrastive loss between image and text embeddings.

        Parameters:
        -----------
        images : torch.Tensor
            The input images of shape (batch_size, n_channels, height, width).
        texts : list of str
            The input text sequences.
        device : torch.device
            The device (CPU or GPU) on which to perform computations.

        Returns:
        --------
        torch.Tensor
            The contrastive loss between image and text embeddings.
        """
        # Encode images and texts
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(texts, device=self.device)

        # Apply projection heads to the embeddings
        image_embeddings = self.image_projection_head(image_embeddings)
        text_embeddings = self.text_projection_head(text_embeddings)

        # Normalize the embeddings
        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

        # Compute scaled pairwise cosine similarities
        logits = (image_embeddings @ text_embeddings.transpose(-2, -1)) * torch.exp(
            self.temperature
        )

        # Create labels for the symmetric loss function
        labels = torch.arange(logits.shape[0]).to(self.device)  # shape: (batch_size,)

        # Compute symmetric loss
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = (loss_i + loss_t) / 2

        return loss

    def load_weights(self):
        checkpoint_path = "./weights/best_checkpoint.pth"
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch.device("cpu")
                )
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
