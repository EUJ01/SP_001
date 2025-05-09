import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models.encode import PositionalEncode, FourierEncode, RoPE_Encoder
from data import KNOWN_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN, ST_MAP, coord_transform_GPS_UTM

S_COLS = ST_MAP['spatial']
T_COLS = ST_MAP['temporal']

def load_transfer_feature(model, UTM_region, spatial_middle_coord, poi_embed, poi_coors):
    model.UTM_region = UTM_region
    model.spatial_middle_coord = spatial_middle_coord
    model.poi_embed_mat = poi_embed
    model.poi_coors = poi_coors
    return model

class TrajFM(nn.Module):
    def __init__(self, embed_size, d_model, poi_embed, poi_coors, rope_layer, UTM_region, spatial_middle_coord, scale, user):
        """Trajectory Fundational Model.

        Args:
            embed_size (int): the dimension of learned embedding modules.
            d_model (int): the dimension of the sequential model.
            poi_embed (np.array): pre-defined embedding matrix of all POIs, with shape (n_poi, E).
            poi_coors (np.array): coordinates of all POIs, with shape (n_poi, 2).
            spatial_border (np.array): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
        """
        super().__init__()

        self.poi_coors = poi_coors
        self.UTM_region = UTM_region
        self.spatial_middle_coord = spatial_middle_coord
        self.scale = scale
        self.user = user

        # Embedding layers for mapping raw features into latent embeddings.
        self.spatial_embed_layer = nn.Sequential(nn.Linear(2, embed_size), nn.LeakyReLU(), nn.Linear(embed_size, d_model))

        self.temporal_embed_modules = nn.ModuleList([FourierEncode(embed_size) for _ in range(4)])
        self.temporal_embed_layer = nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 4, d_model))

        self.poi_embed_mat = poi_embed 
        self.poi_embed_layer = nn.Sequential(nn.LayerNorm(poi_embed.shape[1]), nn.Linear(poi_embed.shape[1], d_model))

        self.token_embed_layer = nn.Sequential(nn.Embedding(6, embed_size, padding_idx=5), nn.LayerNorm(embed_size), nn.Linear(embed_size, d_model))
        self.pos_encode_layer = PositionalEncode(d_model)

        # Self-attention layer for aggregating the modals.
        # self.modal_mixer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True), num_layers=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,  # Add the dropout parameter here
            batch_first=True  # Keep this to match your input shape
        )
        self.modal_mixer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.seq_model = RoPE_Encoder(d_model, layers=rope_layer)

        # Prediction modules.
        self.spatial_pred_layer = nn.Sequential(nn.Linear(d_model, 2))
        self.temporal_pred_layer = nn.Sequential(nn.Linear(d_model, 4), nn.Softplus())
        self.token_pred_layers = nn.ModuleList([nn.Linear(d_model, 5) for _ in range(2)])

        # MOD
        if self.user < 70:
            self.user_pred_layers = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.LeakyReLU(),
                nn.Linear(64, self.user))
        elif self.user > 70:
            self.user_pred_layers = nn.Sequential(nn.Linear(d_model, self.user))

    def forward(self, input_seq, positions):
        """
        The forward calculation of TrajFM.

        Args:
            input_seq (torch.FloatTensor): contains the input features of shape (B, L, F, 2).
            Refer to TrajFMPadder for detailed definition.
            postions (torch.LongTensor): represents the dual-layer positions of shape (B, L, 2).

        Returns:
            Tensor: modal hidden sequence with shape (B, L, E).
            Tensor: memory sequence with shape (B, L, E).
        """
        L = input_seq.size(1)
        # Fetch and embed token modal.
        token = input_seq[..., [S_COLS[0], T_COLS[0]], 1].long()  # (B, L, F)
        # Fetch and embed spatial modal, including POIs.
        spatial = input_seq[:, :, S_COLS, 0]  # (B, L, 2)
        # Fetch and embed temporal modal.
        temporal = input_seq[:, :, T_COLS, 0]  # (B, L, 2)
        temporal_token = tokenize_timestamp(temporal)
        modal_h, norm_coord = self.cal_modal_h(spatial, temporal_token, token, positions)

        # Utilizing the padding token to derive a batch mask of shape (B, L).
        batch_mask = token[..., 0] == PAD_TOKEN
        causal_mask = gen_causal_mask(L).to(input_seq.device)

        # Mod 
        mem_seq = self.seq_model(modal_h, norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask)
        # mem_seq = modal_h
        # Mod 

        return modal_h, mem_seq

    def cal_modal_h(self, spatial, temporal_token, token, positions):
        """Calculate modal hidden states with the given features.

        Args:
            spatial (Tensor): spatial features with shape (B, L, 2).
            temporal_token (Tensor): temporal tokens with shape (B, L, 4).
            token (Tensor): spatial and temporal tokens with shape (B, L, 2).
            positions (Tensor): dual-layer position indices with shape (B, L, 2).

        Returns:
            Tensor: the sequence of modal hidden states with shape (B, L, E).
        """
        B = spatial.size(0)
        norm_coord = spatial

        # Embedding of tokens for the spatial and temporal modals.
        token_e = self.token_embed_layer(token)  # (B, L, F, E)

        # Mask used to fill the embedding of features where the features are padding values.
        # Specifically, features where the token is not "KNOWN" or "UNKNOWN".
        feature_e_mask = ~torch.isin(token, torch.tensor([KNOWN_TOKEN, UNKNOWN_TOKEN]).to(token.device))  # (B, L, 2)

        spatial_e = self.spatial_embed_layer(norm_coord)  # (B, L, E)
        spatial_e.masked_fill(feature_e_mask[..., 0].unsqueeze(-1), 0)
        spatial_e += token_e[:, :, 0]

        # Calculate nearest POI of each coordinates.
        poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) - spatial.unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1)
        poi_e = self.poi_embed_layer(self.poi_embed_mat[poi])
        poi_e.masked_fill(feature_e_mask[..., 0].unsqueeze(-1), 0)
        poi_e += token_e[:, :, 0]
        
        # Embed temporal tokens.
        temporal_e_list = [module(temporal_token[..., i]) for i, module in enumerate(self.temporal_embed_modules)]
        temporal_e = torch.cat(temporal_e_list, dim=-1)

        # temporal_e = torch.cat([module(temporal_token[..., i]) for i, module in enumerate(self.temporal_embed_modules)], -1)
        temporal_e = self.temporal_embed_layer(temporal_e)
        temporal_e.masked_fill(feature_e_mask[..., 1].unsqueeze(-1), 0)
        temporal_e += token_e[:, :, 1]

        # Aggregate and mix the hidden states of all modals.
        modal_e = rearrange(torch.stack([spatial_e, temporal_e, poi_e], 2), 'B L F E -> (B L) F E')

        # Mod no POI
        # modal_e = rearrange(torch.stack([spatial_e, temporal_e], 2), 'B L F E -> (B L) F E')
                            
        modal_h = rearrange(self.modal_mixer(modal_e), '(B L) F E -> B L F E', B=B).mean(axis=2)

        # Incorporate dual-layer positional encoding.
        pos_encoding = self.pos_encode_layer(positions[..., 0]) + self.pos_encode_layer(positions[..., 1])
        modal_h += pos_encoding

        return modal_h, norm_coord

    def pred(self, mem_seq, return_raw=True):
        """Predict all features given the hidden sequence produced by the sequential model.

        Args:
            mem_seq (Tensor): memory sequence with shape (B, L, E).

        Returns:
            Tensor: predicted spatial coordinates with shape (B, L, 2).
            Tensor: predicted temporal tokens with shape (B, L, 4).
            List: predicted token distributions, each item is a Tensor with shape (B, L, n_token).
        """
        # pred_spatial = self.spatial_pred_layer(mem_seq)  # B, F, 2

        # pred_temporal_token = self.temporal_pred_layer(mem_seq)
        
        # pred_token = [layer(mem_seq) for layer in self.token_pred_layers]  # each (B, L, n_token)
        # if not return_raw:
        #     pred_token = torch.argmax(torch.stack(pred_token, 2), -1)  # (B, L, 2)
        pred_spatial, pred_temporal_token, pred_token = 0,0,0
        
        # MOD
        mem_seq_pooled = torch.mean(mem_seq, dim=1) 
        pred_user = self.user_pred_layers(mem_seq_pooled)

        return pred_spatial, pred_temporal_token, pred_token, pred_user
    
    
    # MOD
    def user_loss(self, input_seq, target_seq, positions):
        
        """
        The loss function for TrajFM, modified for user classification specifically 
        for a one-hot vector target representing user numbers.
        
        Args:
            input_seq (torch.FloatTensor): Input sequence with shape (B, L, F, 2).
            target_seq (torch.FloatTensor): One-hot encoded user labels with shape (B, num_classes).
            positions: Additional input data for model (depends on your implementation).
            user_labels (torch.LongTensor): User labels as indices with shape (B,).
        """
        # Forward pass
        _, mem_seq = self.forward(input_seq, positions)  # (B, L, E)
        _, _, _, pred_user = self.pred(mem_seq)
        
        pred_indices = torch.argmax(pred_user, dim=1)
        true_indices = torch.argmax(target_seq, dim=1)
        # pred_user = pred_user[:, -1, :]
        
        # print(pred_user.shape)
        # print(true_indices.shape)
        user_loss = F.cross_entropy(pred_user, true_indices)
        return user_loss

    @torch.no_grad()
    def test_user(self, input_seq, target_seq, positions):
        """The user classification test process of TrajFM.
        
        Args:
            input_seq (torch.FloatTensor): Input features of shape (B, L_in, F, 2).
            target_seq (torch.FloatTensor): One-hot encoded user labels of shape (B, num_classes).
            positions (torch.LongTensor): Input dual-layer positions (not used for this purpose).
        
        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        B, L_in = input_seq.size(0), input_seq.size(1)
        
        # Forward pass to get user logits
        _, mem_seq = self.forward(input_seq, positions[:, :L_in])
        _, _, _, pred_user = self.pred(mem_seq)  # Assuming last output is user_logits of shape (B, num_classes)

        # Converting user_logits to class predictions
        # pred_user = pred_user[:, -1, :]
        pred_indices = torch.argmax(pred_user, dim=1)  # shape: (B,)
        true_indices = torch.argmax(target_seq, dim=1)  # shape: (B,)
        
        # print("true", true_indices)
        # print("pred", pred_indices)

        # Calculate ACC@1
        acc_1 = accuracy_score(true_indices.cpu(), pred_indices.cpu())

        # Calculate ACC@5
        top_5_preds = torch.topk(pred_user, 5, dim=1).indices  # shape: (B, 5)
        # Check if true_indices are in the top 5 predictions
        acc_5 = (top_5_preds.eq(true_indices.unsqueeze(1)).sum(dim=1) > 0).float().mean().item()

        # Calculate Precision, Recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(true_indices.cpu(), pred_indices.cpu(), average='macro')
        
        # Return the metrics as a dictionary
        return {
            'ACC@1': acc_1,
            'ACC@5': acc_5,
            'Macro-P': precision,
            'Macro-R': recall,
            'Macro-F1': f1
        }
    #MOD

def gen_causal_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()

def tokenize_timestamp(t):
    """Calcualte temporal tokens given the timestamp and delta time.

    Args:
        t (Tensor): raw temporal features with shape (..., 2), the two channels representing 
        the timestamp and time difference with regard to the first point in seconds, respectively.

    Returns:
        Tensor: shape (..., 4) with channels representing the week, hour, minute, 
        and time difference with regard to the first point in minutes, respectively.
    """
    week = t[..., 0] % (7 * 24 * 60 * 60) / (24 * 60 * 60)
    hour = t[..., 0] % (24 * 60 * 60) / (60 * 60)
    minute = t[..., 0] % (60 * 60) / 60
    d_minute = t[..., 1] / 60
    return torch.stack([week, hour, minute, d_minute], -1)