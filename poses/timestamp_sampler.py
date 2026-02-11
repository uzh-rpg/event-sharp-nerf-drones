import torch
from torch import nn


class TimestampSampler(nn.Module):
    def __init__(self, num_samples, exposure_time_us):
        super().__init__()
        assert num_samples % 2 == 1, "num_samples must be odd"
        self.num_samples = num_samples
        self.exposure_time_us = exposure_time_us

        # Defines relative anchor times  [1, num_samples]
        times = torch.linspace(-exposure_time_us//2, exposure_time_us//2, num_samples).reshape(1, -1).double()
        # By convention (Deblur-NeRF) the first value should correspond to the center of the exposure time
        sorting_idx = list(range(num_samples))
        sorting_idx = [sorting_idx.pop(num_samples//2), *sorting_idx]  # place the center at the first position
        self.register_buffer("anchor_times", times[:, sorting_idx])  # resort the anchor times
        self.register_buffer("anchor_weights", torch.ones(1, num_samples) / num_samples)  # all weight the same

    def forward(self, time, *args, **kwargs):
        # time.shape = [N]
        time = time.reshape(-1, 1)  # [N, 1]
        sampled_times = time + self.anchor_times  # [N, num_samples]
        sampled_weights = self.anchor_weights.expand(time.shape[0], -1)  # [N, num_samples]
        return sampled_times, sampled_weights

class LearnedTimestampSampler(TimestampSampler):
    def __init__(self, num_samples, exposure_time_us, num_images, view_embed,
                 num_layers, hidden_dim, predict_weigths=True, use_anchors=False):
        super().__init__(num_samples, exposure_time_us)

        self.predict_weigths = predict_weigths
        self.view_embed = view_embed
        self.num_images = num_images
        self.use_anchors = use_anchors

        self.in_channels = view_embed.out_channels
        # if no weight are predicted, predict the times for all positions apart from the center
        # if weights are predicted, predict the times as before but the weights also for the center
        self.out_channels = (num_samples-1) if not predict_weigths else 2*num_samples-1

        self.mlp = nn.ModuleList([nn.Linear(self.in_channels, hidden_dim)] +
                                 [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.out = nn.Linear(hidden_dim, self.out_channels)

        # Defines relative anchor times  [1, num_samples]
        nocenter_idx = list(range(num_samples))
        nocenter_idx.pop(num_samples // 2)  # remove the center, we only predict the rest for the time anchors
        self.register_buffer("anchor_times", torch.linspace(-0.5, 0.5, num_samples)[nocenter_idx].reshape(1, -1))
        self.anchors_step = 1 / (num_images-1)

    def forward(self, time, images_idx, *args, **kwargs):
        view_feature = self.view_embed(images_idx.squeeze(-1)) # [N, in_channels]
        for layer in self.mlp:
            view_feature = layer(view_feature)
            view_feature = torch.relu(view_feature)
        out = self.out(view_feature)
        out = torch.sigmoid(out)

        if self.predict_weigths:
            sampled_times = out[:, :self.num_samples-1]  # [N, num_samples-1]
            sampled_weights = out[:, self.num_samples-1:]  # [N, num_samples]  # we predict also the center's weight
        else:
            sampled_times = out  # [N, num_samples-1]
            sampled_weights = self.anchor_weights.expand(time.shape[0], -1)  # [N, num_samples]

        if self.use_anchors:
            # The network predicts an offset from the anchor times which spans one step before and after the anchor
            sampled_times = sampled_times * 2 * self.anchors_step - self.anchors_step + self.anchor_times
            sampled_times = torch.clamp(sampled_times, -0.5, 0.5)

        # Add the center time in the first position (0 relative delay)  -> [N, num_samples]
        sampled_times = torch.cat([torch.zeros(time.shape[0], 1, device=time.device), sampled_times], dim=1)

        # If no anchors are used the network directly predicts the relative times
        sampled_times = time + sampled_times * self.exposure_time_us

        return sampled_times, sampled_weights
    

class HybridTimestampSampler(nn.Module):
    def __init__(self, num_samples, exposure_time_us, num_images, view_embed,
                 num_layers, hidden_dim, predict_weigths=True, use_anchors=False):
        super().__init__()
        assert num_samples % 2 == 1, "num_samples must be odd"
        self.num_samples = num_samples
        self.exposure_time_us = exposure_time_us
        self.view_embed = view_embed
        self.num_images = num_images
        self.use_anchors = use_anchors
        self.predict_weigths = predict_weigths

        self.in_channels = view_embed.out_channels
        # if no weight are predicted, predict the times for all positions apart from the center
        # if weights are predicted, predict the times as before but the weights also for the center

        n_time_ch = (num_samples-1) if use_anchors else 0
        n_weight_ch = num_samples if predict_weigths else 0
        self.out_channels = n_time_ch + n_weight_ch

        if self.out_channels != 0:
            self.mlp = nn.ModuleList([nn.Linear(self.in_channels, hidden_dim)] +
                                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
            self.out = nn.Linear(hidden_dim, self.out_channels)

            # Defines relative anchor times  [1, num_samples]
            nocenter_idx = list(range(num_samples))
            nocenter_idx.pop(num_samples // 2)  # remove the center, we only predict the rest for the time anchors
            self.register_buffer("anchor_times_learned", torch.linspace(-0.5, 0.5, num_samples)[nocenter_idx].reshape(1, -1))
            self.anchors_step = 1 / (num_images-1)

        # Defines relative anchor times  [1, num_samples]
        times = torch.linspace(-exposure_time_us//2, exposure_time_us//2, num_samples).reshape(1, -1).double()
        # By convention (Deblur-NeRF) the first value should correspond to the center of the exposure time
        sorting_idx = list(range(num_samples))
        sorting_idx = [sorting_idx.pop(num_samples//2), *sorting_idx]  # place the center at the first position
        self.register_buffer("anchor_times_uniform", times[:, sorting_idx])  # resort the anchor times
        self.register_buffer("anchor_weights", torch.ones(1, num_samples) / num_samples)  # all weight the same

    def forward(self, time, images_idx, *args, **kwargs):
        
        if self.out_channels != 0:
            view_feature = self.view_embed(images_idx.squeeze(-1)) # [N, in_channels]
            for layer in self.mlp:
                view_feature = layer(view_feature)
                view_feature = torch.relu(view_feature)
            out = self.out(view_feature)
            out = torch.sigmoid(out)

        if self.predict_weigths:
            sampled_weights = out[:, :self.num_samples] # [N, num_samples], including the center
        else:
            sampled_weights = self.anchor_weights.expand(time.shape[0], -1)  # [N, num_samples]
        
        if self.use_anchors:
            anchor_shift = out[:, self.num_samples:] if self.predict_weigths else out
            sampled_times = anchor_shift * 2 * self.anchors_step - self.anchors_step + self.anchor_times_learned
            # The network predicts an offset from the anchor times which spans one step before and after the anchor
            sampled_times = torch.clamp(sampled_times, -0.5, 0.5)
            # Add the center time in the first position (0 relative delay)  -> [N, num_samples]
            sampled_times = torch.cat([torch.zeros(time.shape[0], 1, device=time.device), sampled_times], dim=1)
            # If no anchors are used the network directly predicts the relative times
            sampled_times = time + sampled_times * self.exposure_time_us
        else:
            # time.shape = [N]
            time = time.reshape(-1, 1)  # [N, 1]
            sampled_times = time + self.anchor_times_uniform  # [N, num_samples]
        
        
        return sampled_times, sampled_weights