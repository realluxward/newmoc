from vector_quantize_pytorch.vector_quantize_pytorch import *
def my_gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True,
    inference_topk = 1,
):
    if training:
        return gumbel_sample(logits, temperature, stochastic, straight_through, reinmax, dim, training)
    
    dtype, size = logits.dtype, logits.shape[dim]
    sampling_logits = logits

    # ind = sampling_logits.argmax(dim = dim)
    _, topk_ind = torch.topk(sampling_logits, k=inference_topk, dim=dim)
    _, bottom_ind = torch.topk(-sampling_logits, k=inference_topk, dim=dim)  # (1, B, K)
    ind = topk_ind[:, :, 0]

    top_and_bottom_ind = torch.cat([topk_ind, bottom_ind], dim=-1)
    return ind, top_and_bottom_ind

class MyEuclideanCodebook(EuclideanCodebook):
    def my_forward_for_inference(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype
        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))
        self.init_embed_(flatten, mask = mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask = mask)
        embed = self.embed if self.learnable_codebook else self.embed.detach()
        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        dist = -cdist(flatten, embed)
        embed_ind, embed_ind_topk = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')
        quantize = batched_embedding(embed_ind, embed)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        dist = unpack_one(dist, ps, 'h * d')

        print('using my impl')
        return quantize, embed_ind, embed_ind_topk, dist
    
class MyVectorQuantize(VectorQuantize):
    def __init__(
        self,
        dim,
        codebook_size,
        inference_topk=1,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer: Callable[..., Optimizer] = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0. # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super(VectorQuantize, self).__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

        self.sync_update_v = sync_update_v

        codebook_class = MyEuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = partial(
            my_gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through,
            inference_topk = inference_topk,
        )

        if not exists(sync_codebook):
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,  ## specical change
            ema_update = ema_update
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last


    def my_forward_for_inference(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
    ):
        orig_input = x
        only_one = x.ndim == 2
        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)
        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        x = self._codebook.transform_input(x)
        # quantize
        quantize, embed_ind, embed_ind_topk, distances = self._codebook.my_forward_for_inference(x, **dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        ))
        return quantize, embed_ind, embed_ind_topk

