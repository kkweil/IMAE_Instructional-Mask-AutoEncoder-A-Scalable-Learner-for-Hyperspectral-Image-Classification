import torch
import faiss
from torch import nn
import time
from timm.models.vision_transformer import Block, Mlp
from models.PixelEmbed import PixelEmbed, PosCNN
import numpy as np
from models.LinearComb import Linear_comb
from common.datautils import *


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    # losses = faiss.vector_to_array(clus.obj)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = data

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    return ReassignedDataset(image_indexes, pseudolabels, dataset)


class Deepcluster(nn.Module):
    def __init__(self, name=None, in_chans=None, K=9,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.name = name
        self.K = K
        # self.max_len = max_size ** 2
        # TODO: encoder
        self.patch_embed = PixelEmbed(in_channels=in_chans, embed_dim=encoder_embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = PosCNN(in_chans=encoder_embed_dim, embed_dim=encoder_embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len+1, encoder_embed_dim),
        #                               requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)
        self.linear_comb = Linear_comb(embed_dim=encoder_embed_dim)

        # --------------------------------------------------------------------------------------------------------------
        # TODO: modify decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = PosCNN(in_chans=decoder_embed_dim, embed_dim=decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        # self.decoder_blocks.append(Block(30, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer))
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True)
        # --------------------------------------------------------------------------------------------------------------
        mlp_hidden_dim = int(in_chans * mlp_ratio)
        self.mlp1 = Mlp(in_features=in_chans, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.mlp2 = Mlp(in_features=in_chans, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.head = nn.Linear(encoder_embed_dim, K)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def encoder_forward(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def decoder_forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        '''
        线性embedding后，应该将其和可学习的mask tokens堆叠到一起，形成完整的token。
        然后进行unshuffle
        接着将cls token加到开头，输入trm网络
        最后进行一次层归一化和线性层输出结果
        '''
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  # learnable vector
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x_ = self.decoder_pos_embed(x_)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        y = x[:, 1:, :]
        y_center = x[:, :1, :]
        return y, y_center

    def forward_loss(self, imgs, pred, pred_center, mask=None):
        """
        imgs: [N,C, H, W]
        pred: [N, H*W, C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = imgs.squeeze().reshape(imgs.shape[0], imgs.shape[1], -1).transpose(-1, -2)  # N,L,D
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_r = (pred - target) ** 2
        loss_r = loss_r.sum(-1).mean()  # [N, L], mean loss per pixel

        pred_center = self.mlp1(pred_center.squeeze())
        target_center = self.mlp2(target[:, target.shape[1] // 2, :])
        loss_center = (pred_center - target_center) ** 2
        loss_center = loss_center.sum(-1).mean()
        # index = pred.shape[1]//2
        # loss_center = (pred[:, index] - target[:, index])**2
        # loss_center = loss_center.mean(dim=-1, keepdim=True)

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss_all = loss_r + loss_center
        return loss_all, loss_r.detach(), loss_center.detach()

    def compute_features(self, dataloader, N):
        with torch.no_grad():
            for i, (input_tensor, _) in enumerate(dataloader):
                input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
                aux, _, _ = self.encoder_forward(input_var, mask_ratio=0)
                aux = self.linear_comb(aux)
                if i == 0:
                    features = np.zeros((N, aux.shape[1]), dtype='float32')

                aux = aux.astype('float32')
                if i < len(dataloader) - 1:
                    features[i * 3000: (i + 1) * 3000] = aux
                else:
                    # special treatment for final batch
                    features[i * 3000:] = aux
        return features

    def cluster(self, dataloader, dataset):
        clustering = Kmeans(self.k)
        features = self.compute_features(dataloader, len(dataset))
        cluster_loss = clustering.cluster(features)
        pesudo = clustering.images_lists
        train_dataset = cluster_assign(pesudo, dataset.imgs)
        sampler = UnifLabelSampler(len(dataset), pesudo)
        train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler, shuffle=False, pin_memory=True)
        return cluster_loss, train_dataloader

    def reconstructe(self, imgs, mask_ratio):
        latent, mask, ids_restore = self.encoder_forward(imgs, mask_ratio)
        pred, pred_center = self.decoder_forward(latent, ids_restore)
        loss_all, loss_r, loss_center = self.forward_loss(imgs, pred, pred_center)
        pred = pred.reshape(pred.shape[0], int(np.sqrt(pred.shape[1])), int(np.sqrt(pred.shape[1])), -1)
        pred = torch.einsum('nhwc->nchw', pred)
        return pred, mask, loss_all, loss_r, loss_center


if __name__ == '__main__':
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    imp = r'../data/Indian_pines_corrected.mat'
    gtp = r'../data/Indian_pines_gt.mat'
    # imp = r'../data/PaviaU.mat'
    # gtp = r'../data/PaviaU_gt.mat'
    class_num = 16
    pretrain = True
    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=15, sample_mode='ratio', train_ratio=0.1,
                        sample_points=30, merge=None, rmbg=False)
    dataset(spectral=False)
    model = Deepcluster(name='hsimae_15p_204c_sstiny_model', in_chans=200, K=class_num,
                        encoder_embed_dim=256, encoder_depth=4,
                        encoder_num_heads=8,
                        decoder_embed_dim=128, decoder_depth=2,
                        decoder_num_heads=4,
                        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

    init_dataset = HSIDataset(np.concatenate([dataset.x_test_patch, dataset.x_train_patch]), dataset.gt,
                              np.concatenate([dataset.coordinate_test, dataset.coordinate_train]))

    init_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    if pretrain:
        pretrain_dict = torch.load(r'runs/exp1/best.pth')
        curent_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if (k in curent_dict) and (k not in ['patch_embed.conv2d_1.weight'])}
        # hsiclf_15p_204c_stiny_model.load_state_dict(torch.load(r'../runs/exp1/best.pth'), strict=False)
        curent_dict.update(pretrain_dict)
        model.load_state_dict(curent_dict)
