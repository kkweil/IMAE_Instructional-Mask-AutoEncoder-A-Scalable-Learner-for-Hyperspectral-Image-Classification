import faiss
import numpy as np
from torch import nn
import time
from models.clf import Classfier
from common.datautils import *
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt


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


@torch.no_grad()
def compute_features(dataloader, model, N):
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * 3000: (i + 1) * 3000] = aux
        else:
            # special treatment for final batch
            features[i * 3000:] = aux
    return features


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


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


def train(loader, model, crit, opt):
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.head.parameters(),
        lr=8e-3)
    avg_loss = 0
    for i, (input_tensor, target) in enumerate(loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        avg_loss = (avg_loss * i + loss) / (i + 1)
        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
    return avg_loss


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
    model = Classfier(name='hsimae_15p_204c_stiny_model', in_chans=200, class_num=class_num,
                      encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=8,
                      mlp_ratio=4., norm_layer=nn.LayerNorm)
    test_dataset = HSIDataset(np.concatenate([dataset.x_test_patch, dataset.x_train_patch]), dataset.gt,
                              np.concatenate([dataset.coordinate_test, dataset.coordinate_train]))
    test_dataloader = DataLoader(test_dataset, batch_size=3000, shuffle=False, pin_memory=True)

    if pretrain:
        pretrain_dict = torch.load(r'runs/exp1/best.pth')
        clf_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if (k in clf_dict) and (k not in ['patch_embed.conv2d_1.weight'])}
        # hsiclf_15p_204c_stiny_model.load_state_dict(torch.load(r'../runs/exp1/best.pth'), strict=False)
        clf_dict.update(pretrain_dict)
        model.load_state_dict(clf_dict)
    # check device. move model to GPU
    if torch.cuda.is_available():
        model.cuda()
    fd = int(model.head.weight.size()[1])
    model.head = None
    # model.fc = nn.Sequential(*list(model.fc.children())[:-1])

    optimizer = torch.optim.AdamW([
        {'params': model.patch_embed.parameters(), 'lr': 1e-5},
        {'params': model.pos_embed.parameters(), 'lr': 1e-5},
        {'params': model.blocks.parameters(), 'lr': 1e-5},
        {'params': model.norm.parameters()},
        {'params': model.linear_comb.parameters()},
        # {'params': model.fc.parameters()},
        # {'params': model.head.parameters()},    # move head layer
    ], lr=8e-3)

    criterion = nn.CrossEntropyLoss()

    deepcluster = Kmeans(9)
    loss = []
    cluster_losses = []
    for i in range(200):
        model.head = None
        model.fc = nn.Sequential(*list(model.fc.children())[:-1])

        features = compute_features(test_dataloader, model, len(test_dataset))
        # features = preprocess_features(features, pca=128)

        cluster_loss = deepcluster.cluster(features)

        pesudo = deepcluster.images_lists
        train_dataset = cluster_assign(pesudo, test_dataset.imgs)
        sampler = UnifLabelSampler(len(test_dataset), pesudo)
        train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler, shuffle=False, pin_memory=True)

        # mlp = list(model.fc.children())
        # mlp.append(nn.ReLU(inplace=True).cuda())
        # model.fc = nn.Sequential(*mlp)
        model.head = nn.Linear(fd, len(deepcluster.images_lists))
        model.head.weight.data.normal_(0, 0.01)
        model.head.bias.data.zero_()
        model.head.cuda()

        avgloss = train(train_dataloader, model, criterion, optimizer)
        loss.append(avgloss)
        cluster_losses.append(cluster_loss)
        print(f'Epoch{i+1} CrossEntropyLoss: {avgloss}   Cluster_Loss:{cluster_loss}')


        #  可视化
        pesudo = {idx: itm for idx, itm in enumerate(deepcluster.images_lists)}

        result = np.zeros_like(dataset.gt)

        coordinate = np.concatenate([dataset.coordinate_test, dataset.coordinate_train])
        cluster_labels = []
        for i, v in enumerate(coordinate):
            cluster_label = 0
            for key, value in pesudo.items():
                if i in value:
                    cluster_label = key
                    cluster_labels.append(key)
            result[v[0], v[1]] = cluster_label + 1
        labels = test_dataset.labels

        print('NMI:', normalized_mutual_info_score(labels, cluster_labels))

        # plt.subplot(121)
        # plt.imshow(result, cmap='jet')
        # plt.title('prediction')
        # plt.subplot(122)
        # plt.imshow(dataset.gt, cmap='jet')
        # plt.title('ground_truth')
        # plt.show()

    a = 0
