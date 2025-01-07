import faiss
import numpy as np
from torch import nn
import time
from common.datautils import *
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from models.Deepcluster import *
import tqdm

if __name__ == '__main__':
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    imp = r'../data/Indian_pines_corrected.mat'
    gtp = r'../data/Indian_pines_gt.mat'
    # imp = r'../data/PaviaU.mat'
    # gtp = r'../data/PaviaU_gt.mat'
    class_num = 16
    pretrain = True
    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=13, sample_mode='ratio', train_ratio=0.1,
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

    init_dataloader = DataLoader(init_dataset, batch_size=128, shuffle=False, pin_memory=True)
    # check device. move model to GPU
    if torch.cuda.is_available():
        model.cuda()
    if pretrain:
        pretrain_dict = torch.load(r'runs/exp1/best.pth')
        clf_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if (k in clf_dict) and (v.shape == clf_dict[k].shape)}

        clf_dict.update(pretrain_dict)
        model.load_state_dict(clf_dict)

    optimizer_re = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.05)
    optimizer_re.zero_grad()

    criterion = nn.CrossEntropyLoss()
    epoches = 10

    for epoch in range(epoches):
        model.head = None
        mean_loss = 0
        mean_loss_r = 0
        mean_loss_c = 0
        model.train()
        for batch, X in enumerate(init_dataloader):
            # Compute prediction and loss
            X = X[1].cuda()
            _, _, loss, loss_r, loss_c = model.reconstructe(X, 0.5)
            # Backpropagation
            loss.backward()
            optimizer_re.step()
            optimizer_re.zero_grad()
            mean_loss = (mean_loss * batch + loss.item()) / (batch + 1)

            mean_loss_r = (mean_loss_r * batch + loss_r.item()) / (batch + 1)

            mean_loss_c = (mean_loss_c * batch + loss_c.item()) / (batch + 1)

        print(
            f'Epoch[{epoch + 1}/{epoches}] avg_loss:{round(mean_loss, 3)},r_loss:{round(mean_loss_r, 3)},c_loss:{round(mean_loss_c, 3)}')


