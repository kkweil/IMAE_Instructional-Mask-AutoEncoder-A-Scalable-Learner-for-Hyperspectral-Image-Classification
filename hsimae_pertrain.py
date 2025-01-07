from common.engine_pertrain import *
from common.datautils import GF5Dataset
from torch.utils.data import DataLoader
from models.HsiMAE import hsimae_15p_204c_stiny_model
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import ConcatDataset
from common.tools import BatchSchedulerSampler
import os

# dist.init_process_group(backend="gloo|nccl")

RecordRootPath = r'record/mae_pertrain/'
ModelArchive = r'ModelArchive/mae_pertrain/'

if __name__ == '__main__':
    # imp = r'data/Salinas_corrected.mat'
    # gtp = r'data/Salinas_gt.mat'
    # dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=15, sample_mode='ratio', sample_ratio=0.1,
    #                     sample_points=None, merge=None, rmbg=False)
    # dataset(spectral=False)
    # train_dataloader = DataLoader(dataset=dataset.x_train_patch.astype('float32'),
    #                               batch_size=32,
    #                               shuffle=True,
    #                               num_workers=0,
    #                               )
    # test_dataloader = DataLoader(dataset=dataset.x_test_patch.astype('float32'),
    #                              batch_size=32,
    #                              shuffle=True,
    #                              num_workers=0,
    #                              )


    tr_root = r'data/GF5_patches/train'
    tr_list = os.listdir(tr_root)
    tr_dataset_list = [GF5Dataset(img_root=os.path.join(tr_root, path)) for path in tr_list]
    te_root = r'data/GF5_patches/test'
    te_list = os.listdir(te_root)
    te_dataset_list = [GF5Dataset(img_root=os.path.join(te_root, path)) for path in te_list]
    tr_concat_dataset = ConcatDataset(tr_dataset_list)
    te_concat_dataset = ConcatDataset(te_dataset_list)

    batch_size = 128

    train_dataloader = DataLoader(dataset=tr_concat_dataset,
                                  sampler=BatchSchedulerSampler(dataset=tr_concat_dataset, batch_size=batch_size),
                                  batch_size=batch_size,
                                  shuffle=False)

    test_dataloader = DataLoader(dataset=te_concat_dataset,
                                 sampler=BatchSchedulerSampler(dataset=te_concat_dataset, batch_size=batch_size),
                                 batch_size=batch_size,
                                 shuffle=False)
    model = torch.nn.parallel.DataParallel(hsimae_15p_204c_stiny_model, device_ids=[0, 1], output_device=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    epoch = 100
    train_loss = []
    valid_loss = []
    best_loss = float('inf')

    if not os.path.exists(ModelArchive):
        os.makedirs(ModelArchive)

    for i in range(epoch):
        train_loss.append(simple_trainer(hsimae_15p_204c_stiny_model, train_dataloader, optimizer, i, epoch))
        valid_loss.append(test_loop(hsimae_15p_204c_stiny_model, test_dataloader, i, epoch))
        if valid_loss[i] < best_loss:
            torch.save(hsimae_15p_204c_stiny_model.state_dict(), ModelArchive + f'best.pth')
        torch.save(hsimae_15p_204c_stiny_model.state_dict(), ModelArchive + 'last.pth')
    recoder(Rootpath=RecordRootPath, ModelArchive=ModelArchive, model=hsimae_15p_204c_stiny_model,
            train_loss=train_loss, valid_loss=valid_loss)

    plt.plot(train_loss, label='train_loss')
    plt.plot(valid_loss, label='valid_loss')
    plt.legend()
    plt.show()
    a = 0
