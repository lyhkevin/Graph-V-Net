import torch
import argparse
from utils.train_loader import *
import os
from models.vision_transformer import *
from models.graph_vnet import *
import logging

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr_p', type=float, default=1e-5, help='learning rate of the patch encoder')
        self.parser.add_argument('--lr_g', type=float, default=1e-3, help='learning rate of the Graph V-Net')
        self.parser.add_argument('--min_lr', type=float, default=1e-7, help='learning rate of the Graph V-Net')
        self.parser.add_argument("--batch_size", default=10, type=int)
        self.parser.add_argument("--warmup_epochs", default=2, type=int)
        self.parser.add_argument("--epoch", default=30, type=int)
        self.parser.add_argument('--channels', default=384)
        self.parser.add_argument('--num_class', default=4)
        self.parser.add_argument("--num_workers", default=8, type=int)
        self.parser.add_argument("--patch_size", default=224, type=int)
        self.parser.add_argument("--data_root", default='./dataset/finetune/')
        self.parser.add_argument("--train_samples",default=['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','06','13','16','18','20','test2','test9'])
        self.parser.add_argument("--val_samples", default=['test3','test6','test7','test8'])
        self.parser.add_argument("--test_samples", default=['01','04','05','09','11','12','17','test1','test5'])
        self.parser.add_argument('--weight_save_path', type=str, default='./weight/finetune/')
        self.parser.add_argument('--log_path', type=str, default='./log/finetune.log')
        self.parser.add_argument('--vit_path', type=str, default='./weight/pretrain/checkpoint.pth', help='oversample the regions for each class')
        self.parser.add_argument("--weight_save_interval", default=2)
        self.parser.add_argument("--augment", default=True)
        self.parser.add_argument("--oversample", default=[1,2,2,2], help='oversample the regions for each class')
        
    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
pool = nn.AvgPool2d(kernel_size=2, stride=2)

# get soft labels for coarse-to-fine prediction
def get_labels(label_8): 
    label_8 = rearrange(label_8, 'b h w d -> b d h w')
    label_4 = pool(label_8)
    label_2 = pool(label_4)
    label_1 = pool(label_2)
    label_1 = rearrange(label_1, 'b d h w -> (b h w) d')
    label_2 = rearrange(label_2, 'b d h w -> (b h w) d')
    label_4 = rearrange(label_4, 'b d h w -> (b h w) d')
    label_8 = rearrange(label_8, 'b d h w -> (b h w) d')
    return label_1, label_2, label_4, label_8

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    opt = Options().get_opt()
    os.makedirs(opt.weight_save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, data_loader = get_train_loader(batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.num_workers, opt=opt)
    patch_encoder = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    vnet = Graph_VNet(opt).to(device)
    
    lr_schedule_p = cosine_scheduler(opt.lr_p, opt.min_lr, opt.epoch, len(data_loader),warmup_epochs=opt.warmup_epochs)
    lr_schedule_g = cosine_scheduler(opt.lr_g, opt.min_lr, opt.epoch, len(data_loader),warmup_epochs=opt.warmup_epochs)

    state_dict = torch.load(opt.vit_path, map_location=torch.device(device))
    patch_encoder.load_state_dict(state_dict, strict=False)
    
    logging.basicConfig(filename=opt.log_path,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    #freeze the weight of the first six layers
    for param in patch_encoder.parameters():
        param.requires_grad = False
    for i in range(6, 12):
        for param in patch_encoder.blocks[i].named_parameters():
            param[1].requires_grad = True
            
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    optimizer_p = torch.optim.Adam(patch_encoder.parameters(), lr=opt.lr_p)
    optimizer_g = torch.optim.Adam(vnet.parameters(), lr=opt.lr_g)
    patch_encoder.train()
    vnet.train()

    for epoch in range(0, opt.epoch):
        for i, (img, x, label) in enumerate(data_loader):
            
            it = len(data_loader) * epoch + i  # global training iteration
            for j, param_group in enumerate(optimizer_p.param_groups):
                param_group["lr"] = lr_schedule_p[it]
            for j, param_group in enumerate(optimizer_g.param_groups):
                param_group["lr"] = lr_schedule_g[it]
                
            optimizer_p.zero_grad()
            optimizer_g.zero_grad()
            
            batch_size = x.size()[0]
            x = rearrange(x, 'b n c h w -> (b n) c h w')
            x = x.to(device)
            label = label.to(device)
            label_1, label_2, label_4, label_8 = get_labels(label) # the label for prediction at different graph levels

            latent = patch_encoder(x)
            latent = rearrange(latent, '(b h w) d -> b d h w', b=batch_size, h=8, w=8)
            pred_1, pred_2, pred_4, pred_8 = vnet(latent)

            loss_1 = loss_function(pred_1, label_1)
            loss_2 = loss_function(pred_2, label_2)
            loss_4 = loss_function(pred_4, label_4)
            loss_8 = loss_function(pred_8, label_8)

            loss = loss_1 + loss_2 + loss_4 + loss_8
            loss.backward()
            optimizer_p.step()
            optimizer_g.step()

            print("[Epoch %d/%d] [Batch %d/%d] [loss_1: %f] [loss_2: %f] [loss_4: %f] [loss_8: %f] [lr_p: %f] [lr_r: %f]"
                  % (epoch, opt.epoch, i, len(data_loader), loss_1.item(), loss_2.item(), loss_4.item(), loss_8.item(),
                     get_lr(optimizer_p), get_lr(optimizer_g)))
            
            logging.info("[Epoch %d/%d] [Batch %d/%d] [loss_1: %f] [loss_2: %f] [loss_4: %f] [loss_8: %f] [lr_p: %f] [lr_r: %f]"
                  % (epoch, opt.epoch, i, len(data_loader), loss_1.item(), loss_2.item(), loss_4.item(), loss_8.item(),
                     get_lr(optimizer_p), get_lr(optimizer_g)))
            
        if (epoch + 1) % opt.weight_save_interval == 0:
            torch.save(patch_encoder.state_dict(), opt.weight_save_path + str(epoch + 1) + 'patch_encoder.pth')
            torch.save(vnet.state_dict(), opt.weight_save_path + str(epoch + 1) + 'vnet.pth')
            
    torch.save(patch_encoder.state_dict(), opt.weight_save_path + 'patch_encoder.pth')
    torch.save(vnet.state_dict(), opt.weight_save_path + 'vnet.pth')