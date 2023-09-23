import argparse
from vision_transformer import *
import torchvision
from torchvision import transforms
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--patch_size", default=16, type=int)
        self.parser.add_argument("--mode", default='train')
        self.parser.add_argument("--img_size", default=(480,480), type=int)
        self.parser.add_argument('--weight_path', type=str, default='../weight/checkpoint.pth')
        self.parser.add_argument('--img_save_path', type=str, default='./attention_maps/')
        self.parser.add_argument("--img_path", default='./imgs/')
        self.parser.add_argument("--checkpoint_key", default="student", type=str, help='Key to use in the checkpoint (example: "teacher")')
        self.opt = self.parser.parse_args(args=[])

    def get_opt(self):
        return self.opt

def main():
    opt = Option().get_opt()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #load the checkpoint (architecture: vit small)
    vit = VisionTransformer(patch_size=opt.patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    state_dict = torch.load(opt.weight_path, map_location=torch.device(device))
    vit.load_state_dict(state_dict,strict=False)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(opt.img_size)
    ])
    img_paths = glob(opt.img_path + '*.png')
    count = 1
    for img_path in img_paths:
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = img_transform(img).unsqueeze(dim=0)
        w_featmap, h_featmap = int(opt.img_size[0] / opt.patch_size), int(opt.img_size[1] / opt.patch_size)
        img = img.to(device)
        feature = vit(img)
        attentions = vit.get_last_selfattention(img)
        nh = attentions.shape[1] # number of head
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        save_path = opt.img_save_path + str(count) + '/'
        os.makedirs(save_path, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=False, scale_each=True),save_path + 'img.png')
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=opt.patch_size, mode="nearest")[0].cpu().detach().numpy()
        for j in range(nh):
            fname = os.path.join(save_path, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
        count += 1

if __name__ == "__main__":
    main()
