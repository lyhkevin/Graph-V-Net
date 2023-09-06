from models.vision_transformer import *
from utils.test_loader import *
from tqdm import tqdm
from einops import repeat
from utils.plot import *
from utils.evaluation import *
from models.graph_vnet import *

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--batch_size", default=5, type=int)
        self.parser.add_argument("--mode", default='test')
        self.parser.add_argument("--num_workers", default=4, type=int)
        self.parser.add_argument("--patch_size", default=224, type=int)
        self.parser.add_argument("--data_root", default='../dataset/finetune/')
        self.parser.add_argument("--train_samples",default=['01', '06', '09', '11', '13', '16', '17', 'A02', 'A03', 'A04', 'A05', 'A06', 'A08', 'A10', 'test3', 'test5', 'test7'])
        self.parser.add_argument("--val_samples", default=['12', '19', 'test2', 'test6'])
        self.parser.add_argument("--test_samples", default=['04', '05', '18', '20', 'test1', 'test9', 'A01', 'A07', 'A09'])
        self.parser.add_argument("--subject_id", default=None)
        self.parser.add_argument('--channels', default=384)
        self.parser.add_argument('--num_class', default=4)
        self.parser.add_argument('--vit_path', type=str, default='../weight/finetune/patch_encoder.pth')
        self.parser.add_argument('--vnet_path', type=str, default='../weight/finetune/vnet.pth')
        self.parser.add_argument('--img_save_path', type=str, default='../snapshot/test/')
        self.parser.add_argument("--augment", default=False)
        self.opt = self.parser.parse_args(args=[])

    def get_opt(self):
        return self.opt

if __name__ == '__main__':

    opt = Options().get_opt()
    os.makedirs(opt.img_save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    softmax = nn.Softmax(dim=-1).to(device)
    patch_encoder = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    vnet = Graph_VNet(opt).to(device)
    patch_encoder.load_state_dict(torch.load(opt.vit_path, map_location=torch.device(device)), strict=False)
    vnet.load_state_dict(torch.load(opt.vnet_path, map_location=torch.device(device)), strict=False)
    vnet.eval()
    patch_encoder.eval()

    gts = []
    preds = []

    for subject_id in opt.test_samples:

        opt.subject_id = subject_id
        dataset, data_loader = get_test_loader(batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.num_workers, opt=opt)
        thumbnail, pix_annotation, patch_annotation, pix_prediction, patch_prediction, thumbnail_img = dataset.get_label()
        prediction = np.zeros((patch_annotation.shape[0], patch_annotation.shape[1], 4))

        for i, (img, (x,y)) in enumerate(tqdm(data_loader)):
            batch_size = img.size()[0]
            img = rearrange(img, 'b n c h w -> (b n) c h w')
            img = img.to(device)
            with torch.no_grad():
                latent = patch_encoder(img)
                latent = rearrange(latent, '(b h w) d -> b d h w', b=batch_size, h=8, w=8)
                _,_,_,pred = vnet(latent)
                pred = softmax(pred)
                pred = rearrange(pred, '(b h w) d -> b h w d', b=batch_size, h=8, w=8)
                for j in range(batch_size):
                    pred_region = pred[j].cpu().numpy()
                    patch_prediction[x[j]:x[j] + 8, y[j]:y[j] + 8, :] += pred_region

        row, col = patch_prediction.shape[0], patch_prediction.shape[1]
        patch_prediction = torch.from_numpy(patch_prediction)
        patch_prediction = rearrange(patch_prediction, 'h w c -> c h w')
        patch_prediction = torch.nn.functional.interpolate(patch_prediction.unsqueeze(0), size=(4 * row, 4 * col), mode='bilinear', align_corners=False).squeeze(0)
        patch_prediction = patch_prediction.numpy()
        patch_prediction = np.argmax(patch_prediction, axis=0)
        row, col = pix_annotation.shape[0], pix_annotation.shape[1]

        patch_prediction = repeat(patch_prediction, 'h w -> (h x) (w y)', x=8, y=8)
        pix_prediction[0:0 + patch_prediction.shape[0], 0:0 + patch_prediction.shape[1]] = patch_prediction

        gts.append(pix_annotation)
        preds.append(pix_prediction)

        prediction_img = drawing_mask(pix_prediction, thumbnail)
        annotation_mask = drawing_annotation_mask(pix_annotation, thumbnail)
        prediction_mask = drawing_annotation_mask(pix_prediction, thumbnail)

        thumbnail_img.save(opt.img_save_path + subject_id + '_gt.png')
        prediction_img.save(opt.img_save_path + subject_id + '_prediction_overlayed.png')
        annotation_mask.save(opt.img_save_path + subject_id + '_annotation_mask.png')
        prediction_mask.save(opt.img_save_path + subject_id + '_prediction_mask.png')

    evaluation(preds, gts, opt.img_save_path)