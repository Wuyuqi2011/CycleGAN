import torch
from torchvision import transforms
from models import GeneratorL2H, GeneratorH2L
from torch.utils.data import DataLoader
from datasets import ImageDataset
import PIL

data_dir = './data/shale'
model_path_hr = './saved_models/netG_A2B_5.pth'
outdir = './outs'

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    netG_A2B = GeneratorL2H(1, 1).to(device)
    netG_A2B.load_state_dict(torch.load(model_path_hr))

    batch_size = 1
    transform = [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))]
    dataloader = DataLoader(ImageDataset(data_dir, transform=transform, unaligned=True),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    # print("hello")
    with torch.no_grad():
        # print(5)
        for i, batch in enumerate(dataloader):
            # print(i,batch)
            # if i == 1:
            #     break
            # print(4)
            real_A = batch['A']
            real_A = real_A.to(device)
            img = netG_A2B(real_A)
            # print(3)
            real_A = (real_A.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            real_A = PIL.Image.fromarray(real_A[0, :, :, 0].cpu().numpy(), 'L').save(f'{outdir}/{i:06d}_lr.png')
            # print('1')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0, :, :, 0].cpu().numpy(), 'L').save(f'{outdir}/hr_{i:06d}.png')
            # print(2)


