import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
import torchvision.transforms as T
from U_net import UNet
import time
from tqdm import tqdm
import os
from torchvision.utils import make_grid

#
# path = r"C:\Users\vigen\PycharmProjects\pythonProject\car_data"
# for j in os.listdir(path):
#     new_path = path + "\\" + j
#     # print(new_path)
#     for i in os.listdir(new_path):
#         new_path_new = new_path + "\\" + i
#         for k in os.listdir(new_path_new):
#             n1_path = new_path_new + "\\" + k
#             os.rename(n1_path, new_path + "\\" + k)

class create_dataset(Dataset):
    def __init__(self, path):
        imgs = []
        for img_name in os.listdir(path):
            full_path = path + "\\" + img_name
            imgs.append(full_path)

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        img = T.PILToTensor()(img)
        img = T.Resize((64, 64))(img)
        img = img.float()
        img = img / 255
        return img




class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1 - self.betas
        self.alpha_heads = torch.cumprod(self.alphas, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_heads[t])[:, None, None, None]
        minus_sqrt_alpha_hat = torch.sqrt(1 - self.alpha_heads[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + minus_sqrt_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n, ))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():

            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.timesteps)):
                t = (torch.ones(n)*i).long().to(self.device)
                e_t = model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alpha_heads[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
                else:
                    noise = torch.zeros((n, 3, self.img_size, self.img_size)).to(self.device)
                x = (1/torch.sqrt(alpha))*(x - e_t * (1 - alpha)/torch.sqrt(1 - alpha_hat)) + noise * torch.sqrt(beta)
        x = (x.clamp(-1, 1) + 1)/2
        x = (x*255).type(torch.uint8)
        model.train()
        return x


def train():
    torch.cuda.empty_cache()
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:22"

    batch_size_train = 9
    batch_size_test = 9

    train_dataset = create_dataset(r"car_data\car_data\train")
    # test_dataset = create_dataset(r"C:\Users\vigen\PycharmProjects\pythonProject\car_data\test")

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                                drop_last=True, num_workers=8)
    # test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, drop_last=True)
    device = torch.device("cuda")
    model = UNet()
    model.load_state_dict(torch.load("model_90"))
    model = model.to(device)

    epochs = 100
    lr = 3e-4
    criterion = torch.nn.MSELoss()
    diffusion = Diffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for num, img in enumerate(train_dataset):
            img = img.to(device)
            t = diffusion.sample_timesteps(batch_size_train).to(device)
            noise_image, noise = diffusion.noise_images(img, t)
            pred = model(noise_image, t)
            loss = criterion(noise, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if num % 100 == 0:
                print(f"Epoch {epoch} Num {num}  Loss {loss}")
        torch.save(model.state_dict(), f"model_{epoch+20}")
        sampling(model, 4, epochs)


def sampling(model, n, epoch):
    diffusion = Diffusion()
    device = torch.device("cuda")
    # model = UNet(device="cuda").to(device)
    # model.load_state_dict(torch.load("model_3"))
    print(1)
    imgs = diffusion.sample(model, n)
    print(2)
    grid = make_grid(imgs, nrow=4)
    print(3)
    grid = T.ToPILImage()(grid)
    grid.show()
    grid.save(f"{epoch}_result1.png")


if __name__ == "__main__":
    train()
