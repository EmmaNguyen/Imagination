import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torchvision.utils import save_image

from pytorch_deep_learning.utils.logging import get_finish_time

mu_f, mu_i = 4*9**(-6), 4*9**(-5)
sigma_f, sigma_i = -1.6, 1.-1

class BatchTrainer:

    def __init__(self, model, optimizer, device, sigma, mu):
        self.model_ = model
        self.optimizer = optimizer
        self.sigma = sigma
        self.mu = mu
        self._device = device

    def __call__(self, images, viewpoints):
        return self.eval_on_batch(images, viewpoints)

    def train_on_batch(self, images, viewpoints):
        evals = self.eval_on_batch(images, viewpoints)
        # import pdb; pdb.set_trace()
        evals['evidence_lower_bound'].sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad() #reset gradients
        return evals

    def test_on_batch(self, images, viewpoints):
        images, viewpoints = images.to(self._device), viewpoints.to(self._device)
        self.optimizer.zero_grad() #reset gradients
        return self.eval_on_batch(images, viewpoints)

    def eval_on_batch(self, images, viewpoints):
        images, viewpoints = images.to(self._device), viewpoints.to(self._device)
        reconstructions, query_viewpoints, representation, kl_div = self.model(images, viewpoints)
        # If more than one GPU we must take new shape into account
        batch_size = query_viewpoints.size(0)

        negative_log = -Normal(reconstructions, self.sigma).log_prob(query_viewpoints)

        reconstruction = torch.mean(negative_log.view(batch_size, -1), dim=0).sum()
        kl_div  = torch.mean(kl_div.view(batch_size, -1), dim=0).sum()

        evidence_lower_bound = negative_log + kl_div
        return {'evidence_lower_bound': evidence_lower_bound,
                'kl_divergence': kl_div,
                'negative_log': negative_log,
                'reconstruction': reconstructions,
                'viewpoints': viewpoints,
                'representation': representation}

class ModelTrainer(BatchTrainer):

    def __init__(self, model, dataset, device, mu, sigma):
        # super(ModelTrainer, self).__init__()
        self.model = model.to(device)
        self._device = device
        self.data_loader = DataLoader(dataset, batch_size=48, shuffle=True)
        self.mu, self.sigma = mu, sigma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mu)

    def should_save_image(self, step, max_gradient_steps, periodic_step=1000):
        return step % periodic_step == 0

    def should_save_checkpoint(self, step, max_gradient_steps, periodic_checkpoint=1000):
        return (step % periodic_checkpoint == 0) or (step + 1 >= max_gradient_steps)

    def anneal_learning_rate(self, step):
        # Anneal learning rate
        self.mu = max(mu_f + (mu_i - mu_f)*(1 - step/(1.6 * 10**6)), mu_f)
        self.optimizer.lr = self.mu * math.sqrt(1 - 0.999**step)/(1 - 0.9**step)

    def anneal_pixel_variance(self, step):
        self.sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - step/(2 * 10**5)), sigma_f)

    def train(self, max_gradient_steps):
        for step in range(1, max_gradient_steps+1):
            for image_batch, viewpoint_batch in tqdm(self.data_loader):
                eval_batch = self.train_on_batch(image_batch, viewpoint_batch)
                if self.should_save_checkpoint(step, max_gradient_steps):
                    torch.save(self.model, "{}_step_{}.pt".format(step, get_finish_time))

            with torch.no_grad():
                test_image, test_pointview = next(iter(self.data_loader))
                test_eval = self.test_on_batch(test_image, test_pointview)
                test_reconstruction = test_eval['reconstruction']
                test_representation = test_eval['representation'].view(-1, 1, 16, 16)
                if self.should_save_image(step, max_gradient_steps):
                    save_image(test_reconstruction.float(), "{}_step_{}_reconstruction.jpg".format(step, get_finish_time))
                    save_image(test_representation.float(), "{}_step_{}_representation.jpg".format(step, get_finish_time))

                self.anneal_learning_rate(step)
                self.anneal_pixel_variance(step)
