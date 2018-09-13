from torch.utils.data import DataLoader
from torch.distributions import Normal

class BatchTrainer:
    def __init__(self, model, dataset, optimizer, device=None):
        self.model_ = model
        self.optimizer = optimizer
        self._device = device if device else torch.device('cpu')

    def __call__(self, images):
        return self.eval_on_batch(images)

    def train_on_batch(self, images, viewpoints):
        self.optimizer.zero_grad() #reset gradients
        evals = self.eval_on_batch(images, viewpoints)
        evals['evidence_lower_bound'].backward()
        self.optimizer.step()
        return evals

    def test_on_batch(self, images, viewpoints):
        self.optimizer.zero_grad() #reset gradients
        return self.eval_on_batch(images, viewpoints)

    def eval_on_batch(self, images, viewpoints):
        images, viewpoints = images.to(self._device), viewpoints.to(self.viewpoints)
        reconstructions, query_viewpoints, representation, kl_div = self.model_(images, viewpoints)
        negative_log = -Normal(reconstructions, sigma).log_prob(query_viewpoints)
        evidence_lower_bound = negative_log + kl_div
        return {'evidence_lower_bound': evidence_lower_bound,
                'kl_divergence': kl_div,
                'negative_log': negative_log,
                'reconstructions': reconstructions,
                'viewpoints': viewpoints,
                'representation': representation}

class ModelTrainer(BatchTrainer):
    def __init__(self, model, dataset, device):
        super(ModelTrainer, self).__init__()
        self.model_ = model
        self.data_loader = DataLoader(dataset, \
            batch_size=args.batch_size, shuffle=True)

        sigma_f, sigma_i = 0.7, 2.0
        mu_f, mu_i = 5*10**(-5), 5*10**(-4)
        mu, sigma = mu_f, sigma_f

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=mu)

        None
    def should_save_image(self, step):
        return step % 1000 == 0

    def should_save_checkpoint(self, step):
        return (step % 10000 == 10000) or (step + 1 == self.gradient_steps)

    def anneal_learning_rate(self):
        # Anneal learning rate
        mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
        self.optimizer.lr = mu * math.sqrt(1 - 0.999**s)/(1 - 0.9**s)

    def anneal_pixel_variance(self):
        sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)

    def train(self, device):
        for step in range(self.gradient_steps):

            for image_batch, viewpoint_batch in tqdm(self.data_loader):
                batch_eval = self.train_on_batch(image_batch, viewpoint_batch)
                if should_save_checkpoint(step):
                    torch.save(self.model, "model-{}-{}.pt".format(step,"0T00T00"))

            with torch.no_grad():
                test_image, test_pointview = next(iter(self.data_loader))
                test_reconstruction, test_representation = self.test_on_batch(test_image, test_pointview)['reconstructions', 'representation']
                test_representation = test_representation.view(-1, 1, 16, 16)
                if should_save_image:
                    save_image(test_reconstruction.float(), "reconstruction.jpg")
                    save_image(test_representation.float(), "representation.jpg")

                self.anneal_learning_rate()
                self.anneal_pixel_variance()
