import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_discriminator(discriminator, imgs, latent_vector):
    latent_vector = torch.ones(imgs.size()) #hot fix
    # import pdb; pdb.set_trace()
    vector = torch.cat([imgs, latent_vector], 1)
    return discriminator(vector)

def get_MNIST(opt):
    os.makedirs('../../data/mnist', exist_ok=True)
    return torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=opt.batch_size, shuffle=True)

def get_sample_data(generator, imgs, latent_dim):
    return generator(Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))), Variable(imgs)

# def train_discriminator(discriminator, fake_data, real_data=None):
#     if real_data is not None:
#         return discriminator(fake_data), discriminator(real_data)
#     else:
#         return discriminator(fake_data)

def get_loss_discriminator(discriminator, fake_imgs, z, real_imgs, fake_z):
    adversarial_loss = nn.BCELoss()
    # minibatch_size = discriminator_real.size()[0]
    minibatch_size = real_imgs.size()[0]
    valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(minibatch_size, 1).fill_(0.0), requires_grad=False)
    real_loss = adversarial_loss(train_discriminator(discriminator, real_imgs, fake_z), valid)
    fake_loss = adversarial_loss(train_discriminator(discriminator, fake_imgs.detach(), z), fake)
    return (real_loss + fake_loss) / 2

def get_loss_generator(discriminator, fake_imgs, z, real_imgs, fake_z):
    objection = nn.BCELoss()
    minibatch_size = fake_imgs.size()[0]
    # minibatch_size = opt.batch_size
    valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
    valid_prediction = train_discriminator(discriminator, fake_imgs, z)
    import pdb; pdb.set_trace()
    return objection(valid_prediction, valid)

def print_progress(discriminator_loss, generator_loss, iteration, periodic_iteration=10000):
    if iteration % periodic_iteration == 0:
            print("Iteration: {0}; discriminator_loss: {1}; generator_loss: {2}".\
                format(iteration, \
                discriminator_loss.data.numpy(),
                generator_loss.data.numpy()))

def view_generated_data(generator, iteration, num_picture=16, periodic_iteration=10000):
    if iteration % 1000 == 0:
        sample_data = generator.data.numpy()[:num_picture]
        fig = plt.figure(figsize=(4, 4))
        grid = gridspec.GridSpec(4, 4)
        grid.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(sample_data):
            ax = plt.subplot(grid[i])
            plt.axis('off')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            plt.imshow(sample.reshape(28,28), cmap='Greys_r')

        if not os.path.exists('out_emma/'):
            os.makedirs('out_emma/')

        plt.savefig("out_emma/{}.png".format(str(iteration).zfill(3)), bbox_inches='tight')
        plt.close(fig)
