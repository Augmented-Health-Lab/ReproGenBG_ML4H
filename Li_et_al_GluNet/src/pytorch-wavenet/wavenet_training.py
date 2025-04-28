import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from wavenet_modules import *
import os
from torch.autograd import Variable


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor):
        self.model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.logger = logger
        self.logger.trainer = self
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype
        self.ltype = ltype

    # def train(self,
    #           batch_size=32,
    #           epochs=10,
    #           continue_training_at_step=0):
    #     self.model.train()
    #     self.dataloader = torch.utils.data.DataLoader(self.dataset,
    #                                                   batch_size=batch_size,
    #                                                   shuffle=True,
    #                                                   num_workers=8,
    #                                                   pin_memory=False)
    #     step = continue_training_at_step
    #     for current_epoch in range(epochs):
    #         print("epoch", current_epoch)
    #         tic = time.time()
    #         for (x, target) in iter(self.dataloader):
    #             x = Variable(x.type(self.dtype))
    #             target = Variable(target.view(-1).type(self.ltype))

    #             output = self.model(x)
    #             loss = F.cross_entropy(output.squeeze(), target.squeeze())
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             loss = loss.data[0]

    #             if self.clip is not None:
    #                 torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
    #             self.optimizer.step()
    #             step += 1

    #             # time step duration:
    #             if step == 100:
    #                 toc = time.time()
    #                 print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

    #             if step % self.snapshot_interval == 0:
    #                 if self.snapshot_path is None:
    #                     continue
    #                 time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    #                 torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string)

    #             self.logger.log(step, loss)
    def train(self, batch_size=32, epochs=10, continue_training_at_step=0, dataloader=None):
        self.model.train()
        self.dataloader = dataloader
        step = continue_training_at_step

        for current_epoch in range(epochs):
            print(f"Epoch {current_epoch} start")
            for batch_idx, (x, target) in enumerate(self.dataloader):
                x = Variable(x.type(self.dtype))
                target = Variable(target.type(torch.LongTensor))  # Ensure target is LongTensor

                output = self.model(x)

                loss = F.cross_entropy(output, target)  # No need to squeeze if shapes are correct
                self.optimizer.zero_grad()
                loss.backward()

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()

                step += 1

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is not None:
                        time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                        torch.save(self.model, os.path.join(self.snapshot_path, f'{self.snapshot_name}_{time_string}'))

                self.logger.log(step, loss.item())

                if batch_idx % 10 == 0:
                    print(f"Epoch: {current_epoch} [{batch_idx * len(x)}/{len(self.dataloader.dataset)} "
                        f"({100. * batch_idx / len(self.dataloader):.0f}%)]\tLoss: {loss.item():.6f}")

        return self.model



    def validate(self):
        print("Enter the validation")
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        for (x, target) in iter(self.dataloader):
            x = Variable(x.type(self.dtype))
            target = Variable(target.view(-1).type(self.ltype))

            output = self.model(x)
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            total_loss += loss.item()
            # total_loss += loss.data[0]

            predictions = torch.max(output, 1)[1].view(-1)
            correct_pred = torch.eq(target, predictions)
            accurate_classifications += torch.sum(correct_pred).item()
        # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
        # print("average loss: ", total_loss / len(self.dataloader))
        avg_loss = total_loss / len(self.dataloader)
        
        avg_accuracy = accurate_classifications / (len(self.dataset)*self.dataset.target_length)
        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_accuracy


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    samples = np.stack(samples, axis=0)
    return samples




def test_single_batch(self):
    # Ensure the model is in evaluation mode
    self.model.eval()
    
    # Fetch a single batch of data
    for x, target in self.dataloader:
        # Convert to variables
        x = Variable(x)
        target = Variable(target)

        # Forward pass to get output/logits
        output = self.model(x)

        # Optionally apply softmax for probabilities
        probabilities = torch.softmax(output, dim=1)

        # Print outputs
        print("Output logits:", output)
        print("Probabilities:", probabilities)
        print("Target labels:", target)

        # Break after the first batch to only test one batch
        break



