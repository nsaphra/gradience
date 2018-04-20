import torch.nn as nn
from collections import namedtuple
import torch
import pandas
import numpy
from torch.autograd import Variable

class AnalysisHook:
    def __init__(self, key, vocab_size, running_count,
                 l1=False, l2=False, mean=False, magnitude=False, range=False, median=False, variance=False):
        Stat = namedtuple('Stat', ['name', 'func'])

        self.stat_functions = []

        if l1:
            self.stat_functions.append(Stat('l1', lambda x: x.norm(1, dim=-1, keepdim=True)))
        if l2:
            self.stat_functions.append(Stat('l2', lambda x: x.norm(2, dim=-1, keepdim=True)))
        if mean:
            self.stat_functions.append(Stat('mean', lambda x: x.mean(dim=-1, keepdim=True)))
        if magnitude:
            self.stat_functions.append(Stat('magnitude', lambda x: x.abs().max(dim=-1, keepdim=True)[0]))
        if range:
            self.stat_functions.append(Stat('range', lambda x: x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]))
        if median:
            self.stat_functions.append(Stat('median', lambda x: x.median(dim=-1, keepdim=True)[0]))
        if variance:
            self.stat_functions.append(Stat('variance', lambda x: x.var(dim=-1, keepdim=True)))

        self.running_count = running_count
        self.running_stats = torch.cuda.FloatTensor(vocab_size, len(self.stat_functions)).fill_(0)

        self.key = key
        self.vocab_size = vocab_size

    def dummy_hook(self, layer, grad_input, grad_output):
        print(layer)
        for key, parameter in layer.named_parameters():
            print(key, parameter.size())

        print("output")
        print([x.size() if x is not None else None for x in grad_output])
        print("input")
        print([x.size() if x is not None else None for x in grad_input])

    def store_stats(self, module, grad_input, grad_output):
        for gradient_idx, gradient in enumerate(grad_input):
            if gradient is None or gradient.size(0) != self.sequence_length:
                continue

            new_stats = torch.stack([stat.func(gradient.data).view_as(self.word_sequence) for stat in self.stat_functions], dim=1)
            self.running_stats.index_add_(0, self.word_sequence, new_stats)
            break

    def register_hook(self, module):
        self.handle = module.register_backward_hook(self.store_stats)

    def remove_hook(self):
        self.handle.remove()

    def clear_stats(self):
        self.running_stats = torch.zeros((self.vocab_size, len(self.stat_functions)))

    def set_word_sequence(self, input_sequence, sequence_length):
        self.word_sequence = input_sequence
        self.sequence_length = sequence_length

    def serialize_stats(self):
        self.running_stats /= self.running_count.view(-1, 1).expand_as(self.running_stats)
        frame = pandas.DataFrame(self.running_stats.cpu().numpy(), columns=[self.key + '_' + stat.name for stat in self.stat_functions])
        return frame

class GradientAnalyzer:
    """
    exists so we can add the AnalysisHook onto the layers, and clear them
    out at different points
    hooks are the analysis hook handles
    """
    def __init__(self, model,
                 l1=False, l2=False, mean=False, magnitude=False, range=False, median=False, variance=False):
        self.model = model
        self.hooks = {}
        for module in model.modules():
            if type(module) is nn.modules.sparse.Embedding:
                self.vocab_size = module.weight.size(0)
                # assume only one embedding layer
                break
        if self.vocab_size is None:
            raise Exception('no embedding layer')
        self.running_count = torch.cuda.FloatTensor(self.vocab_size).fill_(0)

        self.l1 = l1
        self.l2 = l2
        self.mean = mean
        self.magnitude = magnitude
        self.range = range
        self.median = median
        self.variance = variance

    @staticmethod
    def module_output_size(module):
        # return the size of the final parameters in the module,
        # or 0 if there are no parameters
        output_size = 0
        for key, parameter in module.named_parameters():
            if key.find('weight') < 0:
                continue
            output_size = parameter.size(-1)
        return output_size

    def add_hooks_recursively(self, parent_module: nn.Module, prefix=''):
        # add hooks to the modules in a network recursively
        for module_key, module in parent_module.named_children():
            module_key = prefix + module_key
            output_size = self.module_output_size(module)
            if output_size == 0:
                continue
            self.hooks[module_key] = AnalysisHook(module_key, self.vocab_size, self.running_count,
            l1=self.l1, l2=self.l2, mean=self.mean, magnitude=self.magnitude, range=self.range, median=self.median, variance=self.variance)

            self.hooks[module_key].register_hook(module)
            self.add_hooks_recursively(module, prefix=module_key)

    def set_word_sequence(self, module, input, output):
        sequence = input[0].data
        unrolled_sequence = sequence.view(-1)
        sequence_length = sequence.size(0)
        increment = torch.cuda.FloatTensor([1])
        for key, hook in self.hooks.items():
            hook.set_word_sequence(unrolled_sequence, sequence_length)
        self.running_count.index_add_(0, unrolled_sequence, increment.expand_as(unrolled_sequence))

    def add_hooks_to_model(self):
        self.add_hooks_recursively(self.model)
        self.model.register_forward_hook(self.set_word_sequence)

    def remove_hooks(self):
        for key, hook in self.hooks.items():
            hook.remove_hook()
            self.hooks[key].remove()

    def compute_and_clear(self, idx2word, fname):
        print('printing final statistics to ', fname)
        frame = pandas.DataFrame(idx2word, columns=['word'])
        frame['total_count'] = pandas.Series(self.running_count.cpu().numpy(), index=frame.index)

        for key, hook in self.hooks.items():
            frame = pandas.concat([frame, hook.serialize_stats()], axis=1, join_axes=[frame.index])
            hook.clear_stats()
        frame.set_index('word')
        with open(fname, 'w') as file:
            frame.to_csv(file, encoding='utf-8')

        self.running_count = torch.cuda.FloatTensor(self.vocab_size).fill_(0)
