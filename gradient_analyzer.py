import torch.nn as nn
from collections import namedtuple, defaultdict
import torch
import pandas
import numpy
from torch.autograd import Variable

class AnalysisHook:
    def __init__(self, key, analyzer):
        Stat = namedtuple('Stat', ['name', 'func'])

        self.stat_functions = []

        def fano_factor(x):
            mean = x.abs().mean(dim=-1, keepdim=True)
            square_mean = x.pow(2).mean(dim=-1, keepdim=True)

            return mean/square_mean - mean

        if analyzer.l1:
            self.stat_functions.append(Stat('l1', lambda x: x.norm(1, dim=-1, keepdim=True)))
        if analyzer.l2:
            self.stat_functions.append(Stat('l2', lambda x: x.norm(2, dim=-1, keepdim=True)))
        if analyzer.mean:
            self.stat_functions.append(Stat('mean', lambda x: x.mean(dim=-1, keepdim=True)))
        if analyzer.magnitude:
            self.stat_functions.append(Stat('magnitude', lambda x: x.abs().max(dim=-1, keepdim=True)[0]))
        if analyzer.range:
            self.stat_functions.append(Stat('range', lambda x: x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]))
        if analyzer.median:
            self.stat_functions.append(Stat('median', lambda x: x.median(dim=-1, keepdim=True)[0]))
        if analyzer.variance:
            self.stat_functions.append(Stat('variance', lambda x: x.var(dim=-1, keepdim=True)))
        if analyzer.fano:
            self.stat_functions.append(Stat('fano', fano_factor))

        self.running_count = analyzer.running_count
        self.running_stats = {}

        self.key = key
        self.vocab_size = analyzer.vocab_size

        self.normalize_gradient = analyzer.normalize_gradient

        self.analyzer = analyzer

    def dummy_hook(self, layer, grad_input, grad_output):
        print(layer)
        for key, parameter in layer.named_parameters():
            print(key, parameter.size())

        print("output")
        print([x.size() if x is not None else None for x in grad_output])
        print("input")
        print([x.size() if x is not None else None for x in grad_input])

    def forward_hook(self, module, input, output):
        for idx, activation in enumerate(input):
            self.store_stats('%d_forward_in' % idx, activation, self.analyzer.sequence)
        for idx, activation in enumerate(output):
            self.store_stats('%d_forward_out' % idx, activation, self.analyzer.sequence)

    def backward_hook(self, module, grad_input, grad_output):
        for idx, gradient in enumerate(grad_input):
            self.store_stats('%d_backward_in' % idx, gradient, self.analyzer.sequence)
        for idx, gradient in enumerate(grad_output):
            self.store_stats('%d_backward_out' % idx, gradient, self.analyzer.sequence)

    def update_layer_stats(self, gradient_key, gradient, sequence):
        if gradient_key not in self.running_stats:
            print(self.key + '_' + gradient_key, gradient.size())
            self.running_stats[gradient_key] = torch.cuda.FloatTensor(self.vocab_size, len(self.stat_functions)).fill_(0)

        new_stats = torch.stack([stat.func(gradient) for stat in self.stat_functions], dim=1)
        # new_stats: (sequence_length * batch_size) x num_stats
        self.running_stats[gradient_key].index_add_(0, sequence, new_stats)

    def update_individual_cell_stats(self, gradient_key, gradient, sequence):
        if gradient_key not in self.cell_stats:
            self.running_cell_stats[gradient_key] = torch.cuda.FloatTensor(self.vocab_size, gradient.size(-1)).fill_(0)

        new_stats = torch.stack([stat.func(gradient) for stat in self.stat_functions], dim=1)
        # new_stats: (sequence_length * batch_size) x num_stats
        self.running_cell_stats[gradient_key].index_add_(0, sequence, gradient)
        self.running_total_cell_stats[gradient_key] += gradient.sum(0)
        self.running_total_count += 0

    def process_gradient(self, gradient, sequence):
        if gradient is None:
            return None

        if type(gradient) is tuple:
            gradient = torch.stack(gradient, dim=0)

        if type(gradient.data) is not torch.cuda.FloatTensor:
            return None

        if gradient.dim() == 3 and (gradient.size(0) * gradient.size(1)) == (sequence.size(0)):
            # gradient: sequence_length x batch_size x hidden_size
            gradient = gradient.view(sequence.size(0), -1)
        elif gradient.size(0) != sequence.size(0):
            return None
        # gradient: (sequence_length * batch_size) x hidden_size

        if self.normalize_gradient:
            gradient = gradient.data.div(gradient.data.norm(2, dim=-1, keepdim=True).expand_as(gradient.data))
        else:
            gradient = gradient.data

        return gradient

    def store_stats(self, gradient_key, gradient, sequence):
        gradient = self.process_gradient(gradient, sequence)
        if gradient is None:
            return

        self.update_layer_stats(gradient_key, gradient, sequence)

    def register_backward_hook(self, module):
        self.backward_handle = module.register_backward_hook(self.backward_hook)

    def register_forward_hook(self, module):
        self.forward_handle = module.register_forward_hook(self.forward_hook)

    def remove_backward_hook(self):
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def remove_forward_hook(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()

    def clear_stats(self):
        self.running_stats = {}

    def stat_key(self, layer, stat):
        return '_'.join([self.key, str(layer), stat.name])

    def serialize_stats(self):
        frame = pandas.DataFrame()
        for i, layer_stats in self.running_stats.items():
            normalized_stats = layer_stats / self.running_count.view(-1, 1).expand_as(layer_stats)
            frame = pandas.concat([frame,
                           pandas.DataFrame(normalized_stats.cpu().numpy(), columns=[self.stat_key(i, stat) for stat in self.stat_functions])],
                          axis=1)
        return frame

class GradientAnalyzer:
    """
    exists so we can add the AnalysisHook onto the layers, and clear them
    out at different points
    hooks are the analysis hook handles
    """
    def __init__(self, model, normalize_gradient=True, use_input_words=False,
                 l1=False, l2=False, mean=False, magnitude=False, range=False, median=False, variance=False, fano=False):
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
        self.fano = fano

        self.normalize_gradient = normalize_gradient
        self.use_input_words = use_input_words

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

    def set_word_sequence(self, sources, targets):
        if self.use_input_words:
            self.sequence = sources.data.view(-1)
        else:
            self.sequence = targets.data.view(-1)
        sequence_length = self.sequence.size(0)

        increment = torch.cuda.FloatTensor([1])
        self.running_count.index_add_(0, self.sequence, increment.expand_as(targets))

    def add_hooks_recursively(self, parent_module: nn.Module, prefix=''):
        # add hooks to the modules in a network recursively
        for module_key, module in parent_module.named_children():
            module_key = prefix + module_key
            output_size = self.module_output_size(module)
            if output_size == 0:
                continue
            self.hooks[module_key] = AnalysisHook(module_key, self)

            self.hooks[module_key].register_forward_hook(module)
            self.hooks[module_key].register_backward_hook(module)
            self.add_hooks_recursively(module, prefix=module_key)

    def add_hooks_to_model(self):
        self.add_hooks_recursively(self.model)

    def remove_hooks(self):
        for key, hook in self.hooks.items():
            hook.remove_forward_hook()
            hook.remove_backward_hook()
            del self.hooks[key]

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
