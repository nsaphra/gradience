import torch.nn as nn
from collections import namedtuple
import torch
import pandas

class AnalysisHook:
    def __init__(self, key, vocab_size, running_count):
        Stat = namedtuple('Stat', ['name', 'func'])

        self.stat_functions = [
            Stat('l1', lambda x: x.norm(1)),
            Stat('l2', lambda x: x.norm(2)),
            Stat('mean', lambda x: x.mean()),
            Stat('magnitude', lambda x: x.abs().max()),
            Stat('range', lambda x: x.max() - x.min()),
            Stat('median', lambda x: x.median())
        ]

        # TODO maybe find a more efficient way to do this with a pytorch buffer
        self.running_count = running_count
        self.running_stats = torch.zeros((vocab_size, len(self.stat_functions)))

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

            for timestep in range(self.sequence_length):
                word_grad = gradient[timestep]
                word_idx = self.word_sequence[timestep]

                old_count = self.running_count[word_idx] - 1

                def update_mean(stat_idx):
                    stat = self.stat_functions[stat_idx]

                    old_stat = self.running_stats[word_idx][stat_idx]
                    new_stat = stat.func(word_grad)

                    return (old_stat * old_count + new_stat.cpu().data) / self.running_count[word_idx]

                num_stats = len(self.stat_functions)
                self.running_stats[word_idx] = torch.cat([update_mean(i) for i in range(num_stats)]).view(1, num_stats)

    def register_hook(self, module):
        self.handle = module.register_backward_hook(self.store_stats)

    def remove_hook(self):
        self.handle.remove()

    def clear_stats(self):
        self.running_stats = torch.zeros((self.vocab_size, len(self.stat_functions)))

    def set_word_sequence(self, input_sequence):
        self.word_sequence = input_sequence
        self.sequence_length = len(input_sequence)

    def serialize_stats(self):
        frame = pandas.DataFrame(self.running_stats.numpy(), columns=[self.key + '_' + stat.name for stat in self.stat_functions])
        return frame

class GradientAnalyzer:
    """
    exists so we can add the AnalysisHook onto the layers, and clear them
    out at different points
    hooks are the analysis hook handles
    """
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        for module in model.modules():
            if type(module) is nn.modules.sparse.Embedding:
                self.vocab_size = module.weight.size(0)
                # assume only one embedding layer
                break
        if self.vocab_size is None:
            raise Exception('no embedding layer')
        self.running_count = torch.zeros(self.vocab_size)

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
            self.hooks[module_key] = AnalysisHook(module_key, self.vocab_size, self.running_count)

            self.hooks[module_key].register_hook(module)
            self.add_hooks_recursively(module, prefix=module_key)

    def set_word_sequence(self, module, input, output):
        sequence = input[0][:,0].cpu().data
        for key, hook in self.hooks.items():
            hook.set_word_sequence(sequence)
        self.running_count.index_add_(0, sequence, torch.ones(len(sequence)))

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
        frame['total_count'] = pandas.Series(self.running_count.numpy(), index=frame.index)

        for key, hook in self.hooks.items():
            frame = pandas.concat([frame, hook.serialize_stats()], axis=1, join_axes=[frame.index])
            hook.clear_stats()
        frame.set_index('word')
        with open(fname, 'w') as file:
            frame.to_csv(file)

        self.running_count = torch.zeros(self.vocab_size)