# gradience

gradience is a tool for analyzing a language model during backpropagation. This system uses pytorch hooks so it can be applied to any architecture that contains 1 embedding layer. It automatically attaches hooks for analyzing gradients at each module in the architecture. The hook at each module collects average statistics for each word: l1, l1, farno factor, mean, magnitude, range, and median of the gradient. It supports arbitrary batch sizes, as long as the input and output both have dimensions `batch_size x sequence_length`.

## Usage

```python
from gradient_analyzer import GradientAnalyzer

...

analyzer = GradientAnalyzer(model, l1=True, l2=True, variance=True)
analyzer.add_hooks_to_model()

for (input,  output) in corpus:
   analyzer.set_word_sequence(input, output)
   """training code here"""
   ...
   loss.backward()
   ...

analyzer.compute_and_clear(idx2word, "outfile.csv")
analyzer.remove_hooks()
```
