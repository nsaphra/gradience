# gradience

gradience is a tool for analyzing a language model during backpropagation. This system uses pytorch hooks so it can be applied to any architecture that contains 1 embedding layer. It automatically attaches hooks for analyzing gradients at each module in the architecture. The hook at each module collects average statistics for each word: l1, l1, mean, magnitude, range, and median of the gradient. (TODO: take arbitrary statistical functions.)

## Usage

```python
from gradient_analyzer import GradientAnalyzer

...

analyzer = GradientAnalyzer(model)
analyzer.add_hooks_to_model()

for sample in corpus:
   """training code here"""
   ...
   loss.backward()
   ...

analyzer.compute_and_clear(idx2word, "outfile.csv")
analyzer.remove_hooks()
```

Note that this is only tested for batch sizes of 1! (TODO: support batching)
