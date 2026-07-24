from optimization.build_dataset import load_examples

examples = load_examples()

print(examples[0])

print(examples[0].inputs())

print(examples[0].labels())