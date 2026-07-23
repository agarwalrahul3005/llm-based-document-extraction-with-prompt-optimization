import dspy

print("DSPy installed successfully!")

print("Version:", dspy.__version__)

haiku_signature = "subject -> haiku"
haiku_generator = dspy.Predict(haiku_signature)
result = haiku_generator(subject="computer science")
print(result.haiku)