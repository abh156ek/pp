from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")
print(pipe("The company's stock performance is excellent."))
