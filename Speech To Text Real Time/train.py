import os
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Set your audio directory and CSV path
audio_dir = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/az/clips"
csv_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/dataset/common_voices_az_azb_sentences.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Add full audio paths
df["audio"] = df["path"].apply(lambda x: os.path.join(audio_dir, x))

# Create Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Cast the audio column
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load model and processor
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Preprocess function
def preprocess(batch):
    audio = batch["audio"]["array"]  # Already a NumPy array
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(preprocess)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-azb-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    predict_with_generate=True,  # recommended for seq2seq
    generation_max_length=225,  # adjust as needed
)
# breakpoint()
# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()
model.save_pretrained("./whisper-azb-finetuned")
processor.save_pretrained("./whisper-azb-finetuned")
