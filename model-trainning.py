from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import datasets
import sacrebleu

def load_custom_dataset(en_path, pt_path, num_samples=50000):
    en_lines, pt_lines = [], []
    
    with open(en_path, "r", encoding="utf-8") as f_en, open(pt_path, "r", encoding="utf-8") as f_pt:
        for _ in range(num_samples):
            en_line = f_en.readline().strip()
            pt_line = f_pt.readline().strip()
            if not en_line or not pt_line:
                break
            en_lines.append(en_line)
            pt_lines.append(pt_line)
    
    return datasets.Dataset.from_dict({"translation": [{"en": en, "pt": pt} for en, pt in zip(en_lines, pt_lines)]})

en_file = "./datasets/en-pt.txt/Tatoeba.en-pt.en"
pt_file = "./datasets/en-pt.txt/Tatoeba.en-pt.pt"

dataset = load_custom_dataset(en_file, pt_file, num_samples=50000).train_test_split(test_size=0.1)

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")

def tokenize_function(examples):
    inputs = tokenizer([ex["en"] for ex in examples["translation"]], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer([ex["pt"] for ex in examples["translation"]], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./opus-mt-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

model.save_pretrained("./opus-mt-finetuned")
tokenizer.save_pretrained("./opus-mt-finetuned")

print("✅ Fine-tuning concluído! Modelo salvo em './opus-mt-finetuned'")

def compute_bleu(model, tokenizer, dataset, num_samples=100):
    model.eval()
    references, hypotheses = [], []
    for i in range(min(num_samples, len(dataset["test"]))):
        en_text = dataset["test"]["translation"][i]["en"]
        pt_text = dataset["test"]["translation"][i]["pt"]
        inputs = tokenizer(en_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            translated = model.generate(**inputs)
        decoded_translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        references.append([pt_text])
        hypotheses.append(decoded_translation)
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references).score
    print(f"🔹 BLEU Score: {bleu_score}")

compute_bleu(model, tokenizer, dataset)