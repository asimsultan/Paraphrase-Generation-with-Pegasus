
import torch
import argparse
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from utils import get_device, tokenize_data, ParaphraseDataset
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu

def main(model_path, data_path):
    # Load Model and Tokenizer
    model = PegasusForConditionalGeneration.from_pretrained(model_path)
    tokenizer = PegasusTokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = load_dataset('csv', data_files={'validation': data_path})
    tokenized_datasets = dataset.map(lambda x: tokenize_data(tokenizer, x, max_length=128), batched=True)

    # DataLoader
    eval_dataset = ParaphraseDataset(tokenized_datasets['validation'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_bleu_score = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for gen, ref in zip(generated_texts, reference_texts):
                    total_bleu_score += sentence_bleu([ref.split()], gen.split())
                    total_samples += 1

        avg_bleu_score = total_bleu_score / total_samples
        return avg_bleu_score

    # Evaluate
    avg_bleu_score = evaluate(model, eval_loader, device)
    print(f'Average BLEU Score: {avg_bleu_score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
