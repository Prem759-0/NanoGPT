#!/usr/bin/env python3
"""
scripts/train_assistant.py
Reproducible LoRA / QLoRA Fine-Tuning Pipeline for Sovereign AI Assistant.
Designed for Google Colab (Free T4 / A100 GPU) and local CUDA machines.

Features:
- Memory-efficient 4-bit quantization via Unsloth or standard BitsAndBytes/PEFT
- Native tokenizer.apply_chat_template() formatting (NO manual special token concatenation)
- Train & validation split loss tracking to prevent overfitting
- Checkpointing and resumable adapter saving
- Automated post-training smoke test
- Optional GGUF (q4_k_m) export for fast local CPU deployment
"""
import os
import sys
import json
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check_environment():
    """Checks and logs available compute resources."""
    print("=" * 65)
    print("   SOVEREIGN AI TRAINING ENVIRONMENT DIAGNOSTICS")
    print("=" * 65)
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"GPU Device:      {device_name} ({vram_gb} GB VRAM)")
        bf16 = torch.cuda.is_bf16_supported()
        print(f"BF16 Supported:  {bf16}")
    else:
        print("⚠️ Warning: No CUDA GPU detected. Fine-tuning an LLM requires a GPU (e.g. Google Colab T4).")
    print("=" * 65 + "\n")

def load_split_dataset(dataset_path: str):
    """Loads a split JSON file and prepares it for Dataset format."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}. Run scripts/prepare_dataset.py first.")
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} examples from {os.path.basename(dataset_path)}")
    return records

def train(args):
    check_environment()

    # Try importing Unsloth (fastest fine-tuning for Colab)
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("⚡ Using Unsloth accelerated training engine.")
    except ImportError:
        print("ℹ️ Unsloth not installed. Falling back to standard Hugging Face PEFT + TRL.")

    # Load Train and Validation Splits
    train_records = load_split_dataset(args.train_path)
    val_records = load_split_dataset(args.val_path) if os.path.isfile(args.val_path) else None

    # Step 1: Model & Tokenizer Initialization
    max_seq_length = args.max_seq_length
    print(f"\nLoading base model: {args.base_model} (max_seq_length: {max_seq_length})...")

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if args.load_in_4bit else None

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Format Datasets using Native Chat Template
    from datasets import Dataset

    def format_records(records_list):
        formatted = []
        for r in records_list:
            msgs = r.get("messages")
            if not msgs:
                msgs = [
                    {"role": "user", "content": r.get("instruction", "")},
                    {"role": "assistant", "content": r.get("response", "")}
                ]
            try:
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted.append({"text": text})
            except Exception as e:
                # Fallback if chat template is missing from tokenizer
                inst = msgs[0]["content"] if len(msgs) > 0 else ""
                resp = msgs[1]["content"] if len(msgs) > 1 else ""
                formatted.append({"text": f"User: {inst}\nAssistant: {resp}"})
        return Dataset.from_list(formatted)

    print("Formatting train & validation splits with model chat template...")
    train_dataset = format_records(train_records)
    val_dataset = format_records(val_records) if val_records else None
    print(f"Ready: {len(train_dataset)} train samples, {len(val_dataset) if val_dataset else 0} val samples.")

    # Step 3: Configure SFTTrainer
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=max(5, int(args.max_steps * 0.05)),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=10,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=max(20, args.max_steps // 5) if val_dataset else None,
        save_strategy="steps",
        save_steps=max(50, args.max_steps // 2),
        output_dir=args.output_dir,
        report_to="none",
        seed=args.seed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    # Step 4: Run Fine-Tuning
    print(f"\nStarting fine-tuning for {args.max_steps} steps...")
    train_result = trainer.train()
    print("Fine-tuning completed successfully!")

    # Step 5: Save LoRA Adapter & Tokenizer
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter weights to: {adapter_path}")

    # Step 6: Post-Training Inference Smoke Test
    print("\nRunning Post-Training Smoke Test...")
    test_prompts = [
        "Hi, who created you?",
        "Write a Python function that adds two numbers."
    ]
    if use_unsloth:
        FastLanguageModel.for_inference(model)

    for p in test_prompts:
        msgs = [{"role": "user", "content": p}]
        input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80, temperature=0.3)
        reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  Q: {p}")
        print(f"  A: {reply[:100]}...\n")

    # Step 7: Optional GGUF Export (if Unsloth is used)
    if args.export_gguf:
        if use_unsloth:
            gguf_name = f"{os.path.basename(args.output_dir)}_Q4_K_M"
            print(f"Exporting 4-bit GGUF model: {gguf_name}...")
            try:
                model.save_pretrained_gguf(gguf_name, tokenizer, quantization_method="q4_k_m")
                print(f"✅ Exported GGUF: {gguf_name}.gguf ready for laptop CPU deployment!")
            except Exception as e:
                print(f"⚠️ GGUF export failed: {e}")
        else:
            print("ℹ️ GGUF export requires llama.cpp or Unsloth. Export skipped.")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sovereign Assistant with LoRA/QLoRA.")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--train_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "splits", "train.json"))
    parser.add_argument("--val_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "splits", "val.json"))
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "lora_output"))
    parser.add_argument("--export_gguf", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
