# model = AutoModelForCausalLM.from_pretrained(
#     "beomi/Llama-3-Open-Ko-8B",
#     quantization_config=quant_config,
#     device_map={"": 0}
# )
# model.config.use_cache = False
# model.config.pretraining_tp = 1

# tokenizer = AutoTokenizer.from_pretrained(
#               "beomi/Llama-3-Open-Ko-8B",
#               trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# peft_params = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
    
# training_params = TrainingArguments(
#     output_dir="/results",
#     num_train_epochs = 1, #epoch는 1로 설정
#     max_steps=5000, #max_steps을 5000으로 설정
#     # 리소스 제약때문에 batch size를 타협해야하는 경우가 발생 -> micro batch size를 줄이고,
#  	# accumulated step을 늘려, 적절한 size로 gradient를 구해 weight update
#     # https://www.youtube.com/watch?v=ptlmj9Y9iwE
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     optim="paged_adamw_8bit",
#     warmup_steps=0.03,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=100,
#     push_to_hub=False,
#     report_to='tensorboard',
# )

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_params,
#     dataset_text_field="text",
#     max_seq_length=256,
#     tokenizer=tokenizer,
#     args=training_params,
#     packing=False,
# )

# trainer.train()