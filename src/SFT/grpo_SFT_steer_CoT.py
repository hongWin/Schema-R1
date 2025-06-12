from trl import SFTTrainer
from datasets import load_dataset, Dataset
import json
import pandas as pd
import pyarrow.parquet as pq
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
import pyarrow.parquet as pq
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
# 该代码多卡SFT 设置 cuda visiable
import shutil
from pathlib import Path
from transformers import TrainingArguments
from swanlab.integration.transformers import SwanLabCallback
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"







file_path = "src/SFT_Steer/steer_CoT.xlsx"
# model_path = "/home/LLM/models/Qwen2.5-0.5B-Instruct"
model_path = "/home/LLM/models/Qwen2.5-0.5B-Instruct"

if not model_path:
    raise ValueError("model_path  from  src/SFT_Steer/grpo_SFT_steer_CoT.py 不能为空")

df = pd.read_excel(file_path)
print(df)



def make_conversation(user_prompt, assistance_prompt):
    conversation = []
    conversation0 = {}
    conversation1 = {}
    conversation2 = {}
    system_prompt = """
You are an answer evaluation judge. Please evaluate the quality of the response within the <response>...</response> tags based on the question in the <question>...</question> tags, using the following five criteria: helpfulness, correctness, coherence, complexity, and verbosity in <score>...</score>. Follow this response template:
<think>\n Your Thought Process(Why were these scores given?)\n</think>\n<score>\n#helpfulness: \n#correctness: \n#coherence: \n#complexity: \n#verbosity: \n</score> 
"""
    conversation0["role"] = "system"
    conversation0["content"] = system_prompt
    conversation1["role"] = "user"
    conversation1["content"] = user_prompt + "\n"
    conversation2["role"] = "assistant"
    conversation2["content"] = assistance_prompt
    conversation.append(conversation0)
    conversation.append(conversation1)
    conversation.append(conversation2)
#     print(conversation2["content"])
    return conversation


trans_data = {}
message = []
for index, row in df.iterrows():
    conversation = make_conversation(row["question"],row["answer"])
    message.append(conversation)
trans_data["messages"] = message
trans_data = pd.DataFrame(trans_data)
trans_data = Dataset.from_pandas(trans_data,split="train")

output_dir = "src/SFT_model/steer_CoT_SFT_model"
num_train_epochs = 3
bf16 = True
overwrite_output_dir = True
per_device_train_batch_size = 2
gradient_accumulation_steps = 16
gradient_checkpointing = True
evaluation_strategy = "steps"
learning_rate = 5e-5
weight_decay = 0.01
lr_scheduler_type = "cosine"
warmup_ratio = 0.01
max_grad_norm = 0.3
group_by_length = True
auto_find_batch_size = False
save_steps = 50
logging_steps = 5
load_best_model_at_end= False
packing = False
save_total_limit=1
neftune_noise_alpha=5


training_arguments = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs=num_train_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
#     evaluation_strategy=evaluation_strategy,
    max_grad_norm = max_grad_norm,
    auto_find_batch_size = auto_find_batch_size,
    save_total_limit = save_total_limit,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="none",
    neftune_noise_alpha= neftune_noise_alpha
)


swanlab_callback = SwanLabCallback(
    project="Qwen2.5_1.5B_GRPO_steer_CoT",
    experiment_name="Qwen2.5_1.5B_GRPO_steer_CoT",
    description="Qwen2.5_1.5B_GRPO_steer_CoT",
    config={
        "model": "Qwen2.5-1.5B",
        "dataset": "steer_CoT",
    },
)


model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
trainer = SFTTrainer(
    model= model,
    train_dataset=trans_data,
#     peft_config = peft_config,
    args=training_arguments,
    callbacks=[swanlab_callback],
#     params_mask = params_mask
)


trainer.train()

trainer.model.save_pretrained(output_dir)
# 源文件路径
tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
tokenizer_path = os.path.join(model_path, "tokenizer.json")

tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
tokenizer_path = os.path.join(model_path, "tokenizer.json")

# 确保目标目录存在
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 复制文件
try:
    shutil.copy2(tokenizer_config_path, output_dir)
    shutil.copy2(tokenizer_path, output_dir)
    print("文件复制成功！")
except FileNotFoundError as e:
    print(f"错误：找不到源文件 - {e}")
except Exception as e:
    print(f"发生错误：{e}")  # inference
