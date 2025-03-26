import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer
from modeling_qwen2 import Qwen2ForParaCausalLM
import time
import re
from tqdm import tqdm

additional_prompt = (
    "When you need to sequentially handle multiple parallel steps (the steps are individual, for example, analyzing multiple individual documents, planning multiple branches, evaluating multiple aspects) during the reasoning process, you must strictly adhere to the following format: You need to prefix each step with '####', followed by the step's title, and then a colon ':' (an English colon). After all the steps are completed, you need to output '####%%%%', and only then can you proceed with the subsequent reasoning process."
    "Example 1:\n"
    "Question: [Resumes of A, B, C, D] Please analyze which of the four individuals best meets the requirements.\n"
    "Answer: 'Let us analyze the strengths of each person based on their resumes.\n####Strengths of A:......\n####Strengths of B:......\n####Strengths of C:......\n####Strengths of D:...... \n####%%%%Therefore, I believe that A's resume best meets the requirements.'\n"
    "Example 2:\n"
    "Question: [document 1,2,3,4] Based on the information in the documents, what is the birthday of Jack?\n"
    "Answer: 'Let us analyze each documents.\n####document 1:......\n####document 2:......\n####document 3:......\n####document 4:...... \n####%%%%Therefore, Jack's birthday is 5th, May, 2000.'\n"
    " Note that this example is only used to illustrate the format; the specific content should closely revolve around the Question, ensuring that the analysis process is complete, clear, and well-reasoned."
    " Otherwise, give a complete and clear analysis for this step."
    " Please be careful, do not forget any necessary steps, and ensure every reasoning step is complete and clear."
    " Note that only a branch step should start with ####. If it is a stem or general step, you should not add ####."
    )


def normal_decode(model,tokenizer,prompt):
    full_prompt_with_template=tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)

    input_batch=tokenizer([full_prompt_with_template],padding=True,truncation=True,max_length=10000,return_tensors="pt").to(model.device)

    start_time = time.time()
    output=model.generate(input_ids=input_batch["input_ids"],
                            do_sample=False,
                            max_new_tokens=5000)
    response=tokenizer.batch_decode(output[:,input_batch["input_ids"].size(1):],skip_special_tokens=True)[0]
    print(response)
    print("Normal Time:",time.time()-start_time," length:",len(output[0,input_batch["input_ids"].size(1):]))

    return {"response":response,"time":time.time()-start_time,"length":len(output[0,input_batch["input_ids"].size(1):])}

def parallel_decode(model,tokenizer,prompt):
    full_prompt_with_template=tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)

    input_batch=tokenizer([full_prompt_with_template],padding=True,truncation=True,max_length=10000,return_tensors="pt").to(model.device)

    para_begin_token_ids=tokenizer.encode("####")[0]
    para_end_token_ids=tokenizer.encode("%%%%")[0]
    ellipsis_token_ids=tokenizer.encode("......")[0]
    half_ellipsis_token_ids=tokenizer.encode("...")[0]

    start_time = time.time()

    try:
        output_parallel=model.generate_with_parallel(
                                            input_ids=input_batch["input_ids"],
                              attention_mask=None,
                            do_sample=True,
                          use_cache=True,
                          eos_token_id=tokenizer.eos_token_id,
                          para_begin_token_id=para_begin_token_ids,
                          para_end_token_id=para_end_token_ids,
                          ellipsis_token_id=ellipsis_token_ids,
                            half_ellipsis_token_id=half_ellipsis_token_ids,
                          line_break_token_id=tokenizer.encode("\n\n")[0],
                          colon_token_id=tokenizer.encode(":")[0],
                          stage1_max_new_tokens=10000,
                          #stage1_max_length_for_each_step=8,
                          stage2_max_new_tokens=10000,
                          parallel_decoding_max_length=1000,
                            )
    except:
        return {"response":"Parallel decoding failed","time":time.time()-start_time,"length":0}


    response_parallel=tokenizer.batch_decode(output_parallel[:,input_batch["input_ids"].size(1):],skip_special_tokens=True)[0]

    print(response_parallel)
    print("Parallel Time:",time.time()-start_time," length:",len(output_parallel[0,input_batch["input_ids"].size(1):]))

    return {"response":response_parallel,"time":time.time()-start_time,"length":len(output_parallel[0,input_batch["input_ids"].size(1):])}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--task", type=str, default="retrieval")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    args = parser.parse_args()

    model_path = args.model_path
    task = args.task
    attn_implementation = args.attn_implementation


    model = Qwen2ForParaCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 attn_implementation=attn_implementation,
                                                 device_map="auto"
                                                 )
    #model.generate()
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    tokenizer.padding_side = "left"


    if task=="retrieval":
        df_path= "./datasets/student_resume_logic_retrieval/logic_gpa_resume_10.jsonl"#"../检索probing/hard_retrieval_for_llm/logic-based/data_student/logic_gpa_resume_100.jsonl"
        df=pd.read_json(df_path, lines=True)

        addition_prompt="You should check every student to judge whether he meets the requirement in your reasoning process."

    else:
        df_path= "./datasets/multi-doc-qa/2wikimqa.jsonl"
        df=pd.read_json(df_path, lines=True)

        #将context tokenize后统计长度
        df["context_length"]=df["context"].apply(lambda x:len(tokenizer.encode(x)))
        #统计
        print(df["context_length"].describe())
        #删除长度超过8000的
        df=df[df["context_length"]<8000]
        df.reset_index(drop=True,inplace=True)

        addition_prompt="You should check each document one by one (no matter whether it is relevant), and analyze its content to judge whether it provides information about the question. After analysis, output your final answer in the format of 'Answer: your concise answer."


    for i in tqdm(range(len(df))):

        if task=="retrieval":
            prompt=df.loc[i,"prompt"]

        else:
            prompt=df.loc[i,'context']+"\n\n\n\nQuestion: "+ df.loc[i,'input']


        full_prompt=prompt+"\n"+addition_prompt+"\n"+additional_prompt

        #普通解码
        normal_output=normal_decode(model,tokenizer,full_prompt)
        normal_response,normal_time,normal_length=normal_output["response"],normal_output["time"],normal_output["length"]

        df.loc[i,"normal_response"]=normal_response
        df.loc[i,"normal_time"]=normal_time
        df.loc[i,"normal_length"]=normal_length

        #并行解码
        parallel_output=parallel_decode(model,tokenizer,full_prompt)
        parallel_response,parallel_time,parallel_length=parallel_output["response"],parallel_output["time"],parallel_output["length"]

        df.loc[i,"parallel_response"]=parallel_response
        df.loc[i,"parallel_time"]=parallel_time
        df.loc[i,"parallel_length"]=parallel_length


    save_dir="./results/"
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,os.path.basename(df_path))
    df.to_json(save_path,orient='records',lines=True)



