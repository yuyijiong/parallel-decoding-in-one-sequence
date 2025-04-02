import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer
from modeling_qwen2 import Qwen2ForParaCausalLM
import time
import re
from tqdm import tqdm


additional_prompt = ("The answer format requirement is as follows:"
    "When you need to sequentially handle multiple parallel steps (the steps are individual, for example, analyzing multiple individual documents, planning multiple branches, evaluating multiple aspects) during the reasoning process, you must strictly adhere to the following format: You need to prefix each step with '####', followed by the step's title, and then a colon ':' (an English colon). After all the steps are completed, you need to output '####%%%%', and only then can you proceed with the subsequent reasoning process."
    "Example 1:\n"
    "Question: [Resumes of A, B, C, D] Please analyze which of the four individuals best meets the requirements.\n"
    "Answer: Let us analyze the strengths of each person based on their resumes.\n####Strengths of A:......\n####Strengths of B:......\n####Strengths of C:......\n####Strengths of D:...... \n####%%%%Therefore, I believe that A's resume best meets the requirements.\n"
    "Example 2:\n"
    "Question: [document 1,2,3,4] Based on the information in the documents, what is the birthday of Jack?\n"
    "Answer: Let us analyze each documents.\n####document 1:......\n####document 2:......\n####document 3:......\n####document 4:...... \n####%%%%Therefore, Jack's birthday is 5th, May, 2000.\n"
    "Example 3:\n"
    "Question: Please analyze the development status of China from the aspects of economy, politics, culture, and society.\n"
    "Answer: Let us analyze from four aspects.\n####Economy:......\n####Politics:......\n####Culture:......\n####Society:...... \n####%%%%Therefore, we can conclude that the development status of China is......\n"
    " Note that this example is only used to illustrate the format; the specific content should closely revolve around the Question, ensuring that the analysis process is complete, clear, and well-reasoned."
    "If you see that the part after the colon in a certain step is replaced with an ellipsis, it means that the specific content does not need to be provided for this reasoning step, only the title is required, and you should directly proceed to the next parallel step."
    " Otherwise, give a complete and clear analysis for this step."
    " Please be careful, do not forget any necessary steps, and ensure every reasoning step is complete and clear."
    " Note that only a branch step should start with ####. If it is a stem or general step, you should not add ####."
    )

additional_prompt_zh = ("回答格式要求："
                 "当你在推理过程中需要依次处理多个平行的步骤（平行步骤指的是这些步骤是独立、可并行的，例如，分析多个单独的文档、规划多个分支、评估多个方面）时，必须严格遵照以下格式：你需要在每个步骤前面加上 '####'，然后是步骤的标题，标题后加上冒号 ':'（英文冒号）。最后一个步骤结束后，你需要输出 '####%%%%'，然后才能进行后续的推理过程。"
                 "例子1：\n"
                 "问题：[甲乙丙丁的简历] 请分别分析以上四个人的简历，然后总结出谁最符合要求。\n"
                 "回答：'让我们根据简历依次分析每个人的优劣\n####甲的优劣:......\n####乙的优劣:......\n####丙的优劣:......\n####丁的优劣:...... \n####%%%%因此我认为甲的简历最符合要求。'\n"
                 "例子2：\n"
                 "问题：请从经济、政治、文化、社会四个方面分析中国的发展现状\n"
                 "回答：'让我们从四个方面分别分析\n####经济方面:......\n####政治方面:......\n####文化方面:......\n####社会方面:...... \n####%%%%因此，我们可以得出中国的发展现状是......'\n"

                 "注意此例子仅用于表示格式，具体内容应该紧密围绕具体任务，确保分析过程完整清晰，有理有据。"
                 "如果看到某一步骤冒号后的部分被省略号代替，说明这一步骤不需要给出具体内容，只需要标题，你需要直接进入下一平行步骤。否则，则需要给出完整的步骤"
                 "请注意请勿遗漏任何必要步骤，确保每个推理步骤完整清晰。"
                 "注意只有分支步骤才应该以####开头，如果是主干步骤或一般步骤，不应该加####。"
                 )

def normal_decode(model,tokenizer,prompt):
    full_prompt_with_template=tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)

    input_batch=tokenizer([full_prompt_with_template],padding=True,truncation=True,max_length=10000,return_tensors="pt").to(model.device)

    start_time = time.time()
    output=model.generate(input_ids=input_batch["input_ids"],
                            do_sample=False,
                            max_new_tokens=5000)
    normal_time=time.time()-start_time

    response=tokenizer.batch_decode(output[:,input_batch["input_ids"].size(1):],skip_special_tokens=True)[0]


    #print("Normal Time:",normal_time," length:",len(output[0,input_batch["input_ids"].size(1):]))

    return {"response":response,"time":normal_time,"length":len(output[0,input_batch["input_ids"].size(1):])}

def parallel_decode(model,tokenizer,prompt):
    full_prompt_with_template=tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)

    input_batch=tokenizer([full_prompt_with_template],padding=True,truncation=True,max_length=10000,return_tensors="pt").to(model.device)

    para_begin_token_ids=tokenizer.encode("####")[0]
    para_end_token_ids=tokenizer.encode("%%%%")[0]
    ellipsis_token_ids=tokenizer.encode("......")[0]
    half_ellipsis_token_ids=tokenizer.encode("...")[0]

    start_time = time.time()

    try:
        output_parallel_full=model.generate_with_parallel(
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
                            colon_new_line_token_id=tokenizer.encode(":\n")[0],
                          stage1_max_new_tokens=10000,
                          stage2_max_new_tokens=10000,
                          parallel_decoding_max_length=1000,
                            return_parallel_info=True
                            )

        parallel_time=time.time()-start_time
    except:
        return {"response":"Parallel decoding failed","time":time.time()-start_time,"length":0,"num_para":0,"stage1_response":""}

    output_parallel=output_parallel_full["final_output"]
    num_para=output_parallel_full["num_para"]
    stage1_output=output_parallel_full["stage1_output"]

    response_parallel=tokenizer.batch_decode(output_parallel[:,input_batch["input_ids"].size(1):],skip_special_tokens=True)[0]

    stage1_response=tokenizer.batch_decode(stage1_output[:,input_batch["input_ids"].size(1):],skip_special_tokens=True)[0]

    #print("Parallel Time:",parallel_time," length:",len(output_parallel[0,input_batch["input_ids"].size(1):]))

    return {"response":response_parallel,"time":parallel_time,"length":len(output_parallel[0,input_batch["input_ids"].size(1):]),"num_para":num_para,"stage1_response":stage1_response}


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

        df['full_prompt']=df['prompt'].apply(lambda x: x+"\n"+addition_prompt+"\n"+additional_prompt)

    elif task=="planning":
        df_path= "./datasets/planning/industry_tasks.jsonl"
        df=pd.read_json(df_path, lines=True)

        df['full_prompt']=df['task'].apply(lambda x: "任务："+x+"\n"+additional_prompt_zh)

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

        df['full_prompt']=df['context'].apply(lambda x: x+"\n\n\n\nQuestion: "+ df.loc[i,'input']+"\n"+addition_prompt+"\n"+additional_prompt)


    for i in tqdm(range(len(df))):

        full_prompt=df.loc[i,"full_prompt"]

        #普通解码
        normal_output=normal_decode(model,tokenizer,full_prompt)
        normal_response,normal_time,normal_length=normal_output["response"],normal_output["time"],normal_output["length"]

        df.loc[i,"normal_response"]=normal_response
        df.loc[i,"normal_time"]=normal_time
        df.loc[i,"normal_length"]=normal_length

        #并行解码
        parallel_output=parallel_decode(model,tokenizer,full_prompt)
        parallel_response,parallel_time,parallel_length=parallel_output["response"],parallel_output["time"],parallel_output["length"]
        num_para=parallel_output["num_para"]
        stage1_response=parallel_output["stage1_response"]

        df.loc[i,"parallel_response"]=parallel_response
        df.loc[i,"parallel_time"]=parallel_time
        df.loc[i,"parallel_length"]=parallel_length
        df.loc[i,"num_parallel_branches"]=num_para
        df.loc[i,"stage1_response"]=stage1_response

    save_dir="./results/"+os.path.basename(model_path)+"/"
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,os.path.basename(df_path))
    df.to_json(save_path,orient='records',lines=True)



