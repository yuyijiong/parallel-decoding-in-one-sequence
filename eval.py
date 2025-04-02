import pandas as pd
import re
from tqdm import tqdm
from openai import OpenAI
import time


def get_response(prompts,client,model_name="gpt-4o-2024-08-06",max_tokens=512,temperature=0.8):
    answer_list=[]
    max_retry=3
    for prompt in prompts:
        for i in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    timeout=60,

                )
                answer=response.choices[0].message.content

                break
            except Exception as e:
                print(e)
                print("重试第{}次".format(i+1))
                time.sleep(10)
                answer=""

        answer_list.append(answer)

    return answer_list

def judge_answer(question,ref,pred,api_key,api_model="gpt-4o"):
    base_prompt = ("你是一个擅长评价回答的质量的助手。\n请你以公正的评判者的身份，评估一个AI助手对于用户问题的回答的准确性、有用性、清晰度。"
                   "我会提供问题、参考答案和AI助手的回答。你需要评估AI助手的回答的质量，并给出1-5的分数。\n"
                     "1分:回答完全错误，对于问题毫无帮助，或格式混乱，可读性差\n"
                   "2分:回答部分错误，对于问题部分有帮助，或格式比较混乱，可读性差\n"
                    "3分:回答部分正确，对于问题有一定帮助，或格式一般，可读性一般\n"
                    "4分:回答正确，对于问题有帮助，或格式较好，可读性较好\n"
                   "5分:回答完全正确，对于问题非常有帮助，格式工整，可读性好\n"
                "请务必以 “理由：... 分数：...” 的格式给出你的评价。\n"
                "用户的提问： {question}\n"
                "参考答案：{reference}\n"
                "AI助手的回答：{answer}\n")
    prompt = base_prompt.format(question=question, reference=ref, answer=pred)

    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1",max_retries=3)

    judgement=get_response([prompt],client,model_name=api_model,max_tokens=4000,temperature=0)[0]

    #提取 分数： 后面的数字
    judgement=judgement.split("分数：")[-1]
    time.sleep(5)
    try:
        score=re.search(r"\d+",judgement).group()
        print("score:", score)
        return score

    except:
        print(judgement)
        return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, default="./results/logic_gpa_resume_10.jsonl")
    parser.add_argument("--task", type=str, default="retrieval")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_model", type=str, default="gpt-4o")

    args = parser.parse_args()

    df_path = args.model_path
    task = args.task
    attn_implementation = args.attn_implementation
    api_key = args.api_key
    api_model = args.api_model

    df=pd.read_json(df_path,lines=True)


    #统计normal time 平均值
    print("normal time",df["normal_time"].mean())
    print("parallel time",df["parallel_time"].mean())

    #将length除以time得到速度
    df["normal_speed"]=df["normal_length"]/df["normal_time"]
    df["parallel_speed"]=df["parallel_length"]/df["parallel_time"]

    #统计normal speed 平均值
    print("normal speed",df["normal_speed"].mean())
    print("parallel speed",df["parallel_speed"].mean())

    if task=="retrieval":
        #提取response的最后一行作为answer
        df["normal_answer"]=df["normal_response"].apply(lambda x:x.split("\n")[-1])
        df["parallel_answer"]=df["parallel_response"].apply(lambda x:x.split("\n")[-1])
        df['reference'] = df['gold_keys']
        df['question']= df['prompt'].apply(lambda x:x.split("Question: ")[-1])

    elif task=="multi-document-qa":
        df["normal_answer"]=df["normal_response"].apply(lambda x:x.split("\n")[-1])
        df["parallel_answer"]=df["parallel_response"].apply(lambda x:x.split("\n")[-1])
        df['reference'] = df['answers'].apply(lambda x: x[0])
        df['question']= df['input']
    else:
        #提取response的全部作为answer
        df["normal_answer"]=df["normal_response"]
        df["parallel_answer"]=df["parallel_response"]
        df['reference'] = df['answer']
        df['question']= df['task']



    #评估normal_answer和parallel_answer的质量
    for i in tqdm(range(len(df))):
        normal_answer=df.loc[i,"normal_answer"]
        parallel_answer=df.loc[i,"parallel_answer"]
        ref=df.loc[i,"reference"]
        question=df.loc[i,"question"]

        df.loc[i,"normal_score"]=judge_answer(question,ref,normal_answer,api_key,api_model)
        df.loc[i,"parallel_score"]=judge_answer(question,ref,parallel_answer,api_key,api_model)

    #转化为int
    df["normal_score"]=df["normal_score"].astype(int)
    df["parallel_score"]=df["parallel_score"].astype(int)

    #如果normal_score或parallel_score为0，则删除此行
    df=df[df["normal_score"]!=0]
    df=df[df["parallel_score"]!=0]

    print("normal quality score:",df["normal_score"].mean())
    print("parallel quality score:",df["parallel_score"].mean())

    df.to_json(df_path,orient="records",lines=True)
