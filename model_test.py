import os
import asyncio
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# 设置 DashScope API 密钥（建议使用环境变量）
os.environ["DASHSCOPE_API_KEY"] = "sk-cb534124d0c44e7fb12dcb0271715482"

# 初始化模型
def create_llm():
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=8192,
        extra_body={"enable_thinking": False}
    )

# 提取输入 prompt 的“知识点”、“激活类别”和“用户身份”
def analyze_prompt(prompt):
    llm = create_llm()
    analysis_prompt = PromptTemplate.from_template(
        """
        你是一位智能分析器，擅长从用户输入中提取以下信息：
        1. 知识点：识别用户问题中涉及的具体知识点，如“感冒”、“头疼”等。
        2. 激活类别：判断问题涉及的类别，包括“知识”、“语言”、“认知”、“文化”，可能包含一个或多个类别
                    当输入只有“中医的感冒”这样较为笼统的概念时，同时包含四个内容
                    当输入有“中医的感冒中的语言”明确指向时，指输出其中“语言”一个或几个。
        3. 用户身份：判断用户是“老师”还是“学生”。
        用户的输入是：“{prompt}”
        请返回一个 JSON 对象
        示例如下：
        用户的输入是：我是一名老师，请问如何分析中医的感冒？
        返回：
        {{
            "知识点": "感冒",
            "激活类别": ["知识", "语言", "认知", "文化"],
            "用户身份": "老师" 
        }}
        请你一定只返回json格式！！！！其他什么都不要返回！！！
        """
    )
    messages = [SystemMessage(content=analysis_prompt.format(prompt=prompt))]
    response = llm(messages)
    result = json.loads(response.content)
    knowledge_point = result.get("知识点")
    activated_categories = result.get("激活类别")
    return knowledge_point, activated_categories

# 定制每个类别的 Prompt
def get_category_prompt(category, knowledge_point, knowledge_content=None):
    prompts = {
        "知识": SystemMessage(
            content=f"""
            请你设计一个中医的情景文本，{knowledge_point}文本，不要分点，在一个情景中用较为生活化的语言表达，多一些专业知识
            示例如下：
            感冒
            小李这几天感觉身体不对劲，天气变化大，早上出门没注意添衣，晚上又淋了点雨，第二天就出现了嗓子干痒、鼻子里热热的、轻微头疼的症状。他量了体温，有点低烧，知道自己可能感冒了。
            他妈妈看决定带他去附近的中医诊所看看。老中医先是仔细询问了病情，包括症状出现的时间、性质以及是否有恶寒、发热等感受，并观察了他的面色和舌象。接着通过把脉，了解到小李的脉象浮数，这是典型的外感风热之象。根据中医理论，感冒主要是由于人体正气不足，卫外不固，导致风邪侵袭肺卫所致，而小李的情况属于风热犯肺。
            老中医为小李开具了一副中药方子，主要成分包括金银花（清热解毒）、连翘（疏散风热、清热解毒）、牛蒡子（疏散风热、利咽透疹）、薄荷（疏风散热）、桔梗（宣肺祛痰）等。这些药材组合起来，旨在辛凉解表，清热解毒，以达到驱散外邪、恢复肺卫功能的目的。
            同时，医生还建议小李多喝温开水，保持室内空气流通，避免再次受寒。此外，老中医提到可以适量食用一些具有润肺止咳作用的食物，如梨子，可以帮助缓解咳嗽和喉咙不适。回家后，小李妈妈按照医嘱给他熬制了药汤，并且煮了一碗姜糖水，帮助发汗解表。
            经过几天的调养，小李的症状逐渐减轻，体力也慢慢恢复。
            """
            ),
        "语言": SystemMessage(
            content=f"""
            汉语语言教学专家，擅长处理与“{knowledge_point}”相关的汉语知识点问题。根据我给的这个内容进行语言点生成提取：{knowledge_content}
            要求：进行外国人学习汉语的语言点提取，配合简单的英文解读，要求有一般生词、专名术语、语法示例等
            示例如下：
            一般生词与短语
            添衣 (tiān yī) - Add clothes. To put on more clothes due to cold weather.
            淋了点雨 (lín le diǎn yǔ) - Got caught in a little rain. To get wet by rain slightly.
            嗓子干痒 (sǎng zi gàn yǎng) - Sore and itchy throat. A condition of the throat that feels dry and itchy.
            鼻子里热热的 (bí zi li rè rè de) - Nose feels hot. It refers to the sensation of having a warm or feverish nose.
            头疼 (tóu téng) - Headache. Pain in the head.
            把脉 (bǎ mài) - Take the pulse. Traditional Chinese medicine practice of feeling the pulse to diagnose illness.
            风热感冒 (fēng rè gǎn mào) - Wind-heat cold. A type of common cold characterized by symptoms like sore throat, nasal congestion with yellow mucus, etc.
            专名术语
            金银花 (jīn yín huā) - Honeysuckle. Known for its properties to clear heat and detoxify.
            连翘 (lián qiáo) - Forsythia suspensa. Used for clearing heat and resolving toxins.
            牛蒡子 (niú bàng zǐ) - Great burdock seed. Helps in relieving sore throat and promoting eruption of rashes.
            薄荷 (bò he) - Mint. Often used to disperse wind-heat and clear the head.
            桔梗 (jú gěng) - Balloon flower root. Commonly used for its effects on promoting lung function and expelling phlegm.
            迎香穴 (yíng xiāng xué) - Yingxiang acupoint. Located beside the nostrils, used for treating nasal congestion.
            风池穴 (fēng chí xué) - Fengchi acupoint. Found at the back of the neck, beneficial for headache and neck stiffness.
            语法示例
            得……一下
            例子：他妈妈看他蔫蔫的样子，就说：“这得赶紧调理一下，别拖成重感冒。”
            Translation: Seeing him looking listless, his mother said, "We need to treat this quickly before it turns into a severe cold."
            开了几副……主要是……
            例子：医生开了几副中药，主要是银花、连翘、牛蒡子这些清热解毒的药材。
            Translation: The doctor prescribed several doses of herbal medicine, mainly honeysuckle, forsythia, and great burdock seeds which are good for clearing heat and detoxifying.
            """
            ),
        "认知": SystemMessage(
            content=f"""
            根据我提供的语言点内容，进行对应的中医汉语语言点出题，以下是语言点内容：{knowledge_content}
            仿照以下，做到形式上的完美模仿：
            #一、汉字书写填空
            说明：根据拼音填写正确的汉字。
            他因为生病了，所以买了些中药___热解___。
            中医看病的时候会给病人___脉。
            他的鼻子里热热的，可能是___感冒。
            你最近睡觉前用___草泡脚吗？
            这个药方里有___翘和金___花。
            #二、词语填空
            说明：从括号中选择合适的词语填入空格。
            小李嗓子干痒，医生___他多喝温水，早点休息。（要求 / 建议）
            中医讲究___，不是随便开药。（整体调理 / 辨证论治）
            风池穴在脖子后面，___它可以缓解头痛。（按压 / 按摩）
            吃了清热解毒的中药后，他的症状减轻了。
            吃了清热解毒的中药后，他的___减轻了。（情况 / 症状）
            中医认为感冒是因为外邪侵袭肺卫。
            中医认为感冒是因为___侵袭肺卫。（病菌 / 外邪）
            #三、同义词选择
            说明：选出与划线词意思最相近的一项。
            医生开了几副中药，主要是银花、连翘等药材。
            “主要”的意思是：
            A. 很多 B. 最重要的 C. 一部分 D. 全部
            他感觉胃里暖洋洋的，整个人都轻松了。
            “轻松”的意思是：
            A. 快乐 B. 不舒服 C. 放松 D. 忙碌
            风热感冒的症状是发烧、喉咙痛、流黄鼻涕。
            “症状”的意思是：
            A. 表现 B. 治疗方法 C. 药物反应 D. 诊断结果
            中医通过把脉和看舌象来判断病情。
            “判断”的意思是：
            A. 知道 B. 决定 C. 分析 D. 记录
            小李这几天身体不舒服，可能感冒了。
            “可能”的意思是：
            A. 一定 B. 或许 C. 绝对 D. 已经
            #四、语法造句
            说明：请用下面的语法结构或词语造一个句子。
            得 + 动词
            例句：这得赶紧调理一下。
            请造句：_________________________________________________________
            ...觉得...
            例句：小李喝完觉得胃里暖洋洋的。
            请造句：_________________________________________________________
            主要 + 是/用来...
            例句：这副药主要是用来清热解毒的。
            请造句：_________________________________________________________
            为了 + 目的
            例句：为了增强抵抗力，妈妈给他泡了艾草脚。
            请造句：_________________________________________________________
            不是...而是...
            例句：中医治疗不是压制症状，而是调和身体。
            请造句：_________________________________________________________
            """
            ),
        "文化": SystemMessage(
            content=f"""
            中华传统文化专家，擅长编写“{knowledge_point}”的相关故事，请你根据中华传统文化相关内容，用生活化语言写一段有理有据、有来源有出处的故事，不要分点以文段形式输出”
            示例如下：
            唐朝那会儿，长安城里有个叫孙思邈的大夫，医术高明，百姓都尊他为“药王”。有一年春天，天气忽冷忽热，不少人都感冒了，咳嗽、发烧、鼻塞、喉咙痛，一时间看病的人络绎不绝。有位刚中了进士的年轻人，本该在家好好歇着准备入朝谢恩，却急着赶路进京，结果一到长安就病倒了，头昏脑胀，嗓子干得像冒烟，鼻子也不通气，连饭都吃不下，只好找人推荐去求诊于孙思邈。
            孙思邈一看他这副模样，先没急着开药，而是让他坐下，闭上眼睛，慢慢呼吸，静了一会儿。接着，他拿起银针，在这位书生的脖子后面和手背上轻轻扎了几针，说是帮他散散风热。然后又让徒弟熬了一碗汤药，里面有金银花、连翘、薄荷这些清热解毒的药材，还特意加了一碗生姜红糖水让他喝下去，说能发汗驱寒。
            孙大夫一边给他调理，一边叮嘱他说：“你现在身子虚，最怕再吹风受凉，这几天要早睡早起，别看书看太晚，吃饭也别贪嘴，心也别太急。”年轻人一一照做，果然三天后症状就轻了不少，五天下来整个人精神多了，走路也有力气了。
            后来他问孙思邈：“为何我一路奔波，偏偏到了京城才病？”孙思邈笑着答道：“你身体本就疲惫，风邪趁你正气虚弱时钻了空子。《黄帝内经》里讲‘正气存内，邪不可干’，只要平时注意保养，饮食有节，起居有常，哪那么容易生病？”
            这位书生听了深受启发，从此不仅自己注重养生，还在后来当官的日子里把这些道理讲给百姓听。他常说：“治病不如防病，强身胜过吃药。”这事虽没有写在史书里，但在民间流传已久，成了中医讲究“扶正祛邪”、“治未病”的一个活生生的例子。
            """
            )
    }
    return [prompts[category], HumanMessage(content=knowledge_point)]

# 异步运行单个模型
async def run_model(llm, messages):
    try:
        response = await llm.ainvoke(messages)
        return response.content
    except Exception as e:
        return f"模型运行失败：{str(e)}"

async def run_parallel_models(knowledge_point, categories):
    llm = create_llm()
    results = {}
    
    # 先运行“知识”模型
    knowledge_messages = get_category_prompt("知识", knowledge_point)
    knowledge_output = await run_model(llm, knowledge_messages)
    results["知识"] = knowledge_output

    # 根据“知识”模型输出运行“语言”模型
    if "语言" in categories:
        language_messages = get_category_prompt("语言", knowledge_point, knowledge_output)
        language_output = await run_model(llm, language_messages)
        results["语言"] = language_output
    else:
        language_output = None

    # 根据“语言”模型输出运行“认知”模型
    if "认知" in categories:
        cognition_messages = get_category_prompt("认知", knowledge_point, language_output)
        cognition_output = await run_model(llm, cognition_messages)
        results["认知"] = cognition_output

    # 其余类别（如“文化”）仍根据“知识”模型输出
    for category in categories:
        if category not in ("知识", "语言", "认知"):
            messages = get_category_prompt(category, knowledge_point, knowledge_output)
            cat_output = await run_model(llm, messages)
            results[category] = cat_output
    return results

# 总模型提炼输出
def summarize_outputs(prompt, category_outputs, activated_categories):
    llm = create_llm()
    selected_outputs = {cat: category_outputs[cat] for cat in activated_categories}

    summary_prompt = PromptTemplate.from_template(
        """
        你是教学案例设计专家，根据我提供的{category_outputs}内容，不改动这里面的内容进行完整输出，生成一份完整的个性化教案：
        包括以下几个部分：
        开头为：个性化教案
        1. 教学目标：根据内容，设定明确的教学目标，包括知识点拆解。
        2. 教学内容：根据内容，完整引用[知识]内容，并对[知识]的内容进行解析。
        3. 教学方法：根据内容，完整引用[语言][认知]内容，要有题目，并进行简单提要。
        4. 教学评价：根据内容，反思整体设计思路。
        5. 教学补充：根据内容，完整引用[文化]内容，并作注释。
        请你整理输出的格式，要求规范美观，并且不遗漏删改所引用的[知识][语言][认知][文化]
        """
    )
    messages = [
        SystemMessage(content=summary_prompt.format(
            prompt=prompt,
            category_outputs="\n".join([f"{cat}: {out}" for cat, out in selected_outputs.items()])
        ))
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"提炼失败：{str(e)}"
